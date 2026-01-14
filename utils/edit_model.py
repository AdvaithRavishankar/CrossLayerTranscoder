import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Union

from .clt import CrossLayerTranscoder, CLTConfig, MultiLayerCLT


class EditModel(nn.Module):
    """
    A wrapper class that allows editing/intercepting layers in a wrapped model.

    Args:
        model: (nn.Module) The model to wrap and edit

    Example:
        >>> base_model = torchvision.models.resnet18()
        >>> edit_model = EditModel(base_model)
        >>>
        >>> # Add a hook to modify layer output
        >>> def scale_output(module, input, output):
        ...     return output * 2.0
        >>> edit_model.add_edit("layer1.0.conv1", scale_output)
        >>>
        >>> # Replace a layer entirely
        >>> edit_model.replace_layer("fc", nn.Linear(512, 10))
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._edits: Dict[str, Callable] = {}
        self._handles: Dict[str, torch.utils.hooks.RemovableHandle] = {}

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_layer(self, layer_name: str) -> nn.Module:
        """Get a layer by its dot-separated name."""
        parts = layer_name.split(".")
        module = self.model
        for part in parts:
            module = getattr(module, part)
        return module

    def set_layer(self, layer_name: str, new_layer: nn.Module) -> None:
        """Set a layer by its dot-separated name."""
        parts = layer_name.split(".")
        module = self.model
        for part in parts[:-1]:
            module = getattr(module, part)
        setattr(module, parts[-1], new_layer)

    def add_edit(
        self,
        layer_name: str,
        edit_fn: Callable,
        edit_type: str = "output"
    ) -> None:
        """
        Add an edit hook to a layer.

        Args:
            layer_name: Dot-separated path to the layer (e.g., "layer1.0.conv1")
            edit_fn: Function to apply. For output edits: fn(module, input, output) -> modified_output
                     For input edits: fn(module, input) -> modified_input
            edit_type: "output" for forward hook, "input" for forward pre-hook
        """
        # Remove existing edit if present
        self.remove_edit(layer_name)

        layer = self.get_layer(layer_name)

        if edit_type == "output":
            handle = layer.register_forward_hook(edit_fn)
        elif edit_type == "input":
            handle = layer.register_forward_pre_hook(edit_fn)
        else:
            raise ValueError(f"edit_type must be 'output' or 'input', got {edit_type}")

        self._edits[layer_name] = edit_fn
        self._handles[layer_name] = handle

    def remove_edit(self, layer_name: str) -> None:
        """Remove an edit from a layer."""
        if layer_name in self._handles:
            self._handles[layer_name].remove()
            del self._handles[layer_name]
            del self._edits[layer_name]

    def clear_edits(self) -> None:
        """Remove all edits."""
        for handle in self._handles.values():
            handle.remove()
        self._handles.clear()
        self._edits.clear()

    def replace_layer(self, layer_name: str, new_layer: nn.Module) -> nn.Module:
        """
        Replace a layer with a new layer.

        Args:
            layer_name: Dot-separated path to the layer
            new_layer: The new layer to insert

        Returns:
            The original layer that was replaced
        """
        original = self.get_layer(layer_name)
        self.set_layer(layer_name, new_layer)
        return original

    def list_layers(self) -> Dict[str, nn.Module]:
        """List all named layers in the model."""
        return dict(self.model.named_modules())

    def list_clt_layers(self) -> List[str]:
        """
        List all MLP layers in order that can be used for CLT circuit tracing.

        Identifies layers that are likely MLPs based on common naming conventions
        and structure. Supports various architectures including:
        - timm ViT: blocks.0.mlp, blocks.1.mlp, ...
        - HuggingFace ViT: vit.encoder.layer.0.intermediate, ...
        - GPT-style: transformer.h.0.mlp, ...

        Returns:
            Ordered list of layer names suitable for CLT source/target pairs.

        Example:
            >>> edit_model = EditModel(vit_model)
            >>> mlp_layers = edit_model.list_clt_layers()
            >>> # Add CLT between consecutive layers
            >>> for i in range(len(mlp_layers) - 1):
            ...     clt = edit_model.add_clt(mlp_layers[i], mlp_layers[i+1], hidden_dim=4096)
        """
        clt_layers = []

        for name, module in self.model.named_modules():
            if not name:  # Skip root module
                continue

            # Check naming conventions for MLPs
            name_lower = name.lower()
            name_parts = name_lower.split('.')

            # Common MLP naming patterns across architectures
            mlp_patterns = ['mlp', 'ffn', 'feed_forward', 'intermediate']
            is_mlp_name = any(pattern in name_lower for pattern in mlp_patterns)

            # HuggingFace ViT uses 'intermediate' for the MLP expansion layer
            # and 'output' for the MLP projection - we want 'intermediate'
            is_hf_intermediate = 'intermediate' in name_parts

            # Check structure: has dense layer (HF) or fc1/fc2 (timm)
            has_mlp_structure = (
                hasattr(module, 'fc1') or
                hasattr(module, 'fc2') or
                hasattr(module, 'w1') or
                hasattr(module, 'w2') or
                hasattr(module, 'dense') or  # HuggingFace naming
                (isinstance(module, nn.Sequential) and
                 any(isinstance(child, nn.Linear) for child in module.children()))
            )

            # Skip standalone Linear layers and output projections
            is_leaf_linear = isinstance(module, nn.Linear)
            is_output_layer = 'output' in name_parts and 'intermediate' not in name_parts

            if (is_mlp_name or is_hf_intermediate or has_mlp_structure) and not is_leaf_linear and not is_output_layer:
                # Avoid duplicates (don't add both 'blocks.0.mlp' and 'blocks.0.mlp.fc1')
                is_child_of_existing = any(
                    name.startswith(existing + '.') for existing in clt_layers
                )
                if not is_child_of_existing:
                    # Remove any existing entries that are children of this one
                    clt_layers = [
                        existing for existing in clt_layers
                        if not existing.startswith(name + '.')
                    ]
                    clt_layers.append(name)

        return clt_layers

    def __repr__(self) -> str:
        edits_str = ", ".join(self._edits.keys()) if self._edits else "none"
        return f"EditModel(model={self.model.__class__.__name__}, active_edits=[{edits_str}])"

    # ==================== CLT Integration ====================

    def add_clt(
        self,
        source_layer: str,
        target_layer: str,
        hidden_dim: int,
        k: int = 64,
        clt: Optional[CrossLayerTranscoder] = None,
        collect_activations: bool = True,
        replace_with_reconstruction: bool = False
    ) -> CrossLayerTranscoder:
        """
        Add a Cross-Layer Transcoder between two layers for circuit tracing.

        The CLT intercepts activations from the source layer, encodes them into
        sparse interpretable features, and can optionally replace the target
        layer's input with the reconstruction.

        Args:
            source_layer: Name of the layer to read activations from
            target_layer: Name of the layer whose activations we predict
            hidden_dim: Dimension of sparse feature space
            k: Top-k sparsity (number of active features)
            clt: Pre-trained CLT to use. If None, creates a new one.
            collect_activations: If True, stores activations for training
            replace_with_reconstruction: If True, replaces target input with CLT output

        Returns:
            The CrossLayerTranscoder instance (for training or analysis)

        Example:
            >>> edit_model = EditModel(vit_model)
            >>> clt = edit_model.add_clt(
            ...     source_layer="blocks.0.mlp",
            ...     target_layer="blocks.1.mlp",
            ...     hidden_dim=4096,
            ...     k=64
            ... )
            >>> # Run forward pass to collect activations
            >>> output = edit_model(images)
            >>> # Access stored activations for training
            >>> source_act = edit_model.get_clt_activations("blocks.0.mlp", "source")
            >>> target_act = edit_model.get_clt_activations("blocks.0.mlp", "target")
        """
        # Initialize CLT storage if needed
        if not hasattr(self, '_clts'):
            self._clts: Dict[str, CrossLayerTranscoder] = {}
            self._clt_activations: Dict[str, Dict[str, torch.Tensor]] = {}
            self._clt_configs: Dict[str, Dict] = {}

        # Get dimensions from layers
        source_module = self.get_layer(source_layer)
        target_module = self.get_layer(target_layer)

        # Infer dimensions (works for Linear, Conv2d, etc.)
        source_dim = self._get_output_dim(source_module)
        target_dim = self._get_input_dim(target_module)

        # Create or use provided CLT
        if clt is None:
            config = CLTConfig(
                input_dim=source_dim,
                hidden_dim=hidden_dim,
                output_dim=target_dim,
                num_layers=1,
                k=k
            )
            clt = CrossLayerTranscoder(config)

        # Store CLT and config
        self._clts[source_layer] = clt
        self._clt_configs[source_layer] = {
            'source_layer': source_layer,
            'target_layer': target_layer,
            'collect_activations': collect_activations,
            'replace_with_reconstruction': replace_with_reconstruction
        }
        self._clt_activations[source_layer] = {}

        # Register hook on source layer to capture activations and run CLT
        def source_hook(module, input, output):
            # Flatten if needed (for conv layers)
            act = output
            if act.dim() == 4:  # Conv output: (B, C, H, W)
                act = act.permute(0, 2, 3, 1)  # -> (B, H, W, C)

            if collect_activations:
                self._clt_activations[source_layer]['source'] = act.detach()

            # Run CLT
            clt_out, features, indices = clt(act, return_features=True)
            self._clt_activations[source_layer]['features'] = features.detach() if features is not None else None
            self._clt_activations[source_layer]['clt_output'] = clt_out.detach()

            return output  # Don't modify source output

        # Register hook on target layer to optionally replace input and capture target
        def target_hook(module, input):
            if collect_activations:
                inp = input[0] if isinstance(input, tuple) else input
                if inp.dim() == 4:
                    inp = inp.permute(0, 2, 3, 1)
                self._clt_activations[source_layer]['target'] = inp.detach()

            if replace_with_reconstruction:
                clt_out = self._clt_activations[source_layer].get('clt_output')
                if clt_out is not None:
                    # Reshape back if needed
                    if input[0].dim() == 4:
                        clt_out = clt_out.permute(0, 3, 1, 2)
                    return (clt_out,) + input[1:] if isinstance(input, tuple) else clt_out

            return input

        # Add hooks
        self.add_edit(source_layer, source_hook, edit_type="output")

        # Register target hook directly (can't use add_edit since target_layer != hook key)
        target_module = self.get_layer(target_layer)
        handle = target_module.register_forward_pre_hook(target_hook)
        hook_key = f"{source_layer}_clt_target"
        self._handles[hook_key] = handle
        self._edits[hook_key] = target_hook

        return clt

    def get_clt(self, source_layer: str) -> Optional[CrossLayerTranscoder]:
        """Get the CLT associated with a source layer."""
        if hasattr(self, '_clts'):
            return self._clts.get(source_layer)
        return None

    def get_clt_activations(
        self,
        source_layer: str,
        activation_type: str = "features"
    ) -> Optional[torch.Tensor]:
        """
        Get stored activations from a CLT.

        Args:
            source_layer: The source layer name used when adding the CLT
            activation_type: One of "source", "target", "features", "clt_output"

        Returns:
            The requested activation tensor, or None if not available
        """
        if hasattr(self, '_clt_activations'):
            layer_acts = self._clt_activations.get(source_layer, {})
            return layer_acts.get(activation_type)
        return None

    def remove_clt(self, source_layer: str) -> None:
        """Remove a CLT and its associated hooks."""
        if hasattr(self, '_clts') and source_layer in self._clts:
            self.remove_edit(source_layer)
            self.remove_edit(f"{source_layer}_clt_target")
            del self._clts[source_layer]
            del self._clt_configs[source_layer]
            del self._clt_activations[source_layer]

    def clear_clts(self) -> None:
        """Remove all CLTs."""
        if hasattr(self, '_clts'):
            for source_layer in list(self._clts.keys()):
                self.remove_clt(source_layer)

    def _get_output_dim(self, module: nn.Module) -> int:
        """Infer output dimension of a module."""
        if hasattr(module, 'out_features'):
            return module.out_features
        elif hasattr(module, 'out_channels'):
            return module.out_channels
        elif hasattr(module, 'embed_dim'):
            return module.embed_dim
        elif hasattr(module, 'hidden_size'):
            return module.hidden_size
        # HuggingFace modules with 'dense' sublayer (e.g., ViTIntermediate)
        elif hasattr(module, 'dense') and hasattr(module.dense, 'out_features'):
            return module.dense.out_features
        # Try to find any Linear sublayer and get its output dim
        else:
            for child in module.children():
                if isinstance(child, nn.Linear):
                    return child.out_features
            raise ValueError(f"Cannot infer output dimension for {type(module)}")

    def _get_input_dim(self, module: nn.Module) -> int:
        """Infer input dimension of a module."""
        if hasattr(module, 'in_features'):
            return module.in_features
        elif hasattr(module, 'in_channels'):
            return module.in_channels
        elif hasattr(module, 'embed_dim'):
            return module.embed_dim
        elif hasattr(module, 'hidden_size'):
            return module.hidden_size
        # HuggingFace modules with 'dense' sublayer (e.g., ViTIntermediate)
        elif hasattr(module, 'dense') and hasattr(module.dense, 'in_features'):
            return module.dense.in_features
        # Try to find any Linear sublayer and get its input dim
        else:
            for child in module.children():
                if isinstance(child, nn.Linear):
                    return child.in_features
            raise ValueError(f"Cannot infer input dimension for {type(module)}")
