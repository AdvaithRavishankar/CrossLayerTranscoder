import torch
import torch.nn as nn
import os
from typing import Dict, List, Optional
from tqdm import tqdm

from .clt import CrossLayerTranscoder, CLTConfig, clt_loss

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class CLTModel(nn.Module):
    """
    A wrapper class for adding Cross-Layer Transcoders (CLTs) to a model for circuit tracing.

    Args:
        model: (nn.Module) The model to wrap

    Example:
        >>> clt_model = CLTModel(vit_model)
        >>> clt_model.add_all_clts(hidden_dim=4096, k=64)
        >>> clt_model.train_clts(dataloader, num_epochs=10)
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._clts: Dict[str, CrossLayerTranscoder] = {}
        self._clt_activations: Dict[str, Dict[str, torch.Tensor]] = {}
        self._clt_configs: Dict[str, Dict] = {}
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
        num_clts = len(self._clts)
        return f"CLTModel(model={self.model.__class__.__name__}, num_clts={num_clts})"

    # ==================== CLT Methods ====================

    def add_clt(
        self,
        source_layer: str,
        target_layer: str,
        hidden_dim: int,
        k: int = 64,
        clt: Optional[CrossLayerTranscoder] = None
    ) -> CrossLayerTranscoder:
        """
        Add a Cross-Layer Transcoder between two layers for circuit tracing.

        Args:
            source_layer: Name of the layer to read activations from
            target_layer: Name of the layer whose activations we predict
            hidden_dim: Dimension of sparse feature space
            k: Top-k sparsity (number of active features)
            clt: Pre-trained CLT to use. If None, creates a new one.

        Returns:
            The CrossLayerTranscoder instance
        """
        # Get dimensions from layers
        source_module = self.get_layer(source_layer)
        target_module = self.get_layer(target_layer)

        # Infer dimensions
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
            'target_layer': target_layer
        }
        self._clt_activations[source_layer] = {}

        # Register hook on source layer to capture activations and run CLT
        def source_hook(module, input, output):
            act = output
            if act.dim() == 4:  # Conv output: (B, C, H, W)
                act = act.permute(0, 2, 3, 1)  # -> (B, H, W, C)

            self._clt_activations[source_layer]['source'] = act.detach()

            # Run CLT
            clt_out, features, indices = clt(act, return_features=True)
            self._clt_activations[source_layer]['features'] = features.detach() if features is not None else None
            self._clt_activations[source_layer]['clt_output'] = clt_out.detach()

            return output

        # Register hook on target layer to capture target activations
        def target_hook(module, input):
            inp = input[0] if isinstance(input, tuple) else input
            if inp.dim() == 4:
                inp = inp.permute(0, 2, 3, 1)
            self._clt_activations[source_layer]['target'] = inp.detach()
            return input

        # Add hooks
        source_handle = source_module.register_forward_hook(source_hook)
        self._handles[source_layer] = source_handle

        target_handle = target_module.register_forward_pre_hook(target_hook)
        self._handles[f"{source_layer}_target"] = target_handle

        return clt

    def add_all_clts(
        self,
        hidden_dim: int = 4096,
        k: int = 64,
        device: Optional[torch.device] = None,
        verbose: bool = True
    ) -> Dict[str, CrossLayerTranscoder]:
        """
        Add CLTs between all adjacent layers identified by list_clt_layers().

        Args:
            hidden_dim: Dimension of sparse feature space for all CLTs
            k: Top-k sparsity (number of active features)
            device: Device to place CLTs on. If None, uses model's device.
            verbose: If True, prints progress

        Returns:
            Dictionary mapping source layer names to their CLTs
        """
        layers = self.list_clt_layers()
        if len(layers) < 2:
            raise ValueError(f"Need at least 2 layers for CLT, found {len(layers)}")

        if verbose:
            print(f"Found {len(layers)} layers, creating {len(layers) - 1} CLTs")

        if device is None:
            device = next(self.model.parameters()).device

        # Clear any existing CLTs
        self.clear_clts()
        print("="*40)
        print("ADDING Cross Layer Transcoders (CLT)")
        print("="*40)
        # Create CLTs between adjacent layers
        for i in range(len(layers) - 1):
            source_layer = layers[i]
            target_layer = layers[i + 1]
            clt = self.add_clt(
                source_layer=source_layer,
                target_layer=target_layer,
                hidden_dim=hidden_dim,
                k=k
            )
            clt.to(device)
            if verbose:
                print(f"  Created CLT: {source_layer} -> {target_layer}")

        print("="*40)
        print()

        return self._clts

    def train_clts(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_epochs: int = 10,
        lr: float = 1e-3,
        l1_coef: float = 1e-3,
        device: Optional[torch.device] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train all CLTs using the provided dataloader.

        Args:
            dataloader: DataLoader providing input batches for training.
            num_epochs: Number of training epochs
            lr: Learning rate for optimizer
            l1_coef: L1 sparsity coefficient for loss
            device: Device to train on. If None, uses model's device.
            verbose: If True, prints training progress

        Returns:
            Dictionary mapping source layer names to their loss history
        """
        if not self._clts:
            raise ValueError("No CLTs added. Call add_all_clts() first.")

        if device is None:
            device = next(self.model.parameters()).device

        # Freeze base model parameters (only train CLTs)
        for param in self.model.parameters():
            param.requires_grad = False

        print("Freezing base model's weights for CLT training")

        # Create optimizer for all CLT parameters
        all_clt_params = []
        for clt in self._clts.values():
            all_clt_params.extend(clt.parameters())
        optimizer = torch.optim.Adam(all_clt_params, lr=lr)

        # Training loop
        self.model.eval()  # Keep base model frozen
        for clt in self._clts.values():
            clt.train()

        total_clt_params = sum(p.numel() for p in all_clt_params)

        print(f"Set Optimizer for CLT parameters. Total number of CLT params: {total_clt_params}")
        print(f"Training on {device}")
        print()

        loss_history = {layer: [] for layer in self._clts.keys()}

        print("="*40)
        print("Training CLTs")
        print("="*40)
        print()
        
        for epoch in range(num_epochs):
            total_epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", disable=not verbose)
            for batch in pbar:
                # Handle different batch formats
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    with torch.no_grad():
                        _ = self.model(**batch)
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                    with torch.no_grad():
                        _ = self.model(inputs)
                else:
                    batch = batch.to(device)
                    with torch.no_grad():
                        _ = self.model(batch)

                # Compute loss for each CLT
                optimizer.zero_grad()
                batch_loss = 0.0

                for source_layer, clt in self._clts.items():
                    source_act = self.get_clt_activations(source_layer, "source")
                    target_act = self.get_clt_activations(source_layer, "target")

                    if source_act is None or target_act is None:
                        continue

                    # Forward through CLT
                    clt_output, features, _ = clt(source_act, return_features=True)

                    # Compute loss
                    loss, _ = clt_loss(
                        clt_output, target_act, features,
                        l1_coef=l1_coef
                    )

                    batch_loss += loss
                    loss_history[source_layer].append(loss.item())

                # Backward and optimize
                if batch_loss > 0:
                    batch_loss.backward()
                    optimizer.step()
                    total_epoch_loss += batch_loss.item()
                    num_batches += 1
                    pbar.set_postfix(loss=f"{batch_loss.item():.6f}")

        # Set CLTs to eval mode
        for clt in self._clts.values():
            clt.eval()

        if verbose:
            print("Training complete!")

        return loss_history

    def get_clt(self, source_layer: str) -> Optional[CrossLayerTranscoder]:
        """Get the CLT associated with a source layer."""
        return self._clts.get(source_layer)

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
        layer_acts = self._clt_activations.get(source_layer, {})
        return layer_acts.get(activation_type)

    def remove_clt(self, source_layer: str) -> None:
        """Remove a CLT and its associated hooks."""
        if source_layer in self._clts:
            # Remove hooks
            if source_layer in self._handles:
                self._handles[source_layer].remove()
                del self._handles[source_layer]
            target_key = f"{source_layer}_target"
            if target_key in self._handles:
                self._handles[target_key].remove()
                del self._handles[target_key]

            del self._clts[source_layer]
            del self._clt_configs[source_layer]
            del self._clt_activations[source_layer]

    def clear_clts(self) -> None:
        """Remove all CLTs."""
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
        elif hasattr(module, 'dense') and hasattr(module.dense, 'out_features'):
            return module.dense.out_features
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
        elif hasattr(module, 'dense') and hasattr(module.dense, 'in_features'):
            return module.dense.in_features
        else:
            for child in module.children():
                if isinstance(child, nn.Linear):
                    return child.in_features
            raise ValueError(f"Cannot infer input dimension for {type(module)}")

    def analyze_layer_activations(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: Optional[torch.device] = None,
        verbose: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze which layers are activated for each input in the dataset.

        Runs the dataset through the model and records the mean activation
        magnitude for each CLT layer per input sample.

        Args:
            dataloader: DataLoader providing input batches
            device: Device to run on. If None, uses model's device.
            verbose: If True, prints progress

        Returns:
            Dictionary with:
                - 'activations': Tensor of shape (num_samples, num_layers)
                - 'layer_names': List of layer names
                - 'peak_layers': Tensor of shape (num_samples,) where each value is the
                  layer index with the highest activation for that sample
        """
        if not self._clts:
            raise ValueError("No CLTs added. Call add_all_clts() first.")

        if device is None:
            device = next(self.model.parameters()).device

        self.model.eval()
        for clt in self._clts.values():
            clt.eval()

        layer_names = list(self._clts.keys())
        all_activations = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Handle different batch formats
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    _ = self.model(**batch)
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                    _ = self.model(inputs)
                else:
                    batch = batch.to(device)
                    _ = self.model(batch)

                # Collect activation magnitudes for each layer
                batch_size = batch[0].shape[0] if isinstance(batch, (list, tuple)) else \
                            batch['pixel_values'].shape[0] if isinstance(batch, dict) and 'pixel_values' in batch else \
                            batch.shape[0]

                batch_activations = torch.zeros(batch_size, len(layer_names))

                for layer_idx, source_layer in enumerate(layer_names):
                    features = self.get_clt_activations(source_layer, "features")
                    if features is not None:
                        # Mean activation magnitude per sample
                        # features shape: (batch, seq_len, hidden_dim) or (batch, hidden_dim)
                        if features.dim() == 3:
                            act_magnitude = features.abs().mean(dim=(1, 2))  # (batch,)
                        else:
                            act_magnitude = features.abs().mean(dim=1)  # (batch,)
                        batch_activations[:, layer_idx] = act_magnitude.cpu()

                all_activations.append(batch_activations)

                if verbose and (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1} batches")

        activations = torch.cat(all_activations, dim=0)

        # Compute peak layer for each sample (index of layer with max activation)
        peak_layers = activations.argmax(dim=1)  # (num_samples,)

        if verbose:
            print(f"Analyzed {activations.shape[0]} samples across {len(layer_names)} layers")

        return {
            'activations': activations,
            'layer_names': layer_names,
            'peak_layers': peak_layers
        }

    def plot_layer_activation_histogram(
        self,
        dataloader: torch.utils.data.DataLoader,
        output_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        figsize: tuple = (12, 6),
        verbose: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Plot a histogram showing which layers are most activated across the dataset.

        Args:
            dataloader: DataLoader providing input batches
            output_path: Path to save the figure. If None, displays interactively.
            device: Device to run on. If None, uses model's device.
            figsize: Figure size as (width, height)
            verbose: If True, prints progress

        Returns:
            Dictionary containing activation data (same as analyze_layer_activations)

        Example:
            >>> clt_model.add_all_clts(hidden_dim=4096, k=64)
            >>> clt_model.train_clts(train_loader, num_epochs=10)
            >>> clt_model.plot_layer_activation_histogram(val_loader, "outputs/activations.png")
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required. Install with: pip install matplotlib")

        # Analyze activations
        result = self.analyze_layer_activations(dataloader, device, verbose)
        activations = result['activations']
        layer_names = result['layer_names']

        # Compute mean activation per layer across all samples
        mean_per_layer = activations.mean(dim=0).numpy()
        std_per_layer = activations.std(dim=0).numpy()

        # Create short layer names for display
        short_names = [name.split('.')[-1] + f" ({i})" for i, name in enumerate(layer_names)]

        # Plot histogram
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Bar plot of mean activations per layer
        ax1 = axes[0]
        x = range(len(layer_names))
        ax1.bar(x, mean_per_layer, yerr=std_per_layer, capsize=3, alpha=0.7)
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Mean Activation Magnitude')
        ax1.set_title('Mean CLT Activation by Layer')
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_names, rotation=45, ha='right')

        # Heatmap of activations per sample (subsample if too many)
        ax2 = axes[1]
        max_samples = 100
        if activations.shape[0] > max_samples:
            indices = torch.linspace(0, activations.shape[0] - 1, max_samples).long()
            plot_activations = activations[indices].numpy()
        else:
            plot_activations = activations.numpy()

        im = ax2.imshow(plot_activations.T, aspect='auto', cmap='viridis')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Layer')
        ax2.set_title('Layer Activations per Input')
        ax2.set_yticks(range(len(layer_names)))
        ax2.set_yticklabels(short_names)
        plt.colorbar(im, ax=ax2, label='Activation Magnitude')

        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            if verbose:
                print(f"Saved figure to {output_path}")

        plt.show()

        return result
