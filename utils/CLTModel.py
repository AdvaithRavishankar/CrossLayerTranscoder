import torch
import torch.nn as nn
import os
from typing import Dict, List, Optional
from tqdm import tqdm

from .clt import CrossLayerTranscoderSystem, clt_system_loss

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class CLTModel(nn.Module):
    """
    A wrapper class for adding Cross-Layer Transcoders (CLTs) to a model for circuit tracing.

    This implements the full cross-layer architecture from Anthropic's paper where:
    - Each layer l has features that read from the residual stream at layer l
    - Features from layer l can write to ALL subsequent layers l, l+1, ..., L
    - Output at layer l: y_hat^l = Σ_{l'=1}^{l} W_dec^{l'→ℓ} a^{l'}

    Args:
        model: (nn.Module) The model to wrap

    Example:
        >>> clt_model = CLTModel(vit_model)
        >>> clt_model.add_clt_system(hidden_dim=4096, k=64)
        >>> clt_model.train_clts(dataloader, num_epochs=10)
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._clt_system: Optional[CrossLayerTranscoderSystem] = None
        self._layer_names: List[str] = []
        self._residual_activations: Dict[str, torch.Tensor] = {}
        self._mlp_outputs: Dict[str, torch.Tensor] = {}
        self._clt_features: List[torch.Tensor] = []
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._config: Dict = {}

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
        has_clt = self._clt_system is not None
        num_layers = len(self._layer_names) if has_clt else 0
        return f"CLTModel(model={self.model.__class__.__name__}, num_layers={num_layers}, has_clt={has_clt})"

    # ==================== CLT Methods ====================

    def add_clt_system(
        self,
        hidden_dim: int = 4096,
        k: int = 64,
        device: Optional[torch.device] = None,
        verbose: bool = True
    ) -> CrossLayerTranscoderSystem:
        """
        Add a Cross-Layer Transcoder system spanning all MLP layers.

        This creates the full cross-layer architecture where features from each
        layer can contribute to reconstructing outputs at all subsequent layers.

        Args:
            hidden_dim: Dimension of sparse feature space
            k: Top-k sparsity (number of active features per layer)
            device: Device to place CLT on. If None, uses model's device.
            verbose: If True, prints progress

        Returns:
            The CrossLayerTranscoderSystem instance
        """
        # Clear any existing CLT system
        self.clear_clts()

        self._layer_names = self.list_clt_layers()
        num_layers = len(self._layer_names)

        if num_layers < 2:
            raise ValueError(f"Need at least 2 layers for CLT, found {num_layers}")

        if verbose:
            print(f"Found {num_layers} MLP layers for CLT system")

        if device is None:
            device = next(self.model.parameters()).device

        # Get dimensions from first MLP layer
        first_mlp = self.get_layer(self._layer_names[0])
        input_dim = self._get_input_dim(first_mlp)
        output_dim = self._get_output_dim(first_mlp)

        # Create the cross-layer transcoder system
        self._clt_system = CrossLayerTranscoderSystem(
            num_layers=num_layers,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            k=k
        )
        self._clt_system.to(device)

        self._config = {
            'num_layers': num_layers,
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'k': k
        }

        # Register hooks to capture residual stream and MLP outputs
        self._register_hooks()

        print("=" * 40)
        print("ADDING Cross Layer Transcoder System")
        print("=" * 40)
        if verbose:
            for i, name in enumerate(self._layer_names):
                print(f"  Layer {i}: {name}")
            print(f"\nArchitecture:")
            print(f"  - {num_layers} encoders (one per layer)")
            print(f"  - {num_layers * (num_layers + 1) // 2} decoders (cross-layer connections)")
            print(f"  - Hidden dim: {hidden_dim}, Top-k: {k}")
        print("=" * 40)
        print()

        return self._clt_system

    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        for layer_idx, layer_name in enumerate(self._layer_names):
            mlp_module = self.get_layer(layer_name)

            # Hook to capture input to MLP (residual stream)
            def make_pre_hook(idx, name):
                def hook(module, input):
                    inp = input[0] if isinstance(input, tuple) else input
                    if inp.dim() == 4:  # Conv output: (B, C, H, W)
                        inp = inp.permute(0, 2, 3, 1)  # -> (B, H, W, C)
                    self._residual_activations[name] = inp.detach()
                    return input
                return hook

            # Hook to capture output of MLP
            def make_post_hook(idx, name):
                def hook(module, input, output):
                    out = output
                    if out.dim() == 4:
                        out = out.permute(0, 2, 3, 1)
                    self._mlp_outputs[name] = out.detach()
                    return output
                return hook

            pre_handle = mlp_module.register_forward_pre_hook(make_pre_hook(layer_idx, layer_name))
            post_handle = mlp_module.register_forward_hook(make_post_hook(layer_idx, layer_name))
            self._handles.extend([pre_handle, post_handle])

    def train_clts(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_epochs: int = 10,
        lr: float = 1e-3,
        l1_coef: float = 1e-3,
        sparsity_c: float = 1.0,
        device: Optional[torch.device] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the CLT system using the provided dataloader.

        Args:
            dataloader: DataLoader providing input batches for training.
            num_epochs: Number of training epochs
            lr: Learning rate for optimizer
            l1_coef: λ coefficient for sparsity penalty
            sparsity_c: c hyperparameter in tanh(c * ||W_dec|| * a)
            device: Device to train on. If None, uses model's device.
            verbose: If True, prints training progress

        Returns:
            Dictionary with loss history
        """
        if self._clt_system is None:
            raise ValueError("No CLT system added. Call add_clt_system() first.")

        if device is None:
            device = next(self.model.parameters()).device

        # Freeze base model parameters (only train CLT)
        for param in self.model.parameters():
            param.requires_grad = False

        print("Freezing base model's weights for CLT training")

        # Create optimizer for CLT parameters
        optimizer = torch.optim.Adam(self._clt_system.parameters(), lr=lr)

        total_clt_params = sum(p.numel() for p in self._clt_system.parameters())
        print(f"Set Optimizer for CLT parameters. Total number of CLT params: {total_clt_params}")
        print(f"Training on {device}")
        print()

        # Training loop
        self.model.eval()
        self._clt_system.train()

        loss_history = {
            'total': [],
            'reconstruction': [],
            'sparsity': []
        }

        print("=" * 40)
        print("Training CLT System")
        print("=" * 40)
        print()

        for epoch in range(num_epochs):
            total_epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", disable=not verbose)
            for batch in pbar:
                # Forward pass through base model to capture activations
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

                # Gather residual stream inputs for each layer
                layer_inputs = [
                    self._residual_activations[name]
                    for name in self._layer_names
                ]

                # Gather MLP output targets for each layer
                targets = [
                    self._mlp_outputs[name]
                    for name in self._layer_names
                ]

                # Forward through CLT system
                optimizer.zero_grad()
                result = self._clt_system(layer_inputs, return_features=True)
                outputs = result['outputs']
                all_features = result['features']

                # Store features for analysis
                self._clt_features = [f.detach() for f in all_features]

                # Compute loss
                loss, loss_dict = clt_system_loss(
                    outputs, targets, all_features,
                    self._clt_system,
                    l1_coef=l1_coef,
                    sparsity_c=sparsity_c
                )

                # Backward and optimize
                loss.backward()
                optimizer.step()

                total_epoch_loss += loss.item()
                num_batches += 1

                loss_history['total'].append(loss_dict['total'].item())
                loss_history['reconstruction'].append(loss_dict['reconstruction'].item())
                loss_history['sparsity'].append(loss_dict['sparsity'].item())

                pbar.set_postfix(loss=f"{loss.item():.6f}")

        # Set CLT to eval mode
        self._clt_system.eval()

        if verbose:
            print("Training complete!")

        return loss_history

    def get_clt_system(self) -> Optional[CrossLayerTranscoderSystem]:
        """Get the CLT system."""
        return self._clt_system

    def get_clt_activations(
        self,
        layer_idx: int,
        activation_type: str = "features"
    ) -> Optional[torch.Tensor]:
        """
        Get stored activations from the CLT system.

        Args:
            layer_idx: The layer index (0-indexed)
            activation_type: One of "residual", "mlp_output", "features"

        Returns:
            The requested activation tensor, or None if not available
        """
        if layer_idx >= len(self._layer_names):
            return None

        layer_name = self._layer_names[layer_idx]

        if activation_type == "residual":
            return self._residual_activations.get(layer_name)
        elif activation_type == "mlp_output":
            return self._mlp_outputs.get(layer_name)
        elif activation_type == "features":
            if layer_idx < len(self._clt_features):
                return self._clt_features[layer_idx]
        return None

    def clear_clts(self) -> None:
        """Remove the CLT system and hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._clt_system = None
        self._layer_names = []
        self._residual_activations = {}
        self._mlp_outputs = {}
        self._clt_features = []
        self._config = {}

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
        elif hasattr(module, 'fc2') and hasattr(module.fc2, 'out_features'):
            return module.fc2.out_features
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
        elif hasattr(module, 'fc1') and hasattr(module.fc1, 'in_features'):
            return module.fc1.in_features
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
        if self._clt_system is None:
            raise ValueError("No CLT system added. Call add_clt_system() first.")

        if device is None:
            device = next(self.model.parameters()).device

        self.model.eval()
        self._clt_system.eval()

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

                # Forward through CLT to get features
                layer_inputs = [
                    self._residual_activations[name]
                    for name in self._layer_names
                ]
                result = self._clt_system(layer_inputs, return_features=True)
                all_features = result['features']

                # Collect activation magnitudes for each layer
                batch_size = all_features[0].shape[0]
                batch_activations = torch.zeros(batch_size, len(self._layer_names))

                for layer_idx, features in enumerate(all_features):
                    # Mean activation magnitude per sample
                    if features.dim() == 3:
                        act_magnitude = features.abs().mean(dim=(1, 2))
                    else:
                        act_magnitude = features.abs().mean(dim=1)
                    batch_activations[:, layer_idx] = act_magnitude.cpu()

                all_activations.append(batch_activations)

                if verbose and (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1} batches")

        activations = torch.cat(all_activations, dim=0)
        peak_layers = activations.argmax(dim=1)

        if verbose:
            print(f"Analyzed {activations.shape[0]} samples across {len(self._layer_names)} layers")

        return {
            'activations': activations,
            'layer_names': self._layer_names,
            'peak_layers': peak_layers
        }

    def plot_layer_activation_histogram(
        self,
        dataloader: torch.utils.data.DataLoader = None,
        output_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        figsize: tuple = (12, 6),
        verbose: bool = True,
        result: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Plot a histogram showing which layers are most activated across the dataset.

        Args:
            dataloader: DataLoader providing input batches. Not needed if result is provided.
            output_path: Path to save the figure. If None, displays interactively.
            device: Device to run on. If None, uses model's device.
            figsize: Figure size as (width, height)
            verbose: If True, prints progress
            result: Pre-computed result from analyze_layer_activations or load_clts.
                    If provided, skips recomputation and uses this data directly.

        Returns:
            Dictionary containing activation data (same as analyze_layer_activations)

        Example:
            >>> clt_model.add_clt_system(hidden_dim=4096, k=64)
            >>> clt_model.train_clts(train_loader, num_epochs=10)
            >>> clt_model.plot_layer_activation_histogram(val_loader, "outputs/activations.png")
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required. Install with: pip install matplotlib")

        # Use pre-computed result or analyze activations
        if result is None:
            if dataloader is None:
                raise ValueError("Either dataloader or result must be provided")
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

    def save_clts(self, path: str, result: Optional[Dict] = None, verbose: bool = True):
        """
        Save CLT weights and optionally analysis results.

        Args:
            path: Path to save the checkpoint file (.pt)
            result: Optional analysis result dict from analyze_layer_activations
            verbose: If True, prints save confirmation

        Example:
            >>> clt_model.train_clts(dataloader, num_epochs=10)
            >>> result = clt_model.analyze_layer_activations(dataloader)
            >>> clt_model.save_clts('checkpoint.pt', result=result)
        """
        if self._clt_system is None:
            raise ValueError("No CLT system to save. Call add_clt_system() first.")

        checkpoint = {
            'clt_state_dict': self._clt_system.state_dict(),
            'config': self._config,
            'layer_names': self._layer_names,
        }
        if result is not None:
            checkpoint['analysis_result'] = result

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(checkpoint, path)
        if verbose:
            print(f"Saved CLT checkpoint to {path}")

    def load_clts(self, path: str, verbose: bool = True) -> Optional[Dict]:
        """
        Load CLT weights and optionally analysis results.

        Args:
            path: Path to the checkpoint file (.pt)
            verbose: If True, prints load confirmation

        Returns:
            Analysis result dict if it was saved, otherwise None

        Example:
            >>> clt_model = CLTModel(vit)
            >>> result = clt_model.load_clts('checkpoint.pt')
        """
        checkpoint = torch.load(path, weights_only=False)

        # Recreate CLT system from config
        config = checkpoint['config']
        self._layer_names = checkpoint['layer_names']

        device = next(self.model.parameters()).device

        self._clt_system = CrossLayerTranscoderSystem(
            num_layers=config['num_layers'],
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            k=config['k']
        )
        self._clt_system.load_state_dict(checkpoint['clt_state_dict'])
        self._clt_system.to(device)
        self._config = config

        # Re-register hooks
        self._register_hooks()

        if verbose:
            print(f"Loaded CLT checkpoint from {path}")

        return checkpoint.get('analysis_result', None)
