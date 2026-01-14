import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class CLTConfig:
    """Configuration for Cross-Layer Transcoder."""
    input_dim: int
    hidden_dim: int  # Sparse feature dimension (typically much larger than input_dim)
    output_dim: int
    num_layers: int  # Number of layers this CLT spans
    k: int = 64  # Top-k sparsity for feature selection
    tied_weights: bool = False  # Whether encoder/decoder weights are tied


class CrossLayerTranscoder(nn.Module):
    """
    Cross-Layer Transcoder (CLT) based on Anthropic's Circuit Tracing.
    https://transformer-circuits.pub/2025/attribution-graphs/methods.html

    The CLT learns sparse, interpretable features that explain how information
    flows across layers in a neural network. It consists of:

    1. Encoder: Maps layer activations to sparse feature space
    2. Sparse features: High-dimensional, sparse representation
    3. Decoder: Maps sparse features back to predict next layer activations

    The key insight is that by training transcoders across layers (not just
    within a single layer like SAEs), we can trace how features compose
    and transform through the network.

    Args:
        config: CLTConfig with architecture parameters
    """

    def __init__(self, config: CLTConfig):
        super().__init__()
        self.config = config

        # Encoder: maps input activations to sparse feature pre-activations
        self.encoder = nn.Linear(config.input_dim, config.hidden_dim, bias=True)

        # Decoder: maps sparse features to output predictions
        if config.tied_weights:
            self.decoder = None  # Will use encoder.weight.T
        else:
            self.decoder = nn.Linear(config.hidden_dim, config.output_dim, bias=True)

        # Learnable threshold for feature activation (optional)
        self.threshold = nn.Parameter(torch.zeros(config.hidden_dim))

        # Feature magnitudes for normalization
        self.register_buffer('feature_norms', torch.ones(config.hidden_dim))

        # Running statistics for dead feature detection
        self.register_buffer('feature_activation_count', torch.zeros(config.hidden_dim))
        self.register_buffer('total_samples', torch.tensor(0.0))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        # Kaiming initialization for encoder
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)

        if self.decoder is not None:
            nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity='linear')
            nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input activations to sparse features.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim) or (batch, input_dim)

        Returns:
            features: Sparse feature activations (batch, ..., hidden_dim)
            topk_indices: Indices of active features (batch, ..., k)
        """
        # Get pre-activations
        pre_acts = self.encoder(x)  # (batch, ..., hidden_dim)

        # Apply ReLU with learnable threshold
        features = F.relu(pre_acts - self.threshold)

        # Top-k sparsity
        if self.config.k < self.config.hidden_dim:
            topk_values, topk_indices = torch.topk(features, self.config.k, dim=-1)

            # Create sparse mask
            sparse_features = torch.zeros_like(features)
            sparse_features.scatter_(-1, topk_indices, topk_values)
            features = sparse_features
        else:
            topk_indices = torch.arange(self.config.hidden_dim, device=x.device)
            topk_indices = topk_indices.expand(*features.shape[:-1], -1)

        return features, topk_indices

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features to output activations.

        Args:
            features: Sparse feature activations (batch, ..., hidden_dim)

        Returns:
            Output predictions (batch, ..., output_dim)
        """
        if self.config.tied_weights:
            # Use transposed encoder weights
            return F.linear(features, self.encoder.weight.T[:self.config.output_dim])
        else:
            return self.decoder(features)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through the transcoder.

        Args:
            x: Input activations from source layer
            return_features: Whether to return intermediate features

        Returns:
            output: Predicted activations for target layer
            features: Sparse feature activations (if return_features=True)
            topk_indices: Active feature indices (if return_features=True)
        """
        features, topk_indices = self.encode(x)
        output = self.decode(features)

        # Update activation statistics during training
        if self.training:
            self._update_feature_stats(features)

        if return_features:
            return output, features, topk_indices
        return output, None, None

    def _update_feature_stats(self, features: torch.Tensor):
        """Track which features are being used."""
        with torch.no_grad():
            # Flatten batch dimensions
            flat_features = features.view(-1, self.config.hidden_dim)
            active = (flat_features > 0).float().sum(dim=0)
            self.feature_activation_count += active
            self.total_samples += flat_features.shape[0]

    def get_dead_features(self, threshold: float = 0.01) -> torch.Tensor:
        """Return indices of features that rarely activate."""
        if self.total_samples == 0:
            return torch.tensor([], dtype=torch.long)
        activation_rate = self.feature_activation_count / self.total_samples
        return (activation_rate < threshold).nonzero(as_tuple=True)[0]

    def get_feature_directions(self) -> torch.Tensor:
        """Get the decoder directions for each feature (what each feature represents)."""
        if self.config.tied_weights:
            return self.encoder.weight.T
        return self.decoder.weight.T  # (hidden_dim, output_dim)

    def attribute_to_features(
        self,
        x: torch.Tensor,
        target_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute attribution of output to each sparse feature.

        Args:
            x: Input activations
            target_idx: If specified, attribute only this output dimension

        Returns:
            Attribution scores for each feature
        """
        features, _ = self.encode(x)

        if target_idx is not None:
            # Attribution to specific output dimension
            if self.config.tied_weights:
                weights = self.encoder.weight[target_idx]
            else:
                weights = self.decoder.weight[target_idx]
            return features * weights
        else:
            # Full attribution matrix
            return features  # Shape: (batch, ..., hidden_dim)


class MultiLayerCLT(nn.Module):
    """
    Multi-layer Cross-Layer Transcoder that spans multiple consecutive layers.

    This implements the full circuit tracing setup where we have transcoders
    between each pair of adjacent layers, allowing us to trace feature
    composition through the network.
    """

    def __init__(
        self,
        layer_dims: List[int],
        hidden_dim: int,
        k: int = 64,
        tied_weights: bool = False
    ):
        """
        Args:
            layer_dims: Dimensions of each layer [input_dim, layer1_dim, ..., output_dim]
            hidden_dim: Sparse feature dimension for all transcoders
            k: Top-k sparsity
            tied_weights: Whether to tie encoder/decoder weights
        """
        super().__init__()

        self.layer_dims = layer_dims
        self.num_transcoders = len(layer_dims) - 1

        # Create transcoder for each layer transition
        self.transcoders = nn.ModuleList([
            CrossLayerTranscoder(CLTConfig(
                input_dim=layer_dims[i],
                hidden_dim=hidden_dim,
                output_dim=layer_dims[i + 1],
                num_layers=1,
                k=k,
                tied_weights=tied_weights
            ))
            for i in range(self.num_transcoders)
        ])

    def forward(
        self,
        layer_activations: List[torch.Tensor],
        return_all_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Run all transcoders on their respective layer activations.

        Args:
            layer_activations: List of activations from each layer
            return_all_features: Whether to return intermediate features

        Returns:
            Dictionary containing predictions and optionally features
        """
        predictions = []
        all_features = []

        for i, (transcoder, act) in enumerate(zip(self.transcoders, layer_activations[:-1])):
            pred, features, _ = transcoder(act, return_features=return_all_features)
            predictions.append(pred)
            if return_all_features:
                all_features.append(features)

        result = {'predictions': predictions}
        if return_all_features:
            result['features'] = all_features

        return result

    def compute_reconstruction_loss(
        self,
        layer_activations: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute MSE loss between predictions and actual next-layer activations."""
        total_loss = 0.0

        for i, (transcoder, input_act) in enumerate(zip(self.transcoders, layer_activations[:-1])):
            target_act = layer_activations[i + 1]
            pred, _, _ = transcoder(input_act)
            total_loss += F.mse_loss(pred, target_act)

        return total_loss / self.num_transcoders


def clt_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    features: torch.Tensor,
    l1_coef: float = 1e-3,
    reconstruction_coef: float = 1.0
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute CLT training loss.

    Args:
        output: Transcoder output predictions
        target: Actual layer activations to predict
        features: Sparse feature activations
        l1_coef: Coefficient for L1 sparsity penalty
        reconstruction_coef: Coefficient for reconstruction loss

    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary of individual loss components
    """
    # Reconstruction loss (how well we predict the next layer)
    reconstruction_loss = F.mse_loss(output, target)

    # L1 sparsity loss (encourage sparse features)
    l1_loss = features.abs().mean()

    # Total loss
    total_loss = reconstruction_coef * reconstruction_loss + l1_coef * l1_loss

    loss_dict = {
        'reconstruction': reconstruction_loss,
        'l1_sparsity': l1_loss,
        'total': total_loss
    }

    return total_loss, loss_dict
