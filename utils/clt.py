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


class CLTEncoder(nn.Module):
    """
    Encoder for a single layer in the Cross-Layer Transcoder.

    Maps residual stream activations at layer ℓ to sparse feature activations
    using JumpReLU activation: a^ℓ = JumpReLU(W_enc^ℓ x^ℓ)
    """

    def __init__(self, input_dim: int, hidden_dim: int, k: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k

        # Encoder weights: maps input activations to feature pre-activations
        self.weight = nn.Linear(input_dim, hidden_dim, bias=True)

        # Learnable threshold for JumpReLU
        self.threshold = nn.Parameter(torch.zeros(hidden_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight.weight, nonlinearity='relu')
        nn.init.zeros_(self.weight.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input activations to sparse features using JumpReLU.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim) or (batch, input_dim)

        Returns:
            features: Sparse feature activations (batch, ..., hidden_dim)
            topk_indices: Indices of active features (batch, ..., k)
        """
        # Get pre-activations
        pre_acts = self.weight(x)  # (batch, ..., hidden_dim)

        # Apply JumpReLU: x * (x > threshold) - zeroes below threshold without subtracting
        features = pre_acts * (pre_acts > self.threshold).float()

        # Top-k sparsity
        if self.k < self.hidden_dim:
            topk_values, topk_indices = torch.topk(features, self.k, dim=-1)

            # Create sparse mask
            sparse_features = torch.zeros_like(features)
            sparse_features.scatter_(-1, topk_indices, topk_values)
            features = sparse_features
        else:
            topk_indices = torch.arange(self.hidden_dim, device=x.device)
            topk_indices = topk_indices.expand(*features.shape[:-1], -1)

        return features, topk_indices


class CLTDecoder(nn.Module):
    """
    Decoder for Cross-Layer Transcoder.

    Maps sparse features from layer ℓ' to contribute to reconstruction of layer ℓ.
    This is W_dec^{ℓ'→ℓ} in the paper.
    """

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.weight = nn.Linear(hidden_dim, output_dim, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight.weight, nonlinearity='linear')

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features to output contribution.

        Args:
            features: Sparse feature activations (batch, ..., hidden_dim)

        Returns:
            Output contribution (batch, ..., output_dim)
        """
        return self.weight(features)

    def get_feature_norms(self) -> torch.Tensor:
        """Get L2 norm of decoder weights for each feature (for sparsity penalty)."""
        # weight shape: (output_dim, hidden_dim)
        # Return shape: (hidden_dim,) - norm of each feature's decoder vector
        return self.weight.weight.norm(dim=0)


class CrossLayerTranscoder(nn.Module):
    """
    Cross-Layer Transcoder (CLT) based on Anthropic's Circuit Tracing.
    https://transformer-circuits.pub/2025/attribution-graphs/methods.html

    The CLT learns sparse, interpretable features that explain how information
    flows across layers in a neural network.

    Key architecture (from paper):
    - Each feature in layer ℓ reads from residual stream at layer ℓ using encoder
    - Each feature can write to ALL subsequent layers ℓ, ℓ+1, ..., L using separate decoders
    - Output at layer ℓ: ŷ^ℓ = Σ_{ℓ'=1}^{ℓ} W_dec^{ℓ'→ℓ} a^{ℓ'}

    This class represents a single-layer encoder with potentially multiple output decoders.
    For the full cross-layer setup, use CrossLayerTranscoderSystem.

    Args:
        config: CLTConfig with architecture parameters
    """

    def __init__(self, config: CLTConfig):
        super().__init__()
        self.config = config

        # Encoder for this layer
        self.encoder = CLTEncoder(config.input_dim, config.hidden_dim, config.k)

        # Decoders for each output layer (this layer can write to num_layers output layers)
        # decoder[i] writes to layer (current_layer + i)
        self.decoders = nn.ModuleList([
            CLTDecoder(config.hidden_dim, config.output_dim)
            for _ in range(config.num_layers)
        ])

        # Bias for the primary output (layer 0 decoder)
        self.output_bias = nn.Parameter(torch.zeros(config.output_dim))

        # Running statistics for dead feature detection
        self.register_buffer('feature_activation_count', torch.zeros(config.hidden_dim))
        self.register_buffer('total_samples', torch.tensor(0.0))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input activations to sparse features."""
        return self.encoder(x)

    def decode(self, features: torch.Tensor, output_layer_idx: int = 0) -> torch.Tensor:
        """
        Decode sparse features to output activations for a specific output layer.

        Args:
            features: Sparse feature activations (batch, ..., hidden_dim)
            output_layer_idx: Which output layer to decode for (0 = immediate next layer)

        Returns:
            Output predictions (batch, ..., output_dim)
        """
        if output_layer_idx >= len(self.decoders):
            raise ValueError(f"output_layer_idx {output_layer_idx} >= num_layers {len(self.decoders)}")

        output = self.decoders[output_layer_idx](features)
        if output_layer_idx == 0:
            output = output + self.output_bias
        return output

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        output_layer_idx: int = 0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through the transcoder.

        Args:
            x: Input activations from source layer
            return_features: Whether to return intermediate features
            output_layer_idx: Which output layer to decode for

        Returns:
            output: Predicted activations for target layer
            features: Sparse feature activations (if return_features=True)
            topk_indices: Active feature indices (if return_features=True)
        """
        features, topk_indices = self.encode(x)
        output = self.decode(features, output_layer_idx)

        # Update activation statistics during training
        if self.training:
            self._update_feature_stats(features)

        if return_features:
            return output, features, topk_indices
        return output, None, None

    def _update_feature_stats(self, features: torch.Tensor):
        """Track which features are being used."""
        with torch.no_grad():
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

    def get_decoder_norms(self, output_layer_idx: int = 0) -> torch.Tensor:
        """Get L2 norm of decoder weights for each feature."""
        return self.decoders[output_layer_idx].get_feature_norms()

    def get_all_decoder_norms(self) -> torch.Tensor:
        """
        Get concatenated decoder norms across all output layers.
        This is ||W_dec,i|| in the paper's sparsity penalty.

        Returns:
            Tensor of shape (hidden_dim,) with summed norms across all decoders
        """
        total_norms = torch.zeros(self.config.hidden_dim, device=self.encoder.weight.weight.device)
        for decoder in self.decoders:
            total_norms += decoder.get_feature_norms()
        return total_norms


class CrossLayerTranscoderSystem(nn.Module):
    """
    Full Cross-Layer Transcoder system spanning multiple layers.

    This implements the complete architecture from the paper where:
    - Each layer ℓ has an encoder that maps residual stream to features
    - Features from layer ℓ can contribute to reconstructing MLP outputs at layers ℓ, ℓ+1, ..., L
    - Output at layer ℓ: ŷ^ℓ = Σ_{ℓ'=1}^{ℓ} W_dec^{ℓ'→ℓ} a^{ℓ'}

    Args:
        num_layers: Number of layers (L in the paper)
        input_dim: Dimension of residual stream
        hidden_dim: Dimension of sparse feature space
        output_dim: Dimension of MLP output to reconstruct
        k: Top-k sparsity
    """

    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        k: int = 64
    ):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.k = k

        # Create encoder for each layer
        self.encoders = nn.ModuleList([
            CLTEncoder(input_dim, hidden_dim, k)
            for _ in range(num_layers)
        ])

        # Create decoders: decoder[ℓ'][ℓ] maps features from layer ℓ' to output at layer ℓ
        # Only need decoders where ℓ' <= ℓ (features can only write to same or later layers)
        # Stored as: decoders[source_layer][target_offset] where target = source + target_offset
        self.decoders = nn.ModuleList([
            nn.ModuleList([
                CLTDecoder(hidden_dim, output_dim)
                for target_offset in range(num_layers - source_layer)
            ])
            for source_layer in range(num_layers)
        ])

        # Output biases for each layer
        self.output_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(output_dim))
            for _ in range(num_layers)
        ])

        # Statistics tracking
        self.register_buffer('feature_activation_counts', torch.zeros(num_layers, hidden_dim))
        self.register_buffer('total_samples', torch.tensor(0.0))

    def encode_layer(self, x: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode activations at a specific layer."""
        return self.encoders[layer_idx](x)

    def decode_to_layer(
        self,
        all_features: List[torch.Tensor],
        target_layer: int
    ) -> torch.Tensor:
        """
        Decode features from all previous layers to reconstruct target layer output.

        Implements: ŷ^ℓ = Σ_{ℓ'=1}^{ℓ} W_dec^{ℓ'→ℓ} a^{ℓ'}

        Args:
            all_features: List of feature tensors, one per layer up to target_layer
            target_layer: Which layer's MLP output to reconstruct (0-indexed)

        Returns:
            Reconstructed MLP output for target_layer
        """
        output = self.output_biases[target_layer].clone()

        # Sum contributions from all layers ℓ' <= target_layer
        for source_layer in range(target_layer + 1):
            if source_layer < len(all_features) and all_features[source_layer] is not None:
                target_offset = target_layer - source_layer
                decoder = self.decoders[source_layer][target_offset]
                output = output + decoder(all_features[source_layer])

        return output

    def forward(
        self,
        layer_inputs: List[torch.Tensor],
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full CLT system.

        Args:
            layer_inputs: List of residual stream activations at each layer
            return_features: Whether to return intermediate features

        Returns:
            Dictionary with:
                - 'outputs': List of reconstructed MLP outputs for each layer
                - 'features': List of sparse features for each layer (if return_features)
        """
        # Encode all layers
        all_features = []
        all_indices = []
        for layer_idx, x in enumerate(layer_inputs):
            features, indices = self.encode_layer(x, layer_idx)
            all_features.append(features)
            all_indices.append(indices)

            if self.training:
                self._update_feature_stats(features, layer_idx)

        # Decode to each layer
        outputs = []
        for target_layer in range(self.num_layers):
            output = self.decode_to_layer(all_features, target_layer)
            outputs.append(output)

        result = {'outputs': outputs}
        if return_features:
            result['features'] = all_features
            result['indices'] = all_indices

        return result

    def _update_feature_stats(self, features: torch.Tensor, layer_idx: int):
        """Track which features are being used at each layer."""
        with torch.no_grad():
            flat_features = features.view(-1, self.hidden_dim)
            active = (flat_features > 0).float().sum(dim=0)
            self.feature_activation_counts[layer_idx] += active
            self.total_samples += flat_features.shape[0]

    def get_decoder_norms_for_layer(self, source_layer: int) -> torch.Tensor:
        """
        Get total decoder norm for features at a source layer.

        This sums ||W_dec^{ℓ'→ℓ}|| across all target layers ℓ >= ℓ'.
        Used for the sparsity penalty in the paper.
        """
        total_norms = torch.zeros(self.hidden_dim, device=self.encoders[0].weight.weight.device)
        for target_offset in range(len(self.decoders[source_layer])):
            total_norms += self.decoders[source_layer][target_offset].get_feature_norms()
        return total_norms


def clt_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    features: torch.Tensor,
    decoder_norms: Optional[torch.Tensor] = None,
    l1_coef: float = 1e-3,
    sparsity_c: float = 1.0,
    reconstruction_coef: float = 1.0
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute CLT training loss following the paper's formulation.

    Loss = L_MSE + λ * L_sparsity

    Where:
    - L_MSE = Σ_ℓ ||ŷ^ℓ - y^ℓ||²
    - L_sparsity = λ * Σ_ℓ Σ_i tanh(c * ||W_dec,i|| * a_i^ℓ)

    Args:
        output: Transcoder output predictions
        target: Actual layer activations to predict
        features: Sparse feature activations
        decoder_norms: L2 norms of decoder weights per feature (for weighted sparsity)
        l1_coef: λ coefficient for sparsity penalty
        sparsity_c: c hyperparameter in tanh(c * ||W_dec|| * a)
        reconstruction_coef: Coefficient for reconstruction loss

    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary of individual loss components
    """
    # Reconstruction loss (how well we predict the MLP output)
    reconstruction_loss = F.mse_loss(output, target)

    # Sparsity loss following paper: tanh(c * ||W_dec,i|| * a_i)
    if decoder_norms is not None:
        # Weight activations by decoder norms
        weighted_activations = sparsity_c * decoder_norms * features.abs()
        sparsity_loss = torch.tanh(weighted_activations).mean()
    else:
        # Fallback to simple L1 if decoder norms not provided
        sparsity_loss = features.abs().mean()

    # Total loss
    total_loss = reconstruction_coef * reconstruction_loss + l1_coef * sparsity_loss

    loss_dict = {
        'reconstruction': reconstruction_loss,
        'sparsity': sparsity_loss,
        'total': total_loss
    }

    return total_loss, loss_dict


def clt_system_loss(
    outputs: List[torch.Tensor],
    targets: List[torch.Tensor],
    all_features: List[torch.Tensor],
    clt_system: CrossLayerTranscoderSystem,
    l1_coef: float = 1e-3,
    sparsity_c: float = 1.0
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute loss for the full CrossLayerTranscoderSystem.

    Implements the paper's loss:
    - L_MSE = Σ_ℓ ||ŷ^ℓ - y^ℓ||²
    - L_sparsity = λ * Σ_ℓ Σ_i tanh(c * ||W_dec,i^ℓ|| * a_i^ℓ)

    Args:
        outputs: List of reconstructed outputs for each layer
        targets: List of actual MLP outputs for each layer
        all_features: List of sparse features for each layer
        clt_system: The CLT system (to get decoder norms)
        l1_coef: λ sparsity coefficient
        sparsity_c: c hyperparameter

    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary of individual loss components
    """
    total_reconstruction = 0.0
    total_sparsity = 0.0

    for layer_idx, (output, target, features) in enumerate(zip(outputs, targets, all_features)):
        # Reconstruction loss for this layer
        total_reconstruction += F.mse_loss(output, target)

        # Sparsity loss weighted by decoder norms
        decoder_norms = clt_system.get_decoder_norms_for_layer(layer_idx)
        weighted_activations = sparsity_c * decoder_norms * features.abs()
        total_sparsity += torch.tanh(weighted_activations).mean()

    total_loss = total_reconstruction + l1_coef * total_sparsity

    loss_dict = {
        'reconstruction': total_reconstruction,
        'sparsity': total_sparsity,
        'total': total_loss
    }

    return total_loss, loss_dict
