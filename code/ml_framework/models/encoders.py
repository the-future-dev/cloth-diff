from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.models.common.mlp import MLP
from diffusion_policy.models.common.cnn import CNNBackbone


class ObsEncoder(ABC, nn.Module):
    """Abstract base class for observation encoders.

    Encoders take raw observations and produce embeddings that can be fed to policy backbones.
    Different encoder types can handle different observation modalities (lowdim, image, etc.).
    """

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations to embeddings.

        Args:
            obs: Raw observations of arbitrary shape

        Returns:
            embeddings: [B, T, output_dim] where B=batch, T=sequence length
        """
        pass

    @property
    def device(self) -> torch.device:
        """Get device of encoder parameters."""
        return next(self.parameters()).device if any(True for _ in self.parameters()) else torch.device('cpu')

def create_encoder(encoder_type: str, **kwargs) -> ObsEncoder:
    """Factory function to create encoders.

    Args:
        encoder_type: Type of encoder ('identity', 'mlp', 'cnn')
        **kwargs: Arguments passed to encoder constructor

    Returns:
        ObsEncoder instance
    """
    encoder_type = encoder_type.lower()

    if encoder_type == 'identity':
        return IdentityEncoder(**kwargs)
    elif encoder_type == 'mlp':
        return MLPEncoder(**kwargs)
    elif encoder_type == 'cnn':
        return CNNEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. "
                        f"Supported types: 'identity', 'mlp', 'cnn'")

class IdentityEncoder(ObsEncoder):
    """Identity encoder that passes observations through unchanged.

    Useful when observations are already in the desired embedding format.
    """

    def __init__(self, input_dim: int):
        super().__init__(output_dim=input_dim)
        self.input_dim = input_dim
        # No learnable parameters - this is a pass-through

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Pass observations through unchanged.

        Args:
            obs: [B, T, input_dim] observations

        Returns:
            obs: [B, T, input_dim] unchanged observations
        """
        return obs


class MLPEncoder(ObsEncoder):
    """MLP encoder for low-dimensional observations.

    Takes vector observations and projects them to embedding space.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple = (256, 256)):
        super().__init__(output_dim=output_dim)
        self.input_dim = input_dim
        self.mlp = MLP(input_dim, output_dim, hidden=hidden_dims)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations using MLP.

        Args:
            obs: [B, T, input_dim] observations

        Returns:
            embeddings: [B, T, output_dim] encoded observations
        """
        # obs shape: [B, T, input_dim]
        B, T = obs.shape[:2]

        # Flatten batch and time for MLP processing
        obs_flat = obs.reshape(B * T, -1)  # [B*T, input_dim]

        # Encode
        encoded_flat = self.mlp(obs_flat)  # [B*T, output_dim]

        # Reshape back to sequence format
        encoded = encoded_flat.reshape(B, T, self.output_dim)  # [B, T, output_dim]

        return encoded


class CNNEncoder(ObsEncoder):
    """CNN encoder for image observations.

    Takes image observations and produces feature embeddings.
    """

    def __init__(self, input_channels: int, output_dim: int,
                 conv_channels: tuple = (32, 64, 128),
                 kernel_sizes: tuple = (3, 3, 3),
                 strides: tuple = (2, 2, 2)):
        super().__init__(output_dim=output_dim)
        self.input_channels = input_channels

        self.conv_net = CNNBackbone(
            input_channels=input_channels,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            strides=strides
        )

        # We'll determine the flattened size dynamically in the first forward pass
        self._conv_output_size = None
        self.fc = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode image observations using CNN.

        Args:
            obs: [B, T, C, H, W] image observations

        Returns:
            embeddings: [B, T, output_dim] encoded observations
        """
        # obs shape: [B, T, C, H, W]
        B, T, C, H, W = obs.shape

        # Initialize FC layer on first forward pass
        if self.fc is None:
            conv_output_size = self.conv_net.get_flattened_size((C, H, W))
            self.fc = nn.Linear(conv_output_size, self.output_dim).to(obs.device)
            self._conv_output_size = conv_output_size

        # Reshape to process all images at once: [B*T, C, H, W]
        obs_flat = obs.reshape(B * T, C, H, W)

        # Pass through conv layers
        conv_features = self.conv_net(obs_flat)  # [B*T, channels, h, w]

        # Flatten spatial dimensions
        conv_flat = conv_features.reshape(B * T, -1)  # [B*T, conv_output_size]

        # Final linear projection
        encoded_flat = self.fc(conv_flat)  # [B*T, output_dim]

        # Reshape back to sequence format
        encoded = encoded_flat.reshape(B, T, self.output_dim)  # [B, T, output_dim]

        return encoded
