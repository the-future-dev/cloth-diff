from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn

class CNNBackbone(nn.Module):
    """CNN backbone for image feature extraction.
    
    Can be used for image encoding, feature extraction, or any convolutional processing.
    """
    
    def __init__(self, input_channels: int, 
                 conv_channels: Tuple[int, ...] = (32, 64, 128), 
                 kernel_sizes: Tuple[int, ...] = (3, 3, 3),
                 strides: Tuple[int, ...] = (2, 2, 2),
                 use_batch_norm: bool = True):
        super().__init__()
        self.input_channels = input_channels
        
        # Build convolutional layers
        conv_layers = []
        in_channels = input_channels
        
        for out_channels, kernel_size, stride in zip(conv_channels, kernel_sizes, strides):
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2))
            conv_layers.append(nn.ReLU())
            if use_batch_norm:
                conv_layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
        
        self.conv_net = nn.Sequential(*conv_layers)
        
        # We'll determine the output size dynamically
        self._output_shape = None
    
    def _get_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Calculate output shape of conv layers for given input shape."""
        with torch.no_grad():
            # Create dummy input: [1, C, H, W]
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.conv_net(dummy_input)
            return dummy_output.shape[1:]  # [C, H, W]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process images through CNN.
        
        Args:
            x: [B, C, H, W] or [B*T, C, H, W] images
            
        Returns:
            [B, C', H', W'] or [B*T, C', H', W'] feature maps
        """
        return self.conv_net(x)
    
    def get_flattened_size(self, input_shape: Tuple[int, int, int]) -> int:
        """Get the total number of features after flattening."""
        if self._output_shape is None:
            self._output_shape = self._get_output_shape(input_shape)
        return self._output_shape[0] * self._output_shape[1] * self._output_shape[2]