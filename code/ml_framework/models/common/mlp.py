from __future__ import annotations
import torch
import torch.nn as nn

class MLP(nn.Module):
    """Multi-Layer Perceptron for general use.
    
    Can be used for encoding, decoding, or any feed-forward computation.
    """
    
    def __init__(self, in_dim: int, out_dim: int, hidden: tuple = (256, 256)):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)