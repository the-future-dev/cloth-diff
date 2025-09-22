from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from diffusion_policy.models.common.positional_encoding import PositionalEncoding

class TransformerBackbone(nn.Module):
    """Transformer encoder backbone for sequence-to-sequence processing.
    
    Can be used for policies, encoders, or any sequence modeling task.
    """
    
    def __init__(self, d_model: int = 256, nhead: int = 4, num_layers: int = 6, 
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process sequence through transformer encoder.
        
        Args:
            x: [B, T, d_model] input sequence
            src_key_padding_mask: [B, T] mask for padded positions
            
        Returns:
            [B, T, d_model] encoded sequence
        """
        x = self.pos(x)
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)