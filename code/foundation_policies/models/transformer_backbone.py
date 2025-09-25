from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from foundation_policies.models.positional_encoding import PositionalEncoding

class TransformerBackbone(nn.Module):
    """Enhanced transformer encoder-decoder backbone with layer norm and residual connections."""
    def __init__(self, input_dim: int, action_dim: int, d_model: int = 256,
                 nhead: int = 4, num_layers: int = 6, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.d_model = d_model
        
        self.src_proj = nn.Linear(input_dim, d_model)
        self.tgt_proj = nn.Linear(action_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        self.output_proj = nn.Linear(d_model, action_dim)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.src_proj.weight)
        nn.init.zeros_(self.src_proj.bias)
        nn.init.xavier_uniform_(self.tgt_proj.weight)
        nn.init.zeros_(self.tgt_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        for layer in self.transformer.encoder.layers:
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.xavier_uniform_(layer.linear2.weight)
        for layer in self.transformer.decoder.layers:
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.xavier_uniform_(layer.linear2.weight)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src = self.src_proj(src)
        src = self.pos_enc(src)
        src = self.layer_norm(src)
        
        tgt = self.tgt_proj(tgt)
        tgt = self.pos_enc(tgt)
        tgt = self.layer_norm(tgt)
        
        out = self.transformer(src, tgt, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        out = self.output_proj(out)
        return out