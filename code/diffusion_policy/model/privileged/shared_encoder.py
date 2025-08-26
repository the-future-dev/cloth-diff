import torch
import torch.nn as nn

class MLPSharedEncoder(nn.Module):
    """
    Simple per‐timestep MLP over fused features [B,T,D_in]→[B,T,D_out].
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        activation: str = 'relu'
    ):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 1:
                if activation == 'relu':
                    layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        # expose output dimensionality to override cond_dim downstream
        self.output_dim = output_dim

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        B, T, _ = feats.shape
        x = feats.view(B * T, -1)
        y = self.net(x)
        return y.view(B, T, -1)


class TransformerSharedEncoder(nn.Module):
    """
    nn.TransformerEncoder over the sequence of fused features.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # preserve output dimension (same as d_model)
        self.output_dim = d_model

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, T, d_model] → [T, B, d_model]
        x = feats.permute(1, 0, 2)
        y = self.encoder(x)
        return y.permute(1, 0, 2)


class PerceiverSharedEncoder(nn.Module):
    """
    Perceiver‐style cross‐attention encoder mapping [B,T,D_in] → [B,T,latent_dim]
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = None,
        num_latents: int = None,
        cross_heads: int = 1,
        num_self_layers: int = 1,
        self_n_heads: int = 1,
        dim_feedforward: int = None,
        dropout: float = 0.0
    ):
        super().__init__()
        # defaults
        latent_dim = latent_dim or input_dim
        num_latents = num_latents or input_dim
        dim_feedforward = dim_feedforward or (latent_dim * 4)

        # learnable latent embeddings: [M, D_latent]
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        # cross-attention: queries=latents, keys/values=input features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=cross_heads,
            kdim=input_dim,
            vdim=input_dim,
            dropout=dropout,
            batch_first=False
        )
        # self-attention stack on latents
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=self_n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.self_attn = nn.TransformerEncoder(layer, num_layers=num_self_layers)
        # output feature size per time step
        self.output_dim = latent_dim

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: [B, T, input_dim]
        returns: [B, T, latent_dim]
        """
        B, T, _ = feats.shape
        # prepare latents: [M, B, D_latent]
        lat = self.latents.unsqueeze(1).expand(-1, B, -1)
        # keys/values: [T, B, input_dim]
        kv = feats.permute(1, 0, 2)
        # cross-attend: lat <- Attn(lat, kv, kv)
        lat, _ = self.cross_attn(lat, kv, kv)
        # self-attend on latents
        lat = self.self_attn(lat)
        # pool latents to a single vector per batch
        pooled = lat.mean(dim=0)               # [B, D_latent]
        # replicate across T time steps
        cond = pooled.unsqueeze(1).repeat(1, T, 1)  # [B, T, D_latent]
        return cond

class MultiModalCrossAttentionEncoder(nn.Module):
    """
    Shared encoder using cross-attention to fuse image and privileged state features.
    Input: [B, T, img_dim], [B, T, state_dim]
    Output: [B, T, fused_dim]
    """
    def __init__(self,
                 img_dim: int,
                 state_dim: int,
                 fused_dim: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        # Project inputs to common dimension
        self.img_proj = nn.Linear(img_dim, fused_dim)
        self.state_proj = nn.Linear(state_dim, fused_dim)

        # Create cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=fused_dim,
                nhead=num_heads,
                dim_feedforward=fused_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        # Output dimensionality
        self.output_dim = fused_dim

    def forward(self, img_feats: torch.Tensor, state_feats: torch.Tensor) -> torch.Tensor:
        """
        img_feats: [B, T, img_dim]
        state_feats: [B, T, state_dim]
        Returns fused_feats: [B, T, fused_dim]
        """
        B, T, _ = img_feats.shape

        # Project to common embedding space
        img_tokens = self.img_proj(img_feats)       # [B, T, D]
        state_tokens = self.state_proj(state_feats) # [B, T, D]

        # Concatenate for cross-attention
        fused_tokens = img_tokens + state_tokens    # [B, T, D]

        # Pass through transformer layers
        for layer in self.cross_attn_layers:
            fused_tokens = layer(fused_tokens)

        return fused_tokens

def get_shared_encoder(name: str, **kwargs) -> nn.Module:
    """Factory to pick one of the above encoders."""
    if name == 'mlp':
        return MLPSharedEncoder(
            input_dim=kwargs['input_dim'],
            hidden_dims=kwargs['hidden_dims'],
            output_dim=kwargs['output_dim'],
            activation=kwargs.get('activation','relu')
        )
    if name == 'transformer':
        return TransformerSharedEncoder(
            d_model=kwargs['d_model'],
            nhead=kwargs['nhead'],
            num_layers=kwargs['num_layers'],
            dim_feedforward=kwargs['dim_feedforward'],
            dropout=kwargs.get('dropout',0.1)
        )
    if name == 'cross_attention':
        return MultiModalCrossAttentionEncoder(
            img_dim=kwargs['img_dim'],
            state_dim=kwargs['state_dim'],
            fused_dim=kwargs['d_model'],
            num_heads=kwargs['nhead'],
            num_layers=kwargs['num_layers'],
            dropout=kwargs.get('dropout',0.1)
        )
    if name == 'perceiver':
        return PerceiverSharedEncoder(
            input_dim=kwargs['input_dim'],
            latent_dim=kwargs.get('latent_dim'),
            num_latents=kwargs.get('num_latents'),
            cross_heads=kwargs.get('cross_heads',1),
            num_self_layers=kwargs.get('num_self_layers',1),
            self_n_heads=kwargs.get('self_n_heads',1),
            dim_feedforward=kwargs.get('dim_feedforward'),
            dropout=kwargs.get('dropout',0.0)
        )
    raise ValueError(f"Unknown shared_encoder_type={name}")