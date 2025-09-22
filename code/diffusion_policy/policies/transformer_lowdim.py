from __future__ import annotations
from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.policies.base_lowdim import BaseLowdimPolicy
from diffusion_policy.models.common.transformer import TransformerBackbone
from diffusion_policy.models.encoders import ObsEncoder, create_encoder

class TransformerLowDimPolicy(BaseLowdimPolicy):
    def __init__(self, obs_dim: int, action_dim: int, horizon: int,
                 n_action_steps: int, n_obs_steps: int,
                 # Encoder configuration
                 encoder_type: str = "identity",
                 encoder_kwargs: Optional[Dict] = None,
                 # Transformer configuration
                 d_model: int = 256, nhead: int = 2, num_layers: int = 6,
                 ff_dim: int = 1024, dropout: float = 0.1):
        # Create encoder
        encoder_kwargs = encoder_kwargs or {}
        if encoder_type == "identity":
            encoder_kwargs["input_dim"] = obs_dim
            d_model_input = obs_dim
        elif encoder_type == "mlp":
            encoder_kwargs.setdefault("input_dim", obs_dim)
            encoder_kwargs.setdefault("output_dim", d_model)
            d_model_input = d_model
        else:
            raise ValueError(f"Unsupported encoder_type '{encoder_type}' for low-dim policy")
        
        obs_encoder = create_encoder(encoder_type, **encoder_kwargs)
        super().__init__(obs_encoder=obs_encoder)
    
    # horizon: training crop length; model predicts actions for each of these steps
    # n_obs_steps: context length; at inference we return actions starting at (n_obs_steps - 1)
    # n_action_steps: number of action steps returned per predict_action call
        # Validation
        if not (horizon > 0):
            raise ValueError("horizon must be > 0")
        if not (1 <= n_obs_steps <= horizon):
            raise ValueError(f"n_obs_steps must be in [1, horizon]; got {n_obs_steps} vs {horizon}")
        if not (1 <= n_action_steps <= horizon - (n_obs_steps - 1)):
            raise ValueError("n_action_steps must fit inside horizon after observation window")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps

        # Backbone and output layers
        self.backbone = TransformerBackbone(d_model_input, nhead, num_layers, ff_dim, dropout)
        self.readout = nn.Linear(d_model_input, action_dim)

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _forward_impl(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B, T, Do] -> encode -> [B, T, d_model] -> backbone -> [B, T, d_model] -> readout -> [B, T, Da]
        encoded_obs = self.obs_encoder(obs) if self.obs_encoder else obs
        h = self.backbone(encoded_obs)  # [B, T, d_model]
        out = self.readout(h)  # [B, T, Da]
        # take first horizon steps as predictions
        return out[:, :self.horizon]

    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        obs = batch["obs"]
        action = batch["action"]
        pred = self._forward_impl(obs)
        target = action[:, :self.horizon]
        loss = F.mse_loss(pred, target)
        return loss, {"mse": float(loss.detach())}

    @torch.no_grad()
    def predict_action(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        o = obs["obs"]
        pred = self._forward_impl(o)
        # Slice out the chunk to execute now; typical MPC executes 1 step or this whole chunk
        start = max(0, self.n_obs_steps - 1)
        end = start + self.n_action_steps
        return {"action": pred[:, start:end]}
