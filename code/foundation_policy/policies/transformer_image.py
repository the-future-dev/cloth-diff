from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_framework.interfaces.base_image import BaseImagePolicy
from foundation_policy.models.transformer_backbone import TransformerBackbone
from ml_framework.data.normalizer import IdentityNormalizer

class TransformerImagePolicy(BaseImagePolicy):
    """Non-diffusion transformer policy for image.
    """
    def __init__(self, feat_dim: int, action_dim: int, horizon: int,
                 n_action_steps: int, n_obs_steps: int,
                 d_model: int = 256, nhead: int = 4, num_layers: int = 6,
                 ff_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        d_model_input = feat_dim
        if not (horizon > 0):
            raise ValueError("horizon must be > 0")
        if not (1 <= n_obs_steps <= horizon):
            raise ValueError("n_obs_steps must be within horizon")
        if not (1 <= n_action_steps <= horizon - (n_obs_steps - 1)):
            raise ValueError("n_action_steps must fit inside horizon after observation window")
        self.feat_dim = feat_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.backbone = TransformerBackbone(feat_dim, action_dim, d_model, nhead, num_layers, ff_dim, dropout)
        
        # Initialize normalizer locally in this policy
        self.normalizer = IdentityNormalizer()

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _forward_impl(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # If image tensor [B,T,C,H,W], flatten spatial dims; if already features keep as is.
        if obs.dim() == 5:
            B, T, C, H, W = obs.shape
            obs = obs.reshape(B, T, C * H * W)
        pred = self.backbone(obs, action)
        return pred[:, :self.horizon]

    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        obs = batch["obs"]
        action = batch["action"]
        pred = self._forward_impl(obs, action)
        target = action[:, :self.horizon]
        loss = F.mse_loss(pred, target)
        return loss, {"mse": float(loss.detach())}

    @torch.no_grad()
    def predict_action(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        o = obs["obs"]
        B = o.shape[0]
        action_tensor = torch.zeros(B, self.horizon, self.action_dim, device=o.device)
        pred = self._forward_impl(o, action_tensor)
        start = max(0, self.n_obs_steps - 1)
        end = start + self.n_action_steps
        return {"action": pred[:, start:end]}
