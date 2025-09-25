from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_framework.interfaces.base_lowdim import BaseLowdimPolicy
from foundation_policies.models.transformer_backbone import TransformerBackbone
from ml_framework.data.normalizer import IdentityNormalizer

class TransformerLowDimPolicy(BaseLowdimPolicy):
    """Enhanced transformer policy for low-dimensional observations."""
    def __init__(self, obs_dim: int, action_dim: int, horizon: int,
                 n_action_steps: int, n_obs_steps: int,
                 d_model: int = 256, nhead: int = 4, num_layers: int = 6,
                 ff_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        
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
        
        self.normalizer = IdentityNormalizer()
        self.backbone = TransformerBackbone(obs_dim, action_dim, d_model, nhead, num_layers, ff_dim, dropout)
        
        # Initialize weights
        # (backbone handles its own initialization)

    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        obs = batch["obs"]
        action = batch["action"]
        
        # Normalize observations
        norm_obs = self.normalizer.normalize(obs)
        pred = self.backbone(norm_obs, action)
        target = action[:, :self.horizon]
        
        # MSE loss + L1 regularization + temporal smoothness
        mse_loss = F.mse_loss(pred, target)
        l1_loss = sum(p.abs().mean() for p in self.parameters()) * 1e-5
        smooth_loss = F.mse_loss(pred[:, 1:], pred[:, :-1]) * 0.1
        total_loss = mse_loss + l1_loss + smooth_loss
        
        return total_loss, {
            "mse_loss": float(mse_loss.detach()),
            "l1_loss": float(l1_loss.detach()),
            "smooth_loss": float(smooth_loss.detach()),
            "total_loss": float(total_loss.detach())
        }

    def _forward_impl(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pred = self.backbone(obs, action)
        return pred[:, :self.horizon]

    @torch.no_grad()
    def predict_action(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        o = obs["obs"]
        norm_o = self.normalizer.normalize(o)
        B = norm_o.shape[0]
        action_tensor = torch.zeros(B, self.horizon, self.action_dim, device=norm_o.device)
        pred = self.backbone(norm_o, action_tensor)
        start = max(0, self.n_obs_steps - 1)
        end = start + self.n_action_steps
        action = self.normalizer.unnormalize(pred[:, start:end])
        return {"action": action}