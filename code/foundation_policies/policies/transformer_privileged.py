from __future__ import annotations
from typing import Dict, Optional
import torch

from ml_framework.interfaces.base_privileged import BasePrivilegedPolicy
from foundation_policies.policies.transformer_image import TransformerImagePolicy
from foundation_policies.policies.transformer_lowdim import TransformerLowDimPolicy
from ml_framework.data.normalizer import IdentityNormalizer

class TransformerPrivilegedPolicy(BasePrivilegedPolicy):
    """Train with state+image, infer with image-only using transformer backbones."""
    def __init__(self,
                 image_feat_dim: int,
                 lowdim_obs_dim: int,
                 action_dim: int,
                 horizon: int,
                 n_action_steps: int,
                 n_obs_steps: int,
                 lowdim_weight: float = 1.0,
                 d_model: int = 256,
                 nhead: int = 4,
                 num_layers: int = 6,
                 ff_dim: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        if lowdim_weight < 0:
            raise ValueError("lowdim_weight must be >= 0")
        self.lowdim_weight = lowdim_weight
        
        # Initialize normalizer locally in this policy
        self.normalizer = IdentityNormalizer()
        self.image_branch = TransformerImagePolicy(
            feat_dim=image_feat_dim,
            action_dim=action_dim,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.lowdim_branch = TransformerLowDimPolicy(
            obs_dim=lowdim_obs_dim,
            action_dim=action_dim,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
        for branch in (self.image_branch, self.lowdim_branch):
            try:
                branch.set_normalizer(normalizer)
            except Exception:
                pass

    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        action = batch["action"]
        img = batch.get("image", batch.get("obs"))
        loss_img, comp_img = self.image_branch.compute_loss({"obs": img, "action": action})
        total = loss_img
        components = {"img_mse": float(comp_img.get("mse", float(loss_img.detach())))}
        if "state" in batch:
            loss_low, comp_low = self.lowdim_branch.compute_loss({"obs": batch["state"], "action": action})
            total = loss_img + self.lowdim_weight * loss_low
            components["low_mse"] = float(comp_low.get("mse", float(loss_low.detach())))
            components["weighted_total"] = float(total.detach())
        else:
            components["weighted_total"] = float(total.detach())
        return total, components

    @torch.no_grad()
    def predict_action(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        img = obs.get("image", obs.get("obs"))
        return self.image_branch.predict_action({"obs": img})
