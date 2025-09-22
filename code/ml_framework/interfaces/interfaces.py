from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict
import torch

class Policy(ABC, torch.nn.Module):
    """Single-responsibility policy interface.

    Must implement compute_loss (for training) and predict_action (for rollout).
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor: ...

    @abstractmethod
    def predict_action(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]: ...
