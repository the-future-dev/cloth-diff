from __future__ import annotations
from typing import Tuple
import torch
from torch.utils.data import DataLoader

from diffusion_policy.data.demonstrations import Demonstrations


def make_dataloaders(dataset_path: str, batch_size: int, num_workers: int,
                     is_image_based: bool = False,
                     privileged: bool = False,
                     horizon: int | None = None,
                     cfg: object | None = None) -> Tuple[DataLoader, DataLoader]:
    """Create train/validation dataloaders.

    Uses cfg.val_ratio if present; defaults to 0.05 otherwise.
    Applies pin_memory and persistent_workers when running on CUDA and workers > 0.
    """
    val_ratio = getattr(cfg, 'val_ratio', 0.05) if cfg is not None else 0.05
    assert 0.0 < val_ratio < 0.5, "val_ratio should be in (0, 0.5) for reasonable splits"

    ds = Demonstrations(
        dataset_path,
        is_image_based=is_image_based,
        horizon=horizon,
        privileged=privileged,
        cfg=cfg
    )
    n = len(ds)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    use_cuda = torch.cuda.is_available()
    pin_memory = use_cuda
    persistent_workers = num_workers > 0

    dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    return train_loader, val_loader