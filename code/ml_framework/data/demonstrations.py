"""
Demonstrations dataset (ported and simplified).

Handles state-, image-, and privileged modes with optional horizon cropping.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class Demonstrations(Dataset):
    def __init__(self,
                 file_path: str,
                 is_image_based: bool = False,
                 img_transform=None,
                 horizon: Optional[int] = None,
                 privileged: bool = False,
                 cfg: Optional[Any] = None,
                 deterministic_horizon: bool = True):
        super().__init__()
        self.is_image_based = is_image_based
        self.privileged = privileged
        self.img_transform = img_transform
        self.horizon = horizon
        self.cfg = cfg
        self.deterministic_horizon = deterministic_horizon

        # Load and validate pickle file with comprehensive error handling
        try:
            # First check if file exists and has content
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise EOFError(f"Dataset file is empty (0 bytes): {file_path}")
            
            print(f"Loading dataset from: {file_path} ({file_size:,} bytes)")
            
            with open(file_path, 'rb') as f:
                # Check if file is empty first
                if f.seek(0, 2) == 0:  # Seek to end, if position is 0, file is empty
                    raise EOFError(f"Pickle file is empty: {file_path}")
                f.seek(0)  # Reset to beginning
                
                # Try to load with timeout protection
                data = pickle.load(f)
                
        except EOFError as e:
            raise EOFError(f"Failed to load pickle file '{file_path}': {str(e)}. "
                          "The file may be corrupted, truncated, or empty. "
                          "Please check the file integrity and ensure it was created properly.")
        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError(f"Failed to unpickle file '{file_path}': {str(e)}. "
                                        "The file may be corrupted or created with a different Python version.")
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading pickle file '{file_path}': {str(e)}")

        # Validate that we actually got data
        if data is None:
            raise ValueError(f"Pickle file '{file_path}' loaded successfully but contains None/null data")

        # Additional validation for expected data structure
        if not isinstance(data, (dict, list)):
            raise TypeError(f"Pickle file '{file_path}' contains unexpected data type: {type(data)}. "
                           "Expected dict or list.")

        if self.privileged:
            self.trajectories = self._load_privileged_data(data)
        else:
            self.trajectories = self._load_standard_data(data)

        if len(self.trajectories) == 0:
            raise ValueError("No trajectories loaded from file: empty dataset")

        # Basic shape & key validation
        required_keys = {('state','image','action')} if self.privileged else {('obs','action')}
        sample = self.trajectories[0]
        for combo in required_keys:
            for k in combo:
                if k not in sample:
                    raise KeyError(f"Expected key '{k}' in trajectory sample for mode privileged={self.privileged}")

        # Ensure sequence lengths consistent within each trajectory
        for i, traj in enumerate(self.trajectories[:10]):  # spot check first 10
            lengths = {k: v.shape[0] for k, v in traj.items() if isinstance(v, torch.Tensor) and v.dim() >= 1}
            if len(set(lengths.values())) > 1:
                raise ValueError(f"Inconsistent time lengths in trajectory index {i}: {lengths}")

        # Enforce image size expectations if requested
        if self.is_image_based and getattr(self.cfg, 'env_img_size', None) is not None:
            expected = getattr(self.cfg, 'env_img_size')
            # allow tuple or int
            if isinstance(expected, int):
                expected_hw: Tuple[int,int] = (expected, expected)
            elif isinstance(expected, (tuple, list)) and len(expected) == 2:
                expected_hw = (int(expected[0]), int(expected[1]))
            else:
                raise ValueError(f"env_img_size must be int or (H,W), got {expected}")
            # Validate first trajectory image spatial dims after any transpose (C,H,W)
            if self.privileged:
                img_t = self.trajectories[0]['image']
            else:
                if not self.is_image_based:
                    img_t = None
                else:
                    img_t = self.trajectories[0]['obs']
            if img_t is not None and img_t.dim() >= 4:
                # Expect [T,C,H,W]
                H, W = img_t.shape[-2], img_t.shape[-1]
                if (H, W) != expected_hw:
                    raise ValueError(f"Image size mismatch: expected {expected_hw}, got {(H,W)}")

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.privileged:
            sample = self._get_privileged_sample(idx)
        else:
            sample = self._get_standard_sample(idx)
        return sample

    # ---------- helpers ----------
    def _maybe_trim(self, *tensors: torch.Tensor) -> List[torch.Tensor]:
        """Trim sequences to horizon if provided.

        Deterministic: keep first horizon steps.
        (Random cropping removed for validation stability.)
        """
        if self.horizon is None:
            return list(tensors)
        L = tensors[0].size(0)
        if L <= self.horizon:
            return list(tensors)
        if self.deterministic_horizon:
            return [t[:self.horizon] for t in tensors]
        # Fallback random (not default) if explicitly disabled deterministic mode
        start = torch.randint(0, L - self.horizon + 1, (1,)).item()
        return [t[start:start + self.horizon] for t in tensors]

    def _get_standard_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        traj = self.trajectories[idx]
        obs = traj['obs']
        action = traj['action']
        obs, action = self._maybe_trim(obs, action)

        # If image-based and transform provided, apply per-frame
        if self.is_image_based and self.img_transform is not None:
            # Expect obs as [T, C, H, W] or [T, H, W, C]
            if obs.dim() == 4:
                t_list = []
                for t in range(obs.size(0)):
                    img = obs[t]
                    # if HWC -> CHW for transforms
                    if img.size(0) not in (1, 3):
                        img = img.permute(2, 0, 1).contiguous()
                    img = self.img_transform(img)
                    t_list.append(img)
                obs = torch.stack(t_list, dim=0)

        return {'obs': obs, 'action': action}

    def _get_privileged_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        traj = self.trajectories[idx]
        state, image, action = traj['state'], traj['image'], traj['action']
        state, image, action = self._maybe_trim(state, image, action)
        # Optionally permute image to HWC for transformer-like policies; we leave as-is here.
        return {'state': state, 'image': image, 'action': action}

    def _load_standard_data(self, data: Dict[str, Any]):
        if self.is_image_based:
            # load images
            ob_trajs = data['ob_img_trajs']
            # If mode is 'direct', legacy data often saved as [N,T,H,W,C]; transpose to [N,T,C,H,W]
            mode = getattr(self.cfg, 'load_ob_image_mode', 'direct') if self.cfg else 'direct'
            if mode == 'direct' and ob_trajs is not None and ob_trajs.ndim == 5:
                ob_trajs = np.transpose(ob_trajs, (0, 1, 4, 2, 3))
        else:
            if 'ob_trajs' in data:
                ob_trajs = data['ob_trajs']
            elif 'obs_trajs' in data:
                ob_trajs = data['obs_trajs']
            else:
                raise KeyError("No 'ob_trajs' or 'obs_trajs' in pickle")
        action_trajs = data['action_trajs']

        trajectories = []
        for obs_seq, act_seq in zip(ob_trajs, action_trajs):
            obs_tensor = torch.tensor(obs_seq, dtype=torch.float32)
            action_tensor = torch.tensor(act_seq, dtype=torch.float32)
            trajectories.append({'obs': obs_tensor, 'action': action_tensor})
        return trajectories

    def _load_privileged_data(self, data: Dict[str, Any]):
        if 'ob_trajs' in data:
            state_trajs = data['ob_trajs']
        elif 'obs_trajs' in data:
            state_trajs = data['obs_trajs']
        else:
            raise KeyError("No 'ob_trajs' or 'obs_trajs' in pickle for privileged mode")

        img_trajs = data['ob_img_trajs']
        mode = getattr(self.cfg, 'load_ob_image_mode', 'direct') if self.cfg else 'direct'
        if mode == 'direct' and img_trajs is not None and img_trajs.ndim == 5:
            img_trajs = np.transpose(img_trajs, (0, 1, 4, 2, 3))
        action_trajs = data['action_trajs']

        trajectories = []
        for state_seq, img_seq, act_seq in zip(state_trajs, img_trajs, action_trajs):
            state_tensor = torch.tensor(state_seq, dtype=torch.float32)
            image_tensor = torch.tensor(img_seq, dtype=torch.float32)
            action_tensor = torch.tensor(act_seq, dtype=torch.float32)
            trajectories.append({'state': state_tensor, 'image': image_tensor, 'action': action_tensor})
        return trajectories
