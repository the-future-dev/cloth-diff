"""
Demonstrations dataset module.

Handles loading and processing of expert demonstration data.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class Demonstrations(Dataset):
    """
    Dataset class for loading and processing expert demonstrations.

    Handles both state- and image-based demonstrations with proper formatting.
    """

    def __init__(self, file_path, is_image_based=False, img_transform=None, horizon=None, privileged=False, args=None):
        """
        Initialize demonstrations dataset.

        Args:
            file_path: Path to the pickle file containing demonstrations
            is_image_based: Whether observations are image-based
            img_transform: Image transformations to apply
            horizon: Maximum horizon for trajectory cropping
            privileged: Whether to load privileged (state + image) data
            args: Arguments object for configuration
        """
        self.is_image_based = is_image_based
        self.privileged = privileged
        self.img_transform = img_transform
        self.horizon = horizon
        self.args = args  # Store args for access to model_type etc.

        # load_file now returns a list of episode‐dicts
        self.trajectories = self.load_file(file_path)
        print(f"Loaded {len(self.trajectories)} trajectories")

        # Resize image observations to match env_img_size if needed
        if is_image_based:
            assert args.env_img_size == 32, "Current dataset is 32x32, so env_img_size must be 32 for image-based demonstrations"

    def __len__(self):
        """Return the number of trajectories."""
        return len(self.trajectories)

    def __getitem__(self, idx):
        """Get a single trajectory sample."""
        # Privileged dataset: return state and image separately
        if getattr(self, 'privileged', False):
            return self._get_privileged_sample(idx)

        # Standard dataset: return obs
        return self._get_standard_sample(idx)

    def _get_privileged_sample(self, idx):
        """Get a privileged sample with both state and image observations."""
        data = self.trajectories[idx]
        state = data['state']
        image = data['image']
        action = data['action']

        # Crop to horizon if needed
        if self.horizon is not None and state.size(0) > self.horizon:
            L = state.size(0)
            start = torch.randint(0, L - self.horizon + 1, (1,)).item()
            state = state[start : start + self.horizon]
            image = image[start : start + self.horizon]
            action = action[start : start + self.horizon]

        # Permute image for privileged transformer
        if self.args and self.args.model_type == 'privileged':
            # image: [T, C, H, W] -> [T, H, W, C]
            image = image.permute(0, 2, 3, 1).contiguous()

        return {'state': state, 'image': image, 'action': action}

    def _get_standard_sample(self, idx):
        """Get a standard sample with observation and action data."""
        data = self.trajectories[idx]
        obs = data['obs']
        action = data['action']

        # if the raw trajectory is longer than horizon, randomly crop a window
        if self.horizon is not None and obs.size(0) > self.horizon:
            L = obs.size(0)
            start = torch.randint(0, L - self.horizon + 1, (1,)).item()
            obs = obs[start : start + self.horizon]
            action = action[start : start + self.horizon]

        # --- Fix for transformer‐hybrid policy: Robomimic expects H×W×C
        if self.is_image_based and self.args and self.args.model_type == 'transformer':
            obs = obs.permute(0, 2, 3, 1).contiguous()

        return {'obs': obs, 'action': action}

    def load_file(self, file_path):
        """
        Load demonstrations from pickle file.

        Copy-pasted (and lightly adapted) from run_bc.py:
        - Handles both state- and image-based pickles
        - Respects load_ob_image_mode
        - Returns list of {'obs': Tensor[T,...], 'action': Tensor[T,...]}
        """
        print('loading demonstration data to RAM before training....')
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # privileged: load both state and image trajectories
        if getattr(self, 'privileged', False):
            return self._load_privileged_data(data)

        # standard: load obs trajectories
        return self._load_standard_data(data)

    def _load_privileged_data(self, data):
        """Load privileged data with both state and image trajectories."""
        # load state trajectories
        if 'ob_trajs' in data:
            state_trajs = data['ob_trajs']
        elif 'obs_trajs' in data:
            state_trajs = data['obs_trajs']
        else:
            raise KeyError("No 'ob_trajs' or 'obs_trajs' in pickle for privileged mode")

        # load image trajectories
        if self.args.load_ob_image_mode == 'direct':
            img_trajs = data['ob_img_trajs']
            img_trajs = np.transpose(img_trajs, (0, 1, 4, 2, 3))
        else:
            img_trajs = data['ob_img_trajs']
        action_trajs = data['action_trajs']

        trajectories = []
        for state_seq, img_seq, act_seq in zip(state_trajs, img_trajs, action_trajs):
            state_tensor = torch.tensor(state_seq, dtype=torch.float32)
            image_tensor = torch.tensor(img_seq, dtype=torch.float32)
            action_tensor = torch.tensor(act_seq, dtype=torch.float32)
            trajectories.append({
                'state': state_tensor,
                'image': image_tensor,
                'action': action_tensor
            })
        print('finished loading privileged data.')
        return trajectories

    def _load_standard_data(self, data):
        """Load standard data with observation and action trajectories."""
        print('loading all data to RAM before training....')

        if self.is_image_based:
            if self.args.load_ob_image_mode == 'direct':
                ob_trajs = data['ob_img_trajs']
                ob_trajs = np.transpose(ob_trajs, (0, 1, 4, 2, 3))
            else:
                ob_trajs = data['ob_img_trajs']
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
            trajectories.append({
                'obs': obs_tensor,
                'action': action_tensor
            })
        print('finished loading data.')
        return trajectories
