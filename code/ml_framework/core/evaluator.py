from __future__ import annotations
"""Evaluation utilities for diffusion_policy.

This module provides an Evaluator class for running rollouts in a SoftGym environment
using the current policy checkpoints during training.

Key design goals:
- Minimal dependencies (rely on existing core.env SoftGymEnvSB3)
- Stateless except for environment handle; no global side-effects
- Clean interface for train loop: evaluator.run(policy, epoch) -> metrics dict
- Optional video recording of first episode per evaluation call

Returned metrics (per evaluation invocation):
- eval/normalized_performance_mean
- eval/normalized_performance_std
- eval/avg_reward
- eval/avg_ep_length
Optionally: eval/video (path) if save_video True and writer provided
"""
from typing import Dict, Any, Optional, Tuple
import os
import numpy as np
import torch
from collections import deque
from softgym.utils.visualization import save_numpy_as_gif
from core.env import SoftGymEnvSB3


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


class Evaluator:
    def __init__(self,
                 env_kwargs: Dict[str, Any],
                 max_episode_steps: int,
                 n_obs_steps: int,
                 is_image_based: bool,
                 model_type: str,
                 device: torch.device,
                 video_dir: Optional[str] = None,
                 gif_size: int = 128,
                 verbose: bool = True):
        self.env = SoftGymEnvSB3(**env_kwargs)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.is_image_based = is_image_based
        self.model_type = model_type
        self.device = device
        self.verbose = verbose
        self.gif_size = gif_size
        self.video_dir = _ensure_dir(video_dir) if video_dir else None

    def _init_memory_buffers(self):
        # Only used for privileged or image inference requiring a temporal window.
        return dict(state=None, image=None, generic=None)

    def _update_and_build_obs(self,
                              memory: Dict[str, Any],
                              obs_raw,
                              policy,
                              env_img_size: int) -> Dict[str, torch.Tensor]:
        """Mirror logic from old pipeline but adapted for new policy interfaces."""
        if self.model_type == 'privileged':
            # Privileged inference uses image-only branch for predict_action; we enqueue dummy state if needed
            if memory['state'] is None:
                # allocate deque of zeros for state
                dummy_state = torch.zeros(policy.lowdim_branch.obs_dim, dtype=torch.float32, device=self.device)
                memory['state'] = deque([dummy_state.clone() for _ in range(self.n_obs_steps)], maxlen=self.n_obs_steps)
            if memory['image'] is None:
                img = self.env.get_image(env_img_size, env_img_size)
                img = np.transpose(img, (2, 0, 1))
                init_t = torch.tensor(img, dtype=torch.float32, device=self.device)
                memory['image'] = deque([init_t.clone() for _ in range(self.n_obs_steps)], maxlen=self.n_obs_steps)
            # append latest image
            img = self.env.get_image(env_img_size, env_img_size)
            img = np.transpose(img, (2, 0, 1))
            img_t = torch.tensor(img, dtype=torch.float32, device=self.device)
            memory['image'].append(img_t)
            img_seq = torch.stack(list(memory['image']), dim=0)  # [T, C, H, W]
            batch_img = img_seq.unsqueeze(0)
            return {"image": batch_img}
        elif self.is_image_based:
            if memory['generic'] is None:
                img = self.env.get_image(env_img_size, env_img_size)
                img = np.transpose(img, (2, 0, 1))
                init_t = torch.tensor(img, dtype=torch.float32, device=self.device)
                memory['generic'] = deque([init_t.clone() for _ in range(self.n_obs_steps)], maxlen=self.n_obs_steps)
            img = self.env.get_image(env_img_size, env_img_size)
            img = np.transpose(img, (2, 0, 1))
            img_t = torch.tensor(img, dtype=torch.float32, device=self.device)
            memory['generic'].append(img_t)
            seq = torch.stack(list(memory['generic']), dim=0)
            # For transformer-image we accept either [B,T,C,H,W] or flatten. We'll use [B,T,C,H,W]
            batch = seq.unsqueeze(0)
            # Model expects key 'obs'
            return {"obs": batch}
        else:
            # low-dim state observation
            if memory['generic'] is None:
                s = torch.tensor(obs_raw, dtype=torch.float32, device=self.device)
                memory['generic'] = deque([s.clone() for _ in range(self.n_obs_steps)], maxlen=self.n_obs_steps)
            s = torch.tensor(obs_raw, dtype=torch.float32, device=self.device)
            memory['generic'].append(s)
            seq = torch.stack(list(memory['generic']), dim=0)
            batch = seq.unsqueeze(0)
            return {"obs": batch}

    @torch.no_grad()
    def run(self,
            policy,
            num_episodes: int,
            save_video: bool = False,
            epoch: Optional[int] = None,
            env_img_size: int = 128) -> Tuple[Dict[str, float], Optional[str]]:
        policy.eval()
        total_reward = 0.0
        total_length = 0
        final_perfs = []
        saved_gif_path = None
        first_episode = True

        for ep in range(num_episodes):
            obs = self.env.reset()
            memory = self._init_memory_buffers()
            ep_reward = 0.0
            ep_len = 0
            ep_perfs = []
            frames = []
            policy.reset()

            if save_video and first_episode and self.video_dir:
                frames.append(self.env.get_image(self.gif_size, self.gif_size))

            for step in range(self.max_episode_steps):
                obs_input = self._update_and_build_obs(memory, obs, policy, env_img_size)
                act_dict = policy.predict_action(obs_input)
                action = act_dict['action'][0, 0].detach().cpu().numpy()
                obs, reward, done, info = self.env.step(action)
                ep_reward += reward
                ep_len += 1
                perf = info.get('normalized_performance')
                if isinstance(perf, (list, tuple, np.ndarray)):
                    try:
                        perf = perf[-1]
                    except Exception:
                        perf = float(perf[0]) if len(perf) > 0 else 0.0
                if perf is not None:
                    ep_perfs.append(float(perf))
                if save_video and first_episode and self.video_dir:
                    frames.append(self.env.get_image(self.gif_size, self.gif_size))
                if done:
                    break

            final_perf = ep_perfs[-1] if ep_perfs else 0.0
            final_perfs.append(final_perf)
            total_reward += ep_reward
            total_length += ep_len
            if self.verbose:
                print(f"Eval Episode {ep}: final_norm_perf={final_perf:.4f} reward={ep_reward:.2f} length={ep_len}")
            if save_video and first_episode and self.video_dir and save_numpy_as_gif is not None:
                epoch_str = str(epoch + 1) if epoch is not None else 'eval'
                gif_name = f"epoch_{epoch_str}_perf_{final_perf:.4f}.gif"
                saved_gif_path = os.path.join(self.video_dir, gif_name)
                try:
                    save_numpy_as_gif(np.array(frames), saved_gif_path)
                except Exception as e:
                    print(f"[Evaluator] Failed to save GIF: {e}")
                    saved_gif_path = None
                first_episode = False

        metrics = {
            'eval/normalized_performance_mean': float(np.mean(final_perfs)) if final_perfs else 0.0,
            'eval/normalized_performance_std': float(np.std(final_perfs)) if final_perfs else 0.0,
            'eval/avg_reward': total_reward / max(1, num_episodes),
            'eval/avg_ep_length': total_length / max(1, num_episodes)
        }
        if self.verbose:
            print("\n[Evaluator] Summary:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
        return metrics, saved_gif_path