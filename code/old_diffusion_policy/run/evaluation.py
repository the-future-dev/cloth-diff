"""
Evaluation module for diffusion policies.

Handles evaluation of trained policies in the environment.
"""

import os
import tqdm
import wandb
import numpy as np
import torch
from collections import deque

from core.env import SoftGymEnvSB3
from core.utils import set_seed_everywhere, make_dir
from softgym.utils.visualization import save_numpy_as_gif


class Evaluation:
    """
    Evaluation class for running policy evaluation in SoftGym environments.
    """

    def __init__(self, args, env_kwargs):
        """
        Initialize evaluation environment.

        Args:
            args: Arguments object with evaluation settings
            env_kwargs: Environment configuration dictionary
        """
        # initialize SoftGym environment
        self.env = SoftGymEnvSB3(**env_kwargs)
        self.args = args

        # set evaluation device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # prepare a folder for saving per-run videos if requested
        if args.eval_videos:
            eval_base = (
                os.path.join(args.model_save_dir, args.folder_name)
                if not args.is_eval
                else os.path.dirname(args.test_checkpoint)
            )
            self.eval_video_path = make_dir(os.path.join(eval_base, "videos"))

    def evaluate(self, policy, num_episodes=50, seed=None, save_video=False, epoch=None):
        """
        Evaluate policy over multiple episodes.

        Args:
            policy: Policy to evaluate
            num_episodes: Number of episodes to run
            seed: Random seed for evaluation
            save_video: Whether to save evaluation videos
            epoch: Current training epoch

        Returns:
            avg_perf: Average normalized performance
            std_perf: Standard deviation of normalized performance
            avg_reward: Average reward
            avg_length: Average episode length
            saved_gif_path: Path to saved GIF (or None)
        """
        if seed is not None:
            set_seed_everywhere(seed)

        total_reward = 0.0
        total_length = 0
        total_normalized_performance_final = []

        saved_gif_path = None
        first_episode = True

        for episode in tqdm.tqdm(range(num_episodes), desc="Evaluating"):
            obs = self.env.reset()

            # --- initialize deques for privileged vs. pure image/state inference ---
            memory_state, memory_image, memory = self._initialize_memory_buffers(obs)

            episode_reward = 0
            episode_length = 0
            episode_normalized_perf = []
            frames = []

            if save_video and first_episode:
                frames.append(self.env.get_image(self.args.eval_gif_size, self.args.eval_gif_size))

            policy.reset()

            for step in range(self.args.max_steps):
                obs_dict_input = self._prepare_observation_input(
                    memory_state, memory_image, memory, obs
                )

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict_input)
                    action = action_dict['action'][0, 0].cpu().numpy()

                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                episode_normalized_perf.append(info['normalized_performance'])

                if save_video and first_episode:
                    frames.append(self.env.get_image(self.args.eval_gif_size, self.args.eval_gif_size))

                if done:
                    break

            # final performance for this episode
            ep_perf = episode_normalized_perf[-1]
            total_normalized_performance_final.append(ep_perf)
            total_reward += episode_reward
            total_length += episode_length

            print(f'Episode {episode}, Final Normalized Performance: {ep_perf:.4f}, '
                  f'Reward: {episode_reward:.2f}, Length: {episode_length}')

            if save_video and first_episode:
                # handle missing epoch (e.g. in main_testing where epoch=None)
                epoch_str = str(epoch + 1) if epoch is not None else "test"
                gif_path = os.path.join(
                    self.eval_video_path,
                    f"epoch_{epoch_str}_perf_{ep_perf:.4f}.gif"
                )
                save_numpy_as_gif(np.array(frames), gif_path)
                saved_gif_path = gif_path
                first_episode = False

        # compute statistics
        avg_perf = np.mean(total_normalized_performance_final)
        std_perf = np.std(total_normalized_performance_final)
        avg_reward = total_reward / num_episodes
        avg_length = total_length / num_episodes

        print("\nEvaluation Summary:")
        print(f"Average Final Normalized Performance: {avg_perf:.4f}")
        print(f"Std   Final Normalized Performance: {std_perf:.4f}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Episode Length: {avg_length:.2f}")

        return avg_perf, std_perf, avg_reward, avg_length, saved_gif_path

    def _initialize_memory_buffers(self, initial_obs):
        """Initialize memory buffers for different policy types."""
        memory_state = None
        memory_image = None
        memory = None

        if self.args.model_type == 'privileged':
            # For privileged model evaluation, state is a dummy zero vector
            state_dim = policy.state_encoder.state_dim
            dummy_state_vec = torch.zeros((state_dim,), dtype=torch.float32).to(self.device)
            memory_state = deque([dummy_state_vec.clone() for _ in range(self.args.n_obs_steps)],
                                maxlen=self.args.n_obs_steps)
            # image deque
            init_img = self.env.get_image(self.args.env_img_size, self.args.env_img_size)
            init_img = np.transpose(init_img, (2, 0, 1))  # HWC→CHW
            init_img_tensor = torch.tensor(init_img, dtype=torch.float32).to(self.device)
            memory_image = deque([init_img_tensor.clone() for _ in range(self.args.n_obs_steps)],
                                maxlen=self.args.n_obs_steps)
        elif self.args.is_image_based:
            init_img = self.env.get_image(self.args.env_img_size, self.args.env_img_size)
            init_img = np.transpose(init_img, (2, 0, 1))
            init_tensor = torch.tensor(init_img, dtype=torch.float32).to(self.device)
            memory = deque([init_tensor.clone() for _ in range(self.args.n_obs_steps)],
                          maxlen=self.args.n_obs_steps)
        else:
            init_obs = torch.tensor(initial_obs, dtype=torch.float32).to(self.device)
            memory = deque([init_obs.clone() for _ in range(self.args.n_obs_steps)],
                          maxlen=self.args.n_obs_steps)

        return memory_state, memory_image, memory

    def _prepare_observation_input(self, memory_state, memory_image, memory, current_obs):
        """Prepare observation input dictionary for policy."""
        if self.args.model_type == 'privileged':
            # PRIVILEGED: append state + image streams
            # state_vec is the dummy zero vector
            memory_state.append(memory_state[-1].clone())
            img = self.env.get_image(self.args.env_img_size, self.args.env_img_size)
            img = np.transpose(img, (2, 0, 1))
            img_tensor = torch.tensor(img, dtype=torch.float32).to(self.device)
            memory_image.append(img_tensor)
            state_seq = torch.stack(list(memory_state), dim=0)
            state_batch = state_seq.unsqueeze(0)
            img_seq_from_deque = torch.stack(list(memory_image), dim=0)
            img_seq_permuted = img_seq_from_deque.permute(0, 2, 3, 1).contiguous()
            image_batch = img_seq_permuted.unsqueeze(0)
            obs_dict_input = {'state': state_batch, 'image': image_batch}
        elif self.args.is_image_based:
            # IMAGE-ONLY: use visual buffer
            img = self.env.get_image(self.args.env_img_size, self.args.env_img_size)
            img = np.transpose(img, (2, 0, 1))
            img_tensor = torch.tensor(img, dtype=torch.float32).to(self.device)
            memory.append(img_tensor)
            obs_seq = torch.stack(list(memory), dim=0)
            obs_tensor = obs_seq.unsqueeze(0)
            if self.args.model_type == 'transformer':
                obs_tensor = obs_tensor.permute(0, 1, 3, 4, 2).contiguous()
                obs_dict_input = {'obs': obs_tensor}
            else:
                obs_dict_input = {'obs': obs_tensor}
        else:
            # STATE-ONLY: use low‐dim buffer
            obs_vec = torch.tensor(current_obs, dtype=torch.float32).to(self.device)
            memory.append(obs_vec)
            obs_seq = torch.stack(list(memory), dim=0)
            obs_tensor = obs_seq.unsqueeze(0)
            obs_dict_input = {'obs': obs_tensor}

        return obs_dict_input

    def evaluate_five_seeds(self, policy):
        """
        Evaluate policy over 5 different seeds with 20 episodes each.

        Args:
            policy: Policy to evaluate

        Returns:
            None (prints results and saves to file)
        """
        seeds = [100, 200, 300, 400, 500]
        all_final_normalized_perf = []
        policy.eval()

        episode_counter = 0  # Counter for tracking total episodes across all seeds

        for seed in seeds:
            print(f"\nEvaluating with seed {seed}")
            set_seed_everywhere(seed)

            # Evaluate for 20 episodes per seed
            for ep in tqdm.tqdm(range(20), desc=f"Seed {seed} Episodes"):
                obs = self.env.reset()

                memory_state, memory_image, memory = self._initialize_memory_buffers(obs)

                policy.reset()

                for step in range(self.args.max_steps):
                    obs_dict_input = self._prepare_observation_input(
                        memory_state, memory_image, memory, obs
                    )

                    with torch.no_grad():
                        action_dict = policy.predict_action(obs_dict_input)
                        action = action_dict['action'][0, 0].cpu().numpy()

                    obs, reward, done, info = self.env.step(action)

                    # Safely extract the final normalized performance
                    perf_info = info.get('normalized_performance')
                    if isinstance(perf_info, (list, tuple, np.ndarray)):
                        ep_perf = perf_info[-1] if done else perf_info
                    else:
                        ep_perf = perf_info
                    ep_perf = float(ep_perf)
                    all_final_normalized_perf.append(ep_perf)

                    # Log each individual episode performance to WandB
                    if self.args.wandb:
                        wandb.log(
                            {"eval_5seeds/single_normalized_performance": ep_perf,
                             "eval_5seeds/episode_index": episode_counter},  # Add unique index for each episode
                        )

                    if done:
                        break

                episode_counter += 1  # Increment episode counter after each episode

        # Calculate and print statistics over all 100 episodes (5 seeds * 20 episodes)
        all_final_normalized_perf = np.array(all_final_normalized_perf)

        # Save results to .npy file, mimicking run_bc structure
        ckpt_file_path = self.args.test_checkpoint
        run_name = os.path.basename(os.path.dirname(os.path.dirname(ckpt_file_path)))
        npy_file_path = os.path.join(
            os.path.dirname(ckpt_file_path),
            f'{run_name}_eval_5seeds.npy'
        )
        print(f"Saving 5-seed evaluation results to: {npy_file_path}")
        np.save(npy_file_path, all_final_normalized_perf)

        print('\n!!!!!!! Final Normalized Performance Statistics (100 episodes over 5 seeds) !!!!!!!')
        print(f'Mean: {np.mean(all_final_normalized_perf):.4f}')
        print(f'Std: {np.std(all_final_normalized_perf):.4f}')
        print(f'Median: {np.median(all_final_normalized_perf):.4f}')
        print(f'25th Percentile: {np.percentile(all_final_normalized_perf, 25):.4f}')
        print(f'75th Percentile: {np.percentile(all_final_normalized_perf, 75):.4f}')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        if self.args.wandb:
            wandb.log({
                "eval_5seeds/info_normalized_performance_mean":   np.mean(all_final_normalized_perf),
                "eval_5seeds/info_normalized_performance_std":    np.std(all_final_normalized_perf),
                "eval_5seeds/info_normalized_performance_median": np.median(all_final_normalized_perf),
                "eval_5seeds/info_normalized_performance_25th":   np.percentile(all_final_normalized_perf, 25),
                "eval_5seeds/info_normalized_performance_75th":   np.percentile(all_final_normalized_perf, 75),
            })
