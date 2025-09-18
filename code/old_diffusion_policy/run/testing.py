"""
Testing module for diffusion policies.

Handles evaluation and testing of trained policies.
"""

import os
import torch
import wandb

from diffusion_policy.model.common.normalizer import LinearNormalizer
from .evaluation import Evaluation
from .policy_factory import create_diffusion_policy


def load_checkpoint_for_testing(args, device):
    """
    Load checkpoint and prepare policy for testing.

    Returns:
        policy: Loaded and configured policy
        checkpoint: Checkpoint data
    """
    # Create policy
    policy, _ = create_diffusion_policy(args, None)  # We don't need env_kwargs for testing
    policy = policy.to(device)

    # Load checkpoint
    print(f"Loading checkpoint from: {args.test_checkpoint}")
    checkpoint = torch.load(args.test_checkpoint, map_location=device)
    policy.load_state_dict(checkpoint['model_state_dict'])

    # --- Load Normalizer ---
    if hasattr(policy, 'normalizer') and checkpoint.get('normalizer') is not None:
        print("Loading normalizer state from checkpoint.")
        policy.normalizer.load_state_dict(checkpoint['normalizer'])
        # Ensure normalizer is on the correct device
        policy.normalizer.to(device)
    elif hasattr(policy, 'normalizer'):
         print("Warning: Policy has a normalizer, but no normalizer state found in checkpoint.")

    # Load EMA averaged_model parameters directly into the policy network if EMA was used during training
    if args.use_ema and 'ema_state_dict' in checkpoint:
        print("Loading EMA parameters from checkpoint.")
        if args.model_type == 'privileged':
            policy.model.load_state_dict(checkpoint['ema_state_dict'])
        elif args.is_image_based and args.model_type == 'transformer':
            policy.model.load_state_dict(checkpoint['ema_state_dict'])
        elif args.is_image_based:
            policy.nets['action_model'].load_state_dict(checkpoint['ema_state_dict'])
        else:
            policy.model.load_state_dict(checkpoint['ema_state_dict'])
    elif args.use_ema:
        print("Warning: --use_ema is True, but no 'ema_state_dict' found in checkpoint. Using standard model weights.")

    return policy, checkpoint


def setup_wandb_for_testing(checkpoint, args):
    """
    Set up WandB for testing if checkpoint contains run information.

    Returns:
        Updated args with wandb enabled if applicable
    """
    # ────────────── attach to WandB if this checkpoint contains a run id ──────────────
    run_id   = checkpoint.get('wandb_run_id')
    run_name = checkpoint.get('wandb_run_name')
    if run_id:
        project_id = "cloth-diff"  # must match your training project name
        print(f"Re-attaching to WandB run: id={run_id}, name={run_name}")
        wandb.init(project=project_id,
                   id=run_id,
                   resume="must",
                   name=run_name)
        args.wandb = True  # ensure that downstream logging is enabled

    return args


def run_single_seed_evaluation(args, env_kwargs):
    """
    Run evaluation for a single seed.

    Args:
        args: Arguments object
        env_kwargs: Environment configuration

    Returns:
        avg_normalized_perf: Average normalized performance
        std_normalized_perf: Standard deviation
        avg_reward: Average reward
        avg_ep_length: Average episode length
        saved_gif_path: Path to saved GIF (or None)
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load policy and checkpoint
    policy, checkpoint = load_checkpoint_for_testing(args, device)

    # Set up WandB if applicable
    args = setup_wandb_for_testing(checkpoint, args)

    # Set policy to evaluation mode
    policy.eval()

    # Create evaluator
    evaluator = Evaluation(args, env_kwargs)

    # Run evaluation
    print(f"Running evaluation for {args.num_eval_eps} episodes...")
    # Evaluate and print results
    avg_normalized_perf, std_normalized_perf, avg_reward, avg_ep_length, saved_gif_path = evaluator.evaluate(
        policy,
        num_episodes=args.num_eval_eps,
        save_video=args.eval_videos
    )
    # Final print handled within evaluate method, but can add a summary here if needed
    print("\nSingle-seed Evaluation Complete.")

    return avg_normalized_perf, std_normalized_perf, avg_reward, avg_ep_length, saved_gif_path


def run_multi_seed_evaluation(args, env_kwargs):
    """
    Run evaluation over multiple seeds.

    Args:
        args: Arguments object
        env_kwargs: Environment configuration
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load policy and checkpoint
    policy, checkpoint = load_checkpoint_for_testing(args, device)

    # Set up WandB if applicable
    args = setup_wandb_for_testing(checkpoint, args)

    # Set policy to evaluation mode
    policy.eval()

    # Create evaluator
    evaluator = Evaluation(args, env_kwargs)

    # Run multi-seed evaluation
    print("Running evaluation over 5 seeds (100 episodes total)...")
    evaluator.evaluate_five_seeds(policy)  # This function handles its own printing and saving


def main_testing(args, env_kwargs):
    """
    Main testing function that handles evaluation of trained policies.
    """
    if args.eval_over_five_seeds:
        run_multi_seed_evaluation(args, env_kwargs)
        # close out the WandB run
        if args.wandb:
            wandb.finish()
    else:
        avg_normalized_perf, std_normalized_perf, avg_reward, avg_ep_length, saved_gif_path = run_single_seed_evaluation(
            args, env_kwargs
        )
        # log single-seed metrics back to the attached run
        if args.wandb:
            log_dict = {
                "test/info_normalized_performance_mean": avg_normalized_perf,
                "test/info_normalized_performance_std":  std_normalized_perf,
                "test/avg_reward":                      avg_reward,
                "test/avg_ep_length":                   avg_ep_length,
            }
            if saved_gif_path:
                log_dict["test/eval_video"] = wandb.Video(saved_gif_path, fps=10, format="gif")
            wandb.log(log_dict)
            wandb.finish()
