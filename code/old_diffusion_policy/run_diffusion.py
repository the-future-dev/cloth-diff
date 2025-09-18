"""
Main entry point for diffusion policy training and evaluation.

This module has been refactored to use single-responsibility modules from the `run` package.
"""

from diffusion_policy.run.config import setup_parser, setup_environment
from diffusion_policy.run.training import main_training
from diffusion_policy.run.testing import main_testing

# Set up argument parser and configuration
parser = setup_parser()
args = parser.parse_args()
env_kwargs = setup_environment(args)

# Classes and functions moved to modular files in diffusion_policy/run/

# Main execution logic - delegate to appropriate module based on mode
if __name__ == "__main__":
    if args.is_eval:
        main_testing(args, env_kwargs)
    else:
        main_training(args, env_kwargs)