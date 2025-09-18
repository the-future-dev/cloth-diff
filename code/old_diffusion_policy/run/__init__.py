"""
Run-time helpers for diffusion policy training/evaluation.

This package contains single-responsibility modules extracted from
`run_diffusion.py` for easier testing and maintenance.

Modules:
- config: Argument parsing and environment setup
- policy_factory: Creation of diffusion policies
- demonstrations: Dataset loading and processing
- evaluation: Policy evaluation in environments
- training: Training loop orchestration
- testing: Testing and evaluation setup
"""

from .config import setup_parser, setup_environment
from .policy_factory import create_diffusion_policy
from .demonstrations import Demonstrations
from .evaluation import Evaluation
from .training import main_training
from .testing import main_testing

__all__ = [
    'setup_parser',
    'setup_environment',
    'create_diffusion_policy',
    'Demonstrations',
    'Evaluation',
    'main_training',
    'main_testing'
]


