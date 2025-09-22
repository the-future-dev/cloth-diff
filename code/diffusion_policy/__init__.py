"""diffusion_policy package public API skeleton.

Exports the command-line entrypoint for convenience so users can:

	python -m diffusion_policy --args...

after installing a console script that points to `diffusion_policy.main`.
"""

from diffusion_policy.core.cli import main as main  # re-export

__all__ = ["main"]

