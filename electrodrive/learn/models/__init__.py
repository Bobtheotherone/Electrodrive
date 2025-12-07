"""
Model exports for electrodrive.learn.models.

Note:
- PINNHarmonic implements both the default and "large" residual/checkpointed
  variants; the desired behavior is selected via ExperimentConfig.model.model_type
  and params.
"""

from .pinn_harmonic import PINNHarmonic
from .moi_symbolic import MoISymbolic

__all__ = ["PINNHarmonic", "MoISymbolic"]