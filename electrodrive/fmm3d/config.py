from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

# -------------------------
# FMM / spherical-harmonics limits
# -------------------------
# These constants must stay in sync with the spherical-harmonics backend
# (electrodrive.fmm3d.spherical_harmonics._L_MAX_HARD_LIMIT and friends).
#
# The Laplace FMM backend currently uses:
#   - l_max = p           for P2M / M2M / L2L, and
#   - l_max = 2 * p       for M2L translations (irregular solid harmonics).
#
# The spherical-harmonics implementation enforces l_max <= 32, so we
# must keep 2 * expansion_order <= 32 at the configuration level.
FMM_L_MAX_HARD_LIMIT: int = 32
FMM_M2L_LMAX_FACTOR: int = 2

# -------------------------
# FMM backend configuration
# -------------------------

BackendKind = Literal["cpu", "gpu", "auto"]
PrecisionKind = Literal["single", "double", "mixed"]


@dataclass
class FmmConfig:
    """Configuration for the FMM backend.

    This is designed to be compatible with:

    - :mod:`electrodrive.fmm3d.multipole_operators._validate_fmm_config`
      which expects the attributes:
        * expansion_order
        * mac_theta
        * dtype (torch.float32 or torch.float64)
        * p2p_batch_size (optional, positive int)
    - :mod:`electrodrive.fmm3d.kernels_cpu.apply_p2p_cpu`, which reads
      ``cfg.p2p_batch_size`` as a P2P tile size hint.

    Parameters
    ----------
    kernel:
        String identifier for the kernel; e.g. "laplace_single_layer".
    expansion_order:
        Multipole expansion order p. Must be positive and small enough
        that the spherical-harmonics backend can support both P2M/M2M
        and M2L. For the current Laplace FMM backend this implies

            FMM_M2L_LMAX_FACTOR * expansion_order <= FMM_L_MAX_HARD_LIMIT

        i.e. 2 * p <= 32.
    mac_theta:
        Opening angle for the MAC (multipole acceptance criterion).
        Values in [0.1, 0.9] are typically safe.
    leaf_size:
        Target maximum number of points per leaf box in the FMM tree.
    backend:
        Logical backend selector: "cpu", "gpu", or "auto".
    precision:
        High-level precision policy:
          * "single" -> float32 expansions
          * "double" -> float64 expansions
          * "mixed"  -> currently treated like "single" at the config
                        level; mixed-precision details are handled inside
                        kernels.
    use_fft_m2l:
        Whether to allow FFT-accelerated M2L for large p.
    use_gpu:
        Whether GPU backends are allowed at all.
    use_multi_gpu:
        Whether multi-GPU execution is allowed.
    use_mpi:
        Whether MPI-based distribution is allowed.
    dtype:
        Torch dtype used for multipole/local expansions. This must be
        ``torch.float32`` or ``torch.float64``. If not explicitly
        provided, it is derived from ``precision`` in ``__post_init__``.
    p2p_batch_size:
        Optional soft limit on the number of point–point interactions
        per P2P batch. If ``None``, the CPU kernels fall back to their
        internal defaults.
    """

    # High-level algorithmic settings
    kernel: str = "laplace_single_layer"
    expansion_order: int = 8
    mac_theta: float = 0.5
    leaf_size: int = 64

    # Backend / execution policy
    backend: BackendKind = "auto"
    precision: PrecisionKind = "mixed"
    use_fft_m2l: bool = True
    use_gpu: bool = True
    use_multi_gpu: bool = False
    use_mpi: bool = False

    # Numerics / plumbing required by the rest of the stack
    dtype: Optional[torch.dtype] = None
    p2p_batch_size: Optional[int] = None

    def __post_init__(self) -> None:
        """Fill in derived fields and run basic validation."""
        # Derive dtype from the high-level precision policy if not set.
        if self.dtype is None:
            if self.precision == "double":
                self.dtype = torch.float64
            else:
                # "single" and "mixed" currently share float32 expansions;
                # mixed-precision details (e.g. accumulation in float64)
                # are handled at the kernel level.
                self.dtype = torch.float32

        self.validate()

    def validate(self) -> None:
        """Perform cheap validation of basic parameters.

        More sophisticated validation (e.g. hardware/MPI consistency)
        should live in orchestration code, not here.
        """
        # Expansion / MAC / tree sanity
        if self.expansion_order <= 0:
            raise ValueError("expansion_order must be positive")

        if not (0.1 <= self.mac_theta <= 0.9):
            raise ValueError("mac_theta should be in [0.1, 0.9] for stability")

        if self.leaf_size <= 0:
            raise ValueError("leaf_size must be positive")

        # Spherical-harmonics / multipole-order consistency.
        #
        # The current Laplace FMM backend uses:
        #   - l_max = p for P2M / M2M / L2L, and
        #   - l_max = FMM_M2L_LMAX_FACTOR * p for M2L translations,
        # with the spherical-harmonics backend enforcing
        # l_max <= FMM_L_MAX_HARD_LIMIT.
        #
        # Enforcing this here guarantees that downstream calls to
        # electrodrive.fmm3d.spherical_harmonics cannot hit l_max values
        # that violate their internal hard limit.
        l_max_m2l = FMM_M2L_LMAX_FACTOR * self.expansion_order
        if l_max_m2l > FMM_L_MAX_HARD_LIMIT:
            raise ValueError(
                "expansion_order too large for current spherical-harmonics "
                "backend: M2L uses "
                f"l_max={l_max_m2l} (factor {FMM_M2L_LMAX_FACTOR}×p) "
                f"but FMM_L_MAX_HARD_LIMIT={FMM_L_MAX_HARD_LIMIT}. "
                "Reduce expansion_order or upgrade the spherical-harmonics backend."
            )

        # Dtype must match what multipole_operators._validate_fmm_config expects
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError(
                "dtype must be torch.float32 or torch.float64; "
                f"got {self.dtype!r}"
            )

        # P2P batch size (used by kernels_cpu.apply_p2p_cpu)
        if self.p2p_batch_size is not None and self.p2p_batch_size <= 0:
            raise ValueError("p2p_batch_size must be positive if specified")

        # Backend/precision semantics are enforced via Literal types;
        # no extra checking needed here for now.


__all__ = [
    "FMM_L_MAX_HARD_LIMIT",
    "FMM_M2L_LMAX_FACTOR",
    "BackendKind",
    "PrecisionKind",
    "FmmConfig",
]
