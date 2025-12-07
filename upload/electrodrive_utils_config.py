from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING, Any

import math

# Use TYPE_CHECKING to prevent circular imports at runtime.
# During runtime, we will treat these as generic objects.
if TYPE_CHECKING:
    # We cannot import FmmConfig from fmm3d/config at runtime if modules
    # in fmm3d/ are simultaneously importing from electrodrive.utils.config.
    BackendKind = Literal["external", "torch_tiled", "keops", "auto"]
    PrecisionKind = Literal["float32", "float64"]

    class FmmConfig:
        def __init__(self, **kwargs: Any) -> None:
            ...
        expansion_order: int = 8
        mac_theta: float = 0.5
        leaf_size: int = 64
        dtype: "torch.dtype" = None  # type: ignore[name-defined]
else:
    # Ensure they are defined for runtime use outside of the FMM module
    # (The FMM module should define and correctly export these types)
    BackendKind = object
    PrecisionKind = object
    FmmConfig = object


# -------------------------
# Physics constants (SI)
# -------------------------
EPS_0: float = 8.854187817e-12
K_E: float = 1.0 / (4.0 * math.pi * EPS_0)  # Coulomb's constant

# -------------------------
# Certification thresholds
# (used by core/certify.py and CLI gating)
# -------------------------
# These are *default* per-test tolerances; they can be overridden
# at higher levels (e.g. per-experiment configs, CLI flags, etc.).

# Boundary condition residuals (Dirichlet / Neumann)
EPS_BC: float = 1e-6

# Dual (reciprocity-based) checks
EPS_DUAL: float = 1e-6

# PDE residual (Laplace / Poisson)
EPS_PDE: float = 1e-5

# Energy norm comparisons (e.g. Rayleigh quotient, Dirichlet energy)
EPS_ENERGY: float = 1e-5

# Mean value property (harmonic fields)
EPS_MEAN_VAL: float = 1e-5

# Maximum principle (no spurious overshoot/undershoot)
EPS_MAX_PRINCIPLE: float = 1e-6

# Reciprocity checks (swapping source/obs roles)
EPS_RECIPROCITY: float = 1e-6

# -------------------------
# Defaults (CLI / planners)
# -------------------------
DEFAULT_SEED: int = 42
DEFAULT_SOLVE_DTYPE: str = "float64"


# -------------------------
# Solver configuration dataclasses
# (global / non-FMM-specific)
# -------------------------


@dataclass
class BEMConfig:
    """
    Configuration for the BEM solver.

    Notes
    -----
    - Keep the first three fields (use_gpu, fp64, initial_h) compatible
      with earlier positional construction in the codebase.
    - Extra fields have safe defaults and are typically consumed via kwargs.
    """

    # Execution / numeric precision
    use_gpu: bool = True
    fp64: bool = False

    # Start a bit finer to avoid early plateaus and improve BC/dual checks
    initial_h: float = 0.2

    # Refinement / solve knobs
    # Simple h-refinement policy. Rough heuristic:
    # - increase resolution while certification is improving
    # - stop when h ~ h_min or improvement saturates.
    max_refine_passes: int = 3
    # Target L∞ BC residual used as a soft stopping criterion in bem_solve.
    target_bc_inf_norm: float = 1e-7
    linear_tol_factor: float = 0.1

    # GMRES defaults for matrix-free solves
    gmres_tol: float = 5e-8
    gmres_maxiter: int = 2000
    gmres_restart: int = 256
    # Less chatty progress
    gmres_log_every: int = 10

    # VRAM autotune knobs (matrix-free tiling)
    vram_autotune: bool = True
    # Fractional cap (fallback if target_peak_gb is unset). Be aggressive.
    target_vram_fraction: float = 0.95
    # Hard cap in case the device query fails. Safety bound.
    vram_cap_gb: float = 24.0
    # Ask solver to push tiles up toward this peak allocation if possible.
    # (Kept for back-compat; used only for autotune heuristics.)
    target_peak_gb: float = 20.0

    # Tiling safety & bounds
    min_tile: int = 1024
    max_tile: int = 1 << 18  # 262,144
    # If 0, solver computes a heuristic tile size from the above.
    tile_size: int = 0
    # Divide the estimated free memory by this to leave a safety margin.
    # Smaller divisor => bigger tiles (more VRAM usage).
    tile_mem_divisor: float = 2.0

    # Near-field quadrature controls
    # ------------------------------
    # use_near_quadrature:
    #   Enable higher-accuracy triangle quadrature *only* for evaluation
    #   at arbitrary target points via BEMSolution (non-diff, CPU-only).
    #   The learning / collocation stack enables this when using BEM as an
    #   oracle so that analytic-vs-BEM comparisons near conductor surfaces
    #   are dominated by modelling differences rather than quadrature error.
    #
    # use_near_quadrature_matvec:
    #   Additionally enable near-field corrections inside the GMRES
    #   matvec operator. This is substantially more expensive
    #   (O(#near_pairs) work per matvec) and therefore disabled by default.
    use_near_quadrature: bool = False
    use_near_quadrature_matvec: bool = False
    # Panels i and j (or a panel and a target) are considered "near" when
    # their separation is less than this factor times the sum of their
    # equal-area radii R = sqrt(A/pi).
    near_quadrature_distance_factor: float = 1.5
    # Base order used inside `standard_triangle_quadrature` for the near
    # rules. Currently 1 and 2 are supported.
    near_quadrature_order: int = 2


@dataclass
class MOIConfig:
    """
    Multi-objective / regularization configuration for model selection.

    This is intentionally small and mostly used to share a parsimony
    weight and the list of scalar metrics to consider.
    """

    # Names of scalar objectives to track (e.g. losses / error metrics).
    objectives: tuple[str, ...] = ("residual", "bc", "energy")

    # L2- or complexity-style regularization weight used by some drivers.
    parsimony_lambda: float = 1e-4


@dataclass
class OPConfig:
    """
    High-level optimization run configuration (outer loop).

    This is intentionally generic; the meaning of some fields depends
    on the driver (BayesOpt, ES, random search, etc.).
    """

    # Basic stopping / scheduling
    max_iters: int = 100
    patience: int = 10
    history_window: int = 20
    improvement_tol: float = 1e-3

    # Logging / checkpoints
    log_every: int = 1
    checkpoint_every: int = 10
    keep_best_k: int = 5


@dataclass
class CERTConfig:
    """
    Certification / gating configuration.

    Currently very small; most thresholds live in the EPS_* constants
    above, but this struct lets callers override the minimum pass rate
    required to accept a solution.
    """

    # Fraction of checks that must pass for certification to succeed
    min_pass_rate: float = 0.9


@dataclass
class PINNConfig:
    """
    Configuration for PINN-style PDE surrogates.

    This is deliberately conservative; callers are expected to override
    most of these from experiment configs or sweeps.
    """

    # Network architecture
    hidden_layers: int = 6
    hidden_width: int = 128
    activation: str = "tanh"

    # Training schedule
    epochs: int = 10_000
    batch_size: int = 65_536
    lr: float = 1e-3

    # Loss composition
    # These are logical names for loss terms that the training loop
    # is expected to understand (e.g. "residual", "bc", "energy"...).
    loss_terms: tuple[str, ...] = ("residual", "bc")

    # Loss weights (normalized; tweak as needed in callers)
    residual_weight: float = 1.0
    bc_weight: float = 1.0

    # Optimizer / dtype
    optimizer: str = "adam"
    dtype: str = DEFAULT_SOLVE_DTYPE


__all__ = [
    # physics / thresholds
    "EPS_0",
    "K_E",
    "EPS_BC",
    "EPS_DUAL",
    "EPS_PDE",
    "EPS_ENERGY",
    "EPS_MEAN_VAL",
    "EPS_MAX_PRINCIPLE",
    "EPS_RECIPROCITY",
    "DEFAULT_SEED",
    "DEFAULT_SOLVE_DTYPE",
    # FMM config types (re-exported)
    "BackendKind",
    "PrecisionKind",
    "FmmConfig",
    # solver configs
    "BEMConfig",
    "PINNConfig",
    "MOIConfig",
    "OPConfig",
    "CERTConfig",
]
