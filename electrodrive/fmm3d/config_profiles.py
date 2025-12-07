from __future__ import annotations

"""
Named configuration profiles for the FMM backend.

This module provides a small, opinionated registry of FMM "profiles"
built on top of :class:`FmmConfig`.

The goals are:

- Centralize sensible defaults for different usage modes (preview,
  reference, GPU-heavy runs, etc.).
- Keep construction of :class:`FmmConfig` objects DRY and testable.
- Make it easy for higher-level orchestration code (CLI, sweeps, AI
  search loops) to request a profile by name and optionally override a
  few fields.

The profiles themselves are deliberately conservative and only cover the
Laplace single-layer kernel used by the current FMM/BEM stack. Callers
can always bypass this module and instantiate :class:`FmmConfig`
directly when they need full control.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch

from .config import BackendKind, PrecisionKind, FmmConfig


@dataclass(frozen=True)
class FmmProfile:
    """Lightweight description of a named FMM configuration.

    Parameters
    ----------
    name:
        Canonical profile name (case-insensitive in the public API).
    description:
        Human-readable summary of the intended use case.
    expansion_order:
        Multipole expansion order ``p``.
    mac_theta:
        Opening angle for the multipole-acceptance criterion.
    leaf_size:
        Target maximum number of points per leaf in the FMM tree.
    backend:
        Logical backend selector (``"cpu"``, ``"gpu"``, or ``"auto"``).
    precision:
        High-level precision policy (``"single"``, ``"double"``, or
        ``"mixed"``).
    use_fft_m2l:
        Whether to enable FFT-accelerated M2L for large ``p``.
    use_gpu:
        If ``False``, higher-level orchestration should avoid GPU
        backends altogether even if available.
    use_multi_gpu, use_mpi:
        Reserved for future distributed backends.
    p2p_batch_size:
        Optional soft cap on P2P interactions per batch.
    target_rel_error:
        Informal target relative error for planning / certification;
        not enforced anywhere in this module but useful metadata.
    """

    name: str
    description: str
    expansion_order: int
    mac_theta: float
    leaf_size: int
    backend: BackendKind
    precision: PrecisionKind
    use_fft_m2l: bool = True
    use_gpu: bool = True
    use_multi_gpu: bool = False
    use_mpi: bool = False
    p2p_batch_size: Optional[int] = None
    target_rel_error: float = 1e-3

    def build(
        self,
        *,
        kernel: str = "laplace_single_layer",
        overrides: Optional[Mapping[str, object]] = None,
    ) -> FmmConfig:
        """Instantiate a concrete :class:`FmmConfig` from this profile.

        Parameters
        ----------
        kernel:
            Kernel identifier; defaults to ``"laplace_single_layer"``.
        overrides:
            Optional mapping of field name → value applied on top of the
            profile before validation. This is a convenience for small
            tweaks such as ``{"leaf_size": 32}``.

        Returns
        -------
        cfg:
            A fully validated :class:`FmmConfig` instance.
        """
        cfg = FmmConfig(
            kernel=kernel,
            expansion_order=self.expansion_order,
            mac_theta=self.mac_theta,
            leaf_size=self.leaf_size,
            backend=self.backend,
            precision=self.precision,
            use_fft_m2l=self.use_fft_m2l,
            use_gpu=self.use_gpu,
            use_multi_gpu=self.use_multi_gpu,
            use_mpi=self.use_mpi,
            # Let FmmConfig derive dtype from precision unless the caller
            # overrides it explicitly via ``overrides``.
            dtype=None,
            p2p_batch_size=self.p2p_batch_size,
        )
        if overrides:
            cfg = _apply_fmm_overrides(cfg, overrides)
        return cfg


# ---------------------------------------------------------------------------
# Internal registry
# ---------------------------------------------------------------------------


_FMM_PROFILES: Dict[str, FmmProfile] = {}


def _register_profile(profile: FmmProfile) -> None:
    key = profile.name.lower()
    if key in _FMM_PROFILES:
        raise ValueError(f"Duplicate FMM profile name {profile.name!r}")
    _FMM_PROFILES[key] = profile


def _apply_fmm_overrides(
    cfg: FmmConfig,
    overrides: Mapping[str, object],
) -> FmmConfig:
    """Apply a set of field overrides to an :class:`FmmConfig`.

    This is intentionally strict: unknown fields raise ``KeyError`` and
    the resulting config is re-validated before being returned.
    """
    for field_name, value in overrides.items():
        if not hasattr(cfg, field_name):
            raise KeyError(
                f"Unknown FmmConfig field {field_name!r} in overrides; "
                "check for typos or update config_profiles."
            )
        setattr(cfg, field_name, value)
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

# CPU-centric profiles ------------------------------------------------------


_register_profile(
    FmmProfile(
        name="cpu_preview",
        description=(
            "Fast, low-accuracy CPU settings for small problems, "
            "CLI sanity checks, and unit tests. Uses single-precision "
            "expansions, a relatively loose MAC, and larger leaves."
        ),
        expansion_order=4,
        mac_theta=0.8,
        leaf_size=128,
        backend="cpu",
        precision="single",
        use_fft_m2l=True,
        use_gpu=False,
        use_multi_gpu=False,
        use_mpi=False,
        p2p_batch_size=32_768,
        target_rel_error=5e-2,
    )
)


_register_profile(
    FmmProfile(
        name="cpu_ref",
        description=(
            "Reference CPU configuration aimed at ~1e-4–1e-5 relative "
            "accuracy on well-behaved point clouds. Uses double-precision "
            "expansions, conservative MAC, and moderate leaf size."
        ),
        expansion_order=8,
        mac_theta=0.5,
        leaf_size=64,
        backend="cpu",
        precision="double",
        use_fft_m2l=True,
        use_gpu=False,
        use_multi_gpu=False,
        use_mpi=False,
        p2p_batch_size=16_384,
        target_rel_error=1e-4,
    )
)


# GPU-friendly profiles -----------------------------------------------------


_register_profile(
    FmmProfile(
        name="gpu_mixed",
        description=(
            "Default GPU profile: mixed-precision expansions with "
            "TF32/FP32-friendly settings and moderately aggressive MAC. "
            "Intended for large-scale exploratory runs."
        ),
        expansion_order=6,
        mac_theta=0.7,
        leaf_size=128,
        backend="auto",  # let orchestration pick the concrete device
        precision="mixed",
        use_fft_m2l=True,
        use_gpu=True,
        use_multi_gpu=False,
        use_mpi=False,
        p2p_batch_size=65_536,
        target_rel_error=5e-3,
    )
)


_register_profile(
    FmmProfile(
        name="gpu_ref",
        description=(
            "High-accuracy GPU profile using double-precision expansions "
            "and a more conservative MAC. Slower than 'gpu_mixed' but "
            "appropriate for certification-style runs."
        ),
        expansion_order=8,
        mac_theta=0.5,
        leaf_size=64,
        backend="auto",
        precision="double",
        use_fft_m2l=True,
        use_gpu=True,
        use_multi_gpu=False,
        use_mpi=False,
        p2p_batch_size=32_768,
        target_rel_error=1e-4,
    )
)


# Sensible default for callers that do not care about profiles explicitly.
DEFAULT_FMM_PROFILE_NAME: str = "gpu_mixed"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def available_fmm_profiles() -> tuple[str, ...]:
    """Return the set of registered FMM profile names.

    The names are returned in sorted order to keep CLI help deterministic.
    """
    return tuple(sorted(_FMM_PROFILES.keys()))


def get_fmm_profile(name: Optional[str] = None) -> FmmProfile:
    """Retrieve a registered :class:`FmmProfile` by name.

    Parameters
    ----------
    name:
        Profile name (case-insensitive). If omitted or ``None``, the
        module-level :data:`DEFAULT_FMM_PROFILE_NAME` is used.

    Raises
    ------
    KeyError
        If the requested profile name is unknown.
    """
    if name is None:
        name = DEFAULT_FMM_PROFILE_NAME
    key = name.lower()
    try:
        return _FMM_PROFILES[key]
    except KeyError:
        available = ", ".join(sorted(_FMM_PROFILES.keys()))
        raise KeyError(
            f"Unknown FMM profile {name!r}. Available profiles: {available}"
        ) from None


def make_fmm_config(
    name: Optional[str] = None,
    *,
    kernel: str = "laplace_single_layer",
    overrides: Optional[Mapping[str, object]] = None,
) -> FmmConfig:
    """Construct an :class:`FmmConfig` instance from a named profile.

    This is the main entry point for callers that only need an FMM
    configuration and are happy with the built-in profiles.

    Examples
    --------
    >>> cfg = make_fmm_config("cpu_ref")
    >>> cfg.expansion_order
    8

    Extra keyword arguments can be provided via the ``overrides`` dict:

    >>> cfg = make_fmm_config("cpu_ref", overrides={"leaf_size": 32})
    >>> cfg.leaf_size
    32
    """
    profile = get_fmm_profile(name)
    return profile.build(kernel=kernel, overrides=overrides)


def choose_default_fmm_profile_for_device(
    device: Optional[torch.device | str] = None,
) -> str:
    """Heuristic default FMM profile name for the given device.

    If ``device`` is omitted, the current PyTorch environment is
    inspected. The logic is intentionally simple and side-effect free:
    it never touches global CUDA state beyond ``torch.cuda.is_available``.
    """
    if device is None:
        try:
            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False
        return "gpu_mixed" if cuda_available else "cpu_ref"

    dev = torch.device(device)
    if dev.type == "cuda":
        return "gpu_mixed"
    return "cpu_ref"


__all__ = [
    "FmmProfile",
    "available_fmm_profiles",
    "get_fmm_profile",
    "make_fmm_config",
    "choose_default_fmm_profile_for_device",
    "DEFAULT_FMM_PROFILE_NAME",
]
