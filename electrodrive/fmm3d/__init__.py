"""Tier-3 FMM/H-matrix subsystem for electrodrive.

This package is a scaffold for an industrial-strength 3D FMM implementation
designed to integrate with the existing BEM backend (``electrodrive.core.bem_*``).

Target total size for this module: ~150 LOC
once fully implemented (public API, top-level config, basic utilities).

High-level responsibilities
---------------------------
- Expose a stable, documented public API for FMM-accelerated matvecs.
- Provide factory functions that construct FMM backends from configs.
- Bridge between Python orchestration and low-level CPU/GPU backends.
"""

from __future__ import annotations

from .config import FmmConfig
from .bem_coupling import create_bem_fmm_backend

__all__ = [
    "FmmConfig",
    "create_bem_fmm_backend",
]
