"""Smoke tests for the Tier-3 FMM scaffold.

Target size: ~400 LOC.

These tests are intentionally light and are meant to ensure that:
- imports work
- basic factory functions construct objects
- placeholder matvecs run without crashing.
"""

from __future__ import annotations

import torch

from electrodrive.fmm3d import FmmConfig, create_bem_fmm_backend


def test_smoke_backend_construction() -> None:
    cfg = FmmConfig()
    backend = create_bem_fmm_backend(cfg)
    assert backend.cfg.expansion_order == cfg.expansion_order


def test_smoke_apply_zero() -> None:
    cfg = FmmConfig()
    backend = create_bem_fmm_backend(cfg)
    N = 8
    centroids = torch.zeros(N, 3)
    sigma = torch.zeros(N)
    out = backend.apply(centroids, sigma)
    assert out.shape == (N,)
