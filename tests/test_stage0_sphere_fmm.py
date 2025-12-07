from __future__ import annotations

import pytest
import torch

from electrodrive.core.bem import bem_solve
from electrodrive.core.bem_kernel import (
    DEFAULT_SINGLE_LAYER_KERNEL,
    bem_matvec_gpu,
)
try:
    from electrodrive.fmm3d.bem_fmm import make_laplace_fmm_backend
    HAVE_FMM = True
except Exception:
    HAVE_FMM = False
    make_laplace_fmm_backend = None  # type: ignore
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.config import BEMConfig
from electrodrive.utils.logging import JsonlLogger


def _sphere_spec() -> CanonicalSpec:
    return CanonicalSpec.from_json(
        {
            "domain": "R3",
            "BCs": "Dirichlet",
            "conductors": [
                {
                    "id": 0,
                    "type": "sphere",
                    "center": [0.0, 0.0, 0.0],
                    "radius": 1.0,
                    "potential": 0.0,
                }
            ],
            "charges": [
                {
                    "type": "point",
                    "q": 1.0,
                    "pos": [0.0, 0.0, 1.5],
                }
            ],
        }
    )


@pytest.mark.skipif(not HAVE_FMM, reason="FMM backend not available")
def test_stage0_sphere_fmm_matches_dense(tmp_path):
    spec = _sphere_spec()

    cfg = BEMConfig()
    cfg.use_gpu = False
    cfg.fp64 = True
    cfg.max_refine_passes = 1
    cfg.initial_h = 0.35
    cfg.vram_autotune = False

    logger = JsonlLogger(tmp_path / "logs")
    out = bem_solve(spec, cfg, logger, differentiable=False)
    assert "solution" in out

    sol = out["solution"]
    C = sol._C
    A = sol._A
    sigma = torch.randn(C.shape[0], device=C.device, dtype=C.dtype)

    V_ref = bem_matvec_gpu(
        sigma=sigma,
        src_centroids=C,
        areas=A,
        tile_size=256,
        self_integrals=None,
        backend="torch_tiled",
        kernel=DEFAULT_SINGLE_LAYER_KERNEL,
        use_keops=False,
    )

    fmm = make_laplace_fmm_backend(
        src_centroids=C,
        areas=A,
        max_leaf_size=48,
        theta=0.4,
        use_dipole=True,
        logger=None,
    )

    V_fmm = bem_matvec_gpu(
        sigma=sigma,
        src_centroids=C,
        areas=A,
        tile_size=256,
        self_integrals=None,
        backend="external",
        matvec_impl=fmm.matvec,
        kernel=DEFAULT_SINGLE_LAYER_KERNEL,
        use_keops=False,
    )

    rel = torch.linalg.norm(V_fmm - V_ref) / torch.linalg.norm(V_ref).clamp_min(1e-12)
    assert float(rel) < 1e-2
