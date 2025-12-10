import os

import numpy as np
import pytest
import torch

from electrodrive.learn.collocation import make_collocation_batch_for_spec, BEM_AVAILABLE
from electrodrive.orchestration.parser import CanonicalSpec


def _make_three_layer_spec(eps2: float = 4.0) -> CanonicalSpec:
    spec = {
        "domain": {"bbox": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]},
        "conductors": [],
        "dielectrics": [
            {"name": "region1", "epsilon": 1.0, "z_min": 0.0, "z_max": 5.0},
            {"name": "slab", "epsilon": eps2, "z_min": -0.3, "z_max": 0.0},
            {"name": "region3", "epsilon": 1.0, "z_min": -5.0, "z_max": -0.3},
        ],
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.2]}],
        "BCs": "dielectric_interfaces",
        "symmetry": ["rot_z"],
        "queries": [],
    }
    return CanonicalSpec.from_json(spec)


@pytest.mark.skipif(not BEM_AVAILABLE, reason="BEM not available in this environment.")
def test_layered_collocation_analytic_vs_bem():
    spec = _make_three_layer_spec(eps2=4.0)
    rng_seed = 1234
    n_points = 128
    ratio_boundary = 0.5

    rng1 = np.random.default_rng(rng_seed)
    batch_analytic = make_collocation_batch_for_spec(
        spec,
        n_points=n_points,
        ratio_boundary=ratio_boundary,
        supervision_mode="analytic",
        device=torch.device("cpu"),
        dtype=torch.float32,
        rng=rng1,
    )

    rng2 = np.random.default_rng(rng_seed)
    batch_bem = make_collocation_batch_for_spec(
        spec,
        n_points=n_points,
        ratio_boundary=ratio_boundary,
        supervision_mode="bem",
        device=torch.device("cpu"),
        dtype=torch.float32,
        rng=rng2,
    )

    X_a = batch_analytic["X"]
    X_b = batch_bem["X"]
    if X_b.numel() == 0:
        pytest.skip("BEM oracle returned empty batch")
    assert torch.allclose(X_a, X_b)

    top_z = 0.0
    mask_region1 = X_a[:, 2] >= (top_z - 1e-6)
    Va = batch_analytic["V_gt"][mask_region1]
    Vb = batch_bem["V_gt"][mask_region1]
    if Va.numel() == 0 or Vb.numel() == 0:
        pytest.skip("No region1 points for analytic/BEM comparison")
    rel_err = torch.linalg.norm(Va - Vb) / torch.linalg.norm(Vb).clamp_min(1e-12)
    assert rel_err < 1e-2

    # z-distribution: ensure coverage near both interfaces and below the slab
    z_vals = X_a[:, 2].detach().cpu().numpy()
    bottom_z = -0.3
    h = top_z - bottom_z
    band = 0.1 * h
    near_top = np.sum(np.abs(z_vals - top_z) < band)
    near_bottom = np.sum(np.abs(z_vals - bottom_z) < band)
    below_slab = np.sum(z_vals < bottom_z - band)
    assert near_top > 0 and near_bottom > 0 and below_slab > 0
