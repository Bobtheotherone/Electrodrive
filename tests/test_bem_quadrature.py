import os
import numpy as np
import pytest
import torch

from electrodrive.learn.collocation import (
    make_collocation_batch_for_spec,
    BEM_AVAILABLE,
)
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.learn.encoding import ENCODING_DIM
from electrodrive.utils.config import EPS_0  # unit factor between analytic shortcuts and BEM


# Lightweight-but-accurate BEM config used when treating BEM as an oracle in
# tests. Keep fp64 + near-field corrections for accuracy, but allow a modest
# refinement budget so accuracy expectations in these tests are met reliably.
BEM_TEST_ORACLE_CONFIG = {
    # Use fp64 everywhere for stability.
    "fp64": True,
    # Prefer GPU for the matrix-free operator; near-field corrections will
    # be applied on the host as needed.
    "use_gpu": True,
    # Start finer and allow a few refinement passes to keep accuracy high.
    "initial_h": 0.2,
    "min_refine_passes": 2,
    "max_refine_passes": 3,
    "refine_factor": 0.7,
    "gmres_maxiter": 800,
    "gmres_restart": 128,
    # Enable near-field quadrature at evaluation targets to reduce
    # quadrature error near conductor surfaces.
    "use_near_quadrature": True,
    # Also enable near-field corrections inside the GMRES matvec operator.
    # The far field is still handled by the GPU backend; only the local
    # panel-panel interactions are corrected using refined quadrature.
    "use_near_quadrature_matvec": True,
    # Slightly enlarge the near-field window used in both eval and matvec
    # corrections so that immediate neighbours are treated accurately.
    "near_quadrature_distance_factor": 2.0,
}


def _build_plane_spec() -> CanonicalSpec:
    return CanonicalSpec.from_json(
        {
            "domain": "R3",
            "BCs": "dirichlet",
            "conductors": [
                {
                    "type": "plane",
                    "z": 0.0,
                    "potential": 0.0,
                }
            ],
            "charges": [
                {
                    "type": "point",
                    "q": 1e-9,
                    "pos": [0.1, -0.2, 0.5],
                }
            ],
        }
    )


def _build_sphere_spec() -> CanonicalSpec:
    return CanonicalSpec.from_json(
        {
            "domain": "R3",
            "BCs": "dirichlet",
            "conductors": [
                {
                    "type": "sphere",
                    "radius": 0.75,
                    "potential": 0.0,
                    "center": [0.0, 0.0, 0.0],
                }
            ],
            "charges": [
                {
                    "type": "point",
                    "q": -5e-10,
                    "pos": [0.0, 0.0, 1.5],
                }
            ],
        }
    )


def _build_parallel_planes_spec() -> CanonicalSpec:
    d = 0.75
    return CanonicalSpec.from_json(
        {
            "domain": "R3",
            "BCs": "dirichlet",
            "conductors": [
                {
                    "type": "plane",
                    "z": d,
                    "potential": 0.0,
                },
                {
                    "type": "plane",
                    "z": -d,
                    "potential": 0.0,
                },
            ],
            "charges": [
                {
                    "type": "point",
                    "q": 8e-10,
                    "pos": [0.0, 0.0, 0.1],
                }
            ],
        }
    )


@pytest.mark.parametrize(
    "spec_builder, geom_type",
    [
        (_build_plane_spec, "plane"),
        (_build_sphere_spec, "sphere"),
        (_build_parallel_planes_spec, "parallel_planes"),
    ],
)
def test_collocation_shapes_and_finiteness(spec_builder, geom_type):
    spec = spec_builder()
    n_points = 256
    ratio_boundary = 0.4
    device = torch.device("cpu")
    dtype = torch.float64

    batch = make_collocation_batch_for_spec(
        spec=spec,
        n_points=n_points,
        ratio_boundary=ratio_boundary,
        supervision_mode="analytic",
        device=device,
        dtype=dtype,
    )

    X = batch["X"]
    V = batch["V_gt"]
    is_boundary = batch["is_boundary"]
    mask_finite = batch["mask_finite"]
    encoding = batch["encoding"]

    # Basic shape checks
    assert X.dim() == 2 and X.shape[1] == 3
    assert V.shape == (X.shape[0],)
    assert is_boundary.shape == (X.shape[0],)
    assert mask_finite.shape == (X.shape[0],)
    assert encoding.shape == (X.shape[0], ENCODING_DIM)

    # Dtypes / device
    assert X.dtype == dtype and X.device == device
    assert V.dtype == dtype and V.device == device
    assert encoding.dtype == dtype and encoding.device == device

    # No non-finite values where mask_finite is True
    if mask_finite.any():
        assert torch.isfinite(V[mask_finite]).all()


def test_collocation_bbox_center_tracks_translated_sphere():
    """Ensure translated spheres keep sampling centred on the conductor."""
    center = [1.1, -0.4, 0.75]
    radius = 0.6
    spec = CanonicalSpec.from_json(
        {
            "domain": "R3",
            "BCs": "dirichlet",
            "conductors": [
                {
                    "type": "sphere",
                    "radius": radius,
                    "potential": 0.0,
                    "center": center,
                }
            ],
            "charges": [
                {
                    "type": "point",
                    "q": 4e-10,
                    "pos": [center[0], center[1], center[2] + 1.2],
                }
            ],
        }
    )

    device = torch.device("cpu")
    dtype = torch.float64
    rng = np.random.default_rng(2025)

    batch = make_collocation_batch_for_spec(
        spec=spec,
        n_points=256,
        ratio_boundary=0.5,
        supervision_mode="analytic",
        device=device,
        dtype=dtype,
        rng=rng,
        geom_type="sphere",
    )

    bbox_center = batch["bbox_center"]
    bbox_extent = batch["bbox_extent"]

    expected_center = torch.tensor(center, dtype=dtype)
    assert bbox_center.shape == batch["X"].shape
    assert torch.allclose(
        bbox_center, expected_center.unsqueeze(0).expand_as(bbox_center), atol=1e-9
    )

    expected_extent = 4.0 * radius
    assert torch.allclose(
        bbox_extent,
        torch.full((batch["X"].shape[0],), expected_extent, dtype=dtype),
    )

    # The sampled cloud should be centred on the conductor to within Monte Carlo noise.
    mean_pos = batch["X"].mean(dim=0)
    assert torch.allclose(mean_pos, expected_center, atol=0.15)


@pytest.mark.parametrize(
    "spec_builder, geom_type, needs_eps_scaling",
    [
        (_build_plane_spec, "plane", True),
        (_build_sphere_spec, "sphere", True),
        (_build_parallel_planes_spec, "parallel_planes", True),
    ],
)
def test_analytic_matches_bem_up_to_units(spec_builder, geom_type, needs_eps_scaling):
    """
    For geometries with analytic coverage we expect the analytic shortcut
    and the BEM oracle to agree on a shared set of collocation points.

    Historically the fast analytic paths in the learning stack omit the
    global Coulomb constant K_E, so BEM results differ by a fixed 1/eps0
    scale factor. We take this into account when comparing.
    """
    # If the BEM backend is not available on this machine, skip rather
    # than hard-failing. This keeps the test suite usable on CPU-only
    # or BEM-less setups while still acting as a strong regression test
    # when BEM is present.
    if not BEM_AVAILABLE:
        pytest.skip("BEM backend not available; skipping analytic-vs-BEM sanity check.")

    spec = spec_builder()
    # Trimmed collocation count to keep this BEM-heavy test lightweight while
    # still sampling both boundary and interior regions.
    n_points = 128
    ratio_boundary = 0.5
    device = torch.device("cpu")
    dtype = torch.float64

    # Use two RNGs with the same seed so that analytic and BEM batches see
    # identical collocation points.
    rng1 = np.random.default_rng(1234)
    rng2 = np.random.default_rng(1234)

    batch_analytic = make_collocation_batch_for_spec(
        spec=spec,
        n_points=n_points,
        ratio_boundary=ratio_boundary,
        supervision_mode="analytic",
        device=device,
        dtype=dtype,
        rng=rng1,
        geom_type=geom_type,
    )

    batch_bem = make_collocation_batch_for_spec(
        spec=spec,
        n_points=n_points,
        ratio_boundary=ratio_boundary,
        supervision_mode="bem",
        device=device,
        dtype=dtype,
        rng=rng2,
        geom_type=geom_type,
        bem_oracle_config=BEM_TEST_ORACLE_CONFIG,
    )

    # If the BEM oracle failed to converge (no collocation points returned),
    # treat this as a real bug, not as "unavailable".
    assert batch_bem["X"].numel() > 0, (
        "BEM oracle returned no collocation points for this configuration. "
        "This indicates a BEM failure that must be diagnosed and fixed."
    )

    # Ensure we are comparing like-with-like
    assert torch.allclose(batch_analytic["X"], batch_bem["X"])

    mask = batch_analytic["mask_finite"] & batch_bem["mask_finite"]
    assert mask.any(), (
        "No finite targets to compare between analytic and BEM batches. "
        "Investigate collocation / oracle behavior."
    )

    Va = batch_analytic["V_gt"][mask]
    Vb = batch_bem["V_gt"][mask]

    if needs_eps_scaling:
        Vb_comp = Vb * EPS_0
    else:
        Vb_comp = Vb

    # Relative error should be small; we include a small absolute floor to
    # avoid division by ~0 in regions of very small potential.
    rel_err = torch.max(
        torch.abs(Va - Vb_comp)
        / (torch.abs(Va) + torch.abs(Vb_comp) + torch.tensor(1e-9, dtype=Va.dtype))
    ).item()

    assert rel_err < 1e-2
