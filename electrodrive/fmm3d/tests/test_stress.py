from __future__ import annotations

import math
import time

import pytest
import torch
from torch import Tensor

from electrodrive.utils.config import K_E
from electrodrive.fmm3d.config import FmmConfig
from electrodrive.fmm3d.tree import FmmTree
from electrodrive.fmm3d.interaction_lists import build_interaction_lists
from electrodrive.fmm3d.kernels_cpu import apply_p2p_cpu_tiled
# Use the actual, working factory from our BEM-FMM module
from electrodrive.fmm3d.bem_fmm import make_laplace_fmm_backend, LaplaceFmm3D
from electrodrive.core.bem_kernel import bem_matvec_gpu, DEFAULT_SINGLE_LAYER_KERNEL


# ------------------------------
# Helpers
# ------------------------------


def make_random_points(n: int, seed: int, mode: str = "uniform") -> tuple[Tensor, Tensor]:
    """
    Generate a synthetic system of point charges for FMM stress testing.
    Positions in [-1, 1]^3, charges ~ N(0,1), with different spatial structures.
    """
    torch.manual_seed(seed)
    if mode == "uniform":
        x = 2.0 * torch.rand(n, 3, dtype=torch.float64) - 1.0
    elif mode == "clusters":
        # two clusters separated in space
        n1 = n // 2
        n2 = n - n1
        c1 = torch.tensor([+0.5, 0.0, 0.0], dtype=torch.float64)
        c2 = torch.tensor([-0.5, 0.0, 0.0], dtype=torch.float64)
        x1 = c1 + 0.1 * torch.randn(n1, 3, dtype=torch.float64)
        x2 = c2 + 0.1 * torch.randn(n2, 3, dtype=torch.float64)
        x = torch.cat([x1, x2], dim=0)
    elif mode == "shell":
        # points approximately on a sphere of radius ~0.8
        r = 0.8
        u = torch.randn(n, 3, dtype=torch.float64)
        u = u / u.norm(dim=1, keepdim=True)
        x = r * u + 0.05 * torch.randn(n, 3, dtype=torch.float64)
    else:
        raise ValueError(f"Unknown mode={mode!r}")

    q = torch.randn(n, dtype=torch.float64)
    return x, q


def direct_potential(x: Tensor, q: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Naive O(N^2) direct Laplace potential (for reference on moderate N).

    We include the Coulomb constant K_E so that this matches the
    physical scaling used by the FMM backend; this way, relative
    errors are purely geometric/algorithmic.
    """
    n = x.shape[0]
    # Use K_E to match physical units of FMM
    dx = x.unsqueeze(1) - x.unsqueeze(0)  # [n, n, 3]
    r = dx.norm(dim=-1).clamp_min(eps)    # [n, n]

    # exclude self-interaction: set diagonal to +inf so 1/r -> 0
    idx = torch.arange(n, device=x.device)
    r[idx, idx] = float("inf")

    kernel = K_E / r
    return (kernel * q.view(1, n)).sum(dim=1)


def rel_l2_err(phi: Tensor, phi_ref: Tensor) -> float:
    num = (phi - phi_ref).norm().item()
    den = phi_ref.norm().item()
    if den == 0.0:
        return 0.0 if num == 0.0 else math.inf
    return num / den


def make_test_bem_system(n_panels: int, dtype: torch.dtype, device: torch.device):
    """
    Create a synthetic BEM system (centroids, areas, rhs/sigma) for testing.

    This is intentionally simple and random but stable across runs
    (fixed seed) so that we can use it as a regression harness for
    BEM-FMM coupling.
    """
    torch.manual_seed(42)
    # Random centroids in [-1, 1]^3
    centroids = 2.0 * torch.rand(n_panels, 3, dtype=dtype, device=device) - 1.0
    # Random positive areas
    areas = torch.rand(n_panels, dtype=dtype, device=device) * 0.1 + 0.01
    # Random RHS (charge density sigma)
    rhs = torch.randn(n_panels, dtype=dtype, device=device)
    return (centroids, areas), rhs


# ------------------------------
# Fast Regression Test (Runs by default)
# ------------------------------


def test_fmm_stress_fast_m2l():
    """
    Fast regression test (not marked slow).

    Uses two separated clusters to FORCE valid M2L interactions.
    This ensures we aren't just testing P2P (Near-Field) on small N.

    With the corrected M2L/L2L operators at p=8, we expect very high
    accuracy here; if this test fails, it is a strong signal that
    something is broken in the far-field pipeline.
    """
    # Two clusters of 100 points each, separated by 10 units.
    # Box size ~2.0. Separation > 2.0 -> Guaranteed Multipole usage.
    n_per_cluster = 100
    shift = 10.0

    dtype = torch.float64
    device = torch.device("cpu")

    torch.manual_seed(555)
    # Cluster 1 at origin
    x1 = torch.randn(n_per_cluster, 3, dtype=dtype, device=device) * 0.5
    # Cluster 2 shifted far away
    x2 = torch.randn(n_per_cluster, 3, dtype=dtype, device=device) * 0.5
    x2[:, 0] += shift

    x = torch.cat([x1, x2])
    q_sigma = torch.randn(2 * n_per_cluster, dtype=dtype, device=device)
    areas = torch.ones_like(q_sigma)

    fmm_prod = make_laplace_fmm_backend(
        src_centroids=x,
        areas=areas,
        max_leaf_size=50,  # Small enough to split clusters, large enough to have depth
        theta=0.5,
        expansion_order=8,
    )

    phi_prod = fmm_prod.matvec(
        sigma=q_sigma,
        src_centroids=x,
        areas=areas,
        tile_size=1024,
        self_integrals=None,
    )

    phi_ref = direct_potential(x, q_sigma)
    err = rel_l2_err(phi_prod, phi_ref)

    print(f"\n[FAST-STRESS] n={len(x)}, separated clusters, err={err:.3e}")

    # If M2L (and subsequently L2L) works, this should be < 1e-5.
    # If the far-field operators are broken/flipped, this will be ~1e-1 to 1e0.
    assert err < 1e-5


# ------------------------------
# Slow FMM stress tests
# ------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_points, mode",
    [
        (2048, "uniform"),   # direct still okay
        (4096, "clusters"),  # direct borderline but fine once
        # (16384, "uniform"), # Commented out to keep test runtime reasonable for now
    ],
)
def test_fmm_stress_large_n(n_points: int, mode: str):
    """
    Stress test for the Laplace FMM backend (LaplaceFmm3D) directly:

    - larger N than accuracy tests,
    - different spatial distributions,
    - checks accuracy vs direct (for moderate N).

    This is our main "physics-level" regression test for the FMM point
    backend. It exercises the full pipeline: P2M, M2M, M2L, L2L, L2P,
    plus the interaction-list builder.
    """
    dtype = torch.float64
    device = torch.device("cpu")  # FMM implementation is CPU-only for now

    x, q_sigma = make_random_points(n_points, seed=123, mode=mode)
    x = x.to(dtype=dtype, device=device)
    q_sigma = q_sigma.to(dtype=dtype, device=device)

    # LaplaceFmm3D expects areas and sigma (surface charge density).
    # For point charges q, we can treat area=1.0 and sigma=q.
    areas = torch.ones_like(q_sigma)

    # Production config tuned for large-N accuracy:
    # tighter MAC (theta) reduces far-field truncation error and a larger
    # leaf size keeps interaction-list overhead reasonable for clustered inputs.
    fmm_prod = make_laplace_fmm_backend(
        src_centroids=x,
        areas=areas,
        max_leaf_size=256,
        theta=0.3,
        expansion_order=8,
    )

    t0 = time.perf_counter()
    # Matvec computes V = Sum K(r) * sigma * area.
    # Since area=1, this is Sum K(r) * q.
    phi_prod = fmm_prod.matvec(
        sigma=q_sigma,
        src_centroids=x,
        areas=areas,
        tile_size=1024,
        self_integrals=None,
    )
    wall_fmm_prod = time.perf_counter() - t0

    # Reference: Direct evaluation
    t1 = time.perf_counter()
    # q_sigma is 'q' here since area=1.
    phi_ref = direct_potential(x, q_sigma)
    wall_ref = time.perf_counter() - t1
    ref_type = "direct"

    err = rel_l2_err(phi_prod, phi_ref)

    # Print a clear stress-test line
    print(
        f"\n[STRESS-FMM] n={n_points}, mode={mode}, "
        f"ref={ref_type}, rel_l2_err={err:.3e}, "
        f"t_prod={wall_fmm_prod:.3f}s, t_ref={wall_ref:.3f}s",
    )

    # UPDATED: With the M2L and L2L fixes and p=8, we expect high accuracy.
    # We keep the tolerance tight to catch regressions in the far-field math.
    assert err < 1e-5


# ------------------------------
# BEM + FMM stress tests
# ------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_panels, theta",
    [
        (128, 0.6),   # small system
        (512, 0.6),   # medium system
    ],
)
def test_bem_fmm_stress_scaling(n_panels: int, theta: float):
    """
    Stress test for BEM-FMM coupling via the official bem_matvec_gpu entry point.

    This checks that wiring FMM in as an "external" matvec produces the
    same result (up to a reasonably tight tolerance) as the internal
    dense/tiled BEM evaluation.
    """
    dtype = torch.float64
    device = torch.device("cpu")

    (centroids, areas), sigma = make_test_bem_system(
        n_panels=n_panels,
        dtype=dtype,
        device=device,
    )

    # 1. Create FMM backend
    fmm_backend = make_laplace_fmm_backend(
        src_centroids=centroids,
        areas=areas,
        max_leaf_size=64,
        theta=theta,
        expansion_order=8,
    )

    # 2. Run FMM matvec via the standard kernel API
    t0 = time.perf_counter()
    sol_fmm = bem_matvec_gpu(
        sigma=sigma,
        src_centroids=centroids,
        areas=areas,
        backend="external",
        matvec_impl=fmm_backend.matvec,
        tile_size=1024,
    )
    wall_fmm = time.perf_counter() - t0

    # 3. Reference: Dense Direct BEM
    t1 = time.perf_counter()
    sol_ref = bem_matvec_gpu(
        sigma=sigma,
        src_centroids=centroids,
        areas=areas,
        backend="torch_tiled",
        tile_size=1024,
    )
    wall_ref = time.perf_counter() - t1
    ref_type = "bem_direct"

    # Tolerance: BEM comparisons involve near-field quadrature differences
    # (Analytic vs FMM P2P Centroid). We keep this slightly looser than
    # the pure point test, but still tight enough to catch regressions.
    tol = 1e-2

    err = rel_l2_err(sol_fmm, sol_ref)

    print(
        f"\n[STRESS-BEM] panels={n_panels}, theta={theta}, ref={ref_type}, "
        f"rel_l2_err={err:.3e}, t_fmm={wall_fmm:.3f}s, t_ref={wall_ref:.3f}s",
    )

    assert err < tol
