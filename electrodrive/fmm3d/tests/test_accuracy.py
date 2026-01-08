from __future__ import annotations

"""
Accuracy regression tests for the Tier-3 FMM stack.

These tests are deliberately *numerical* rather than purely structural:
they check that the FMM and BEM–FMM paths agree with high-accuracy
direct (O(N²)) evaluations for a variety of problem sizes and dtypes.
"""

from typing import Iterable, Tuple

import pytest
import torch
from torch import Tensor

from electrodrive.fmm3d import FmmConfig, create_bem_fmm_backend
from electrodrive.fmm3d.sanity_suite import (
    run_tree_and_interaction_lists,
    run_p2p_against_direct,
    run_full_fmm_against_direct,
    run_bem_fmm_against_bem,
    TestResult, # Imported for type hints, but pytest might try to collect it
)
from electrodrive.fmm3d.bem_fmm import make_laplace_fmm_backend
from electrodrive.core.bem_kernel import (
    bem_matvec_gpu,
    DEFAULT_SINGLE_LAYER_KERNEL,
)

# Tell pytest NOT to try and collect 'TestResult' as a test class
TestResult.__test__ = False # type: ignore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_CPU_DEVICE = torch.device("cpu")


def _format_failure(prefix: str, res: TestResult) -> str:
    """Pretty formatter for assertion messages."""
    msg = [
        f"{prefix} failed:",
        f"  ok          = {res.ok}",
        f"  max_abs_err = {res.max_abs_err:.6e}",
        f"  rel_l2_err  = {res.rel_l2_err:.6e}",
    ]
    if res.extra:
        msg.append("  extra:")
        for k, v in sorted(res.extra.items()):
            msg.append(f"    {k}: {v}")
    return "\n".join(msg)


def _dtypes_for_p2p() -> Iterable[torch.dtype]:
    # P2P is extremely accurate; we test both dtypes.
    return (torch.float32, torch.float64)


def _dtypes_for_fmm() -> Iterable[torch.dtype]:
    # FMM accuracy depends on configuration; keep both dtypes to catch
    # mixed-precision regressions.
    return (torch.float32, torch.float64)


# ---------------------------------------------------------------------------
# 1. Tree + interaction lists
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tree_and_interaction_lists_random_cloud(dtype: torch.dtype) -> None:
    """
    Structural sanity: tree and interaction lists must be internally
    consistent for a moderate random point cloud.
    """
    n_points = 512
    res = run_tree_and_interaction_lists(
        n_points=n_points,
        device=_CPU_DEVICE,
        dtype=dtype,
    )
    assert res.ok, _format_failure("tree_and_interaction_lists", res)


# ---------------------------------------------------------------------------
# 2. P2P kernel vs direct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", list(_dtypes_for_p2p()))
def test_p2p_against_direct_random_cloud(dtype: torch.dtype) -> None:
    """
    Tiled P2P kernel must agree with explicit 1/r within tight
    tolerances.

    We use a stricter tolerance for float64 and a slightly relaxed one
    for float32 to account for accumulated rounding.
    """
    n_points = 512
    if dtype == torch.float64:
        tol = 1e-10
    else:
        tol = 1e-6

    res = run_p2p_against_direct(
        n_points=n_points,
        device=_CPU_DEVICE,
        dtype=dtype,
        tol=tol,
    )
    assert res.ok, _format_failure("p2p_against_direct", res)


# ---------------------------------------------------------------------------
# 3. “Full FMM” helper vs direct (point charges)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", list(_dtypes_for_fmm()))
def test_full_fmm_against_direct_random_cloud(dtype: torch.dtype) -> None:
    """
    End-to-end helper test against direct O(N²) evaluation.

    This uses the same helper as the CLI sanity suite via
    ``run_full_fmm_against_direct``. At present that helper is implemented
    as a tree-based direct Coulomb evaluator for backwards compatibility;
    it still exercises the FMM configuration wiring and statistics and
    provides a regression harness for the eventual full FMM pipeline.

    The tolerances here are intentionally fairly strict; if they fail,
    either the configuration (p, theta, leaf_size) or the underlying
    operators will need to be revisited.
    """
    n_points = 512

    # Target tolerance in line with CLI defaults used in the sanity suite.
    tol = 1e-2

    res = run_full_fmm_against_direct(
        n_points=n_points,
        device=_CPU_DEVICE,
        dtype=dtype,
        tol=tol,
        expansion_order=None,
        mac_theta=None,
        leaf_size=None,
        p2p_batch_size=None,
    )
    assert res.ok, _format_failure("fmm_vs_direct", res)


# ---------------------------------------------------------------------------
# 4. BEM FMM vs BEM direct  (LaplaceFmm3D backend)
# ---------------------------------------------------------------------------


def _make_random_panels(
    n_panels: int,
    dtype: torch.dtype,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Construct a simple synthetic BEM geometry: random centroids in a
    cube and small random areas.

    This is intentionally simple; accuracy is judged relative to the
    direct BEM matvec on *the same* discretisation.
    """
    g = torch.Generator(device="cpu").manual_seed(123)
    centroids = torch.rand(
        n_panels,
        3,
        generator=g,
        device=_CPU_DEVICE,
        dtype=dtype,
    ) - 0.5
    areas = torch.rand(
        n_panels,
        generator=g,
        device=_CPU_DEVICE,
        dtype=dtype,
    ) * 0.1
    sigma = torch.randn(
        n_panels,
        generator=g,
        device=_CPU_DEVICE,
        dtype=dtype,
    )
    return centroids, areas, sigma


@pytest.mark.parametrize("n_panels", [64, 128])
def test_laplace_bem_fmm_against_direct(n_panels: int) -> None:
    """
    Compare the LaplaceFmm3D backend (external matvec) against the
    direct BEM matvec for a random panel geometry.

    This is a focused regression test for ``electrodrive.fmm3d.bem_fmm``,
    independent of the older bridge in ``bem_coupling``.
    """
    dtype = torch.float64
    centroids, areas, sigma = _make_random_panels(
        n_panels=n_panels,
        dtype=dtype,
    )

    # Direct BEM reference (torch-tiled backend).
    V_ref = bem_matvec_gpu(
        sigma=sigma,
        src_centroids=centroids,
        areas=areas,
        tile_size=64,
        self_integrals=None,
        logger=None,
        use_keops=False,
        kernel=DEFAULT_SINGLE_LAYER_KERNEL,
        backend="torch_tiled",
    )

    # FMM backend (LaplaceFmm3D).
    fmm = make_laplace_fmm_backend(
        src_centroids=centroids,
        areas=areas,
        max_leaf_size=64,
        theta=0.6,
        use_dipole=True,
        logger=None,
    )

    V_fmm = bem_matvec_gpu(
        sigma=sigma,
        src_centroids=centroids,
        areas=areas,
        tile_size=64,
        self_integrals=None,
        logger=None,
        use_keops=False,
        kernel=DEFAULT_SINGLE_LAYER_KERNEL,
        backend="external",
        matvec_impl=fmm.matvec,
    )

    diff = V_fmm - V_ref
    max_abs_err = float(diff.abs().max().item())
    denom = float(V_ref.norm().item())
    rel_l2 = 0.0 if denom == 0.0 else float(diff.norm().item() / denom)

    # Tolerance mirrors the CLI sanity suite's default tol_bem=1e-1.
    tol = 1e-1
    ok = rel_l2 <= tol

    if not ok:
        msg_lines = [
            "laplace_bem_fmm_against_direct failed:",
            f"  n_panels    = {n_panels}",
            f"  max_abs_err = {max_abs_err:.6e}",
            f"  rel_l2_err  = {rel_l2:.6e}",
        ]
        pytest.fail("\n".join(msg_lines))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for FMM GPU matvec test"
)
def test_laplace_bem_fmm_cuda_sigma_device() -> None:
    dtype = torch.float32
    n_panels = 64
    centroids, areas, sigma = _make_random_panels(
        n_panels=n_panels,
        dtype=dtype,
    )

    fmm = make_laplace_fmm_backend(
        src_centroids=centroids,
        areas=areas,
        max_leaf_size=64,
        theta=0.6,
        use_dipole=True,
        logger=None,
        backend="gpu",
        device="cuda",
    )

    sigma_cuda = sigma.to(device="cuda")
    V_cuda = bem_matvec_gpu(
        sigma=sigma_cuda,
        src_centroids=centroids,
        areas=areas,
        tile_size=64,
        self_integrals=None,
        logger=None,
        use_keops=False,
        kernel=DEFAULT_SINGLE_LAYER_KERNEL,
        backend="external",
        matvec_impl=fmm.matvec,
    )

    assert V_cuda.is_cuda
    assert torch.isfinite(V_cuda).all()

    V_cpu = bem_matvec_gpu(
        sigma=sigma,
        src_centroids=centroids,
        areas=areas,
        tile_size=64,
        self_integrals=None,
        logger=None,
        use_keops=False,
        kernel=DEFAULT_SINGLE_LAYER_KERNEL,
        backend="external",
        matvec_impl=fmm.matvec,
    )

    assert torch.allclose(V_cuda.cpu(), V_cpu, rtol=1e-3, atol=1e-4)


def test_sanity_suite_bem_against_direct_small() -> None:
    """
    Thin wrapper around ``sanity_suite.run_bem_fmm_against_bem`` with
    a reduced panel count to keep the test fast.

    This uses the same configuration as the CLI sanity suite and serves
    as a higher-level integration check.
    """
    n_panels = 128
    dtype = torch.float64
    tol = 1e-1

    res = run_bem_fmm_against_bem(
        n_panels=n_panels,
        device=_CPU_DEVICE,
        dtype=dtype,
        tol=tol,
    )
    assert res.ok, _format_failure("bem_fmm_against_bem", res)


# ---------------------------------------------------------------------------
# 5. Legacy BEM–FMM bridge (create_bem_fmm_backend)
# ---------------------------------------------------------------------------


def test_legacy_bem_backend_zero_input() -> None:
    """
    Regression test for the legacy BEM–FMM bridge:

    - Backend can be constructed from a bare FmmConfig.
    - Zero charges on a degenerate geometry produce zero potential.
    """
    cfg = FmmConfig()
    backend = create_bem_fmm_backend(cfg)

    N = 8
    centroids = torch.zeros(N, 3, device=_CPU_DEVICE)
    sigma = torch.zeros(N, device=_CPU_DEVICE)

    out = backend.apply(centroids, sigma)
    assert out.shape == (N,)
    assert torch.allclose(out, torch.zeros_like(out))


def test_legacy_bem_backend_small_random() -> None:
    """
    Sanity check that the legacy BEM–FMM bridge is at least numerically
    reasonable on a tiny random problem.

    This does *not* enforce a strong accuracy bound (that is the job of
    the LaplaceFmm3D tests above), but it catches gross wiring errors or
    shape/NaN regressions.
    """
    cfg = FmmConfig()
    backend = create_bem_fmm_backend(cfg)

    N = 16
    g = torch.Generator(device="cpu").manual_seed(321)
    centroids = torch.rand(
        N,
        3,
        generator=g,
        device=_CPU_DEVICE,
        dtype=torch.float64,
    )
    sigma = torch.randn(
        N,
        generator=g,
        device=_CPU_DEVICE,
        dtype=torch.float64,
    )

    # BEM reference with torch-tiled backend.
    areas = torch.ones(N, device=_CPU_DEVICE, dtype=torch.float64)
    V_ref = bem_matvec_gpu(
        sigma=sigma,
        src_centroids=centroids,
        areas=areas,
        tile_size=32,
        self_integrals=None,
        logger=None,
        use_keops=False,
        kernel=DEFAULT_SINGLE_LAYER_KERNEL,
        backend="torch_tiled",
    )

    # Legacy/scaffold FMM backend.
    V_fmm = backend.apply(centroids, sigma)

    diff = V_fmm - V_ref
    max_abs_err = float(diff.abs().max().item())
    denom = float(V_ref.norm().item())
    rel_l2 = 0.0 if denom == 0.0 else float(diff.norm().item() / denom)

    # Loose bound: this path is a scaffold / legacy bridge; we only care
    # that it is not completely degenerate.
    tol = 5e-1
    ok = rel_l2 <= tol

    if not ok:
        msg_lines = [
            "legacy_bem_backend_small_random failed:",
            f"  N           = {N}",
            f"  max_abs_err = {max_abs_err:.6e}",
            f"  rel_l2_err  = {rel_l2:.6e}",
        ]
        pytest.fail("\n".join(msg_lines))
