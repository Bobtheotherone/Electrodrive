from __future__ import annotations

import time

import pytest
import torch

from electrodrive.fmm3d.config import FmmConfig
from electrodrive.fmm3d.tree import build_fmm_tree
from electrodrive.fmm3d.interaction_lists import build_interaction_lists
from electrodrive.fmm3d.kernels_cpu import (
    apply_p2p_cpu,
    p2m_cpu,
    m2m_cpu,
    m2l_cpu,
    l2l_cpu,
    l2p_cpu,
)
from electrodrive.fmm3d.kernels_gpu import (
    apply_p2p_gpu,
    p2m_gpu,
    m2m_gpu,
    m2l_gpu,
    l2l_gpu,
    l2p_gpu,
)

# Skip if no CUDA
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required for GPU FMM stress tests.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _disable_fmm_logging() -> None:
    """
    Turn off all the noisy FMM debug / spectral logging so we get
    clean performance numbers.

    This patches the already-imported modules in-place; it only affects
    this test process.
    """
    import electrodrive.fmm3d.multipole_operators as mop
    import electrodrive.fmm3d.kernels_cpu as kcpu
    import electrodrive.fmm3d.kernels_gpu as kgpu

    class _NullLogger:
        def debug(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            pass

        def info(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            pass

        def warning(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            pass

        def error(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            pass

    def _silent_get_logger(logger=None):  # type: ignore[no-untyped-def]
        # Ignore the incoming logger and always return a quiet one.
        return _NullLogger()

    # multipole_operators: stop all spectral / debug logging
    mop.get_logger = _silent_get_logger
    mop.log_spectral_stats = lambda *a, **k: None  # type: ignore[assignment]
    mop.debug_tensor_stats = lambda *a, **k: None  # type: ignore[assignment]
    mop.want_verbose_debug = lambda: False  # type: ignore[assignment]

    # CPU kernels: quiet logger + no debug tensor stats / spectral logs
    kcpu.get_logger = _silent_get_logger
    kcpu.log_spectral_stats = lambda *a, **k: None  # type: ignore[assignment]
    kcpu.debug_tensor_stats = lambda *a, **k: None  # type: ignore[assignment]
    kcpu.want_verbose_debug = lambda: False  # type: ignore[assignment]

    # GPU kernels: quiet logger + no debug tensor stats
    kgpu.get_logger = _silent_get_logger
    kgpu.debug_tensor_stats = lambda *a, **k: None  # type: ignore[assignment]


def make_random_problem(
    n_points: int,
    *,
    seed: int = 2025,
    dtype: torch.dtype = torch.float64,
    expansion_order: int = 10,
    mac_theta: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, FmmConfig]:
    """
    Build a 'BEM-like' random problem for stress testing.

    - Points ~ uniform in [0, 1]^3
    - Charges ~ sigma * area (two random factors)
    - FMM config: p=expansion_order, theta=mac_theta, leaf_size=64
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # Geometry
    points = torch.rand((n_points, 3), generator=g, dtype=dtype)

    # Roughly BEM-ish charge scaling: sigma * area
    sigma = torch.randn(n_points, generator=g, dtype=dtype)
    areas = torch.rand(n_points, generator=g, dtype=dtype)
    charges = sigma * areas

    cfg = FmmConfig(
        expansion_order=expansion_order,
        mac_theta=mac_theta,
        leaf_size=64,
        dtype=dtype,
    )
    return points, charges, cfg


def solve_cpu_full_fmm(
    points: torch.Tensor,
    charges: torch.Tensor,
    cfg: FmmConfig,
) -> torch.Tensor:
    """
    Reference CPU FMM pipeline (far + near), using kernels_cpu.

    Returns potentials in the original point ordering for the pure 1/|r|
    kernel (no K_E scaling).
    """
    # Tree + lists on CPU
    tree = build_fmm_tree(points.cpu(), leaf_size=int(cfg.leaf_size))
    lists = build_interaction_lists(tree, tree, mac_theta=float(cfg.mac_theta))

    # Upward / far field
    multipoles = p2m_cpu(tree=tree, charges=charges, cfg=cfg)
    multipoles = m2m_cpu(tree=tree, multipoles=multipoles, cfg=cfg)

    locals_ = m2l_cpu(
        source_tree=tree,
        target_tree=tree,
        multipoles=multipoles,
        lists=lists,
        cfg=cfg,
    )
    locals_ = l2l_cpu(tree=tree, locals_=locals_, cfg=cfg)
    phi_far_tree = l2p_cpu(tree=tree, locals_=locals_, cfg=cfg)

    # Near field P2P (exact 1/r, self-interactions removed)
    p2p_result = apply_p2p_cpu(
        source_tree=tree,
        target_tree=tree,
        charges_src=multipoles.charges,  # already in tree order
        lists=lists,
        cfg=cfg,
    )
    phi_near_tree = p2p_result.potential

    # Combine in tree order, then map back to original ordering
    phi_total_tree = phi_far_tree + phi_near_tree
    phi_total = tree.map_to_original_order(phi_total_tree)
    return phi_total


def solve_gpu_full_fmm(
    points: torch.Tensor,
    charges: torch.Tensor,
    cfg: FmmConfig,
) -> torch.Tensor:
    """
    GPU FMM pipeline (far + near) using kernels_gpu.

    - Tree + interaction lists are built on CPU.
    - The tree is then moved to CUDA once.
    - Far field and near field run on the GPU.
    - Result is mapped back to original ordering and returned on CPU.
    """
    dtype = charges.dtype

    # 1) Tree + lists on CPU
    tree = build_fmm_tree(points.cpu(), leaf_size=int(cfg.leaf_size))
    lists = build_interaction_lists(tree, tree, mac_theta=float(cfg.mac_theta))

    # 2) Move tree + charges to CUDA
    tree.to("cuda", dtype=dtype)
    charges_cuda = charges.to(tree.device, dtype=dtype)

    # 3) Upward / far field on GPU
    multipoles = p2m_gpu(tree=tree, charges=charges_cuda, cfg=cfg)
    multipoles = m2m_gpu(tree=tree, multipoles=multipoles, cfg=cfg)

    locals_ = m2l_gpu(
        source_tree=tree,
        target_tree=tree,
        multipoles=multipoles,
        lists=lists,
        cfg=cfg,
    )
    locals_ = l2l_gpu(tree=tree, locals_=locals_, cfg=cfg)
    phi_far_tree = l2p_gpu(tree=tree, locals_=locals_, cfg=cfg)

    # 4) Near field on GPU, using tree-ordered charges
    p2p_result = apply_p2p_gpu(
        source_tree=tree,
        target_tree=tree,
        charges_src=multipoles.charges,  # already in tree order on CUDA
        lists=lists,
        cfg=cfg,
    )
    phi_near_tree = p2p_result.potential

    # 5) Combine and map back to original ordering (on CPU)
    phi_total_tree = phi_far_tree + phi_near_tree
    phi_total = tree.map_to_original_order(phi_total_tree)
    return phi_total.cpu()


# ---------------------------------------------------------------------------
# Stress test: N = 64k, theta = 0.5, p = 10
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gpu_fmm_stress_large_n_matches_cpu_64k() -> None:
    """
    Heavy regression test at 'lab' settings:

        N = 64_000 points
        p = 10
        theta = 0.5
        leaf_size = 64

    We compare a full CPU FMM matvec with a full GPU FMM matvec and
    expect agreement up to near-roundoff. All FMM debug/spectral
    logging is disabled so timings reflect normal operating conditions.
    """
    _disable_fmm_logging()

    n_points = 64_000

    points, charges, cfg = make_random_problem(
        n_points=n_points,
        expansion_order=10,
        mac_theta=0.5,
    )

    # CPU FMM
    t0 = time.perf_counter()
    phi_cpu = solve_cpu_full_fmm(points, charges, cfg)
    t1 = time.perf_counter()

    # GPU FMM
    phi_gpu = solve_gpu_full_fmm(points, charges, cfg)
    t2 = time.perf_counter()

    cpu_time = t1 - t0
    gpu_time = t2 - t1
    speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")

    # Timing info (shown when running with -s)
    print(
        f"[gpu stress] n={n_points:6d} | "
        f"CPU FMM: {cpu_time:7.3f}s | "
        f"GPU FMM: {gpu_time:7.3f}s | "
        f"speedup: {speedup:5.1f}x"
    )

    # Relative L2 and max relative error between CPU and GPU potentials
    denom = torch.linalg.norm(phi_cpu)
    rel_l2 = torch.linalg.norm(phi_cpu - phi_gpu) / (denom + 1e-16)
    max_rel = torch.max(
        torch.abs(phi_cpu - phi_gpu) / (torch.abs(phi_cpu) + 1e-16)
    )

    # Same algorithm, different device -> should match very closely.
    # Keep tolerances tight but slightly relaxed vs tiny-N tests.
    assert rel_l2 < 1e-9
    assert max_rel < 1e-8
