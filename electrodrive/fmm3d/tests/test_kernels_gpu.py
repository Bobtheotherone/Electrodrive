from __future__ import annotations

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
    apply_p2p_gpu_tiled,
    p2m_gpu,
    m2m_gpu,
    m2l_gpu,
    l2l_gpu,
    l2p_gpu,
)


# Skip the entire module if there is no CUDA device
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required to exercise kernels_gpu.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_random_problem(
    n_points: int = 512,
    *,
    seed: int = 1234,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, FmmConfig]:
    """
    Build a simple random test problem: points + charges + FMM config.

    We use the same kind of config the BEM backends use
    (p ≈ 8, theta ≈ 0.5, leaf_size ≈ 64).
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # Uniform box [0, 1]^3
    points = torch.rand((n_points, 3), generator=g, dtype=dtype)

    # Fake "BEM-like" charges: sigma * area
    sigma = torch.randn(n_points, generator=g, dtype=dtype)
    areas = torch.rand(n_points, generator=g, dtype=dtype)
    charges = sigma * areas

    cfg = FmmConfig(
        expansion_order=8,
        mac_theta=0.5,
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
    # Tree + lists on CPU (as in LaplaceFmm3D / FmmBemBackend).
    tree = build_fmm_tree(points.cpu(), leaf_size=int(cfg.leaf_size))
    lists = build_interaction_lists(tree, tree, mac_theta=float(cfg.mac_theta))

    # Upward / far-field pass
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

    # Near-field P2P (exact 1/r)
    p2p_result = apply_p2p_cpu(
        source_tree=tree,
        target_tree=tree,
        charges_src=multipoles.charges,  # already in tree order
        lists=lists,
        cfg=cfg,
    )
    phi_near_tree = p2p_result.potential

    # Combine in tree order, then map back.
    phi_total_tree = phi_far_tree + phi_near_tree
    phi_total = tree.map_to_original_order(phi_total_tree)
    return phi_total


def solve_gpu_full_fmm(
    points: torch.Tensor,
    charges: torch.Tensor,
    cfg: FmmConfig,
) -> torch.Tensor:
    """
    GPU FMM pipeline, using kernels_gpu wrappers.

    - Tree + interaction lists are still built on CPU.
    - Tree is then moved to CUDA once.
    - Far-field and near-field run on the GPU.
    - Result is mapped back to original ordering and returned on CPU.
    """
    dtype = charges.dtype

    # 1. Build tree + interaction lists on CPU
    tree = build_fmm_tree(points.cpu(), leaf_size=int(cfg.leaf_size))
    lists = build_interaction_lists(tree, tree, mac_theta=float(cfg.mac_theta))

    # 2. Move geometry to CUDA (lists stay as CPU index pairs)
    tree.to("cuda", dtype=dtype)
    charges_cuda = charges.to(tree.device, dtype=dtype)

    # 3. Upward / far-field on GPU via wrappers
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

    # 4. Near-field P2P on GPU, using the same tree-ordered charges
    p2p_result = apply_p2p_gpu(
        source_tree=tree,
        target_tree=tree,
        charges_src=multipoles.charges,  # already in tree order, on CUDA
        lists=lists,
        cfg=cfg,
    )
    phi_near_tree = p2p_result.potential

    # 5. Combine and map back to original order, then move to CPU
    phi_total_tree = phi_far_tree + phi_near_tree
    phi_total = tree.map_to_original_order(phi_total_tree)
    return phi_total.cpu()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_points", [32, 256])
def test_full_gpu_fmm_matches_cpu(n_points: int) -> None:
    """
    End-to-end regression: GPU FMM == CPU FMM (1/|r| kernel) to
    near machine precision.

    This validates:
      - p2m_gpu / m2m_gpu / m2l_gpu / l2l_gpu / l2p_gpu wrappers
      - apply_p2p_gpu
      - correct use of tree.map_(to/from)_tree_order
      - reuse of CPU-built interaction lists on the GPU
    """
    points, charges, cfg = make_random_problem(n_points=n_points)

    phi_cpu = solve_cpu_full_fmm(points, charges, cfg)
    phi_gpu = solve_gpu_full_fmm(points, charges, cfg)

    # Relative L2 error and max relative error.
    denom = torch.linalg.norm(phi_cpu)
    rel_l2 = torch.linalg.norm(phi_cpu - phi_gpu) / (denom + 1e-16)
    max_rel = torch.max(
        torch.abs(phi_cpu - phi_gpu) / (torch.abs(phi_cpu) + 1e-16)
    )

    # The math is identical, just on a different device; differences
    # should be pure roundoff.
    assert rel_l2 < 1e-11
    assert max_rel < 1e-10


def test_p2p_gpu_matches_cpu_on_same_tree() -> None:
    """
    Focused near-field test: apply_p2p_gpu == apply_p2p_cpu when both
    run on the *same* CPU tree (device='cpu').

    This checks that apply_p2p_gpu_tiled is a drop-in replacement for
    the CPU kernel even before we move the tree to CUDA.
    """
    points, charges, cfg = make_random_problem(n_points=128)
    tree = build_fmm_tree(points.cpu(), leaf_size=int(cfg.leaf_size))
    lists = build_interaction_lists(tree, tree, mac_theta=float(cfg.mac_theta))

    # charges_src must be in tree order for both P2P paths
    charges_tree = tree.map_to_tree_order(charges)

    res_cpu = apply_p2p_cpu(
        source_tree=tree,
        target_tree=tree,
        charges_src=charges_tree.clone(),
        lists=lists,
        cfg=cfg,
    ).potential

    res_gpu = apply_p2p_gpu(
        source_tree=tree,
        target_tree=tree,
        charges_src=charges_tree.clone(),
        lists=lists,
        cfg=cfg,
    ).potential

    assert torch.allclose(res_cpu, res_gpu, rtol=1e-12, atol=1e-12)


def test_p2p_gpu_cuda_matches_cpu_reference() -> None:
    """
    Near-field test with an actual CUDA tree:

      CPU: apply_p2p_cpu on a CPU tree
      GPU: apply_p2p_gpu on the same geometry moved to CUDA

    The interaction lists are built on the CPU tree and reused on the
    CUDA tree via node indices.
    """
    points, charges, cfg = make_random_problem(n_points=256)

    # CPU reference
    tree_cpu = build_fmm_tree(points.cpu(), leaf_size=int(cfg.leaf_size))
    lists = build_interaction_lists(tree_cpu, tree_cpu, mac_theta=float(cfg.mac_theta))
    charges_tree_cpu = tree_cpu.map_to_tree_order(charges)
    p2p_cpu = apply_p2p_cpu(
        source_tree=tree_cpu,
        target_tree=tree_cpu,
        charges_src=charges_tree_cpu,
        lists=lists,
        cfg=cfg,
    ).potential

    # GPU version: move *the same* tree to CUDA, and reuse the
    # already-tree-ordered charges + lists.
    tree_gpu = tree_cpu
    tree_gpu.to("cuda", dtype=charges.dtype)
    charges_tree_gpu = charges_tree_cpu.to(tree_gpu.device)

    p2p_gpu = apply_p2p_gpu(
        source_tree=tree_gpu,
        target_tree=tree_gpu,
        charges_src=charges_tree_gpu,
        lists=lists,
        cfg=cfg,
    ).potential.cpu()

    assert torch.allclose(p2p_cpu, p2p_gpu, rtol=1e-12, atol=1e-12)


def test_p2p_gpu_tiling_equivalence() -> None:
    """
    Stress the tiling logic in apply_p2p_gpu_tiled by forcing very small
    tile sizes, and check that results are identical to a 'no tiling'
    run on the same CUDA tree.

    This is important to catch edge cases in the inner tiling loop.
    """
    points, charges, cfg = make_random_problem(n_points=512)

    # Build tree + lists on CPU, then move to CUDA.
    tree = build_fmm_tree(points.cpu(), leaf_size=int(cfg.leaf_size))
    lists = build_interaction_lists(tree, tree, mac_theta=float(cfg.mac_theta))

    tree.to("cuda", dtype=charges.dtype)
    charges_tree = tree.map_to_tree_order(charges.to(tree.device))

    # 1) Reference: huge tile_size_points => effectively no tiling.
    res_ref = apply_p2p_gpu_tiled(
        source_tree=tree,
        target_tree=tree,
        charges_src=charges_tree,
        lists=lists,
        tile_size_points=10**9,
    ).potential

    # 2) Aggressive tiling: tiny tile_size_points
    res_tiled = apply_p2p_gpu_tiled(
        source_tree=tree,
        target_tree=tree,
        charges_src=charges_tree,
        lists=lists,
        tile_size_points=64,  # forces lots of tiles
    ).potential

    # Compare on CPU
    res_ref_cpu = res_ref.cpu()
    res_tiled_cpu = res_tiled.cpu()

    assert torch.allclose(res_ref_cpu, res_tiled_cpu, rtol=1e-12, atol=1e-12)
