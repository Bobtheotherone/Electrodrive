from __future__ import annotations

"""
Error estimation and control for the 3D Laplace FMM.

This module provides both a-priori and a-posteriori error estimators that are
cheap enough to be used inside benchmarking / autotuning loops while remaining
numerically meaningful.

High-level API
--------------
- estimate_error_from_config:
    Crude a-priori estimate based only on FmmConfig.
- estimate_error_for_tree:
    Geometry-aware a-priori estimate using an existing FmmTree and
    InteractionLists.
- estimate_error_from_samples:
    A-posteriori estimate obtained by comparing FMM results against a
    direct O(N^2) kernel on a small random subset of points.
- choose_expansion_order_for_tol:
    Simple helper that picks the smallest expansion order p whose
    config-only estimate meets a target tolerance.

The a-priori estimates are based on classical Laplace FMM truncation
bounds of the form

    err <= C * rho^{p+1},  where  rho = diam / dist  <  1,

with C treated as O(1) and rho taken either from the MAC parameter
(theta) or from the actual node geometry. These are heuristic but
monotone in (p, rho).

The a-posteriori estimator is fully self-contained: it builds a
temporary tree, runs a full FMM matvec using the CPU reference
backend, and compares against an internal direct kernel that uses the
same Coulomb constant K_E as the rest of electrodrive.
"""

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import math

import torch
from torch import Tensor

from .config import FmmConfig, K_E
from .tree import FmmTree, build_fmm_tree
from .interaction_lists import InteractionLists, build_interaction_lists
from .geometry import compute_node_diameters
from .kernels_cpu import (
    apply_p2p_cpu_tiled,
    p2m_cpu,
    m2m_cpu,
    m2l_cpu,
    l2l_cpu,
    l2p_cpu,
)
from electrodrive.utils.logging import JsonlLogger


__all__ = [
    "ErrorEstimate",
    "estimate_error_from_config",
    "estimate_error_for_tree",
    "estimate_error_from_samples",
    "choose_expansion_order_for_tol",
    "estimate_error_placeholder",
]


@dataclass
class ErrorEstimate:
    """
    Aggregate error metrics for a single FMM configuration.

    The interpretation of the "relative" errors depends on the estimator:
    - For a-priori estimators, these are bounds on a dimensionless
      kernel error (i.e. relative error in the Green's function).
    - For a-posteriori estimators, they are relative errors in the
      computed potentials for a particular test problem.
    """

    max_relative_error: float
    rms_relative_error: float
    details: Dict[str, float]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _clamp_theta(theta: float) -> float:
    """
    Clamp a MAC parameter into the mathematically sensible range (0, 1).

    Values outside (0, 1) are replaced by a safe default; this should
    never happen for a validated FmmConfig, but defensive coding keeps
    the estimators robust to partial configs.
    """
    if not math.isfinite(theta) or theta <= 0.0:
        return 0.5
    if theta >= 1.0:
        return 0.99
    return float(theta)


def _truncation_bound_from_ratio(
    p: int,
    rho: Tensor,
    *,
    safety_factor: float = 2.0,
) -> Tensor:
    """
    Dimensionless truncation error bound C * rho^{p+1} with a safety margin.

    The constant C is not known analytically for arbitrary geometry, but
    for well-separated Laplace FMM interactions it is O(1). We fold
    this into safety_factor and clamp rho to [0, 0.99] to avoid
    numerical surprises.
    """
    if p <= 0:
        raise ValueError(f"expansion order p must be positive, got {p}")

    rho_clamped = torch.clamp(rho, min=0.0, max=0.99).to(torch.float64)
    # Use exp((p+1)*log(rho)) for numerical robustness when p is large.
    exponents = (p + 1) * torch.log(torch.clamp(rho_clamped, min=1e-16))
    # Where rho == 0, the error is exactly zero.
    err = torch.where(
        rho_clamped > 0.0,
        torch.exp(exponents),
        torch.zeros_like(rho_clamped),
    )
    return safety_factor * err


def _direct_potential(
    points_src: Tensor,
    points_tgt: Tensor,
    charges_src: Tensor,
    *,
    exclude_self: bool = True,
) -> Tensor:
    """
    Direct O(N^2) Coulomb potential with the same K_E and self-interaction
    semantics as the reference P2P kernel.

    This mirrors the behavior of :func:`_p2p_block` in
    :mod:`electrodrive.fmm3d.kernels_cpu` and the helper used in
    :mod:`electrodrive.fmm3d.sanity_suite`:

        phi_i = sum_{j != i} K_E * q_j / |x_i - x_j|.

    Parameters
    ----------
    points_src : (N, 3)
        Source locations.
    points_tgt : (M, 3)
        Target locations.
    charges_src : (N,)
        Source charges.
    exclude_self : bool, default=True
        If True and points_src and points_tgt represent the same logical
        point set (N == M), diagonal self-interactions are removed by
        zeroing the diagonal contributions, just like the P2P kernel.
    """
    if (
        points_src.ndim != 2
        or points_tgt.ndim != 2
        or points_src.shape[1] != 3
        or points_tgt.shape[1] != 3
    ):
        raise ValueError(
            "points_src and points_tgt must have shape (N, 3) and (M, 3)"
        )
    if charges_src.ndim != 1 or charges_src.shape[0] != points_src.shape[0]:
        raise ValueError(
            "charges_src must have shape (N,) and match points_src"
        )

    # (M, N, 3)
    diff = points_tgt[:, None, :] - points_src[None, :, :]
    r = torch.linalg.norm(diff, dim=-1)  # (M, N)

    # Numerical safety.
    eps = torch.finfo(r.dtype).eps
    r = torch.clamp(r, min=eps)
    inv_r = 1.0 / r

    # Coulomb scaling, matching kernels_cpu.
    ke = points_tgt.new_tensor(float(K_E), dtype=points_tgt.dtype)
    contrib = ke * charges_src[None, :] * inv_r  # (M, N)

    if exclude_self and points_src.shape[0] == points_tgt.shape[0]:
        # Mirror _p2p_block: if we are in a self-interaction setting,
        # zero out the diagonal contributions.
        n = min(contrib.shape[0], contrib.shape[1])
        if n > 0:
            idx = torch.arange(n, device=contrib.device)
            contrib[idx, idx] = 0.0

    # Sum over sources for each target.
    phi = contrib.sum(dim=-1)
    return phi


def _check_points_and_charges(points: Tensor, charges: Tensor) -> None:
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {tuple(points.shape)}")
    if charges.ndim != 1 or charges.shape[0] != points.shape[0]:
        raise ValueError(
            f"charges must have shape (N,), got {tuple(charges.shape)} "
            f"for N={points.shape[0]}",
        )


# ---------------------------------------------------------------------------
# A-priori estimators
# ---------------------------------------------------------------------------


def estimate_error_from_config(cfg: FmmConfig) -> ErrorEstimate:
    """
    Crude a-priori estimate based only on FmmConfig.

    We approximate the worst-case kernel error over all admissible
    M2L interactions as::

        err_max  ~=  safety * theta^{p+1} / (1 - theta)
        err_rms  ~=  0.5 * err_max

    where theta is the MAC parameter and p is the expansion
    order. This is intentionally simple, monotone in p and theta, and
    cheap to evaluate, making it suitable for coarse autotuning.
    """
    cfg.validate()
    p = int(cfg.expansion_order)
    theta = _clamp_theta(float(cfg.mac_theta))

    if p <= 0:
        raise ValueError("expansion_order must be positive")

    # Avoid division by zero if theta is extremely close to 1.
    denom = max(1.0 - theta, 1e-3)
    base = theta ** (p + 1)
    safety = 2.0
    err_max = safety * base / denom
    err_rms = 0.5 * err_max

    details = {
        "p": float(p),
        "theta": float(theta),
        "heuristic": 1.0,
    }
    return ErrorEstimate(
        max_relative_error=float(err_max),
        rms_relative_error=float(err_rms),
        details=details,
    )


def estimate_error_for_tree(
    tree: FmmTree,
    lists: InteractionLists,
    cfg: FmmConfig,
    *,
    safety_factor: float = 2.0,
) -> ErrorEstimate:
    """
    Geometry-aware a-priori estimate for a specific tree and interaction lists.

    For each M2L node–node interaction (i, j), we compute the separation
    ratio

        rho_ij = max(diam_i, diam_j) / dist(center_i, center_j),

    then bound the kernel truncation error by

        err_ij <= safety * rho_ij^{p+1}.

    The returned max_relative_error is max_ij err_ij and
    rms_relative_error is sqrt(mean_ij err_ij^2). These are still
    bounds on a dimensionless kernel error, not measured potential
    errors for a particular charge distribution.
    """
    cfg.validate()

    centers = tree.node_centers  # (M, 3)
    half_extents = tree.node_half_extents  # (M, 3)
    diameters = compute_node_diameters(half_extents)  # (M,)

    if not lists.m2l_pairs:
        # No far-field interactions => no truncation error.
        return ErrorEstimate(
            max_relative_error=0.0,
            rms_relative_error=0.0,
            details={
                "p": float(cfg.expansion_order),
                "theta": float(cfg.mac_theta),
                "n_m2l_pairs": 0.0,
            },
        )

    device = centers.device
    idx_src = torch.tensor(
        [int(i) for (i, _j) in lists.m2l_pairs],
        dtype=torch.long,
        device=device,
    )
    idx_tgt = torch.tensor(
        [int(j) for (_i, j) in lists.m2l_pairs],
        dtype=torch.long,
        device=device,
    )

    c_src = centers[idx_src]
    c_tgt = centers[idx_tgt]
    d_src = diameters[idx_src]
    d_tgt = diameters[idx_tgt]

    diff = c_src - c_tgt
    dist = torch.linalg.norm(diff, dim=-1)
    eps = torch.finfo(dist.dtype).eps
    dist = torch.clamp(dist, min=10.0 * eps)

    diam_max = torch.maximum(d_src, d_tgt)
    rho = diam_max / dist

    p = int(cfg.expansion_order)
    err_ij = _truncation_bound_from_ratio(p, rho, safety_factor=safety_factor)

    max_err = float(err_ij.max().item())
    rms_err = float(torch.sqrt(torch.mean(err_ij ** 2)).item())

    details = {
        "p": float(p),
        "theta": float(cfg.mac_theta),
        "n_m2l_pairs": float(len(lists.m2l_pairs)),
        "max_rho": float(rho.max().item()),
        "min_rho": float(rho.min().item()),
    }
    return ErrorEstimate(
        max_relative_error=max_err,
        rms_relative_error=rms_err,
        details=details,
    )


# ---------------------------------------------------------------------------
# A-posteriori estimator (point charges)
# ---------------------------------------------------------------------------


def estimate_error_from_samples(
    points: Tensor,
    charges: Tensor,
    cfg: FmmConfig,
    *,
    n_points_sample: int = 512,
    seed: int = 12345,
    logger: Optional[JsonlLogger] = None,
) -> ErrorEstimate:
    """
    Empirical FMM error estimate from a small random subset of points.

    Parameters
    ----------
    points:
        Array of particle positions with shape (N, 3).
    charges:
        Array of charges with shape (N,).
    cfg:
        FMM configuration to evaluate. Only CPU execution is used
        (via the reference kernels_cpu backend).
    n_points_sample:
        Number of points to include in the test system. If N is
        smaller than this, all points are used.
    seed:
        Seed for the CPU-side RNG used to select the subset.
    logger:
        Optional JsonlLogger that receives a single event
        carrying the raw metrics.

    Returns
    -------
    ErrorEstimate
        Relative errors between FMM and direct potentials for the
        sampled system.
    """
    _check_points_and_charges(points, charges)
    cfg.validate()

    N = points.shape[0]
    if N == 0:
        return ErrorEstimate(
            max_relative_error=0.0,
            rms_relative_error=0.0,
            details={"n_points": 0.0},
        )

    n_sample = min(int(n_points_sample), int(N))
    if n_sample <= 0:
        raise ValueError(f"n_points_sample must be positive, got {n_points_sample}")

    # Work on CPU; tree building and reference kernels are CPU-only.
    points_cpu = points.detach().to("cpu", non_blocking=False)
    charges_cpu = charges.detach().to("cpu", non_blocking=False)

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    if n_sample == N:
        idx = torch.arange(N, dtype=torch.long, device=points_cpu.device)
    else:
        perm = torch.randperm(N, generator=g, device=points_cpu.device)
        idx = perm[:n_sample]

    points_sub = points_cpu[idx]
    charges_sub = charges_cpu[idx]

    # Build tree and interaction lists.
    tree = build_fmm_tree(points_sub, leaf_size=int(cfg.leaf_size), max_depth=32)
    tree.verify()

    mac_theta_val = float(getattr(cfg, "mac_theta", 0.5))
    p2p_batch_val = int(getattr(cfg, "p2p_batch_size", 32768))

    lists = build_interaction_lists(tree, tree, mac_theta=mac_theta_val)

    # Far-field via multipole pipeline.
    stats = None  # MultipoleOpStats is optional here; we only care about error.
    multipoles = p2m_cpu(tree, charges_sub, cfg, stats=stats)
    multipoles = m2m_cpu(tree, multipoles, cfg, stats=stats)
    locals_ = m2l_cpu(tree, tree, multipoles, cfg, stats=stats)
    locals_ = l2l_cpu(tree, locals_, cfg, stats=stats)
    phi_far_tree = l2p_cpu(tree, locals_, cfg, stats=stats)
    phi_far_tree = phi_far_tree * float(K_E)

    # Near-field via tiled P2P using the same interaction lists.
    charges_tree = tree.map_to_tree_order(charges_sub)
    p2p_res = apply_p2p_cpu_tiled(
        source_tree=tree,
        target_tree=tree,
        charges_src=charges_tree,
        lists=lists,
        tile_size_points=p2p_batch_val,
        logger=None,
        out=None,
    )
    phi_p2p_tree = p2p_res.potential

    phi_fmm_tree = phi_far_tree + phi_p2p_tree
    # Map FMM potentials back to the original order of the subset.
    phi_fmm = tree.map_to_original_order(phi_fmm_tree)

    # Direct reference on the same subset.
    phi_direct = _direct_potential(
        points_src=points_sub,
        points_tgt=points_sub,
        charges_src=charges_sub,
        exclude_self=True,
    )

    diff = phi_fmm - phi_direct
    max_abs_err = float(diff.abs().max().item())
    denom = float(phi_direct.norm().item())
    rel_l2 = 0.0 if denom == 0.0 else float(diff.norm().item() / denom)

    # For an a-posteriori estimate we take max_abs_err / ||phi_direct||_max
    # as a proxy for max_relative_error.
    denom_inf = float(phi_direct.abs().max().item())
    rel_inf = 0.0 if denom_inf == 0.0 else float(max_abs_err / denom_inf)

    details = {
        "n_points_total": float(N),
        "n_points_sample": float(n_sample),
        "expansion_order": float(cfg.expansion_order),
        "mac_theta": float(cfg.mac_theta),
        "leaf_size": float(cfg.leaf_size),
        "p2p_batch_size": float(p2p_batch_val),
        "rel_l2": rel_l2,
        "rel_inf": rel_inf,
        "max_abs_err": max_abs_err,
    }

    if logger is not None:
        logger.info("fmm_error_estimate", **details)

    return ErrorEstimate(
        max_relative_error=rel_inf,
        rms_relative_error=rel_l2,
        details=details,
    )


# ---------------------------------------------------------------------------
# Simple config search helper
# ---------------------------------------------------------------------------


def choose_expansion_order_for_tol(
    base_cfg: FmmConfig,
    target_rel_error: float,
    *,
    p_min: int = 2,
    p_max: int = 32,
) -> Tuple[FmmConfig, ErrorEstimate]:
    """
    Pick the smallest expansion order p whose config-only estimate
    meets target_rel_error.

    This is intentionally cheap and purely a-priori; callers that care
    about tight guarantees should follow up with
    estimate_error_from_samples.
    """
    if target_rel_error <= 0.0:
        raise ValueError("target_rel_error must be positive")

    best_cfg = None
    best_est = None

    for p in range(int(p_min), int(p_max) + 1):
        cfg = replace(base_cfg, expansion_order=p)
        est = estimate_error_from_config(cfg)
        if est.max_relative_error <= target_rel_error:
            best_cfg = cfg
            best_est = est
            break

        # Keep track of the last evaluated config in case we never hit the target.
        best_cfg = cfg
        best_est = est

    assert best_cfg is not None and best_est is not None
    return best_cfg, best_est


# ---------------------------------------------------------------------------
# Backwards compatible placeholder alias
# ---------------------------------------------------------------------------


def estimate_error_placeholder(cfg: FmmConfig) -> ErrorEstimate:
    """
    Backwards-compatible alias for estimate_error_from_config.

    Older callers that imported estimate_error_placeholder will now
    receive a meaningful heuristic estimate instead of a hard-coded
    zero error.
    """
    return estimate_error_from_config(cfg)