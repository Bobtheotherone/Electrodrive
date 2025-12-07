from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Any

import torch
from torch import Tensor

from electrodrive.utils.config import K_E
from electrodrive.fmm3d.config import FmmConfig
from electrodrive.utils.logging import JsonlLogger
from electrodrive.fmm3d.tree import FmmTree
from electrodrive.fmm3d.interaction_lists import InteractionLists
from electrodrive.fmm3d.multipole_operators import (
    MultipoleCoefficients,
    LocalCoefficients,
    MultipoleOpStats,
    m2m as _m2m,
    m2l as _m2l,
    l2l as _l2l,
    _make_workspace_for_tree,
    num_harmonics,
    _get_or_init_global_scale,  # NEW: global scale helper
)
from electrodrive.fmm3d.logging_utils import (
    log_spectral_stats,
    want_verbose_debug,
    ConsoleLogger,
    debug_tensor_stats,
    get_logger,
)

__all__ = [
    "P2PResult",
    "apply_p2p_cpu",
    "apply_p2p_cpu_tiled",
    "p2m_cpu",
    "m2m_cpu",
    "m2l_cpu",
    "l2l_cpu",
    "l2p_cpu",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _env_var_true(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# P2P near-field on CPU
# ---------------------------------------------------------------------------


@dataclass
class P2PResult:
    """
    Container for near-field P2P evaluation on CPU.

    Attributes
    ----------
    potential : (N_tgt,) tensor
        Potential at each target point (tree ordering).
    n_pairs : int
        Number of leaf–leaf node pairs processed.
    n_interactions : int
        Total number of point–point interactions accumulated.
    """

    potential: Tensor
    n_pairs: int
    n_interactions: int


def _p2p_block(
    x_tgt: Tensor,
    x_src: Tensor,
    q_src: Tensor,
    *,
    exclude_self: bool = False,
) -> Tensor:
    """
    Compute P2P contribution with heavy debug instrumentation.

    Crucially, this evaluates the pure kernel 1/r. The Coulomb constant K_E
    is NOT applied here; it must be applied by the caller (e.g. BEM layer)
    to the final summed potential.

    Parameters
    ----------
    x_tgt : (n_tgt, 3) tensor
    x_src : (n_src, 3) tensor
    q_src : (n_src,) tensor
    exclude_self : bool
        If True, and if x_tgt and x_src refer to the same point set
        (n_tgt == n_src), diagonal self-interactions are removed.

    Returns
    -------
    phi : (n_tgt,) tensor
        Potential at each target (pure 1/r).
    """
    if x_tgt.ndim != 2 or x_tgt.shape[1] != 3:
        raise ValueError(f"x_tgt must have shape (n_tgt, 3), got {tuple(x_tgt.shape)}")
    if x_src.ndim != 2 or x_src.shape[1] != 3:
        raise ValueError(f"x_src must have shape (n_src, 3), got {tuple(x_src.shape)}")
    if q_src.ndim != 1 or q_src.shape[0] != x_src.shape[0]:
        raise ValueError(
            f"q_src must have shape (n_src,), got {tuple(q_src.shape)} "
            f"for n_src={x_src.shape[0]}",
        )

    # Broadcasted difference: (n_tgt, n_src, 3)
    diff = x_tgt[:, None, :] - x_src[None, :, :]
    r = torch.linalg.norm(diff, dim=-1)

    # --- SINGULARITY WARNING ---
    # Check for extremely small r BEFORE clamping (potential singularity)
    min_r = r.min().item()
    if min_r < 1e-12:
        # Only warn if we are NOT excluding self-interactions.
        # If exclude_self is True, r=0 is expected on the diagonal.
        if not exclude_self and want_verbose_debug():
            lg = get_logger(None)
            if lg:
                lg.warning(
                    f"[P2P-WARN] Near-singularity detected! min_r={min_r:.4e}. "
                    f"exclude_self={exclude_self}. Potential blowup likely."
                )

    # Numerical safety.
    eps = torch.finfo(r.dtype).eps
    r = torch.clamp(r, min=eps)
    inv_r = 1.0 / r

    # P2P kernel is 1/r. K_E is applied at the top level BEM/FMM driver.
    # (n_tgt, n_src)
    contrib = q_src[None, :] * inv_r

    if exclude_self:
        # For self-interactions we expect n_tgt == n_src.
        n = min(contrib.shape[0], contrib.shape[1])
        if n > 0:
            idx = torch.arange(n, device=contrib.device)
            contrib[idx, idx] = 0.0

    # --- EXPLOSION DETECTOR ---
    # Check if this specific block generated crazy values
    block_max = contrib.abs().max().item()
    if block_max > 1e15 and want_verbose_debug():
        lg = get_logger(None)
        if lg:
            lg.error(f"[P2P-CRITICAL] Explosion detected in _p2p_block!")
            lg.debug(f"  Max contribution: {block_max:.4e}")
            lg.debug(f"  Min distance r:   {min_r:.4e} (eps={eps:.1e})")
            lg.debug(f"  Max charge q:     {q_src.abs().max().item():.4e}")
            lg.debug(f"  Block size:       {x_tgt.shape[0]} x {x_src.shape[0]}")

            # Find the culprit index
            bad_indices = torch.nonzero(contrib.abs() > 1e15, as_tuple=False)
            if bad_indices.numel() > 0:
                i, j = bad_indices[0]
                i, j = i.item(), j.item()
                lg.debug(f"  Culprit indices: tgt={i}, src={j}")
                lg.debug(f"  Target coords:   {x_tgt[i].tolist()}")
                lg.debug(f"  Source coords:   {x_src[j].tolist()}")
                lg.debug(f"  Charge:          {q_src[j].item():.4e}")
                lg.debug(f"  Distance:        {r[i, j].item():.4e}")
    # --------------------------

    # Sum over sources for each target.
    return contrib.sum(dim=1)


def apply_p2p_cpu_tiled(
    source_tree: FmmTree,
    target_tree: FmmTree,
    charges_src: Tensor,
    lists: InteractionLists,
    *,
    tile_size_points: int = 16384,
    logger: Optional[JsonlLogger] = None,
    out: Optional[Tensor] = None,
) -> P2PResult:
    """
    Evaluate near-field P2P interactions on CPU given interaction lists.

    This is the main CPU near-field kernel used by the FMM layer.
    It operates purely on the SoA representation of :class:`FmmTree`
    and the node-based interaction lists computed by
    :mod:`electrodrive.fmm3d.interaction_lists`.
    """
    if lists.truncated:
        raise RuntimeError(
            "apply_p2p_cpu_tiled: interaction lists were truncated "
            "(mac_theta too small or max_pairs too low). "
            "Rebuild the InteractionLists with a larger max_pairs or "
            "relax the MAC parameter (mac_theta).",
        )

    if source_tree.device.type != "cpu" or target_tree.device.type != "cpu":
        raise RuntimeError(
            "apply_p2p_cpu_tiled currently supports CPU trees only. "
            f"Got source_tree.device={source_tree.device}, "
            f"target_tree.device={target_tree.device}.",
        )

    # P2P implies source charges match source points.
    # charges_src must be in TREE ORDER (as returned by p2m_cpu in MultipoleCoefficients).
    if charges_src.shape[0] != source_tree.n_points:
        raise ValueError(
            "charges_src length must equal source_tree.n_points "
            f"(got {charges_src.shape[0]} vs {source_tree.n_points}).",
        )

    if charges_src.device != source_tree.device:
        raise ValueError(
            "charges_src must live on the same device as source_tree.points. "
            f"Got charges_src.device={charges_src.device}, "
            f"source_tree.device={source_tree.device}.",
        )

    if source_tree.dtype != charges_src.dtype:
        raise ValueError(
            f"charges_src.dtype ({charges_src.dtype}) must match "
            f"source_tree.dtype ({source_tree.dtype}).",
        )
    if target_tree.dtype != source_tree.dtype:
        raise ValueError(
            f"source_tree and target_tree must have same dtype; "
            f"got {source_tree.dtype} vs {target_tree.dtype}.",
        )

    if tile_size_points <= 0:
        # Treat non-positive values as "no tiling" and let the kernel
        # handle full leaf–leaf blocks.
        tile_size_points = source_tree.n_points * max(1, target_tree.n_points)

    # Centralized logger resolution (replaces manual checks)
    logger = get_logger(logger)

    device = target_tree.device
    dtype = target_tree.dtype

    # Interaction pairs live on CPU; indices are small and cheap to move.
    # We convert them to tensors FIRST to ensure .shape access is safe.
    p2p_pairs, _ = lists.as_tensors(device=torch.device("cpu"))
    num_pairs = int(p2p_pairs.shape[0])

    # Now it is safe to log
    if logger is not None:
        logger.debug(f"Entered apply_p2p_cpu_tiled. Pairs to process: {num_pairs}")

    if out is None:
        phi_tgt = torch.zeros(target_tree.n_points, device=device, dtype=dtype)
    else:
        if out.shape != (target_tree.n_points,):
            raise ValueError(
                f"out has shape {tuple(out.shape)}, expected "
                f"({target_tree.n_points},).",
            )
        phi_tgt = out
        phi_tgt.zero_()

    # Use robust debug stats (name, tensor, logger)
    debug_tensor_stats("charges_src", charges_src, logger)
    debug_tensor_stats("p2p_pairs", p2p_pairs.to(torch.int64), logger)

    pts_src = source_tree.points
    pts_tgt = target_tree.points

    node_ranges_src = source_tree.node_ranges.to(torch.int64)
    node_ranges_tgt = target_tree.node_ranges.to(torch.int64)

    total_interactions: int = 0

    # Loop over leaf–leaf node pairs.
    for k in range(num_pairs):
        src_idx = int(p2p_pairs[k, 0].item())
        tgt_idx = int(p2p_pairs[k, 1].item())

        s0 = int(node_ranges_src[src_idx, 0].item())
        s1 = int(node_ranges_src[src_idx, 1].item())
        t0 = int(node_ranges_tgt[tgt_idx, 0].item())
        t1 = int(node_ranges_tgt[tgt_idx, 1].item())

        if s1 <= s0 or t1 <= t0:
            continue  # empty node

        x_src_leaf = pts_src[s0:s1]
        x_tgt_leaf = pts_tgt[t0:t1]
        q_src_leaf = charges_src[s0:s1]

        ns = s1 - s0
        nt = t1 - t0
        if ns <= 0 or nt <= 0:
            continue

        total_interactions += ns * nt
        exclude_self = (source_tree is target_tree) and (src_idx == tgt_idx)

        # For self-interactions we avoid tiling so that diagonal removal
        # in _p2p_block is well-defined and inexpensive (leaf sizes are small).
        if (ns * nt) <= tile_size_points or exclude_self:
            phi_block = _p2p_block(
                x_tgt=x_tgt_leaf,
                x_src=x_src_leaf,
                q_src=q_src_leaf,
                exclude_self=exclude_self,
            )
            phi_tgt[t0:t1] += phi_block
            continue

        # Tile in the target dimension only. This keeps x_src contiguous
        # and preserves cache locality while bounding the work per tile.
        tile_tgt = max(1, tile_size_points // ns)
        if tile_tgt >= nt:
            # Degenerates to the non-tiled path; keep logic simple.
            phi_block = _p2p_block(
                x_tgt=x_tgt_leaf,
                x_src=x_src_leaf,
                q_src=q_src_leaf,
                exclude_self=exclude_self,
            )
            phi_tgt[t0:t1] += phi_block
            continue

        for t_start in range(t0, t1, tile_tgt):
            t_end = min(t_start + tile_tgt, t1)
            x_tgt_block = pts_tgt[t_start:t_end]
            phi_block = _p2p_block(
                x_tgt=x_tgt_block,
                x_src=x_src_leaf,
                q_src=q_src_leaf,
                exclude_self=False,  # self-case handled above
            )
            phi_tgt[t_start:t_end] += phi_block

    debug_tensor_stats("phi_tgt (P2P Result)", phi_tgt, logger)

    if logger is not None:
        logger.debug(
            f"apply_p2p_cpu_tiled completed. Pairs={num_pairs}, "
            f"Interactions={total_interactions}"
        )

    return P2PResult(
        potential=phi_tgt,
        n_pairs=num_pairs,
        n_interactions=total_interactions,
    )


def apply_p2p_cpu(
    source_tree: FmmTree,
    target_tree: FmmTree,
    charges_src: Tensor,
    lists: InteractionLists,
    *,
    cfg: Optional[FmmConfig] = None,
    logger: Optional[JsonlLogger] = None,
    out: Optional[Tensor] = None,
) -> P2PResult:
    """
    Convenience wrapper around :func:`apply_p2p_cpu_tiled`.

    This reads ``tile_size_points`` from :class:`FmmConfig` if provided
    and forwards to :func:`apply_p2p_cpu_tiled`.
    """
    tile_size = 16384
    if cfg is not None and getattr(cfg, "p2p_batch_size", None) is not None:
        # Heuristic: reuse autotuned batch size if available; otherwise
        # fall back to a reasonably cache-friendly default.
        tile_size = int(cfg.p2p_batch_size)

    # No manual logger instantiation needed here; tiled implementation uses get_logger(logger)
    return apply_p2p_cpu_tiled(
        source_tree=source_tree,
        target_tree=target_tree,
        charges_src=charges_src,
        lists=lists,
        tile_size_points=tile_size,
        logger=logger,
        out=out,
    )


# ---------------------------------------------------------------------------
# Tree-aware multipole / local operators on CPU
# ---------------------------------------------------------------------------


def p2m_cpu(
    tree: FmmTree,
    charges: Tensor,
    cfg: FmmConfig,
    *,
    stats: Optional[MultipoleOpStats] = None,
) -> MultipoleCoefficients:
    """
    CPU implementation of P2M using correct packed layout via MultipoleOperators.

    UPDATED: uses a single global scale (shared with M2M/M2L/L2L/L2P) obtained
    via _get_or_init_global_scale so that all translation matrices and
    expansions are scale-consistent.
    """
    # Basic consistency checks.
    if tree.device.type != "cpu":
        raise RuntimeError(f"p2m_cpu supports CPU trees only. Got {tree.device}.")

    if charges.shape[0] != tree.n_points:
        raise ValueError(
            f"charges len {charges.shape[0]} != tree.n_points {tree.n_points}"
        )

    # 1. Reorder charges to tree order so they align with tree.points
    charges_tree = tree.map_to_tree_order(charges)

    # 2. Setup workspace and derived types
    op = _make_workspace_for_tree(tree, cfg)

    # 3. Allocate packed multipoles: (n_nodes, (p+1)^2)
    P2 = num_harmonics(op.p)
    multipoles = torch.zeros(
        tree.n_nodes,
        P2,
        dtype=op.complex_dtype,
        device=tree.device,
    )

    # 4. Iterate leaves and compute
    leaf_indices = tree.leaf_indices().tolist()
    ranges = tree.node_ranges.to(torch.int64)
    centers = tree.node_centers

    # Use a single global scale shared across P2M/M2M/M2L/L2L/L2P for this cfg/tree
    global_scale = float(_get_or_init_global_scale(cfg, tree))

    for idx in leaf_indices:
        i = int(idx)
        start = int(ranges[i, 0].item())
        end = int(ranges[i, 1].item())

        if end <= start:
            continue

        pts = tree.points[start:end]
        q = charges_tree[start:end]
        c = centers[i]

        # MultipoleOperators.p2m uses global_scale to match translation matrices
        M_leaf = op.p2m(pts, q, c, global_scale)
        multipoles[i] = M_leaf

    if stats:
        stats.incr("p2m_calls", float(len(leaf_indices)))
        stats.merge_from(op.stats)

    # --- INSTRUMENTATION ---
    # Use centralized logger (or None to auto-resolve via env vars)
    log_spectral_stats(get_logger(None), "P2M_OUTPUT", multipoles, op.p)
    # -----------------------

    return MultipoleCoefficients(
        data=multipoles,
        tree=tree,
        charges=charges_tree,
        p=op.p,
        dtype=op.dtype,
        complex_dtype=op.complex_dtype,
    )


def m2m_cpu(
    tree: FmmTree,
    multipoles: MultipoleCoefficients,
    cfg: FmmConfig,
    *,
    stats: Optional[MultipoleOpStats] = None,
) -> MultipoleCoefficients:
    """
    Wrapper for M2M. Delegates to the tree-level operator in multipole_operators.
    """
    # Delegate
    res = _m2m(tree=tree, multipoles=multipoles, cfg=cfg, stats=stats)

    # --- INSTRUMENTATION ---
    log_spectral_stats(get_logger(None), "M2M_OUTPUT", res.data, res.p)
    # -----------------------

    return res


def m2l_cpu(
    source_tree: FmmTree,
    target_tree: FmmTree,
    multipoles: MultipoleCoefficients,
    lists: Optional[InteractionLists] = None,
    cfg: Optional[FmmConfig] = None,
    *,
    stats: Optional[MultipoleOpStats] = None,
) -> LocalCoefficients:
    """
    Wrapper for M2L. Delegates to the tree-level operator in multipole_operators.

    Updated to accept precomputed interaction lists so we can share the
    same lists as P2P and avoid silently truncated far-field pairs.
    """
    if cfg is None:
        raise ValueError("m2l_cpu requires a valid FmmConfig object 'cfg'.")

    # If the caller passes explicit interaction lists (e.g. from the BEM
    # glue layer), use them; otherwise let the tree-level M2L build its
    # own lists.
    if lists is None:
        res = _m2l(
            source_tree=source_tree,
            target_tree=target_tree,
            multipoles=multipoles,
            cfg=cfg,
            stats=stats,
        )
    else:
        # Optional safety: fail loudly if the lists are truncated instead
        # of silently dropping far-field interactions.
        if getattr(lists, "truncated", False):
            raise RuntimeError(
                "m2l_cpu: interaction lists were truncated. "
                "Increase max_pairs or relax mac_theta."
            )

        res = _m2l(
            source_tree=source_tree,
            target_tree=target_tree,
            multipoles=multipoles,
            cfg=cfg,
            stats=stats,
            lists=lists,  # requires matching signature in multipole_operators.m2l
        )

    # --- INSTRUMENTATION ---
    log_spectral_stats(get_logger(None), "M2L_OUTPUT", res.data, res.p)
    # -----------------------

    return res


def l2l_cpu(
    tree: FmmTree,
    locals_: LocalCoefficients,
    cfg: FmmConfig,
    *,
    stats: Optional[MultipoleOpStats] = None,
) -> LocalCoefficients:
    """
    Wrapper for L2L. Delegates to the tree-level operator in multipole_operators.
    """
    # Delegate
    res = _l2l(tree=tree, locals_=locals_, cfg=cfg, stats=stats)

    # --- INSTRUMENTATION ---
    log_spectral_stats(get_logger(None), "L2L_OUTPUT", res.data, res.p)
    # -----------------------

    return res


def l2p_cpu(
    tree: FmmTree,
    locals_: LocalCoefficients,
    cfg: FmmConfig,
    *,
    stats: Optional[MultipoleOpStats] = None,
) -> Tensor:
    """
    CPU implementation of L2P using packed layout consistency via MultipoleOperators.

    UPDATED: uses the same global scale as P2M/M2M/M2L/L2L (via _get_or_init_global_scale),
    so that local expansions are evaluated consistently with the translation
    matrices. Returns potentials in tree order matching tree.points; the caller is
    responsible for mapping back to original order if needed.
    """
    if tree.device.type != "cpu":
        raise RuntimeError(f"l2p_cpu supports CPU trees only. Got {tree.device}.")

    # 1. Setup workspace
    op = _make_workspace_for_tree(tree, cfg)

    # 2. Allocate result in tree order
    potential_tree = torch.zeros(tree.n_points, dtype=op.dtype, device=tree.device)

    leaf_indices = tree.leaf_indices().tolist()
    ranges = tree.node_ranges.to(torch.int64)
    centers = tree.node_centers
    L_data = locals_.data  # Packed (N_nodes, P2)

    # --- INSTRUMENTATION CHECK ---
    # Check inputs before processing
    temp_logger = get_logger(None)
    if temp_logger and L_data.abs().max() > 1e10:
        temp_logger.error(
            f"L2P_INPUT: Locals are already blown up! Max={L_data.abs().max():.2e}"
        )
    # -----------------------------

    # Use the same global scale as other tree-level operators
    global_scale = float(_get_or_init_global_scale(cfg, tree))

    # 3. Iterate leaves
    for idx in leaf_indices:
        i = int(idx)
        start = int(ranges[i, 0].item())
        end = int(ranges[i, 1].item())
        if end <= start:
            continue

        pts = tree.points[start:end]
        c = centers[i]
        L_leaf = L_data[i]  # Packed (P2,)

        # MultipoleOperators.l2p expects packed L and the global scale
        V_leaf = op.l2p(L_leaf, pts, c, global_scale)
        potential_tree[start:end] = V_leaf

    if stats:
        stats.incr("l2p_calls", float(len(leaf_indices)))
        stats.merge_from(op.stats)

    # --- INSTRUMENTATION OUTPUT ---
    log_spectral_stats(temp_logger, "L2P_FINAL_POTENTIAL", potential_tree, 0)  # p=0 for scalar field
    # ------------------------------

    # 4. Return in tree order. The caller (e.g. LaplaceFmm3D) will combine this with
    #    near-field P2P contributions and then call tree.map_to_original_order once
    #    on the combined potential.
    return potential_tree
