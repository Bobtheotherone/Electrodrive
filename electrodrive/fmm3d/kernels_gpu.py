from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from electrodrive.fmm3d.config import FmmConfig
from electrodrive.utils.logging import JsonlLogger
from electrodrive.fmm3d.tree import FmmTree
from electrodrive.fmm3d.interaction_lists import InteractionLists
from electrodrive.fmm3d.multipole_operators import (
    MultipoleCoefficients,
    LocalCoefficients,
    MultipoleOpStats,
    p2m as _p2m_tree,
    m2m as _m2m_tree,
    m2l as _m2l_tree,
    l2l as _l2l_tree,
    l2p as _l2p_tree,
)
from electrodrive.fmm3d.logging_utils import (
    debug_tensor_stats,
    get_logger,
)

# Reuse the P2PResult container and the low-level 1/r block kernel from the
# CPU module. The implementation there is device-agnostic: as long as the
# input tensors live on the same device (CPU or CUDA), it will execute on
# that device.
from electrodrive.fmm3d.kernels_cpu import P2PResult, _p2p_block


__all__ = [
    "P2PResult",
    "apply_p2p_gpu",
    "apply_p2p_gpu_tiled",
    "p2m_gpu",
    "m2m_gpu",
    "m2l_gpu",
    "l2l_gpu",
    "l2p_gpu",
]


# ---------------------------------------------------------------------------
# P2P near-field on GPU (device-agnostic, runs on CPU or CUDA)
# ---------------------------------------------------------------------------


def apply_p2p_gpu_tiled(
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
    Evaluate near-field P2P interactions given interaction lists.

    This is the GPU-aware counterpart of :func:`apply_p2p_cpu_tiled`. It
    shares the same semantics and numerical behaviour, but operates on
    whatever device the trees live on (CPU or CUDA). In particular, if
    ``source_tree.points`` and ``target_tree.points`` are on a CUDA
    device, all heavy work happens on the GPU.

    Notes
    -----
    - The kernel evaluates the *pure* 1/|r| potential. The Coulomb
      constant K_E is **not** applied here; it must be applied by the
      caller (e.g. BEM layer) to the combined near- and far-field
      contributions.
    - Interaction lists are still constructed on the CPU via
      :func:`build_interaction_lists`. This function only consumes the
      node-index pairs exposed by :class:`InteractionLists`.
    """
    if lists.truncated:
        raise RuntimeError(
            "apply_p2p_gpu_tiled: interaction lists were truncated "
            "(mac_theta too small or max_pairs too low). "
            "Rebuild the InteractionLists with a larger max_pairs or "
            "relax the MAC parameter (mac_theta).",
        )

    # P2P implies source charges match source points.
    # charges_src must be in TREE ORDER (as returned by p2m_* in MultipoleCoefficients).
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
    if target_tree.device != source_tree.device:
        raise ValueError(
            "source_tree and target_tree must live on the same device "
            f"(got {source_tree.device} vs {target_tree.device}).",
        )

    if tile_size_points <= 0:
        # Treat non-positive values as "no tiling" and let the kernel
        # handle full leaf–leaf blocks.
        tile_size_points = source_tree.n_points * max(1, target_tree.n_points)

    # Centralized logger resolution (reuses the same helper as the CPU path).
    logger = get_logger(logger)

    device = target_tree.device
    dtype = target_tree.dtype

    # Interaction pairs live naturally on the CPU; indices are small and
    # cheap to move. We keep them on CPU and only use Python int() to
    # index into the (possibly CUDA) node_ranges arrays.
    p2p_pairs, _ = lists.as_tensors(device=torch.device("cpu"))
    num_pairs = int(p2p_pairs.shape[0])

    if logger is not None:
        logger.debug(
            f"Entered apply_p2p_gpu_tiled. "
            f"Pairs to process: {num_pairs}, device={device}, dtype={dtype}."
        )

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

    # Debug instrumentation mirrors the CPU implementation.
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

    debug_tensor_stats("phi_tgt (P2P GPU Result)", phi_tgt, logger)

    if logger is not None:
        logger.debug(
            "apply_p2p_gpu_tiled completed. "
            f"Pairs={num_pairs}, Interactions={total_interactions}"
        )

    return P2PResult(
        potential=phi_tgt,
        n_pairs=num_pairs,
        n_interactions=total_interactions,
    )


def apply_p2p_gpu(
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
    Convenience wrapper around :func:`apply_p2p_gpu_tiled`.

    Behaviour is identical to :func:`apply_p2p_cpu` except that the work
    is carried out on the device of ``source_tree`` / ``target_tree``.
    In particular, if the trees live on a CUDA device, this routine
    becomes a GPU P2P kernel.
    """
    tile_size = 16384
    if cfg is not None and getattr(cfg, "p2p_batch_size", None) is not None:
        # Heuristic: reuse autotuned batch size if available; otherwise
        # fall back to a reasonably cache-friendly default.
        tile_size = int(cfg.p2p_batch_size)

    return apply_p2p_gpu_tiled(
        source_tree=source_tree,
        target_tree=target_tree,
        charges_src=charges_src,
        lists=lists,
        tile_size_points=tile_size,
        logger=logger,
        out=out,
    )


# ---------------------------------------------------------------------------
# Tree-aware multipole / local operators on GPU
# ---------------------------------------------------------------------------


def _ensure_device_dtype(x: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
    """
    Utility to move tensors to (device, dtype).

    Higher-level code is responsible for ensuring that such transfers are
    not on the critical path (typically we move all state to CUDA once at
    backend setup).
    """
    if x.device == device and x.dtype == dtype:
        return x
    return x.to(device=device, dtype=dtype)


def p2m_gpu(
    tree: FmmTree,
    charges: Tensor,
    cfg: FmmConfig,
    *,
    stats: Optional[MultipoleOpStats] = None,
) -> MultipoleCoefficients:
    """
    GPU-aware P2M wrapper.

    This delegates to the tree-level :func:`multipole_operators.p2m`
    implementation, which already supports arbitrary devices. The only
    additional logic here is to ensure that ``charges`` lives on the same
    device/dtype as the tree.
    """
    if charges.shape[0] != tree.n_points:
        raise ValueError(
            f"charges length {charges.shape[0]} != tree.n_points {tree.n_points}"
        )

    device = tree.device
    dtype = tree.dtype
    charges_dev = _ensure_device_dtype(charges, device, dtype)

    # Tree-level p2m handles global scale and logging. :contentReference[oaicite:3]{index=3}
    return _p2m_tree(tree=tree, charges=charges_dev, cfg=cfg, stats=stats)


def m2m_gpu(
    tree: FmmTree,
    multipoles: MultipoleCoefficients,
    cfg: FmmConfig,
    *,
    stats: Optional[MultipoleOpStats] = None,
) -> MultipoleCoefficients:
    """
    GPU-aware wrapper for tree-level M2M.

    Provided the underlying :class:`FmmTree` and multipole coefficients
    live on the same device (typically CUDA), this will run there.
    """
    if multipoles.tree is not tree:
        raise ValueError(
            "m2m_gpu expects multipoles.tree to be the same FmmTree instance "
            "as the 'tree' argument."
        )
    if multipoles.data.device != tree.device:
        raise ValueError(
            "m2m_gpu expects multipoles.data to live on the same device as tree."
        )

    return _m2m_tree(tree=tree, multipoles=multipoles, cfg=cfg, stats=stats)


def m2l_gpu(
    source_tree: FmmTree,
    target_tree: FmmTree,
    multipoles: MultipoleCoefficients,
    lists: InteractionLists,
    cfg: FmmConfig,
    *,
    stats: Optional[MultipoleOpStats] = None,
) -> LocalCoefficients:
    """
    GPU-aware wrapper for tree-level M2L.

    Parameters
    ----------
    source_tree, target_tree:
        Geometry trees. For symmetric BEM applications these are
        typically the same tree instance.
    multipoles:
        Multipole coefficients defined on ``source_tree``.
    lists:
        Precomputed :class:`InteractionLists` built on the CPU. These
        encode the node–node M2L and P2P pairs. We always require
        explicit lists for the GPU path to avoid trying to rebuild them
        on a CUDA tree.
    cfg:
        FMM configuration (expansion order, MAC parameter, etc.).
    stats:
        Optional :class:`MultipoleOpStats` accumulator.

    Returns
    -------
    LocalCoefficients
        Local expansions attached to ``target_tree`` nodes.
    """
    if getattr(lists, "truncated", False):
        raise RuntimeError(
            "m2l_gpu: interaction lists were truncated. "
            "Increase max_pairs or relax mac_theta."
        )

    # NOTE: we intentionally do *not* call the tree-level m2l() without
    # lists: building interaction lists requires CPU trees. 
    return _m2l_tree(
        source_tree=source_tree,
        target_tree=target_tree,
        multipoles=multipoles,
        cfg=cfg,
        stats=stats,
        lists=lists,
    )


def l2l_gpu(
    tree: FmmTree,
    locals_: LocalCoefficients,
    cfg: FmmConfig,
    *,
    stats: Optional[MultipoleOpStats] = None,
) -> LocalCoefficients:
    """
    GPU-aware wrapper for tree-level L2L.

    This is a thin pass-through; all the heavy lifting (and logging) is
    handled by :func:`multipole_operators.l2l`.
    """
    if locals_.tree is not tree:
        raise ValueError(
            "l2l_gpu expects locals_.tree to be the same FmmTree instance "
            "as the 'tree' argument."
        )
    if locals_.data.device != tree.device:
        raise ValueError(
            "l2l_gpu expects locals_.data to live on the same device as tree."
        )

    return _l2l_tree(tree=tree, locals_=locals_, cfg=cfg, stats=stats)


def l2p_gpu(
    tree: FmmTree,
    locals_: LocalCoefficients,
    cfg: FmmConfig,
    *,
    stats: Optional[MultipoleOpStats] = None,
) -> Tensor:
    """
    GPU-aware wrapper for tree-level L2P.

    Returns potentials in **tree order** for the pure 1/|r| kernel.
    Any physics constant K_E must be applied by the caller.
    """
    if locals_.tree is not tree:
        raise ValueError(
            "l2p_gpu expects locals_.tree to be the same FmmTree instance "
            "as the 'tree' argument."
        )
    if locals_.data.device != tree.device:
        raise ValueError(
            "l2p_gpu expects locals_.data to live on the same device as tree."
        )

    return _l2p_tree(tree=tree, locals_=locals_, cfg=cfg, stats=stats)
