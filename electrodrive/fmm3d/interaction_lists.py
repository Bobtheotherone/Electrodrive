from __future__ import annotations

"""
Interaction lists and MAC (multipole acceptance criterion).

This module implements the geometric part of a 3D FMM based on a
tree-of-boxes representation:

- classification of node–node interactions into
  - near-field (P2P / U-list)
  - far-field (M2L / V/W/X-lists)
- construction of SoA-friendly interaction lists that can later be
  consumed by high-performance CPU/GPU kernels.

Design notes
------------
- Dual-tree traversal here is *CPU-only*. Trees must live on CPU.
- The logic is kernel-agnostic (Laplace, Helmholtz, Yukawa, ...).
- Phase-3 implementations may move this traversal to C++/CUDA while
  preserving the public API and MAC semantics.

API compatibility
-----------------
Historically, this module exposed

    build_interaction_lists(tree, mac_theta=...)

while newer call sites use

    build_interaction_lists(source_tree, target_tree, mac_theta=...)

To remain compatible with both styles, the current signature is:

    build_interaction_lists(
        source_tree,
        target_tree=None,
        mac_theta=0.5,
        ...
    )

If ``target_tree`` is ``None``, it defaults to ``source_tree``.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .geometry import compute_node_diameter
from .tree import FmmTree, TreeNode


# ---------------------------------------------------------------------------
# Telemetry / instrumentation
# ---------------------------------------------------------------------------


@dataclass
class InteractionStats:
    """
    Lightweight counters for interaction-list construction.

    Purely diagnostic; useful for validating MAC choices, tree quality,
    and for basic performance telemetry.
    """

    num_p2p_pairs: int = 0
    num_m2l_pairs: int = 0
    num_stack_pops: int = 0
    max_stack_size: int = 0
    num_mac_accept: int = 0
    num_mac_reject: int = 0
    num_leaf_leaf: int = 0
    truncated: bool = False  # True if max_pairs was hit

    def as_dict(self) -> Dict[str, int | bool]:
        """Return a plain-Python dict of counters."""
        return {
            "num_p2p_pairs": self.num_p2p_pairs,
            "num_m2l_pairs": self.num_m2l_pairs,
            "num_stack_pops": self.num_stack_pops,
            "max_stack_size": self.max_stack_size,
            "num_mac_accept": self.num_mac_accept,
            "num_mac_reject": self.num_mac_reject,
            "num_leaf_leaf": self.num_leaf_leaf,
            "truncated": self.truncated,
        }


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class InteractionLists:
    """
    Container for FMM interaction lists on a tree-of-boxes.

    The lists are expressed purely in terms of node indices so they can be
    consumed by either CPU or GPU backends without Python object traversal.

    Self-interactions
    -----------------
    When source and target trees are identical, leaf self-pairs
    ``(i, i)`` are *included* in ``p2p_pairs`` and ``u_list`` and are also
    flagged via ``u_self_mask[i] = True``. Downstream P2P kernels are
    responsible for excluding panel/particle self terms (e.g. ``i == j``)
    if the underlying physics requires it.
    """

    # Global flattened lists of node–node pairs
    p2p_pairs: List[Tuple[int, int]]
    m2l_pairs: List[Tuple[int, int]]

    # Per-target-node lists (indices into source tree)
    mac_theta: float
    u_list: List[List[int]]
    v_list: List[List[int]]
    w_list: List[List[int]]
    x_list: List[List[int]]

    # Per-target-node self-interaction flag (for same-tree case)
    # u_self_mask[t] is True iff (t, t) appears in p2p_pairs.
    u_self_mask: Optional[List[bool]] = field(default=None, repr=False)

    # Whether max_pairs truncated traversal (soft guard)
    truncated: bool = False

    # Optional telemetry
    stats: Optional[InteractionStats] = field(default=None, repr=False)

    # Cached tensor views (lazily materialized)
    _p2p_tensor: Optional[Tensor] = field(default=None, init=False, repr=False)
    _m2l_tensor: Optional[Tensor] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # Convenience properties                                             #
    # ------------------------------------------------------------------ #

    @property
    def num_p2p_pairs(self) -> int:
        """Total number of near-field node–node pairs."""
        return len(self.p2p_pairs)

    @property
    def num_m2l_pairs(self) -> int:
        """Total number of far-field (M2L) node–node pairs."""
        return len(self.m2l_pairs)

    # ------------------------------------------------------------------ #
    # Tensor views                                                       #
    # ------------------------------------------------------------------ #

    def as_tensors(
        self,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Return ``(p2p_pairs, m2l_pairs)`` as ``LongTensor`` of shape ``(K, 2)``.
        """
        if device is not None:
            device = torch.device(device)

        def _ensure_tensor(
            cache: Optional[Tensor],
            pairs: List[Tuple[int, int]],
        ) -> Tensor:
            if cache is not None:
                if device is None or cache.device == device:
                    return cache.to(device) if device is not None else cache

            if pairs:
                t = torch.tensor(pairs, dtype=torch.long)
            else:
                t = torch.empty((0, 2), dtype=torch.long)
            if device is not None:
                t = t.to(device)
            return t

        self._p2p_tensor = _ensure_tensor(self._p2p_tensor, self.p2p_pairs)
        self._m2l_tensor = _ensure_tensor(self._m2l_tensor, self.m2l_pairs)
        return self._p2p_tensor, self._m2l_tensor

    def to(self, device: torch.device | str) -> "InteractionLists":
        """
        Convenience wrapper around :meth:`as_tensors` that returns ``self``.
        """
        self.as_tensors(device=torch.device(device))
        return self


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_leaf(tree: FmmTree, node_idx: int) -> bool:
    """Return ``True`` if the given node has no children."""
    node: TreeNode = tree.nodes[node_idx]
    return all(child is None for child in node.children)


def _node_diameter(tree: FmmTree, node_idx: int) -> float:
    """
    Scalar diameter of a node (Python float).

    Prefer the SoA radius stored on :class:`FmmTree`; fall back to the
    per-node half-extent if the SoA arrays are not populated.

    This helper is used only in MAC evaluation and is therefore
    explicitly CPU-only.
    """
    # Preferred path: SoA radii produced by FmmTree._build_soa().
    if (
        hasattr(tree, "node_radii")
        and isinstance(tree.node_radii, Tensor)
        and tree.node_radii.numel() > 0
    ):
        if tree.node_radii.device.type != "cpu":
            raise RuntimeError("MAC evaluation requires CPU node_radii tensors.")
        return float((2.0 * tree.node_radii[node_idx]).item())

    # Fallback: compute from the TreeNode's half-extent.
    node: TreeNode = tree.nodes[node_idx]
    half_extent = node.half_extent
    if isinstance(half_extent, Tensor) and half_extent.device.type != "cpu":
        raise RuntimeError("MAC evaluation requires CPU half_extent tensors.")
    d = compute_node_diameter(half_extent)
    return float(d.item())


def _node_distance(
    src_tree: FmmTree,
    src_idx: int,
    tgt_tree: FmmTree,
    tgt_idx: int,
) -> float:
    """
    Euclidean distance between node centers (Python float).

    Uses the SoA node-centers representation when available.
    This helper is used only in MAC evaluation and is therefore
    explicitly CPU-only.
    """
    if (
        hasattr(src_tree, "node_centers")
        and hasattr(tgt_tree, "node_centers")
        and isinstance(src_tree.node_centers, Tensor)
        and isinstance(tgt_tree.node_centers, Tensor)
        and src_tree.node_centers.numel() > 0
        and tgt_tree.node_centers.numel() > 0
    ):
        if (
            src_tree.node_centers.device.type != "cpu"
            or tgt_tree.node_centers.device.type != "cpu"
        ):
            raise RuntimeError("MAC evaluation requires CPU node_centers tensors.")
        diff = src_tree.node_centers[src_idx] - tgt_tree.node_centers[tgt_idx]
        return float(torch.linalg.vector_norm(diff).item())

    # Fallback to per-node centers.
    src_c = src_tree.nodes[src_idx].center
    tgt_c = tgt_tree.nodes[tgt_idx].center
    if isinstance(src_c, Tensor) and src_c.device.type != "cpu":
        raise RuntimeError("MAC evaluation requires CPU node centers.")
    if isinstance(tgt_c, Tensor) and tgt_c.device.type != "cpu":
        raise RuntimeError("MAC evaluation requires CPU node centers.")
    return float(torch.linalg.vector_norm(src_c - tgt_c).item())


def _mac_satisfied(
    src_tree: FmmTree,
    src_idx: int,
    tgt_tree: FmmTree,
    tgt_idx: int,
    theta: float,
) -> bool:
    """
    Return ``True`` if the canonical geometric MAC accepts the pair.

    MAC:
        max(d_src, d_tgt) / dist(center_src, center_tgt) < theta
    """
    if theta <= 0.0:
        raise ValueError("mac_theta must be positive")

    dist = _node_distance(src_tree, src_idx, tgt_tree, tgt_idx)
    if dist == 0.0:
        # Overlapping centers (same box) are *never* admissible for M2L.
        return False

    d_src = _node_diameter(src_tree, src_idx)
    d_tgt = _node_diameter(tgt_tree, tgt_idx)
    s = max(d_src, d_tgt)
    return s / dist < theta


def _point_count(tree: FmmTree, node_idx: int) -> int:
    """Number of points covered by a node."""
    node: TreeNode = tree.nodes[node_idx]
    return max(0, int(node.end - node.start))


def _choose_refinement_side(
    src_tree: FmmTree,
    src_idx: int,
    tgt_tree: FmmTree,
    tgt_idx: int,
) -> str:
    """
    Heuristic for dual-tree traversal: which side to refine?
    """
    src_leaf = _is_leaf(src_tree, src_idx)
    tgt_leaf = _is_leaf(tgt_tree, tgt_idx)

    if src_leaf and not tgt_leaf:
        return "tgt"
    if tgt_leaf and not src_leaf:
        return "src"
    if src_leaf and tgt_leaf:
        return "src"

    d_src = _node_diameter(src_tree, src_idx)
    d_tgt = _node_diameter(tgt_tree, tgt_idx)

    if d_src > d_tgt * 1.01:
        return "src"
    if d_tgt > d_src * 1.01:
        return "tgt"

    n_src = _point_count(src_tree, src_idx)
    n_tgt = _point_count(tgt_tree, tgt_idx)
    if n_src >= n_tgt:
        return "src"
    return "tgt"


def _assert_cpu_tree(tree: FmmTree, name: str) -> None:
    """
    Ensure that the tree's geometry tensors live on CPU.

    Dual-tree traversal in this module is CPU-only. This helper is
    intentionally conservative and will raise if it detects CUDA-backed
    tensors for either the SoA centers or the per-node centers.
    """
    if (
        hasattr(tree, "node_centers")
        and isinstance(tree.node_centers, Tensor)
        and tree.node_centers.numel() > 0
    ):
        if tree.node_centers.device.type != "cpu":
            raise RuntimeError(
                f"interaction_lists: {name} must reside on CPU "
                f"(got device={tree.node_centers.device})."
            )
        return

    # Fallback: inspect the root node center, if available.
    if tree.nodes:
        center = tree.nodes[tree.root].center
        if isinstance(center, Tensor) and center.device.type != "cpu":
            raise RuntimeError(
                f"interaction_lists: {name} must reside on CPU "
                f"(got device={center.device})."
            )


# TODO(optimization): Dual-tree traversal should be implemented in
# C++/CUDA for production; this Python version is a correctness spec.


def _build_interaction_lists(
    source_tree: FmmTree,
    target_tree: FmmTree,
    mac_theta: float,
    *,
    max_pairs: Optional[int],
    sort_pairs: bool,
) -> InteractionLists:
    """
    Internal worker that builds near- and far-field interaction lists.

    The public :func:`build_interaction_lists` wrapper adds policy for
    how truncation is surfaced to callers.
    """
    if mac_theta <= 0.0:
        raise ValueError("mac_theta must be positive")
    if mac_theta >= 1.0:
        raise ValueError("mac_theta should be < 1.0 for a meaningful MAC")

    stats = InteractionStats()

    num_src_nodes = len(source_tree.nodes)
    num_tgt_nodes = len(target_tree.nodes)

    # Early exit for empty trees.
    if num_src_nodes == 0 or num_tgt_nodes == 0:
        stats.num_p2p_pairs = 0
        stats.num_m2l_pairs = 0
        lists = InteractionLists(
            p2p_pairs=[],
            m2l_pairs=[],
            mac_theta=float(mac_theta),
            u_list=[[] for _ in range(num_tgt_nodes)],
            v_list=[[] for _ in range(num_tgt_nodes)],
            w_list=[[] for _ in range(num_tgt_nodes)],
            x_list=[[] for _ in range(num_tgt_nodes)],
            u_self_mask=[False for _ in range(num_tgt_nodes)],
            truncated=False,
            stats=stats,
        )
        return lists

    p2p_pairs: List[Tuple[int, int]] = []
    m2l_pairs: List[Tuple[int, int]] = []

    stack: List[Tuple[int, int]] = [(source_tree.root, target_tree.root)]
    stats.max_stack_size = max(stats.max_stack_size, len(stack))

    def _maybe_push_pair(src_idx: int, tgt_idx: int) -> None:
        if max_pairs is not None and (len(p2p_pairs) + len(m2l_pairs)) >= max_pairs:
            stats.truncated = True
            return
        stack.append((src_idx, tgt_idx))
        stats.max_stack_size = max(stats.max_stack_size, len(stack))

    same_tree = source_tree is target_tree

    while stack:
        src_idx, tgt_idx = stack.pop()
        stats.num_stack_pops += 1

        src_leaf = _is_leaf(source_tree, src_idx)
        tgt_leaf = _is_leaf(target_tree, tgt_idx)

        same_node = same_tree and (src_idx == tgt_idx)

        mac_ok = False
        if not same_node:
            mac_ok = _mac_satisfied(
                source_tree, src_idx, target_tree, tgt_idx, mac_theta
            )
            if mac_ok:
                stats.num_mac_accept += 1
            else:
                stats.num_mac_reject += 1

        if mac_ok:
            m2l_pairs.append((src_idx, tgt_idx))
            continue

        if src_leaf and tgt_leaf:
            stats.num_leaf_leaf += 1
            # Leaf–leaf interactions, including same-node pairs when
            # source_tree is target_tree. Downstream P2P kernels are
            # responsible for excluding i == j contributions if needed.
            p2p_pairs.append((src_idx, tgt_idx))
            continue

        side = _choose_refinement_side(source_tree, src_idx, target_tree, tgt_idx)
        if side == "src":
            children = source_tree.nodes[src_idx].children
            for child_idx in children:
                if child_idx is not None:
                    _maybe_push_pair(child_idx, tgt_idx)
        else:
            children = target_tree.nodes[tgt_idx].children
            for child_idx in children:
                if child_idx is not None:
                    _maybe_push_pair(src_idx, child_idx)

    def _dedup(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        # Deterministic deduplication: sort the unique set of pairs.
        return sorted(set(pairs))

    p2p_pairs = _dedup(p2p_pairs)
    m2l_pairs = _dedup(m2l_pairs)

    if sort_pairs:
        # Stable ordering: sort by (tgt, src) so that all interactions
        # for a given target node form a contiguous block.
        p2p_pairs = sorted(p2p_pairs, key=lambda ij: (ij[1], ij[0]))
        m2l_pairs = sorted(m2l_pairs, key=lambda ij: (ij[1], ij[0]))

    # Rebuild per-target lists from the deduplicated, optionally-sorted
    # global pair lists. Also construct the self-interaction mask.
    u_list: List[List[int]] = [[] for _ in range(num_tgt_nodes)]
    v_list: List[List[int]] = [[] for _ in range(num_tgt_nodes)]
    w_list: List[List[int]] = [[] for _ in range(num_tgt_nodes)]
    x_list: List[List[int]] = [[] for _ in range(num_tgt_nodes)]
    u_self_mask: List[bool] = [False for _ in range(num_tgt_nodes)]

    for src_idx, tgt_idx in p2p_pairs:
        u_list[tgt_idx].append(src_idx)
        if same_tree and src_idx == tgt_idx:
            u_self_mask[tgt_idx] = True

    for src_idx, tgt_idx in m2l_pairs:
        v_list[tgt_idx].append(src_idx)
        w_list[tgt_idx].append(src_idx)
        x_list[tgt_idx].append(src_idx)

    stats.num_p2p_pairs = len(p2p_pairs)
    stats.num_m2l_pairs = len(m2l_pairs)

    lists = InteractionLists(
        p2p_pairs=p2p_pairs,
        m2l_pairs=m2l_pairs,
        mac_theta=float(mac_theta),
        u_list=u_list,
        v_list=v_list,
        w_list=w_list,
        x_list=x_list,
        u_self_mask=u_self_mask,
        truncated=stats.truncated,
        stats=stats,
    )

    if lists.truncated:
        # Truncation in this context almost always means either a broken
        # MAC or a degenerate tree producing an explosion of pairs.
        raise RuntimeError(
            "Interaction list truncated—MAC is likely misconfigured or pair count exploded."
        )

    return lists


def build_interaction_lists(
    source_tree: FmmTree,
    target_tree: Optional[FmmTree] = None,
    mac_theta: float = 0.5,
    *,
    max_pairs: Optional[int] = None,
    sort_pairs: bool = True,
) -> InteractionLists:
    """
    Public API for building FMM interaction lists.

    Parameters
    ----------
    source_tree:
        Source FmmTree.
    target_tree:
        Target FmmTree. If ``None``, this defaults to ``source_tree``,
        which preserves compatibility with the older API
        ``build_interaction_lists(tree, mac_theta=...)``.
    mac_theta:
        MAC parameter (opening angle). Must be in (0, 1).
    max_pairs:
        Optional soft cap on total number of node–node pairs. If this
        is exceeded, traversal is marked as truncated and a RuntimeError
        is raised by the worker.
    sort_pairs:
        If True, node pairs are sorted by (tgt, src) to make per-target
        slices contiguous.

    Notes
    -----
    Trees must live on CPU. This is enforced to avoid accidental device
    synchronisation when calling into Python from CUDA tensors.
    """
    if target_tree is None:
        target_tree = source_tree

    _assert_cpu_tree(source_tree, "source_tree")
    _assert_cpu_tree(target_tree, "target_tree")

    return _build_interaction_lists(
        source_tree=source_tree,
        target_tree=target_tree,
        mac_theta=mac_theta,
        max_pairs=max_pairs,
        sort_pairs=sort_pairs,
    )


def verify_interaction_lists(
    source_tree: FmmTree,
    target_tree: FmmTree,
    lists: InteractionLists,
) -> None:
    """
    Debug helper: sanity-check interaction lists against the trees.

    This is intended for tests and assertions, not for hot paths.
    Raises ``AssertionError`` if an inconsistency is found.
    """
    num_src_nodes = len(source_tree.nodes)
    num_tgt_nodes = len(target_tree.nodes)

    # Global pair ranges
    for src_idx, tgt_idx in lists.p2p_pairs:
        assert 0 <= src_idx < num_src_nodes, "p2p src_idx out of range"
        assert 0 <= tgt_idx < num_tgt_nodes, "p2p tgt_idx out of range"

    for src_idx, tgt_idx in lists.m2l_pairs:
        assert 0 <= src_idx < num_src_nodes, "m2l src_idx out of range"
        assert 0 <= tgt_idx < num_tgt_nodes, "m2l tgt_idx out of range"

    p2p_set = set(lists.p2p_pairs)
    m2l_set = set(lists.m2l_pairs)

    # Per-target lists must be consistent with global lists.
    for tgt_idx in range(num_tgt_nodes):
        for src_idx in lists.u_list[tgt_idx]:
            assert (src_idx, tgt_idx) in p2p_set, "u_list not subset of p2p_pairs"
        for src_idx in lists.v_list[tgt_idx]:
            assert (src_idx, tgt_idx) in m2l_set, "v_list not subset of m2l_pairs"
        for src_idx in lists.w_list[tgt_idx]:
            assert (src_idx, tgt_idx) in m2l_set, "w_list not subset of m2l_pairs"
        for src_idx in lists.x_list[tgt_idx]:
            assert (src_idx, tgt_idx) in m2l_set, "x_list not subset of m2l_pairs"

    # Self-interaction mask must agree with p2p_pairs for same-tree case.
    if source_tree is target_tree and lists.u_self_mask is not None:
        for tgt_idx in range(num_tgt_nodes):
            has_self = (tgt_idx, tgt_idx) in p2p_set
            assert lists.u_self_mask[tgt_idx] == has_self, "u_self_mask mismatch"


__all__ = [
    "InteractionStats",
    "InteractionLists",
    "build_interaction_lists",
    "verify_interaction_lists",
]
