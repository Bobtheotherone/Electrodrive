"""Spatial tree / octree data structures for the 3D FMM.

This module is the geometric backbone of the Tier-3 FMM stack. It
turns an unstructured set of source (or target) points into a hierarchical
octree that later phases depend on:

- multipoles / local expansions attach coefficients to tree nodes,
- CPU / GPU kernels traverse the tree and operate on node-local SoA data,
- multi-GPU / MPI partition the tree in Morton / octree order.

The tree is purely geometric: it knows about points, bounding boxes,
levels, and index ranges; it does *not* depend on kernel type or multipole
order. This keeps the separation of concerns clean:

- ``geometry.py``            → bounding boxes / Morton keys
- ``tree.py`` (this module)  → spatial hierarchy and index ranges
- ``interaction_lists.py``   → MAC and node–node classification
- ``multipole_operators.py`` → P2M / M2M / M2L / L2L / L2P

All tensors are standard PyTorch tensors and can live on CPU or GPU.
The reference implementation here builds the tree with Python + PyTorch
loops and is **designed to run on the CPU**. A future C++ / CUDA
implementation can reuse the same data layout and API.

Typical workflow:

1. Build tree on CPU via :func:`build_fmm_tree`.
2. Move tree to GPU via :meth:`FmmTree.to` for kernel evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import warnings

import torch
from torch import Tensor

from .geometry import (
    BoundingBox,
    BoundingBoxArray,
    compute_bounding_box,
)

__all__ = [
    "TreeNode",
    "TreeStatistics",
    "FmmTree",
    "build_fmm_tree",
]


# ---------------------------------------------------------------------------
# Node dataclass (object-oriented view)
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    """A single node in the adaptive octree.

    This is a light-weight, object-oriented view of the tree. Core FMM
    kernels work on the SoA fields of :class:`FmmTree` instead.

    Attributes
    ----------
    index:
        Integer index of this node in the global node arrays.
    parent:
        Parent node index or ``None`` for the root.
    level:
        Depth of the node in the tree (root has ``level = 0``).
    center:
        Tensor of shape ``(3,)`` with the box center.
    half_extent:
        Tensor of shape ``(3,)`` with half side lengths.
    start, end:
        Integer indices such that the points belonging to this node are
        exactly those in ``points[index_map[start:end]]`` in *original*
        point order; equivalently ``points_tree[start:end]`` where
        ``points_tree = points[index_map]``.
    children:
        Tuple of length 8 containing the indices of child nodes or
        ``None`` for missing children.
    """

    index: int
    parent: Optional[int]
    level: int
    center: Tensor
    half_extent: Tensor
    start: int
    end: int
    children: Tuple[Optional[int], ...]  # len 8

    @property
    def is_leaf(self) -> bool:
        """Return True if this node has no children."""
        return all(c is None for c in self.children)

    @property
    def n_points(self) -> int:
        """Number of points contained in this node."""
        return self.end - self.start

    # Backwards-compatibility alias for older code that expects ``half_size``.
    @property
    def half_size(self) -> Tensor:
        """Alias for :attr:`half_extent` (for compatibility)."""
        return self.half_extent


# ---------------------------------------------------------------------------
# Tree statistics / instrumentation
# ---------------------------------------------------------------------------


@dataclass
class TreeStatistics:
    """Summary statistics for an :class:`FmmTree`.

    Fields
    ------
    n_points:
        Total number of points in the tree.
    n_nodes:
        Total number of nodes.
    n_leaves:
        Number of leaf nodes.
    max_depth:
        Maximum level index present in the tree (root has level 0), or
        ``-1`` for an empty tree.
    nodes_per_level:
        1D tensor where ``nodes_per_level[l]`` is the number of nodes at
        level ``l``.
    leaf_size_min, leaf_size_max, leaf_size_mean:
        Basic statistics over the number of points per leaf node.
    """

    n_points: int
    n_nodes: int
    n_leaves: int
    max_depth: int
    nodes_per_level: Tensor
    leaf_size_min: int
    leaf_size_max: int
    leaf_size_mean: float

    def as_dict(self) -> dict:
        """Return a Python ``dict`` representation (for logging / JSON)."""
        return {
            "n_points": self.n_points,
            "n_nodes": self.n_nodes,
            "n_leaves": self.n_leaves,
            "max_depth": self.max_depth,
            "nodes_per_level": self.nodes_per_level.clone(),
            "leaf_size_min": self.leaf_size_min,
            "leaf_size_max": self.leaf_size_max,
            "leaf_size_mean": self.leaf_size_mean,
        }


# ---------------------------------------------------------------------------
# FmmTree container (SoA view)
# ---------------------------------------------------------------------------


class FmmTree:
    """Adaptive octree over a set of 3D points.

    The tree is built by recursively subdividing axis-aligned boxes
    until either:

    - the number of points in a node is ``<= leaf_size``, or
    - the depth reaches ``max_depth``.

    Geometry and tree topology are stored in *structure-of-arrays (SoA)*
    form for use by later CPU/GPU kernels. A list of :class:`TreeNode`
    objects is also kept for debugging and high-level algorithms that
    want an object-oriented view.

    Notes
    -----
    - ``global_bbox`` stored on the tree is a tight bounding box over all
      input points. It is used as metadata and as a canonical frame for
      Morton keys or domain decomposition; subdivision itself is based on
      local node bounding boxes.
    - Indexing conventions:

        * ``points`` stored on the tree are in **tree order**.
        * ``tree_to_original[i]`` gives the original point index for
          point ``i`` in tree order.
        * ``original_to_tree[j]`` gives the tree-order index for the
          original point index ``j``.

    - Typical usage is:

        .. code-block:: python

            tree = build_fmm_tree(points_cpu)
            tree.to("cuda")  # move to GPU before kernels
    """

    # NOTE: callers should not instantiate FmmTree directly; use
    # :func:`build_fmm_tree` or a future factory that accepts a full
    # :class:`FmmConfig`.

    def __init__(
        self,
        points: Tensor,
        *,
        index_map: Tensor,
        nodes: List[TreeNode],
        leaf_size: int,
        max_depth: int,
        global_bbox: BoundingBox,
    ) -> None:
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {tuple(points.shape)}")

        self.points = points

        # Permutation between original and tree order.
        #
        # - tree_to_original[i] == original index of point at tree position i.
        # - original_to_tree[j] == tree position of point j in the original array.
        #
        # For an empty tree both are zero-length tensors.
        self.index_map = index_map  # tree_index -> original_index
        self.inverse_index_map = torch.empty_like(index_map)
        if index_map.numel() > 0:
            self.inverse_index_map[index_map] = torch.arange(
                index_map.shape[0],
                device=index_map.device,
                dtype=index_map.dtype,
            )

        self.nodes: List[TreeNode] = nodes
        self.leaf_size = int(leaf_size)
        self.max_depth = int(max_depth)
        self.global_bbox = global_bbox

        # Build SoA representation used by interaction lists + kernels.
        self._build_soa()
        # Ensure SoA layout is device/dtype-consistent.
        self._assert_soa_consistent()

    # ------------------------- basic properties -------------------------

    @property
    def device(self) -> torch.device:
        return self.points.device

    @property
    def dtype(self) -> torch.dtype:
        return self.points.dtype

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def root(self) -> int:
        """Index of the root node (or -1 for an empty tree)."""
        return 0 if self.n_nodes > 0 else -1

    # Explicit names for the permutation vectors, for readability.
    @property
    def tree_to_original(self) -> Tensor:
        """Permutation: tree index → original index."""
        return self.index_map

    @property
    def original_to_tree(self) -> Tensor:
        """Permutation: original index → tree index."""
        return self.inverse_index_map

    # --------------------- structure-of-arrays view ---------------------

    def _build_soa(self) -> None:
        """Populate SoA fields from the Python ``TreeNode`` list.

        Arrays created
        --------------
        - ``node_centers``      : ``(M, 3)``
        - ``node_half_extents`` : ``(M, 3)``
        - ``node_radii``        : ``(M,)``
        - ``node_levels``       : ``(M,)`` (int32)
        - ``node_parents``      : ``(M,)`` (int64, -1 for root)
        - ``node_children``     : ``(M, 8)`` (int64, -1 for missing)
        - ``node_ranges``       : ``(M, 2)`` (int64, [start, end))
        - ``node_is_leaf``      : ``(M,)`` (bool)
        """
        M = len(self.nodes)
        device = self.points.device
        dtype = self.points.dtype

        if M == 0:
            self.node_centers = torch.empty(0, 3, device=device, dtype=dtype)
            self.node_half_extents = torch.empty(0, 3, device=device, dtype=dtype)
            self.node_radii = torch.empty(0, device=device, dtype=dtype)
            self.node_levels = torch.empty(0, device=device, dtype=torch.int32)
            self.node_parents = torch.empty(0, device=device, dtype=torch.int64)
            self.node_children = torch.empty(0, 8, device=device, dtype=torch.int64)
            self.node_ranges = torch.empty(0, 2, device=device, dtype=torch.int64)
            self.node_is_leaf = torch.empty(0, device=device, dtype=torch.bool)
            return

        centers = torch.empty(M, 3, device=device, dtype=dtype)
        half_extents = torch.empty(M, 3, device=device, dtype=dtype)
        radii = torch.empty(M, device=device, dtype=dtype)
        levels = torch.empty(M, device=device, dtype=torch.int32)
        parents = torch.full((M,), -1, device=device, dtype=torch.int64)
        children = torch.full((M, 8), -1, device=device, dtype=torch.int64)
        ranges = torch.empty(M, 2, device=device, dtype=torch.int64)
        is_leaf = torch.empty(M, device=device, dtype=torch.bool)

        for node in self.nodes:
            i = int(node.index)
            centers[i] = node.center.to(device=device, dtype=dtype)
            half_extents[i] = node.half_extent.to(device=device, dtype=dtype)
            # Conservative node radius: distance to box corner with a tiny
            # safety factor to guard against floating-point shrinkage in
            # extremely thin boxes.
            radii[i] = torch.linalg.norm(half_extents[i]) * (1.0 + 1e-12)
            levels[i] = int(node.level)
            parents[i] = -1 if node.parent is None else int(node.parent)
            ranges[i, 0] = int(node.start)
            ranges[i, 1] = int(node.end)
            is_leaf[i] = node.is_leaf
            for octant, child in enumerate(node.children):
                if child is not None:
                    children[i, octant] = int(child)

        self.node_centers = centers
        self.node_half_extents = half_extents
        self.node_radii = radii
        self.node_levels = levels
        self.node_parents = parents
        self.node_children = children
        self.node_ranges = ranges
        self.node_is_leaf = is_leaf

    def _assert_soa_consistent(self) -> None:
        """Internal: enforce device/dtype consistency across SoA tensors.

        This check is critical before wiring GPU kernels; it guarantees
        that all floating-point fields share the same dtype/device, and
        that integer / boolean fields have the expected dtypes on the
        same device.
        """
        dev = self.points.device
        float_dtype = self.points.dtype

        # Floating-point tensors must all share (device, dtype).
        float_tensors = [
            self.points,
            self.node_centers,
            self.node_half_extents,
            self.node_radii,
            self.global_bbox.center,
            self.global_bbox.half_extent,
        ]
        for t in float_tensors:
            if t.device != dev or t.dtype != float_dtype:
                raise RuntimeError(
                    "FmmTree SoA inconsistency: floating-point tensor "
                    f"has (device={t.device}, dtype={t.dtype}) != "
                    f"(device={dev}, dtype={float_dtype})."
                )

        # Integer tensors: must be on same device and consistent dtypes.
        int64_tensors = [
            self.index_map,
            self.inverse_index_map,
            self.node_parents,
            self.node_children,
            self.node_ranges,
        ]
        for t in int64_tensors:
            if t.device != dev or t.dtype != torch.int64:
                raise RuntimeError(
                    "FmmTree SoA inconsistency: integer tensor must be "
                    f"(device={dev}, dtype=torch.int64), got "
                    f"(device={t.device}, dtype={t.dtype})."
                )

        if self.node_levels.device != dev or self.node_levels.dtype != torch.int32:
            raise RuntimeError(
                "FmmTree SoA inconsistency: node_levels must be int32 on "
                f"{dev}, got (device={self.node_levels.device}, "
                f"dtype={self.node_levels.dtype})."
            )

        if self.node_is_leaf.device != dev or self.node_is_leaf.dtype != torch.bool:
            raise RuntimeError(
                "FmmTree SoA inconsistency: node_is_leaf must be bool on "
                f"{dev}, got (device={self.node_is_leaf.device}, "
                f"dtype={self.node_is_leaf.dtype})."
            )

    # ------------------------- convenience APIs -------------------------

    def bounding_boxes(self) -> BoundingBoxArray:
        """Return a :class:`BoundingBoxArray` for all nodes."""
        return BoundingBoxArray(
            centers=self.node_centers,
            half_extents=self.node_half_extents,
            radii=self.node_radii,
        )

    def leaf_indices(self) -> Tensor:
        """Return indices of leaf nodes as a 1D tensor."""
        if self.node_is_leaf.numel() == 0:
            return torch.empty(0, device=self.device, dtype=torch.int64)
        return torch.nonzero(self.node_is_leaf, as_tuple=False).view(-1).to(torch.int64)

    def leaves(self) -> List[TreeNode]:
        """Return leaf nodes as a Python list."""
        return [self.nodes[int(i)] for i in self.leaf_indices()]

    def map_to_tree_order(self, values: Tensor) -> Tensor:
        """Reorder a per-point array from original to tree order.

        Parameters
        ----------
        values:
            Tensor whose first dimension has length ``N == self.n_points``,
            containing data indexed in the *original* order (the order in
            which points were passed to :func:`build_fmm_tree`).

        Returns
        -------
        Tensor
            View of ``values`` reordered such that index ``k`` matches
            ``self.points[k]`` (tree order).
        """
        if values.shape[0] != self.n_points:
            raise ValueError(
                f"values.shape[0] = {values.shape[0]} does not match n_points = {self.n_points}"
            )
        if self.index_map.numel() == 0:
            return values
        return values[self.index_map]

    def map_to_original_order(self, values: Tensor) -> Tensor:
        """Inverse of :meth:`map_to_tree_order`.

        Parameters
        ----------
        values:
            Tensor whose first dimension has length ``N == self.n_points``,
            containing data indexed in *tree* order.

        Returns
        -------
        Tensor
            View of ``values`` such that index ``i`` corresponds to the
            original ordering of input points.
        """
        if values.shape[0] != self.n_points:
            raise ValueError(
                f"values.shape[0] = {values.shape[0]} does not match n_points = {self.n_points}"
            )
        if self.index_map.numel() == 0:
            return values
        return values[self.inverse_index_map]

    # Backwards-compatibility alias for older code paths.
    def map_from_tree_order(self, values: Tensor) -> Tensor:
        """Alias for :meth:`map_to_original_order`.

        Historically, some callers used :meth:`map_from_tree_order` to
        denote a mapping from tree order back to the original point
        order. New code should use :meth:`map_to_original_order`
        instead; this method exists solely to keep older tests and
        utilities working without modification.
        """
        return self.map_to_original_order(values)

    def to(
        self,
        device: Union[str, torch.device],
        dtype: Optional[torch.dtype] = None,
    ) -> "FmmTree":
        """Move all tree tensors to a new device / dtype.

        This mirrors the semantics of :meth:`torch.nn.Module.to`:
        the operation is in-place and returns ``self`` for convenience.

        Parameters
        ----------
        device:
            Target device (e.g. ``"cpu"``, ``"cuda"`` or ``torch.device``).
        dtype:
            Optional target floating-point dtype (e.g. ``torch.float32``).
            If ``None``, the current ``self.dtype`` is preserved.

        Returns
        -------
        FmmTree
            The same instance, modified in-place.
        """
        device = torch.device(device)
        target_dtype = self.points.dtype if dtype is None else dtype

        # Floating-point tensors
        self.points = self.points.to(device=device, dtype=target_dtype)
        self.node_centers = self.node_centers.to(device=device, dtype=target_dtype)
        self.node_half_extents = self.node_half_extents.to(device=device, dtype=target_dtype)
        self.node_radii = self.node_radii.to(device=device, dtype=target_dtype)
        self.global_bbox = BoundingBox(
            center=self.global_bbox.center.to(device=device, dtype=target_dtype),
            half_extent=self.global_bbox.half_extent.to(device=device, dtype=target_dtype),
        )

        # Integer / bool tensors (device-only move)
        self.index_map = self.index_map.to(device=device)
        self.inverse_index_map = self.inverse_index_map.to(device=device)
        self.node_levels = self.node_levels.to(device=device)
        self.node_parents = self.node_parents.to(device=device)
        self.node_children = self.node_children.to(device=device)
        self.node_ranges = self.node_ranges.to(device=device)
        self.node_is_leaf = self.node_is_leaf.to(device=device)

        # Keep TreeNode views consistent.
        for node in self.nodes:
            node.center = node.center.to(device=device, dtype=target_dtype)
            node.half_extent = node.half_extent.to(device=device, dtype=target_dtype)

        # Enforce SoA invariants after migration.
        self._assert_soa_consistent()
        return self

    # ------------------------ diagnostics / stats ------------------------

    def statistics(self) -> TreeStatistics:
        """Compute summary statistics for this tree.

        This is intended for instrumentation, logging, and quick sanity
        checks; it is cheap compared to tree construction and uses only
        existing SoA fields.
        """
        n_points = self.n_points
        n_nodes = self.n_nodes

        if n_nodes == 0:
            nodes_per_level = torch.empty(0, dtype=torch.int64, device=self.device)
            return TreeStatistics(
                n_points=n_points,
                n_nodes=0,
                n_leaves=0,
                max_depth=-1,
                nodes_per_level=nodes_per_level,
                leaf_size_min=0,
                leaf_size_max=0,
                leaf_size_mean=0.0,
            )

        levels = self.node_levels.to(torch.int64)
        max_depth = int(levels.max().item())
        nodes_per_level = torch.bincount(levels, minlength=max_depth + 1)

        leaf_idx = self.leaf_indices()
        n_leaves = int(leaf_idx.numel())
        if n_leaves > 0:
            ranges = self.node_ranges[leaf_idx]
            leaf_sizes = (ranges[:, 1] - ranges[:, 0]).to(torch.int64)
            leaf_size_min = int(leaf_sizes.min().item())
            leaf_size_max = int(leaf_sizes.max().item())
            leaf_size_mean = float(leaf_sizes.to(torch.float32).mean().item())
        else:
            leaf_size_min = leaf_size_max = 0
            leaf_size_mean = 0.0

        return TreeStatistics(
            n_points=n_points,
            n_nodes=n_nodes,
            n_leaves=n_leaves,
            max_depth=max_depth,
            nodes_per_level=nodes_per_level,
            leaf_size_min=leaf_size_min,
            leaf_size_max=leaf_size_max,
            leaf_size_mean=leaf_size_mean,
        )

    def describe(self) -> str:
        """Return a human-readable multi-line description of the tree.

        Examples
        --------
        >>> print(tree.describe())
        FmmTree: n_points=..., n_nodes=..., n_leaves=..., max_depth=...
          leaf_size: min=..., max=..., mean=...
          nodes_per_level: 0:1, 1:8, 2:32, ...
        """
        stats = self.statistics()
        lines = [
            f"FmmTree: n_points={stats.n_points}, "
            f"n_nodes={stats.n_nodes}, n_leaves={stats.n_leaves}, "
            f"max_depth={stats.max_depth}",
            f"  leaf_size: min={stats.leaf_size_min}, "
            f"max={stats.leaf_size_max}, mean={stats.leaf_size_mean:.2f}",
        ]
        if stats.nodes_per_level.numel() > 0:
            levels_str = ", ".join(
                f"{lvl}:{int(cnt)}" for lvl, cnt in enumerate(stats.nodes_per_level.tolist())
            )
            lines.append(f"  nodes_per_level: {levels_str}")
        return "\n".join(lines)

    def verify(
        self,
        *,
        check_bboxes: bool = True,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> None:
        """Check internal invariants of the tree.

        This is a debug / testing helper. It raises :class:`ValueError`
        if any invariant is violated.

        Parameters
        ----------
        check_bboxes:
            If ``True``, recompute bounding boxes from point ranges and
            compare against stored node centers / half extents.
        atol, rtol:
            Absolute and relative tolerances used for bounding-box checks.
        """
        # Ensure SoA tensors live on the same device and use consistent dtypes.
        self._assert_soa_consistent()

        n = self.n_points
        m = self.n_nodes

        # Empty tree invariants.
        if m == 0:
            if n != 0:
                raise ValueError("Empty tree must have zero points.")
            return

        # Permutation arrays must be consistent with n_points.
        if self.index_map.shape[0] != n or self.inverse_index_map.shape[0] != n:
            raise ValueError("Permutation arrays do not match n_points")

        if n > 0:
            expected = torch.arange(n, device=self.device, dtype=self.index_map.dtype)
            if not torch.equal(self.index_map[self.inverse_index_map], expected):
                raise ValueError("index_map and inverse_index_map are not true inverses")

        # Node ranges must be well-formed and within [0, n).
        if self.node_ranges.shape[0] != m:
            raise ValueError("node_ranges length must equal n_nodes")

        starts = self.node_ranges[:, 0]
        ends = self.node_ranges[:, 1]

        if (starts < 0).any() or (ends > n).any() or (starts > ends).any():
            raise ValueError("Invalid node range detected")

        # Leaf coverage: leaf ranges must partition [0, n) into contiguous
        # disjoint segments.
        leaf_mask = self.node_is_leaf
        if leaf_mask.shape[0] != m:
            raise ValueError("node_is_leaf length must equal n_nodes")

        leaf_ranges = self.node_ranges[leaf_mask]
        if n == 0:
            if leaf_ranges.numel() != 0:
                raise ValueError("Non-empty leaf ranges for empty tree")
        else:
            if leaf_ranges.shape[0] == 0:
                raise ValueError("Tree with points must have at least one leaf")
            l_starts = leaf_ranges[:, 0]
            l_ends = leaf_ranges[:, 1]
            order = torch.argsort(l_starts)
            l_starts = l_starts[order]
            l_ends = l_ends[order]
            if int(l_starts[0].item()) != 0 or int(l_ends[-1].item()) != n:
                raise ValueError("Leaf ranges do not cover full [0, N) interval")
            if (l_starts[1:] != l_ends[:-1]).any():
                raise ValueError("Leaf ranges are not contiguous / disjoint")

        # Parent / child topology and level consistency.
        parents = self.node_parents
        children = self.node_children
        levels = self.node_levels

        if parents.shape[0] != m or children.shape[0] != m or levels.shape[0] != m:
            raise ValueError("Topology arrays must have length n_nodes")

        if int(parents[0].item()) != -1:
            raise ValueError("Root node must have parent = -1")

        for i in range(m):
            p = int(parents[i].item())
            if p != -1:
                if p < 0 or p >= m:
                    raise ValueError(f"Invalid parent index {p} for node {i}")
                if not (children[p] == i).any():
                    raise ValueError(f"Node {i} is not listed among children of its parent {p}")

            child_row = children[i]
            has_child = False
            child_indices: List[int] = []
            for c in child_row:
                c_int = int(c.item())
                if c_int < 0:
                    continue
                has_child = True
                child_indices.append(c_int)
                if c_int >= m:
                    raise ValueError(f"Invalid child index {c_int} for node {i}")
                if int(parents[c_int].item()) != i:
                    raise ValueError(f"Parent mismatch: node {c_int} does not point back to {i}")
                if int(levels[c_int].item()) != int(levels[i].item()) + 1:
                    raise ValueError("Child level must be parent level + 1")

            is_leaf = bool(self.node_is_leaf[i].item())
            if is_leaf and has_child:
                raise ValueError(f"Leaf node {i} has children")
            if not is_leaf and not has_child:
                raise ValueError(f"Internal node {i} has no children")

            # Ensure children cover the node's [start, end) range exactly.
            if not is_leaf:
                parent_start = int(self.node_ranges[i, 0].item())
                parent_end = int(self.node_ranges[i, 1].item())
                if parent_end <= parent_start:
                    raise ValueError(f"Internal node {i} has empty range")

                if len(child_indices) == 0:
                    raise ValueError(f"Internal node {i} has no children to cover its range")

                child_ranges: List[Tuple[int, int]] = []
                for c_idx in child_indices:
                    c_start = int(self.node_ranges[c_idx, 0].item())
                    c_end = int(self.node_ranges[c_idx, 1].item())
                    if c_end <= c_start:
                        raise ValueError(
                            f"Child node {c_idx} of parent {i} has empty or "
                            "invalid range"
                        )
                    child_ranges.append((c_start, c_end))

                # Sort by start index to check contiguity and coverage.
                child_ranges.sort(key=lambda r: r[0])

                if child_ranges[0][0] != parent_start or child_ranges[-1][1] != parent_end:
                    raise ValueError(
                        f"Children of node {i} do not cover parent range: "
                        f"parent=({parent_start}, {parent_end}), "
                        f"children[0].start={child_ranges[0][0]}, "
                        f"children[-1].end={child_ranges[-1][1]}"
                    )

                for (prev_s, prev_e), (curr_s, curr_e) in zip(
                    child_ranges, child_ranges[1:]
                ):
                    if curr_s != prev_e:
                        raise ValueError(
                            f"Children of node {i} do not form a contiguous "
                            f"partition: got ...({prev_s}, {prev_e}) then "
                            f"({curr_s}, {curr_e})"
                        )

        # Bounding box consistency check (optional).
        if check_bboxes and n > 0:
            with torch.no_grad():
                for i in range(m):
                    s = int(starts[i].item())
                    e = int(ends[i].item())
                    if e <= s:
                        continue
                    pts = self.points[s:e]
                    mins = pts.min(dim=0).values
                    maxs = pts.max(dim=0).values
                    center = 0.5 * (mins + maxs)
                    half_extent = torch.clamp(0.5 * (maxs - mins), min=0.0)
                    if not torch.allclose(center, self.node_centers[i], atol=atol, rtol=rtol):
                        raise ValueError(f"Center mismatch for node {i}")
                    if not torch.allclose(
                        half_extent, self.node_half_extents[i], atol=atol, rtol=rtol
                    ):
                        raise ValueError(f"Half-extent mismatch for node {i}")


# ---------------------------------------------------------------------------
# Internal helper: adaptive octree construction
# ---------------------------------------------------------------------------


def _build_adaptive_octree(
    points: Tensor,
    leaf_size: int,
    max_depth: int,
) -> Tuple[Tensor, List[TreeNode], BoundingBox]:
    """Build an adaptive octree over a point cloud.

    Parameters
    ----------
    points:
        ``(N, 3)`` tensor of point coordinates (floating point).
    leaf_size:
        Maximum number of points per leaf node.
    max_depth:
        Maximum allowed tree depth (root is level 0). This controls the
        maximum recursion depth of the Python builder; for deeper trees a
        non-recursive C++/CUDA implementation is recommended.

    Returns
    -------
    index_map:
        ``(N,)`` tensor of indices such that ``points[index_map]`` is the
        reordered point array in tree order.
    nodes:
        List of :class:`TreeNode` objects.
    global_bbox:
        :class:`BoundingBox` enclosing all points. This value is not used
        for splitting; it is metadata and a canonical frame for other
        geometry helpers (e.g. Morton keys, domain decomposition).

    Notes
    -----
    The implementation is written in terms of an index array rather
    than physically moving the point tensor at each subdivision. This
    keeps the algorithm :math:`\\mathcal{O}(N \\log N)` in the worst
    case, and exposes a clear path to a future CUDA / C++ re-write.
    """
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {tuple(points.shape)}")
    if not torch.is_floating_point(points):
        raise TypeError(f"points must be floating point, got {points.dtype}")

    N = int(points.shape[0])
    device = points.device
    dtype = points.dtype

    if N == 0:
        # Degenerate empty tree.
        zeros = torch.zeros(3, device=device, dtype=dtype)
        bbox = BoundingBox(center=zeros, half_extent=zeros)
        return torch.empty(0, device=device, dtype=torch.int64), [], bbox

    # Initial global bounding box (tight). Used as metadata and as a
    # canonical frame for other geometry routines; subdivision itself
    # works with per-node bounding boxes.
    mins, maxs = compute_bounding_box(points)
    global_bbox = BoundingBox.from_min_max(mins, maxs)

    indices = torch.arange(N, device=device, dtype=torch.int64)
    nodes: List[TreeNode] = []
    # Scratch buffer reused by all recursive calls; each node operates on a
    # slice ``scratch[start:end]`` of this tensor.
    scratch = torch.empty(N, device=device, dtype=torch.int64)

    # For now we rely on Python recursion; the depth is bounded by max_depth.
    # A production C++/CUDA implementation should replace this by an explicit
    # stack-based traversal.
    def build_node(start: int, end: int, level: int, parent: Optional[int]) -> int:
        """Recursive helper to build a node in the tree."""
        local_idx = indices[start:end]
        local_points = points[local_idx]

        # Compute tight bounding box for this node.
        mins, maxs = compute_bounding_box(local_points)
        center = 0.5 * (mins + maxs)
        half_extent = torch.clamp(0.5 * (maxs - mins), min=0.0)

        node_index = len(nodes)
        node = TreeNode(
            index=node_index,
            parent=parent,
            level=level,
            center=center,
            half_extent=half_extent,
            start=start,
            end=end,
            children=(None,) * 8,
        )
        nodes.append(node)

        n_points = end - start

        # Determine if the box is physically too small to split (degenerate geometry).
        # This prevents infinite recursion on coincident points.
        is_degenerate = float(half_extent.max().item()) < 1e-7

        if n_points <= leaf_size or level >= max_depth or is_degenerate:
            # Leaf node: no further subdivision. Emit a warning if we are
            # forced to stop due to max_depth while still above leaf_size
            # (and the node is not degenerate).
            if level >= max_depth and n_points > leaf_size and not is_degenerate:
                warnings.warn(
                    f"max_depth={max_depth} reached at level={level} with "
                    f"{n_points} points (> leaf_size={leaf_size}); "
                    "consider revisiting tree parameters or discretisation.",
                    RuntimeWarning,
                )
            return node_index

        # Subdivide: classify points into 8 octants relative to the center.
        # NOTE: we use strict ">" so points lying exactly on a splitting
        # plane go to the "lower" child along that axis, which makes
        # splitting deterministic.
        rel = local_points > center  # (n_points, 3) bool
        octant_ids = (
            rel[:, 0].to(torch.int64)
            + 2 * rel[:, 1].to(torch.int64)
            + 4 * rel[:, 2].to(torch.int64)
        )  # (n_points,)

        # Stable partition of indices by octant (counting sort).
        scratch_local = scratch[start:end]
        child_starts_local = torch.empty(8, device=device, dtype=torch.int64)
        child_ends_local = torch.empty(8, device=device, dtype=torch.int64)

        offset = 0
        for octant in range(8):
            mask = octant_ids == octant
            count = int(mask.sum().item())
            start_q = offset
            end_q = offset + count
            if count > 0:
                scratch_local[start_q:end_q] = local_idx[mask]
            child_starts_local[octant] = start_q
            child_ends_local[octant] = end_q
            offset = end_q

        # Write reordered indices back into the global index array.
        indices[start:end] = scratch_local

        # Recursively build children.
        children: List[Optional[int]] = [None] * 8
        for octant in range(8):
            s_local = int(child_starts_local[octant].item())
            e_local = int(child_ends_local[octant].item())
            if e_local <= s_local:
                continue
            s_global = start + s_local
            e_global = start + e_local
            child_index = build_node(s_global, e_global, level + 1, node_index)
            children[octant] = child_index

        # Update children tuple in the stored node.
        nodes[node_index].children = tuple(children)
        return node_index

    with torch.no_grad():
        build_node(0, N, level=0, parent=None)

    return indices, nodes, global_bbox


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------


def build_fmm_tree(
    points: Tensor,
    *,
    leaf_size: int = 64,
    max_depth: int = 32,
) -> FmmTree:
    """Construct an :class:`FmmTree` from a point cloud.

    This is the main entry point used by higher-level BEM drivers. It
    returns a tree where:

    - Each node corresponds to an axis-aligned bounding box.
    - Leaf nodes contain at most ``leaf_size`` points (unless ``max_depth``
      is reached first, in which case a warning is emitted).
    - Point data are stored in contiguous ranges for each node, which
    is essential for SoA-based CPU/GPU kernels.

    Parameters
    ----------
    points:
        Tensor of shape ``(N, 3)`` with point coordinates (floating
        point). **Tree construction must run on the CPU**; if you have
        CUDA points, move them to CPU first (``points.cpu()``) and then
        move the resulting tree to GPU via :meth:`FmmTree.to`.
    leaf_size:
        Maximum number of points per leaf. This is the primary control
        over the tree branching factor and depth. Typical values are
        in the range 32–256.
    max_depth:
        Maximum allowed depth (root is level 0). This prevents
        pathological recursion when the geometry is extremely
        clustered. A conservative default of 32 is used, which is more
        than enough for practical point counts (since ``8**32`` is
        astronomically large). For much deeper trees a non-recursive
        builder (C++/CUDA) is recommended instead of increasing this.

    Returns
    -------
    FmmTree
        The constructed tree with SoA geometry data.

    Notes
    -----
    - If ``points`` is on a CUDA device, this function will raise
      :class:`RuntimeError`. Build on CPU and call ``tree.to('cuda')``
      afterwards.
    - Extremely clustered or coincident points may push the deepest
      nodes towards ``max_depth``. If you observe warnings about
      ``max_depth`` being reached with ``n_points > leaf_size``, consider
      increasing ``leaf_size`` or revisiting the discretisation.
    """
    if max_depth <= 0:
        raise ValueError(f"max_depth must be positive, got {max_depth}")
    # Sanity guard for the recursive Python builder. For very deep trees
    # a non-recursive C++/CUDA implementation should be used instead.
    if max_depth > 64:
        raise ValueError(
            f"max_depth={max_depth} is too large for the recursive Python "
            "builder; use a non-recursive C++/CUDA implementation for such "
            "deep trees."
        )
    if leaf_size <= 0:
        raise ValueError(f"leaf_size must be positive, got {leaf_size}")

    # Explicit CPU-only constraint for the Python builder.
    if points.device.type != "cpu":
        raise RuntimeError(
            "Tree building must run on CPU. "
            "Call build_fmm_tree(points.cpu()) and then use tree.to('cuda')."
        )

    index_map, nodes, global_bbox = _build_adaptive_octree(
        points, leaf_size=leaf_size, max_depth=max_depth
    )
    points_reordered = points[index_map] if index_map.numel() > 0 else points
    return FmmTree(
        points=points_reordered,
        index_map=index_map,
        nodes=nodes,
        leaf_size=leaf_size,
        max_depth=max_depth,
        global_bbox=global_bbox,
    )
