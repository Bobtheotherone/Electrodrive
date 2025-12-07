"""Geometry utilities used by the FMM layer.

This module is Phase 1 of the Electrodrive FMM stack: it provides all
geometry primitives that the higher-level tree, interaction-list, and
multipole code depend on.

Design goals
------------
- Pure PyTorch implementation (CPU / CUDA).
- Shape- and dtype-stable APIs for later C++/CUDA kernels.
- Numerically robust for highly non-uniform BEM meshes.
- Clear mathematical conventions so this file doubles as an executable spec.

Layers
------
1) Low-level helpers
   - Bounding boxes for point clouds.
   - Diameters / radii / extents for tree nodes.
   - Basic distance and separation estimates.

2) Structured geometry objects
   - BoundingBox: single axis-aligned box (center + half_extent).
   - BoundingBoxArray: SoA representation suitable for kernels.

3) Locality / ordering helpers
   - Morton (Z-order) keys.
   - Utilities to reorder data by Morton order.

MAC conventions
---------------
This module defines the canonical geometric multipole acceptance criteria
(MAC) used elsewhere:

- node_separation: distance / (r_i + r_j)
- mac_separation_ratio: sep >= 1/theta
- mac_max_d_over_dist: max(d_i, d_j) / distance < theta

Higher-level modules should express MAC logic in terms of these helpers.

Dtype conventions
-----------------
Unless stated otherwise, geometry routines expect float32 or float64
tensors. Half-precision (float16, bfloat16) and integer dtypes are
deliberately rejected for core routines to avoid FMM instability.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

__all__ = [
    "BoundingBox",
    "BoundingBoxArray",
    "compute_bounding_box",
    "compute_node_diameter",
    "compute_node_diameters",
    "compute_bounding_boxes_for_ranges",
    "node_separation",
    "mac_separation_ratio",
    "mac_max_d_over_dist",
    "morton_keys_from_points",
    "reorder_by_morton_keys",
]

# ---------------------------------------------------------------------------
# Debug / validation helpers
# ---------------------------------------------------------------------------

_DEBUG = os.getenv("ELECTRODRIVE_DEBUG_ASSERTS", "0") not in ("0", "", "false", "False")


def _assert_finite(x: Tensor, name: str) -> None:
    """Debug-only finite check for geometry tensors.

    Enabled when the environment variable ELECTRODRIVE_DEBUG_ASSERTS
    is set to a truthy value. In release runs this is a no-op.
    """
    if not _DEBUG:
        return
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains non-finite values")


# ---------------------------------------------------------------------------
# Small internal helpers (device/dtype consistency)
# ---------------------------------------------------------------------------

_ALLOWED_FLOAT_DTYPES = (torch.float32, torch.float64)


def _ensure_fp32_or_fp64(x: Tensor, name: str) -> None:
    """Ensure tensor has dtype float32 or float64.

    Many FMM kernels rely on at least single-precision; half-precision
    is rejected here to avoid subtle instabilities.
    """
    if not torch.is_floating_point(x):
        raise TypeError(f"{name} must be floating point, got dtype {x.dtype}")
    if x.dtype not in _ALLOWED_FLOAT_DTYPES:
        raise TypeError(f"{name} must have dtype float32 or float64, got {x.dtype}")


def _zeros3(device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return a length-3 zero vector on the given device / dtype."""
    return torch.zeros(3, device=device, dtype=dtype)


def _empty_bbox_array(device: torch.device, dtype: torch.dtype) -> "BoundingBoxArray":
    """Return an empty BoundingBoxArray on the given device / dtype."""
    empty = torch.empty(0, 3, device=device, dtype=dtype)
    radii = torch.empty(0, device=device, dtype=dtype)
    return BoundingBoxArray(centers=empty, half_extents=empty, radii=radii)


# ---------------------------------------------------------------------------
# Basic bounding-box helpers
# ---------------------------------------------------------------------------


def compute_bounding_box(points: Tensor) -> Tuple[Tensor, Tensor]:
    """Return global (min_xyz, max_xyz) for a set of points.

    Parameters
    ----------
    points:
        Tensor of shape (N, 3). Assumed finite (no NaNs or Infs) in
        production; in debug mode a finite check is performed.

    Returns
    -------
    (mins, maxs):
        Tensors of shape (3,) giving the axis-aligned bounding box.

    Empty input
    -----------
    If N == 0, returns a degenerate box at the origin (mins = maxs = 0).
    """
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {tuple(points.shape)}")
    _ensure_fp32_or_fp64(points, "points")

    _assert_finite(points, "points")

    if points.shape[0] == 0:
        device = points.device
        dtype = points.dtype
        zeros = _zeros3(device, dtype)
        return zeros.clone(), zeros.clone()

    mins = points.min(dim=0).values
    maxs = points.max(dim=0).values
    return mins, maxs


def compute_node_diameter(half_extent: Tensor) -> Tensor:
    """Compute a scalar diameter from a half-extent vector.

    Parameters
    ----------
    half_extent:
        Tensor of shape (3,) with half side lengths of an axis-aligned box.

    Returns
    -------
    Tensor
        Scalar diameter of the smallest sphere that contains the box:
        2 * ||half_extent||_2.
    """
    if half_extent.shape != (3,):
        raise ValueError(
            f"half_extent must have shape (3,), got {tuple(half_extent.shape)}"
        )
    _ensure_fp32_or_fp64(half_extent, "half_extent")
    return 2.0 * half_extent.norm()


def compute_node_diameters(half_extents: Tensor) -> Tensor:
    """Batched version of compute_node_diameter.

    Parameters
    ----------
    half_extents:
        Tensor of shape (M, 3) with half side lengths for each box.

    Returns
    -------
    diameters:
        Tensor of shape (M,) with diameters 2 * ||half_extent||_2.
    """
    if half_extents.ndim != 2 or half_extents.shape[-1] != 3:
        raise ValueError(
            f"half_extents must have shape (M, 3), got {tuple(half_extents.shape)}"
        )
    _ensure_fp32_or_fp64(half_extents, "half_extents")
    return 2.0 * torch.linalg.norm(half_extents, dim=-1)


# ---------------------------------------------------------------------------
# Structured bounding-box representations
# ---------------------------------------------------------------------------


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in 3D.

    Canonical representation: (center, half_extent) instead of (mins, maxs).

    Attributes
    ----------
    center:
        Tensor of shape (3,) with the box center.
    half_extent:
        Tensor of shape (3,) with half side lengths (non-negative).
    """

    center: Tensor
    half_extent: Tensor

    def __post_init__(self) -> None:
        if self.center.shape != (3,):
            raise ValueError(
                f"BoundingBox.center must have shape (3,), got {tuple(self.center.shape)}"
            )
        if self.half_extent.shape != (3,):
            raise ValueError(
                "BoundingBox.half_extent must have shape (3,), "
                f"got {tuple(self.half_extent.shape)}"
            )
        _ensure_fp32_or_fp64(self.center, "BoundingBox.center")
        _ensure_fp32_or_fp64(self.half_extent, "BoundingBox.half_extent")

    # ------------ construction helpers ------------

    @classmethod
    def from_min_max(cls, mins: Tensor, maxs: Tensor) -> "BoundingBox":
        """Construct from (mins, maxs) vectors of shape (3,)."""
        if mins.shape != (3,) or maxs.shape != (3,):
            raise ValueError(
                f"mins/maxs must have shape (3,), got {tuple(mins.shape)}, {tuple(maxs.shape)}"
            )
        _ensure_fp32_or_fp64(mins, "mins")
        _ensure_fp32_or_fp64(maxs, "maxs")
        center = 0.5 * (mins + maxs)
        half_extent = 0.5 * (maxs - mins)
        half_extent = torch.clamp(half_extent, min=0.0)  # clamp small negatives
        return cls(center=center, half_extent=half_extent)

    @classmethod
    def from_points(cls, points: Tensor) -> "BoundingBox":
        """Build the tight box enclosing a set of points.

        If points is empty, constructs a degenerate box at the origin.
        """
        mins, maxs = compute_bounding_box(points)
        return cls.from_min_max(mins, maxs)

    # ------------ basic properties ------------

    @property
    def device(self) -> torch.device:
        return self.center.device

    @property
    def dtype(self) -> torch.dtype:
        return self.center.dtype

    @property
    def mins(self) -> Tensor:
        """Return (3,) vector of minimum coordinates."""
        return self.center - self.half_extent

    @property
    def maxs(self) -> Tensor:
        """Return (3,) vector of maximum coordinates."""
        return self.center + self.half_extent

    @property
    def extents(self) -> Tensor:
        """Return full side lengths along each axis."""
        return 2.0 * self.half_extent

    @property
    def radius(self) -> Tensor:
        """Return radius of the circumscribed sphere."""
        return self.half_extent.norm()

    @property
    def diameter(self) -> Tensor:
        """Return diameter of the circumscribed sphere."""
        return 2.0 * self.radius

    @property
    def volume(self) -> Tensor:
        """Return volume of the box."""
        return (2.0 * self.half_extent).prod()

    # ------------ geometric relations ------------

    def expanded(self, rel: float = 0.0, abs_: float = 0.0) -> "BoundingBox":
        """Return an expanded version of the box.

        rel:
            Relative padding factor (e.g. 0.01 → 1% padding).
        abs_:
            Absolute padding in geometry units.
        """
        padding = self.half_extent * float(rel) + float(abs_)
        padding = torch.as_tensor(padding, device=self.device, dtype=self.dtype)
        return BoundingBox(center=self.center, half_extent=self.half_extent + padding)

    def union(self, other: "BoundingBox") -> "BoundingBox":
        """Return the minimal box containing both self and other."""
        mins = torch.minimum(self.mins, other.mins)
        maxs = torch.maximum(self.maxs, other.maxs)
        return BoundingBox.from_min_max(mins, maxs)

    def contains_points(self, points: Tensor) -> Tensor:
        """Return a boolean mask indicating which points are inside the box.

        Parameters
        ----------
        points:
            Tensor of shape (..., 3).

        Returns
        -------
        mask:
            Boolean tensor of shape (...,).
        """
        if points.shape[-1] != 3:
            raise ValueError(f"points must have last dimension 3, got {tuple(points.shape)}")
        mins = self.mins
        maxs = self.maxs
        ge_min = points >= mins
        le_max = points <= maxs
        mask = ge_min & le_max
        return mask.all(dim=-1)

    def distance_to(self, other: "BoundingBox") -> Tensor:
        """Return Euclidean distance between centers of two boxes."""
        return torch.linalg.norm(self.center - other.center)

    def separation_ratio(self, other: "BoundingBox") -> Tensor:
        """Return distance(center_i, center_j) / (radius_i + radius_j)."""
        dist = self.distance_to(other)
        r_sum = self.radius + other.radius
        r_sum = torch.clamp(r_sum, min=torch.finfo(dist.dtype).eps)
        return dist / r_sum


@dataclass
class BoundingBoxArray:
    """SoA representation of many bounding boxes.

    Attributes
    ----------
    centers:
        (M, 3) tensor of box centers.
    half_extents:
        (M, 3) tensor of half side lengths.
    radii:
        (M,) tensor of circumscribed-sphere radii.
    """

    centers: Tensor  # (M, 3)
    half_extents: Tensor  # (M, 3)
    radii: Tensor  # (M,)

    def __post_init__(self) -> None:
        if self.centers.ndim != 2 or self.centers.shape[-1] != 3:
            raise ValueError(f"centers must have shape (M, 3), got {tuple(self.centers.shape)}")
        if self.half_extents.shape != self.centers.shape:
            raise ValueError(
                "half_extents must have same shape as centers, "
                f"got {tuple(self.half_extents.shape)}"
            )
        if self.radii.shape != (self.centers.shape[0],):
            raise ValueError(
                f"radii must have shape (M,), got {tuple(self.radii.shape)} "
                f"for M={self.centers.shape[0]}"
            )
        _ensure_fp32_or_fp64(self.centers, "BoundingBoxArray.centers")
        _ensure_fp32_or_fp64(self.half_extents, "BoundingBoxArray.half_extents")
        _ensure_fp32_or_fp64(self.radii, "BoundingBoxArray.radii")

    # ------------ construction helpers ------------

    @classmethod
    def from_min_max(cls, mins: Tensor, maxs: Tensor) -> "BoundingBoxArray":
        """Construct from batched mins/maxs arrays of shape (M, 3)."""
        if mins.shape != maxs.shape or mins.ndim != 2 or mins.shape[-1] != 3:
            raise ValueError(
                "mins/maxs must have shape (M, 3) and match, "
                f"got {tuple(mins.shape)}, {tuple(maxs.shape)}"
            )
        _ensure_fp32_or_fp64(mins, "mins")
        _ensure_fp32_or_fp64(maxs, "maxs")
        centers = 0.5 * (mins + maxs)
        half_extents = 0.5 * (maxs - mins)
        half_extents = torch.clamp(half_extents, min=0.0)
        radii = torch.linalg.norm(half_extents, dim=-1)
        return cls(centers=centers, half_extents=half_extents, radii=radii)

    @classmethod
    def from_boxes(cls, boxes: Sequence[BoundingBox]) -> "BoundingBoxArray":
        """Construct from a sequence of BoundingBox objects.

        All boxes are cast to the device/dtype of boxes[0]. In debug
        mode we assert that they already agree to catch accidental
        mixing early.
        """
        if len(boxes) == 0:
            # Empty CPU fallback; callers can move to GPU if needed.
            device = torch.device("cpu")
            dtype = torch.float32
            return _empty_bbox_array(device, dtype)

        device = boxes[0].device
        dtype = boxes[0].dtype

        if _DEBUG:
            for i, b in enumerate(boxes[1:], start=1):
                if b.device != device or b.dtype != dtype:
                    raise ValueError(
                        f"BoundingBoxArray.from_boxes: box[{i}] has device/dtype "
                        f"{b.device}/{b.dtype} but expected {device}/{dtype}"
                    )

        centers = torch.stack([b.center.to(device=device, dtype=dtype) for b in boxes], dim=0)
        half_extents = torch.stack(
            [b.half_extent.to(device=device, dtype=dtype) for b in boxes], dim=0
        )
        radii = torch.stack([b.radius.to(device=device, dtype=dtype) for b in boxes], dim=0)
        return cls(centers=centers, half_extents=half_extents, radii=radii)

    # ------------ derived quantities ------------

    @property
    def device(self) -> torch.device:
        return self.centers.device

    @property
    def dtype(self) -> torch.dtype:
        return self.centers.dtype

    @property
    def extents(self) -> Tensor:
        """Full side lengths for each box, shape (M, 3)."""
        return 2.0 * self.half_extents

    @property
    def diameters(self) -> Tensor:
        """Diameters for each box, shape (M,)."""
        return 2.0 * self.radii

    def to_min_max(self) -> Tuple[Tensor, Tensor]:
        """Return batched (mins, maxs) of shape (M, 3)."""
        mins = self.centers - self.half_extents
        maxs = self.centers + self.half_extents
        return mins, maxs

    # ------------ selection / slicing ------------

    def select(self, indices: Tensor) -> "BoundingBoxArray":
        """Return a new BoundingBoxArray with a subset of boxes.

        indices:
            1D integer tensor of indices along the first dimension.
        """
        centers = self.centers[indices]
        half_extents = self.half_extents[indices]
        radii = self.radii[indices]
        return BoundingBoxArray(centers=centers, half_extents=half_extents, radii=radii)

    # ------------ device / dtype helpers ------------

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "BoundingBoxArray":
        """Return a copy moved to device/dtype (like Tensor.to)."""
        centers = self.centers.to(device=device, dtype=dtype)
        half_extents = self.half_extents.to(device=device, dtype=dtype)
        radii = self.radii.to(device=device, dtype=dtype)
        return BoundingBoxArray(centers=centers, half_extents=half_extents, radii=radii)

    def clone(self) -> "BoundingBoxArray":
        """Return a deep copy of this BoundingBoxArray."""
        return BoundingBoxArray(
            centers=self.centers.clone(),
            half_extents=self.half_extents.clone(),
            radii=self.radii.clone(),
        )


# ---------------------------------------------------------------------------
# Batched bounding-box construction for tree nodes
# ---------------------------------------------------------------------------


def compute_bounding_boxes_for_ranges(points: Tensor, ranges: Tensor) -> BoundingBoxArray:
    """Compute bounding boxes for many contiguous index ranges (CPU-only).

    Parameters
    ----------
    points:
        Tensor of shape (N, 3), typically BEM panel centroids or
        source/target points permuted into Morton order. Must reside
        on CPU and have dtype float32 or float64.
    ranges:
        Integer tensor of shape (M, 2); each row is [start, end)
        indices into points with 0 <= start < end <= N.

    Returns
    -------
    BoundingBoxArray
        SoA bounding-box representation for each range.

    Notes
    -----
    This is the bridge between logical tree layout (index ranges) and
    physical geometry storage (SoA arrays) used by kernels.

    Implementation
    --------------
    This is a reference CPU implementation with a Python loop over
    ranges. For Phase 3 it should be replaced by a fused C++/CUDA
    kernel that computes mins/maxs for all ranges in one pass.
    """
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {tuple(points.shape)}")
    if ranges.ndim != 2 or ranges.shape[-1] != 2:
        raise ValueError(f"ranges must have shape (M, 2), got {tuple(ranges.shape)}")

    # Explicitly CPU-only for now: the Python loop and .item() calls
    # would otherwise hard-sync a CUDA device and destroy performance.
    if points.device.type != "cpu":
        raise RuntimeError(
            "compute_bounding_boxes_for_ranges: CPU-only reference implementation."
        )

    _ensure_fp32_or_fp64(points, "points")
    _assert_finite(points, "points")

    if points.shape[0] == 0 or ranges.shape[0] == 0:
        return _empty_bbox_array(points.device, points.dtype)

    device = points.device
    dtype = points.dtype
    ranges = ranges.to(device=device)
    starts = ranges[:, 0].to(torch.int64)
    ends = ranges[:, 1].to(torch.int64)

    centers = torch.empty((ranges.shape[0], 3), device=device, dtype=dtype)
    half_extents = torch.empty_like(centers)

    N = points.shape[0]

    # TODO(optimization): This function has a Python loop over all ranges.
    # In Phase 3 replace this with a fused C++/CUDA kernel that computes
    # mins/maxs for all ranges in a single pass.
    for i in range(ranges.shape[0]):
        s = int(starts[i].item())
        e = int(ends[i].item())
        if not (0 <= s < e <= N):
            raise ValueError(f"Invalid range [{s}, {e}) for N={N}")
        pts = points[s:e]
        mins = pts.min(dim=0).values
        maxs = pts.max(dim=0).values
        centers[i] = 0.5 * (mins + maxs)
        half_extents[i] = 0.5 * (maxs - mins)

    half_extents = torch.clamp(half_extents, min=0.0)
    radii = torch.linalg.norm(half_extents, dim=-1)
    return BoundingBoxArray(centers=centers, half_extents=half_extents, radii=radii)


# ---------------------------------------------------------------------------
# Node separation and MAC helpers
# ---------------------------------------------------------------------------


def node_separation(
    centers_i: Tensor,
    radii_i: Tensor,
    centers_j: Tensor,
    radii_j: Tensor,
) -> Tensor:
    """Compute center-to-center separation in units of sum of radii.

    separation = distance(center_i, center_j) / (r_i + r_j)
    """
    if centers_i.shape[-1] != 3 or centers_j.shape[-1] != 3:
        raise ValueError(
            "centers_i/centers_j must have last dimension 3, "
            f"got {tuple(centers_i.shape)}, {tuple(centers_j.shape)}"
        )
    _ensure_fp32_or_fp64(centers_i, "centers_i")
    _ensure_fp32_or_fp64(centers_j, "centers_j")
    _ensure_fp32_or_fp64(radii_i, "radii_i")
    _ensure_fp32_or_fp64(radii_j, "radii_j")

    diff = centers_i - centers_j  # supports broadcasting
    dist = torch.linalg.norm(diff, dim=-1)
    r_sum = radii_i + radii_j
    r_sum = torch.clamp(r_sum, min=torch.finfo(dist.dtype).eps)
    return dist / r_sum


def mac_separation_ratio(
    centers_i: Tensor,
    radii_i: Tensor,
    centers_j: Tensor,
    radii_j: Tensor,
    theta: float,
) -> Tensor:
    """Symmetric MAC: sep >= 1/theta with sep = dist / (r_i + r_j)."""
    if not (0.0 < theta < 1.0):
        raise ValueError(f"theta must be in (0, 1), got {theta}")
    sep = node_separation(centers_i, radii_i, centers_j, radii_j)
    threshold = 1.0 / float(theta)
    return sep >= threshold


def mac_max_d_over_dist(
    centers_i: Tensor,
    diam_i: Tensor,
    centers_j: Tensor,
    diam_j: Tensor,
    theta: float,
) -> Tensor:
    """Classical FMM MAC: max(d_i, d_j) / dist < theta.

    Returns a boolean tensor where True means "admissible for far-field".
    """
    if theta <= 0.0:
        raise ValueError(f"theta must be positive, got {theta}")
    if centers_i.shape[-1] != 3 or centers_j.shape[-1] != 3:
        raise ValueError(
            "centers_i/centers_j must have last dimension 3, "
            f"got {tuple(centers_i.shape)}, {tuple(centers_j.shape)}"
        )

    _ensure_fp32_or_fp64(centers_i, "centers_i")
    _ensure_fp32_or_fp64(centers_j, "centers_j")
    _ensure_fp32_or_fp64(diam_i, "diam_i")
    _ensure_fp32_or_fp64(diam_j, "diam_j")

    diff = centers_i - centers_j
    dist = torch.linalg.norm(diff, dim=-1)
    eps = torch.finfo(dist.dtype).eps
    dist = torch.clamp(dist, min=eps)

    diam_max = torch.maximum(diam_i, diam_j)
    ratio = diam_max / dist
    return ratio < float(theta)


# ---------------------------------------------------------------------------
# Morton keys and locality-preserving reordering
# ---------------------------------------------------------------------------


def _compute_global_bbox(points: Tensor) -> Tuple[Tensor, Tensor]:
    """Internal helper that returns a non-degenerate bounding box.

    If the point cloud has zero extent along an axis, we pad that axis
    slightly so that normalized coordinates are well-defined.
    """
    _ensure_fp32_or_fp64(points, "points")
    _assert_finite(points, "points")

    mins, maxs = compute_bounding_box(points)
    extent = maxs - mins

    # Use a small relative + absolute padding for degenerate axes.
    # This scales with the cloud size but remains above machine noise
    # for very small geometries.
    eps_rel = points.new_tensor(1e-12)
    max_extent = torch.max(extent.abs())
    padded = eps_rel * (1.0 + max_extent)
    extent = torch.where(extent > 0, extent, padded)

    return mins, mins + extent


def _normalize_points_to_unit_cube(points: Tensor, mins: Tensor, maxs: Tensor) -> Tensor:
    """Map points from [mins, maxs] to [0, 1)^3."""
    _ensure_fp32_or_fp64(points, "points")
    extent = maxs - mins
    eps = torch.finfo(points.dtype).eps
    extent = torch.clamp(extent, min=eps)
    normalized = (points - mins) / extent
    # Numerical safety: clamp exactly to [0, 1).
    return torch.clamp(normalized, 0.0, 1.0 - 1e-9)


def _morton_interleave(ix: Tensor, iy: Tensor, iz: Tensor, level: int) -> Tensor:
    """Interleave 3D integer coordinates into Morton (Z-order) keys.

    ix, iy, iz:
        Integer tensors with values in [0, 2**level).
    level:
        Number of bits per coordinate; for 64-bit keys use level <= 21.
    """
    if level <= 0 or level > 21:
        raise ValueError(f"level must be in [1, 21], got {level}")

    ix = ix.to(torch.int64)
    iy = iy.to(torch.int64)
    iz = iz.to(torch.int64)
    keys = torch.zeros_like(ix, dtype=torch.int64)

    # Only Python loop is over bits; level is capped at 21.
    for bit in range(level):
        shift = bit
        mask = 1 << shift
        bx = (ix & mask) >> shift
        by = (iy & mask) >> shift
        bz = (iz & mask) >> shift

        keys |= bx << (3 * bit)
        keys |= by << (3 * bit + 1)
        keys |= bz << (3 * bit + 2)

    return keys


def morton_keys_from_points(
    points: Tensor,
    *,
    level: int = 16,
    global_bbox: Optional[BoundingBox] = None,
) -> Tuple[Tensor, BoundingBox]:
    """Compute Morton (Z-order) keys for a point cloud.

    Parameters
    ----------
    points:
        Tensor of shape (N, 3), floating point, finite.
    level:
        Morton resolution; 2**level buckets per axis. For octree depth d,
        a common choice is level ≈ d.
    global_bbox:
        Optional BoundingBox specifying the normalization frame.

    Returns
    -------
    keys:
        (N,) int64 tensor of Morton keys.
    bbox:
        BoundingBox used for normalization (global_bbox if provided).

    Empty input
    -----------
    If N == 0:
      - returns keys of shape (0,)
      - returns global_bbox if provided, otherwise a degenerate box at origin.
    """
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {tuple(points.shape)}")

    _ensure_fp32_or_fp64(points, "points")
    _assert_finite(points, "points")

    if points.shape[0] == 0:
        device = points.device
        dtype = points.dtype
        empty_keys = torch.zeros(0, device=device, dtype=torch.int64)
        if global_bbox is None:
            zeros = _zeros3(device, dtype)
            global_bbox = BoundingBox(center=zeros, half_extent=zeros)
        return empty_keys, global_bbox

    if global_bbox is None:
        mins, maxs = _compute_global_bbox(points)
        global_bbox = BoundingBox.from_min_max(mins, maxs)
    else:
        mins, maxs = global_bbox.mins, global_bbox.maxs

    # Normalize into [0, 1)^3 and quantize.
    normalized = _normalize_points_to_unit_cube(points, mins, maxs)
    scale = float(1 << level)
    ijk = torch.floor(normalized * scale).to(torch.int64)
    max_index = (1 << level) - 1
    ijk = torch.clamp(ijk, 0, max_index)

    ix = ijk[:, 0]
    iy = ijk[:, 1]
    iz = ijk[:, 2]
    keys = _morton_interleave(ix, iy, iz, level=level)
    return keys, global_bbox


def reorder_by_morton_keys(
    keys: Tensor,
    *arrays: Tensor,
) -> Tuple[Tensor, ...]:
    """Reorder one or more arrays according to Morton keys.

    Parameters
    ----------
    keys:
        (N,) tensor of integer Morton keys.
    *arrays:
        Tensors whose first dimension is N. They are permuted using the
        same order as keys.

    Returns
    -------
    (order, *reordered_arrays):
        order is the permutation (torch.argsort(keys)), and each
        reordered array is arrays[i][order].
    """
    if keys.ndim != 1:
        raise ValueError(f"keys must be 1D, got shape {tuple(keys.shape)}")
    N = keys.shape[0]
    for arr in arrays:
        if arr.shape[0] != N:
            raise ValueError(
                f"All arrays must have first dimension {N}, got {tuple(arr.shape)}"
            )
    order = torch.argsort(keys)
    reordered = [order]
    for arr in arrays:
        reordered.append(arr[order])
    return tuple(reordered)
