"""
Quadrature helpers for single-layer Laplace BEM on triangular panels.

This module provides:

  - `standard_triangle_quadrature(vertices, order)`:
      Low-order Gaussian-like rules on a single physical triangle.
      Currently supports orders 1 and 2 and returns (points, weights)
      such that integrating a constant is exact:

          ∫_triangle f dS ≈ Σ_i w_i f(x_i),
          Σ_i w_i = area(triangle).

  - `reference_triangle_quadrature(order)`:
      Low-order Gaussian-style rules on the *reference* triangle with
      vertices (0, 0), (1, 0), (0, 1). This is primarily intended for
      GPU / CUDA bindings that want to reuse the same node/weight sets
      on device.

  - `near_singular_quadrature(target, panel_vertices, method="telles", order=2)`:
      A simple adaptive refinement rule for targets close to a panel.
      The current implementation repeatedly refines the subtriangle
      closest to the target and applies `standard_triangle_quadrature`
      on the refined patch family. This clusters nodes near the closest
      point and is sufficient to regularise near-singular behaviour for
      the Laplace single-layer kernel.

  - `self_integral_correction(area)`:
      Equal-area disk approximation for the self-panel integral used as
      a robust diagonal in the matrix-free operator.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from electrodrive.utils.config import K_E

__all__ = [
    "self_integral_correction",
    "standard_triangle_quadrature",
    "reference_triangle_quadrature",
    "near_singular_quadrature",
]


# ---------------------------------------------------------------------------
# Basic geometry helpers
# ---------------------------------------------------------------------------


def _as_triangle_vertices(vertices) -> np.ndarray:
    """
    Convert input to a (3, 3) float64 array of triangle vertices.
    """
    v = np.asarray(vertices, dtype=float)
    if v.shape != (3, 3):
        raise ValueError(f"vertices must have shape (3, 3), got {v.shape!r}")
    return v


def _triangle_area(verts: np.ndarray) -> float:
    """
    Area of a triangle given by three vertices in R^3.
    """
    e1 = verts[1] - verts[0]
    e2 = verts[2] - verts[0]
    return 0.5 * float(np.linalg.norm(np.cross(e1, e2)))


# ---------------------------------------------------------------------------
# Standard Gaussian-style rules on a physical triangle
# ---------------------------------------------------------------------------


def standard_triangle_quadrature(
    vertices,
    order: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian-style quadrature on a single physical triangle.

    Parameters
    ----------
    vertices : array_like, shape (3, 3)
        Triangle vertices in R^3.
    order : int, optional
        Polynomial order indicator (currently 1 or 2).

    Returns
    -------
    points : ndarray, shape (Q, 3)
        Quadrature points in physical coordinates.
    weights : ndarray, shape (Q,)
        Physical weights such that Σ_i weights[i] = area(vertices).

    Notes
    -----
    Implemented rules:

    - order <= 1: 1-point centroid rule (degree-1 exact).
    - order == 2: symmetric 3-point rule (degree-2 exact).

    Higher orders can be added later as needed.
    """
    verts = _as_triangle_vertices(vertices)

    if order <= 1:
        # 1-point centroid rule. We normalise weights so that Σw = 1,
        # then scale by the physical area.
        bary = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=float)
        w_ref = np.array([1.0], dtype=float)
    elif order == 2:
        # Classical degree-2 symmetric rule: 3 points at permutations
        # of (2/3, 1/6, 1/6), equal weights.
        a = 2.0 / 3.0
        b = 1.0 / 6.0
        bary = np.array(
            [
                [a, b, b],
                [b, a, b],
                [b, b, a],
            ],
            dtype=float,
        )
        w_ref = np.full(3, 1.0 / 3.0, dtype=float)
    else:
        raise NotImplementedError(
            f"Triangle quadrature of order {order} is not implemented; "
            "supported orders are 1 and 2."
        )

    area = _triangle_area(verts)
    if area <= 0.0:
        # Degenerate triangle: return a single point with zero weight.
        centroid = np.mean(verts, axis=0, keepdims=True)
        return centroid, np.zeros_like(w_ref)

    # Map barycentric coordinates to physical space.
    points = (
        bary[:, 0:1] * verts[0:1, :]
        + bary[:, 1:2] * verts[1:2, :]
        + bary[:, 2:3] * verts[2:3, :]
    )
    # Scale reference weights so that Σ w = area.
    weights = w_ref * area
    return points, weights


# ---------------------------------------------------------------------------
# Reference triangle rules (for CUDA / GPU bindings)
# ---------------------------------------------------------------------------


def reference_triangle_quadrature(
    order: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quadrature nodes and weights on the reference triangle with vertices
    (0, 0), (1, 0), (0, 1).

    Parameters
    ----------
    order : int, optional
        Polynomial order indicator (currently 1 or 2).

    Returns
    -------
    points : ndarray, shape (Q, 2)
        Quadrature points in (u, v) coordinates on the reference triangle.
    weights : ndarray, shape (Q,)
        Weights such that Σ_i weights[i] equals the reference triangle
        area (0.5).

    Notes
    -----
    Implemented rules mirror `standard_triangle_quadrature`:

    - order <= 1:
        1-point centroid rule at (1/3, 1/3), weight = 0.5.

    - order == 2:
        3-point symmetric rule at:
            (2/3, 1/6), (1/6, 2/3), (1/6, 1/6)
        with equal weights 0.5 / 3 = 1/6.
    """
    if order <= 1:
        pts = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float)
        w = np.array([0.5], dtype=float)
    elif order == 2:
        pts = np.array(
            [
                [2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 2.0 / 3.0],
                [1.0 / 6.0, 1.0 / 6.0],
            ],
            dtype=float,
        )
        w = np.full(3, 1.0 / 6.0, dtype=float)
    else:
        raise NotImplementedError(
            f"Reference triangle quadrature of order {order} is not implemented; "
            "supported orders are 1 and 2."
        )

    return pts, w


# ---------------------------------------------------------------------------
# Near-singular quadrature by adaptive refinement
# ---------------------------------------------------------------------------


def near_singular_quadrature(
    target,
    panel_vertices,
    method: str = "telles",
    order: int = 2,
    *,
    max_depth: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    High-resolution quadrature rule for a target close to a triangle.

    Parameters
    ----------
    target : array_like, shape (3,)
        Evaluation point in R^3.
    panel_vertices : array_like, shape (3, 3)
        Triangle vertices in R^3.
    method : str, optional
        Transformation strategy. Currently only "telles" is recognised
        and maps to a simple adaptive refinement scheme (no exact 1D
        Telles transform is used yet).
    order : int, optional
        Base order used on each refined subtriangle (1 or 2).

    Returns
    -------
    points : ndarray, shape (Q, 3)
        Quadrature points in physical coordinates.
    weights : ndarray, shape (Q,)
        Physical weights; sum to the triangle area.

    Notes
    -----
    Algorithm:
        - Start from the original triangle.
        - For a small fixed number of refinement levels, find the
          subtriangle whose centroid is closest to the target and split
          it into four smaller triangles by connecting edge midpoints.
        - Apply `standard_triangle_quadrature` on each leaf triangle
          and aggregate points and weights.

    This clusters quadrature nodes near the closest point on the panel
    to the target and is sufficient to stabilise near-singular 1/r
    kernels in the Laplace single-layer context for modest accuracies.
    """
    verts = _as_triangle_vertices(panel_vertices)
    tgt = np.asarray(target, dtype=float).reshape(3)

    # Fixed refinement depth; keep a slightly deeper tree for the Telles rule
    # to stabilise near-singular Laplace kernels at on-/near-surface targets.
    depth = max_depth
    if depth is None:
        depth = 4 if method.lower() == "telles" else 1
    if depth <= 0:
        return standard_triangle_quadrature(verts, order=order)

    # Work with a Python list of (3,3) arrays.
    tris = [verts]
    for _ in range(depth):
        # Find the triangle whose centroid is closest to the target.
        centroids = np.array([tri.mean(axis=0) for tri in tris])
        dists = np.linalg.norm(centroids - tgt[None, :], axis=1)
        idx = int(np.argmin(dists))
        tri = tris.pop(idx)

        v0, v1, v2 = tri
        m01 = 0.5 * (v0 + v1)
        m12 = 0.5 * (v1 + v2)
        m20 = 0.5 * (v2 + v0)

        # 4-way subdivision
        tris.append(np.stack([v0, m01, m20]))
        tris.append(np.stack([m01, v1, m12]))
        tris.append(np.stack([m20, m12, v2]))
        tris.append(np.stack([m01, m12, m20]))

    pts_list = []
    w_list = []
    for tri in tris:
        pts_sub, w_sub = standard_triangle_quadrature(tri, order=order)
        if pts_sub.size == 0:
            continue
        pts_list.append(pts_sub)
        w_list.append(w_sub)

    if not pts_list:
        # Fallback: use the base rule.
        return standard_triangle_quadrature(verts, order=order)

    points = np.vstack(pts_list)
    weights = np.concatenate(w_list)
    return points, weights


# ---------------------------------------------------------------------------
# Self-integral diagonal approximation
# ---------------------------------------------------------------------------


def self_integral_correction(area) -> float:
    """
    Return the *integral* value (not divided by area) to be used as the
    diagonal replacement for panel j in the matrix-free operator.

    We approximate each triangle by an equal-area disk of radius
        R = math.sqrt(A / math.pi).

    For the Laplace single-layer kernel

        G(x, y) = K_E / |x - y|,

    the on-surface integral over a uniformly charged disk of radius R has
    the form

        I_self ≈ 2π K_E R,

    which is the potential at the centre due to a unit surface charge
    density. This gives a stable, scale-aware diagonal used by GMRES and
    for constructing an initial guess. It is not a replacement for
    high-order quadrature; it is a robust first pass.
    """
    Aj = float(area)
    if Aj <= 0.0:
        return 0.0
    R = math.sqrt(Aj / math.pi)
    # Approximate ∫_panel G(x, y) dS_y with G(x, y) = K_E / |x - y|
    # using the equal-area disk heuristic: I_self ≈ 2π K_E R.
    return 2.0 * math.pi * K_E * R
