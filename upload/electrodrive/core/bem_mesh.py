"""
Minimal mesh generator for plane and sphere conductors (no external deps).
"""
from typing import Tuple, Optional
from dataclasses import dataclass
import os
import numpy as np

from electrodrive.utils.logging import JsonlLogger
from electrodrive.orchestration.parser import CanonicalSpec


@dataclass
class TriMesh:
    vertices: np.ndarray       # [Nv,3]
    triangles: np.ndarray      # [Nt,3]
    centroids: np.ndarray      # [Nt,3]
    normals: np.ndarray        # [Nt,3]
    areas: np.ndarray          # [Nt]
    conductor_ids: np.ndarray # [Nt]

    @property
    def n_panels(self) -> int:
        return int(self.triangles.shape[0])


def generate_mesh(spec: CanonicalSpec, target_h: float = 0.1, logger: Optional[JsonlLogger] = None) -> TriMesh:
    if logger:
        logger.info("Mesh generation started.", target_h=target_h)

    V_list, T_list, C_list = [], [], []
    tri_offset = 0
    for cond_idx, c in enumerate(spec.conductors):
        typ = c.get("type")
        cid = int(c.get("id", cond_idx))  # prefer user-specified id, else index

        if typ == "plane":
            z = float(c.get("z", 0.0))
            # adapt extent to nearest charge height for better truncation
            dmin = 0.3
            for ch in spec.charges:
                if ch.get("type") == "point":
                    dmin = max(dmin, abs(float(ch["pos"][2] - z)))
            try:
                patch_scale = float(os.getenv("EDE_BEM_PLANE_PATCH_SCALE", "1.0"))
            except Exception:
                patch_scale = 1.0
            L_base = max(8.0, 16.0 * dmin)
            L = patch_scale * L_base
            n = max(8, int(L / max(1e-3, target_h)))
            v, t = _mesh_plane_patch(z, L, n)

        elif typ == "sphere":
            center = np.array(c.get("center", [0.0, 0.0, 0.0]), dtype=float)
            radius = float(c.get("radius", 1.0))
            base = max(1e-3, target_h)
            # Slightly denser angular grid for spheres so that the constant-
            # per-panel single-layer BEM better resolves the induced surface
            # charge, especially near the closest approach to external charges.
            try:
                density_scale = float(os.getenv("EDE_BEM_SPHERE_DENSITY_SCALE", "1.5"))
            except Exception:
                density_scale = 1.5
            density_scale = max(1.0, density_scale)
            n_theta = max(10, int(np.ceil(density_scale * np.pi * radius / base)))
            n_phi   = max(20, int(np.ceil(density_scale * 2.0 * np.pi * radius / base)))
            v, t = _mesh_sphere(center, radius, n_theta, n_phi)

        elif typ in ("torus", "toroid"):
            center = np.array(c.get("center", [0.0, 0.0, 0.0]), dtype=float)
            R = float(c.get("major_radius", c.get("radius", 1.0)))
            a = float(c.get("minor_radius", 0.25 * R))
            base = max(1e-3, target_h)
            # Resolution heuristics: wrap around 2πR for u and 2πa for v.
            try:
                major_scale = float(os.getenv("EDE_BEM_TORUS_MAJOR_SCALE", "1.0"))
            except Exception:
                major_scale = 1.0
            try:
                minor_scale = float(os.getenv("EDE_BEM_TORUS_MINOR_SCALE", "1.0"))
            except Exception:
                minor_scale = 1.0
            n_u = max(24, int(np.ceil(major_scale * (2.0 * np.pi * R) / base)))
            n_v = max(16, int(np.ceil(minor_scale * (2.0 * np.pi * a) / base)))
            v, t = _mesh_torus(center, R, a, n_u, n_v)

        else:
            raise NotImplementedError(f"Unsupported conductor type: {typ}")

        V_list.append(v)
        T_list.append(t + tri_offset)
        C_list.extend([cid] * t.shape[0])
        tri_offset += v.shape[0]

    if not V_list:
        # Handle case with no conductors
        return TriMesh(np.empty((0,3)), np.empty((0,3), dtype=np.int32), np.empty((0,3)), np.empty((0,3)), np.empty(0), np.empty(0, dtype=np.int32))

    vertices       = np.vstack(V_list)
    triangles      = np.vstack(T_list)
    conductor_ids = np.asarray(C_list, dtype=np.int32)

    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    centroids = (v0 + v1 + v2) / 3.0
    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    # Drop degenerate panels (e.g., polar caps in coarse sphere grids)
    mask = areas > 1e-16
    dropped = int((~mask).sum())
    if dropped:
        triangles      = triangles[mask]
        centroids      = centroids[mask]
        cross          = cross[mask]
        areas          = areas[mask]
        conductor_ids = conductor_ids[mask]
        # No need to recompute v0, v1, v2 unless they are used later for something else

    # normals: unit length
    # Added 1e-30 to areas before division to avoid division by zero if any areas are still 0
    normals = cross / (2.0 * areas[:, None] + 1e-30)

    if logger:
        logger.info(
            "Mesh generation complete.",
            n_panels=int(triangles.shape[0]),
            total_area=float(areas.sum()),
            dropped_panels=dropped
        )

    return TriMesh(vertices, triangles, centroids, normals, areas, conductor_ids)


def _mesh_plane_patch(z: float, L: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-L / 2.0, L / 2.0, n + 1)
    y = np.linspace(-L / 2.0, L / 2.0, n + 1)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    vertices = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)], axis=1)

    tris = []
    for i in range(n):
        for j in range(n):
            idx = i * (n + 1) + j
            tris.append([idx, idx + 1, idx + (n + 1)])            # two triangles per cell
            tris.append([idx + 1, idx + (n + 2), idx + (n + 1)])
    return vertices, np.asarray(tris, dtype=np.int32)


def _mesh_sphere(center: np.ndarray, radius: float, n_theta: int, n_phi: int) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, np.pi, n_theta)
    phi   = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    verts = []
    for t in theta:
        for p in phi:
            x = center[0] + radius * np.sin(t) * np.cos(p)
            y = center[1] + radius * np.sin(t) * np.sin(p)
            z = center[2] + radius * np.cos(t)
            verts.append([x, y, z])
    vertices = np.asarray(verts, dtype=float)

    tris = []
    for i in range(n_theta - 1):
        for j in range(n_phi):
            jn = (j + 1) % n_phi
            a = i * n_phi + j
            b = i * n_phi + jn
            c = (i + 1) * n_phi + j
            d = (i + 1) * n_phi + jn
            tris.append([a, b, c])
            tris.append([b, d, c])

    return vertices, np.asarray(tris, dtype=np.int32)


def _mesh_torus(center: np.ndarray, R: float, a: float, n_u: int, n_v: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple tensor-product grid torus mesh parameterised by angles:
      u: around the major circle (0..2π)
      v: around the minor circle (0..2π)
    """
    u = np.linspace(0.0, 2.0 * np.pi, n_u, endpoint=False)
    v = np.linspace(0.0, 2.0 * np.pi, n_v, endpoint=False)
    verts = []
    for uu in u:
        cosu = np.cos(uu)
        sinu = np.sin(uu)
        for vv in v:
            cosv = np.cos(vv)
            sinv = np.sin(vv)
            x = (R + a * cosv) * cosu + center[0]
            y = (R + a * cosv) * sinu + center[1]
            z = a * sinv + center[2]
            verts.append([x, y, z])
    vertices = np.asarray(verts, dtype=float)

    tris = []
    for i in range(n_u):
        inext = (i + 1) % n_u
        for j in range(n_v):
            jnext = (j + 1) % n_v
            a_idx = i * n_v + j
            b_idx = i * n_v + jnext
            c_idx = inext * n_v + j
            d_idx = inext * n_v + jnext
            tris.append([a_idx, b_idx, c_idx])
            tris.append([b_idx, d_idx, c_idx])
    return vertices, np.asarray(tris, dtype=np.int32)
