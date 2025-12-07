import argparse
import json
import math
import pathlib
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from electrodrive.core.bem_mesh import generate_mesh
from electrodrive.core.bem_quadrature import (
    near_singular_quadrature,
    standard_triangle_quadrature,
)
from electrodrive.utils.config import K_E
from tests.test_bem_quadrature import _build_plane_spec, _build_sphere_spec


GeomEntry = Tuple[str, str, Any]


def _near_integral(target: np.ndarray, verts: np.ndarray, order: int = 2) -> float:
    pts, w = near_singular_quadrature(target, verts, method="telles", order=order)
    r = np.linalg.norm(pts - target[None, :], axis=1)
    r = np.maximum(r, 1e-12)
    return float(np.sum((K_E / r) * w))


def _near_integral_refined(
    target: np.ndarray,
    verts: np.ndarray,
    *,
    order: int = 2,
    extra_depth: int = 1,
) -> float:
    """
    Slightly deeper refinement than the default near_singular_quadrature.
    This mirrors the library logic but allows a higher refinement depth
    for a quasi-reference integral.
    """
    tgt = np.asarray(target, dtype=float).reshape(3)
    tri_root = np.asarray(verts, dtype=float).reshape(3, 3)

    max_depth = 3 + max(0, int(extra_depth))
    tris = [tri_root]
    for _ in range(max_depth):
        centroids = np.array([tri.mean(axis=0) for tri in tris])
        dists = np.linalg.norm(centroids - tgt[None, :], axis=1)
        idx = int(np.argmin(dists))
        tri = tris.pop(idx)

        v0, v1, v2 = tri
        m01 = 0.5 * (v0 + v1)
        m12 = 0.5 * (v1 + v2)
        m20 = 0.5 * (v2 + v0)

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
        pts, w = near_singular_quadrature(target, verts, method="telles", order=order)
        r = np.linalg.norm(pts - target[None, :], axis=1)
        r = np.maximum(r, 1e-12)
        return float(np.sum((K_E / r) * w))

    pts = np.vstack(pts_list)
    w = np.concatenate(w_list)
    r = np.linalg.norm(pts - target[None, :], axis=1)
    r = np.maximum(r, 1e-12)
    return float(np.sum((K_E / r) * w))


def _parse_geometries(sel: Iterable[str]) -> List[GeomEntry]:
    mapping = {
        "plane": ("plane", _build_plane_spec, "plane"),
        "sphere": ("sphere", _build_sphere_spec, "sphere"),
    }
    if "all" in sel:
        return list(mapping.values())
    chosen: List[GeomEntry] = []
    for key in sel:
        if key not in mapping:
            raise ValueError(f"Unknown geometry: {key}")
        chosen.append(mapping[key])
    return chosen


def _panel_normal(name: str, mesh, idx: int, spec) -> np.ndarray:
    n = mesh.normals[idx]
    norm = np.linalg.norm(n)
    if norm > 0:
        n = n / norm

    if name == "sphere":
        center = np.array(spec.conductors[0].get("center", [0.0, 0.0, 0.0]), dtype=float)
        radial = mesh.centroids[idx] - center
        norm_r = np.linalg.norm(radial)
        if norm_r > 0:
            n = radial / norm_r
    return n


def _run_geometry(
    name: str,
    builder,
    geom_type: str,
    panel_index: int,
    target_h: float,
    num_distances: int,
    dist_min: float,
    dist_max: float,
    extra_refine: int,
    output_dir: pathlib.Path,
) -> None:
    spec = builder()
    mesh = generate_mesh(spec, target_h=target_h, logger=None)
    if mesh.n_panels == 0:
        raise RuntimeError(f"No panels generated for geometry {name}")

    j = int(min(max(panel_index, 0), mesh.n_panels - 1))
    C_j = mesh.centroids[j]
    A_j = mesh.areas[j]
    verts_j = mesh.vertices[mesh.triangles[j]]
    n = _panel_normal(name, mesh, j, spec)

    sigma_j = 1.0
    distances = np.logspace(
        math.log10(dist_min), math.log10(dist_max), num=num_distances
    )

    V_far: List[float] = []
    V_near: List[float] = []
    V_ref: List[float] = []
    abs_err: List[float] = []
    abs_err_ref: List[float] = []

    for d in distances:
        tgt = C_j + d * n
        r_far = float(np.linalg.norm(tgt - C_j))
        I_far = float(K_E * A_j / max(r_far, 1e-12))
        V_far.append(sigma_j * I_far)

        I_near = _near_integral(tgt, verts_j, order=2)
        v_near = sigma_j * I_near
        V_near.append(v_near)

        I_ref = _near_integral_refined(tgt, verts_j, order=2, extra_depth=extra_refine)
        v_ref = sigma_j * I_ref
        V_ref.append(v_ref)

        abs_err.append(abs(v_near - I_far * sigma_j))
        abs_err_ref.append(abs(v_ref - I_far * sigma_j))

    payload = {
        "geometry": name,
        "panel_index": j,
        "panel_area": float(A_j),
        "distances": distances.tolist(),
        "V_far": V_far,
        "V_near": V_near,
        "V_refined": V_ref,
        "abs_err": abs_err,
        "abs_err_refined": abs_err_ref,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"bem_panel_nearfield_probe_{name}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{name}: panel {j}, {len(distances)} distances -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Near-field quadrature probe for a single BEM panel."
    )
    parser.add_argument(
        "--geometry",
        choices=["all", "plane", "sphere"],
        nargs="+",
        default=["all"],
        help="Geometries to run (default: all).",
    )
    parser.add_argument(
        "--panel-index",
        type=int,
        default=0,
        help="Panel index to probe (default: 0).",
    )
    parser.add_argument(
        "--target-h",
        type=float,
        default=0.2,
        help="Target mesh spacing passed to generate_mesh (default: 0.2).",
    )
    parser.add_argument(
        "--num-distances",
        type=int,
        default=40,
        help="Number of log-spaced distances to probe (default: 40).",
    )
    parser.add_argument(
        "--distance-min",
        type=float,
        default=1e-3,
        help="Minimum offset distance along the normal (default: 1e-3).",
    )
    parser.add_argument(
        "--distance-max",
        type=float,
        default=1e1,
        help="Maximum offset distance along the normal (default: 1e1).",
    )
    parser.add_argument(
        "--extra-refine",
        type=int,
        default=1,
        help="Additional refinement depth for the reference near integral (default: 1).",
    )
    args = parser.parse_args()

    geometries = _parse_geometries(args.geometry)
    out_dir = pathlib.Path(__file__).resolve().parent / "_agent_outputs"

    for name, builder, geom_type in geometries:
        _run_geometry(
            name=name,
            builder=builder,
            geom_type=geom_type,
            panel_index=args.panel_index,
            target_h=args.target_h,
            num_distances=args.num_distances,
            dist_min=args.distance_min,
            dist_max=args.distance_max,
            extra_refine=args.extra_refine,
            output_dir=out_dir,
        )


if __name__ == "__main__":
    main()
