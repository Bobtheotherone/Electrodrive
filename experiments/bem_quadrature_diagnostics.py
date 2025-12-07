import argparse
import json
import math
import pathlib
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from electrodrive.core.bem import bem_solve
from electrodrive.fmm3d.logging_utils import ConsoleLogger
from electrodrive.learn.collocation import (
    _infer_bbox_for_spec,
    make_collocation_batch_for_spec,
)
from electrodrive.utils.config import EPS_0, BEMConfig
from tests.test_bem_quadrature import (
    BEM_TEST_ORACLE_CONFIG,
    _build_parallel_planes_spec,
    _build_plane_spec,
    _build_sphere_spec,
)


GeomEntry = Tuple[str, str, Any]


def _percentile(t: torch.Tensor, q: float) -> float:
    if t.numel() == 0:
        return float("nan")
    return float(torch.quantile(t, q / 100.0).item())


def _charge_boundary_distance(spec, geom_type: str) -> float:
    if not spec.charges:
        return float("inf")
    ch = spec.charges[0]
    if ch.get("type") != "point":
        return float("inf")
    pos = np.array(ch.get("pos", [0.0, 0.0, 0.0]), dtype=float)

    if geom_type in ("plane", "parallel_planes"):
        z_planes = [
            float(c.get("z", 0.0))
            for c in spec.conductors
            if c.get("type") == "plane"
        ]
        if not z_planes:
            return float("inf")
        d = min(abs(pos[2] - zp) for zp in z_planes)
        return float(max(d, 1e-12))

    if geom_type == "sphere":
        for c in spec.conductors:
            if c.get("type") != "sphere":
                continue
            center = np.array(c.get("center", [0.0, 0.0, 0.0]), dtype=float)
            radius = float(c.get("radius", 1.0))
            r0 = np.linalg.norm(pos - center)
            return float(max(abs(r0 - radius), 1e-12))

    return float("inf")


def _solve_bem_meta(spec) -> Dict[str, Any]:
    cfg = BEMConfig()
    if hasattr(cfg, "fp64"):
        cfg.fp64 = True
    if hasattr(cfg, "use_gpu"):
        cfg.use_gpu = False
    if hasattr(cfg, "max_refine_passes"):
        cfg.max_refine_passes = max(int(getattr(cfg, "max_refine_passes", 3) or 3), 3)
    if hasattr(cfg, "min_refine_passes"):
        cfg.min_refine_passes = 1
    if hasattr(cfg, "near_alpha"):
        cfg.near_alpha = 0.0
    if hasattr(cfg, "use_near_quadrature"):
        cfg.use_near_quadrature = True
    if hasattr(cfg, "use_near_quadrature_matvec"):
        cfg.use_near_quadrature_matvec = False

    result = bem_solve(spec, cfg, ConsoleLogger())
    mesh_stats = dict(result.get("mesh_stats", {}))
    gmres_stats = dict(result.get("gmres_stats", {}))
    meta = {
        "n_panels": int(mesh_stats.get("n_panels", 0)),
        "total_area": float(mesh_stats.get("total_area", 0.0) or 0.0),
        "patch_L": (
            float(mesh_stats["patch_L"]) if "patch_L" in mesh_stats else None
        ),
        "bc_residual_linf": float(mesh_stats.get("bc_residual_linf", math.inf)),
        "gmres_iters": int(gmres_stats.get("iters", gmres_stats.get("niter", 0) or 0)),
        "gmres_resid": float(gmres_stats.get("resid", math.nan)),
        "gmres_success": bool(gmres_stats.get("success", True)),
    }
    return meta


def _spec_summary(spec) -> Dict[str, Any]:
    return {
        "conductors": spec.conductors,
        "charges": spec.charges,
    }


def _compute_geom_features(
    point: torch.Tensor,
    spec,
    geom_type: str,
    bem_meta: Dict[str, Any],
    charge_bd_dist: float,
) -> Dict[str, Any]:
    x, y, z = (float(v) for v in point.tolist())
    p_np = np.array([x, y, z], dtype=float)

    r_charge = float("nan")
    if spec.charges:
        ch = spec.charges[0]
        if ch.get("type") == "point":
            pos = np.array(ch.get("pos", [0.0, 0.0, 0.0]), dtype=float)
            r_charge = float(np.linalg.norm(p_np - pos))

    rho = None
    d_plane = None
    R = None
    d_surf = None
    is_near_boundary = False

    if geom_type in ("plane", "parallel_planes"):
        rho = float(math.hypot(x, y))
        z_planes = [
            float(c.get("z", 0.0))
            for c in spec.conductors
            if c.get("type") == "plane"
        ]
        if z_planes:
            d_plane = float(min(abs(z - zp) for zp in z_planes))
            char_len = bem_meta.get("patch_L") or _infer_bbox_for_spec(spec)
            is_near_boundary = d_plane < 0.1 * float(char_len)

    if geom_type == "sphere":
        for c in spec.conductors:
            if c.get("type") != "sphere":
                continue
            center = np.array(c.get("center", [0.0, 0.0, 0.0]), dtype=float)
            radius = float(c.get("radius", 1.0))
            R = float(np.linalg.norm(p_np - center))
            d_surf = float(R - radius)
            is_near_boundary = abs(d_surf) < 0.1 * radius
            break

    is_near_charge = (
        r_charge < 0.5 * charge_bd_dist if math.isfinite(charge_bd_dist) else False
    )

    return {
        "r_charge": r_charge,
        "d_plane": d_plane,
        "rho": rho,
        "R": R,
        "d_surf": d_surf,
        "is_near_boundary": bool(is_near_boundary),
        "is_near_charge": bool(is_near_charge),
    }


def _gather_samples(
    X: torch.Tensor,
    Va: torch.Tensor,
    Vb_SI: torch.Tensor,
    Vb_red: torch.Tensor,
    abs_err: torch.Tensor,
    rel_err: torch.Tensor,
    is_boundary: torch.Tensor,
    spec,
    geom_type: str,
    bem_meta: Dict[str, Any],
    charge_bd_dist: float,
    limit: int,
) -> List[Dict[str, Any]]:
    n = rel_err.numel()
    k = min(limit, n)
    if k <= 0:
        return []
    topk = torch.topk(rel_err, k=k)
    idxs = topk.indices

    samples: List[Dict[str, Any]] = []
    for idx in idxs.tolist():
        p = X[idx]
        samples.append(
            {
                "point": [float(v) for v in p.tolist()],
                "Va": float(Va[idx].item()),
                "Vb_SI": float(Vb_SI[idx].item()),
                "Vb_reduced": float(Vb_red[idx].item()),
                "abs_err": float(abs_err[idx].item()),
                "rel_err": float(rel_err[idx].item()),
                "is_boundary": bool(is_boundary[idx].item()),
                "geom_features": _compute_geom_features(
                    p, spec, geom_type, bem_meta, charge_bd_dist
                ),
            }
        )
    return samples


def _run_geometry(
    name: str,
    builder,
    geom_type: str,
    n_points: int,
    ratio_boundary: float,
    seed: int,
    sample_limit: int,
    output_dir: pathlib.Path,
) -> None:
    spec = builder()
    device = torch.device("cpu")
    dtype = torch.float64

    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)

    batch_a = make_collocation_batch_for_spec(
        spec=spec,
        n_points=n_points,
        ratio_boundary=ratio_boundary,
        supervision_mode="analytic",
        device=device,
        dtype=dtype,
        rng=rng1,
        geom_type=geom_type,
    )
    batch_b = make_collocation_batch_for_spec(
        spec=spec,
        n_points=n_points,
        ratio_boundary=ratio_boundary,
        supervision_mode="bem",
        device=device,
        dtype=dtype,
        rng=rng2,
        geom_type=geom_type,
        bem_oracle_config=BEM_TEST_ORACLE_CONFIG,
    )

    assert torch.allclose(batch_a["X"], batch_b["X"])
    mask = batch_a["mask_finite"] & batch_b["mask_finite"]
    assert mask.any(), "No finite points to compare."

    X = batch_a["X"][mask]
    Va = batch_a["V_gt"][mask]
    Vb_SI = batch_b["V_gt"][mask]
    Vb_red = Vb_SI * EPS_0

    abs_err = torch.abs(Va - Vb_red)
    rel_err = abs_err / (
        torch.abs(Va) + torch.abs(Vb_red) + torch.tensor(1e-9, dtype=Va.dtype)
    )

    bem_meta = _solve_bem_meta(spec)
    charge_bd_dist = _charge_boundary_distance(spec, geom_type)

    stats = {
        "num_points": int(mask.sum().item()),
        "abs_err_max": float(abs_err.max().item()),
        "abs_err_mean": float(abs_err.mean().item()),
        "abs_err_median": _percentile(abs_err, 50),
        "rel_err_max": float(rel_err.max().item()),
        "rel_err_mean": float(rel_err.mean().item()),
        "rel_err_median": _percentile(rel_err, 50),
        "rel_err_p95": _percentile(rel_err, 95),
    }

    samples = _gather_samples(
        X,
        Va,
        Vb_SI,
        Vb_red,
        abs_err,
        rel_err,
        batch_a["is_boundary"][mask],
        spec,
        geom_type,
        bem_meta,
        charge_bd_dist,
        sample_limit,
    )

    payload = {
        "geometry": name,
        "spec_summary": _spec_summary(spec),
        "bem_meta": bem_meta,
        "stats": stats,
        "samples": samples,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"bem_quadrature_{name}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"{name}: rel_err_max={stats['rel_err_max']:.3e}, rel_err_mean={stats['rel_err_mean']:.3e} -> {out_path}"
    )


def _parse_geometries(sel: Iterable[str]) -> List[GeomEntry]:
    mapping = {
        "plane": ("plane", _build_plane_spec, "plane"),
        "sphere": ("sphere", _build_sphere_spec, "sphere"),
        "parallel_planes": ("parallel_planes", _build_parallel_planes_spec, "parallel_planes"),
    }
    if "all" in sel:
        return list(mapping.values())
    chosen: List[GeomEntry] = []
    for key in sel:
        if key not in mapping:
            raise ValueError(f"Unknown geometry: {key}")
        chosen.append(mapping[key])
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collocation-level analytic vs BEM diagnostics."
    )
    parser.add_argument(
        "--geometry",
        choices=["all", "plane", "sphere", "parallel_planes"],
        nargs="+",
        default=["all"],
        help="Geometries to run (default: all).",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=256,
        help="Number of collocation points per geometry (default: 256).",
    )
    parser.add_argument(
        "--ratio-boundary",
        type=float,
        default=0.5,
        help="Boundary sampling ratio (default: 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="RNG seed shared between analytic and BEM batches.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=256,
        help="Maximum number of point samples to include in output (sorted by rel_err).",
    )
    args = parser.parse_args()

    geometries = _parse_geometries(args.geometry)
    out_dir = pathlib.Path(__file__).resolve().parent / "_agent_outputs"

    for name, builder, geom_type in geometries:
        _run_geometry(
            name=name,
            builder=builder,
            geom_type=geom_type,
            n_points=args.n_points,
            ratio_boundary=args.ratio_boundary,
            seed=args.seed,
            sample_limit=args.sample_limit,
            output_dir=out_dir,
        )


if __name__ == "__main__":
    main()
