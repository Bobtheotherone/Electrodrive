#!/usr/bin/env python3
from __future__ import annotations

"""
Local geometry explorer for Stage-1 sphere dimer (R2, d tweaks + basis tweaks).

Takes top candidates (or a default config) and perturbs sphere 2 radius/spacing,
runs discovery with Kelvin ladder + rings, evaluates against oracle BEM, and
aggregates metrics.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from electrodrive.images.search import discover_images, ImageSystem
from electrodrive.images.io import save_image_system
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.orchestration.spec_registry import stage1_sphere_dimer_inside_path
from electrodrive.utils.logging import JsonlLogger
from tools.stage1_sphere_dimer_bem_probe import make_oracle_cfg
from electrodrive.core.bem import bem_solve


BASE_SPEC = stage1_sphere_dimer_inside_path()
OUT_ROOT = Path("runs/stage1_sphere_dimer/local_geometry")


def load_spec(path: Path) -> CanonicalSpec:
    return CanonicalSpec.from_json(json.loads(path.read_text()))


def write_spec(base: CanonicalSpec, R2: float, d: float, z_src: float, out_path: Path) -> Path:
    data = {
        "domain": base.domain,
        "BCs": base.BCs,
        "conductors": [],
        "charges": [],
        "symmetry": base.symmetry,
        "domain_meta": getattr(base, "domain_meta", {}),
    }
    # sphere 1 unchanged
    s0 = next(c for c in base.conductors if c.get("id", 0) == 0)
    data["conductors"].append(s0)
    # sphere 2 updated
    s1 = next(c for c in base.conductors if c.get("id", 1) == 1)
    s1_new = dict(s1)
    s1_new["center"] = [0.0, 0.0, float(d)]
    s1_new["radius"] = float(R2)
    data["conductors"].append(s1_new)
    ch = base.charges[0]
    ch_new = dict(ch)
    ch_new["pos"] = [0.0, 0.0, float(z_src)]
    data["charges"] = [ch_new]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out_path


def sample_gap_belt(R: float, z_mid: float, n_r: int = 8, n_z: int = 8, r_span: float = 0.6) -> np.ndarray:
    rs = np.linspace(0.0, r_span, n_r)
    zs = np.linspace(z_mid - 0.4, z_mid + 0.4, n_z)
    pts = []
    for r in rs:
        for z in zs:
            pts.append([r, 0.0, z])
            pts.append([-r, 0.0, z])
    return np.asarray(pts, dtype=np.float64)


def sample_axis(z_min: float, z_max: float, n: int, exclude: List[Tuple[float, float]]) -> np.ndarray:
    zs = np.linspace(z_min, z_max, n)
    mask = np.ones_like(zs, dtype=bool)
    for zc, r in exclude:
        mask &= np.abs(zs - zc) > r
    zs = zs[mask]
    return np.stack([np.zeros_like(zs), np.zeros_like(zs), zs], axis=1)


def sample_surface(center: Sequence[float], radius: float, n_theta: int = 16, n_phi: int = 32) -> np.ndarray:
    thetas = np.linspace(0.0, math.pi, n_theta)
    phis = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    pts: List[List[float]] = []
    for th in thetas:
        s, c = math.sin(th), math.cos(th)
        for ph in phis:
            pts.append(
                [
                    center[0] + radius * s * math.cos(ph),
                    center[1] + radius * s * math.sin(ph),
                    center[2] + radius * c,
                ]
            )
    return np.asarray(pts, dtype=np.float64)


def diagnostic_points(spec: CanonicalSpec) -> Tuple[np.ndarray, np.ndarray]:
    spheres = [c for c in spec.conductors if c.get("type") == "sphere"]
    s0, s1 = spheres[0], spheres[1]
    c0, c1 = np.asarray(s0["center"]), np.asarray(s1["center"])
    R0, R1 = float(s0["radius"]), float(s1["radius"])
    z0, z1 = c0[2], c1[2]
    d = z1 - z0
    boundary = np.concatenate(
        [
            sample_surface(c0, R0),
            sample_surface(c1, R1),
        ],
        axis=0,
    )
    axis = sample_axis(z_min=z0 - 0.5, z_max=z1 + 0.5, n=41, exclude=[(z0, R0 + 0.05), (z1, R1 + 0.05)])
    belt = sample_gap_belt(R=R0, z_mid=z0 + 0.5 * d, n_r=8, n_z=8, r_span=0.6)
    pts = np.concatenate([boundary, axis, belt], axis=0)
    return pts, belt


def eval_bem(spec: CanonicalSpec, pts: np.ndarray):
    cfg = make_oracle_cfg()
    out_dir = Path("runs/stage1_sphere_dimer/bem_cache_local")
    out_dir.mkdir(parents=True, exist_ok=True)
    with JsonlLogger(out_dir) as logger:
        res = bem_solve(spec, cfg, logger, differentiable=False)
    sol = res["solution"]
    device = sol._device
    dtype = sol._dtype
    with torch.no_grad():
        P = torch.as_tensor(pts, device=device, dtype=dtype)
        V, _ = sol.eval_V_E_batched(P)
    return V.detach().cpu().numpy(), res


def eval_image(system: ImageSystem, pts: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        P = torch.as_tensor(pts, device=system.device, dtype=system.dtype)
        V = system.potential(P)
    return V.detach().cpu().numpy()


def error_stats(V_ref: np.ndarray, V_test: np.ndarray) -> Dict[str, float]:
    abs_err = np.abs(V_test - V_ref)
    rel_err = abs_err / np.maximum(np.abs(V_ref), 1e-6)
    return {
        "mean_rel": float(rel_err.mean()),
        "max_rel": float(rel_err.max()),
    }


def run_candidate(spec_path: Path, config: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    spec = load_spec(spec_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(out_dir)
    basis = config.get("basis_types", ["sphere_kelvin_ladder", "sphere_equatorial_ring"])
    n_max = int(config.get("n_max", 8))
    reg_l1 = float(config.get("reg_l1", 1e-3))
    restarts = int(config.get("restarts", 0))
    system = discover_images(spec=spec, basis_types=basis, n_max=n_max, reg_l1=reg_l1, restarts=restarts, logger=logger)
    save_image_system(system, out_dir / "discovered_system.json", metadata={"config": config})

    pts, belt = diagnostic_points(spec)
    V_ref, bem_out = eval_bem(spec, pts)
    V_img = eval_image(system, pts)
    stats_all = error_stats(V_ref, V_img)
    belt_len = belt.shape[0]
    stats_inner = error_stats(V_ref[-belt_len:], V_img[-belt_len:])

    metrics = {
        "n_images": len(system.elements),
        "mean_rel": stats_all["mean_rel"],
        "inner_mean_rel": stats_inner["mean_rel"],
        "max_rel": stats_all["max_rel"],
        "inner_max_rel": stats_inner["max_rel"],
        "bem_mesh": bem_out.get("mesh_stats", {}),
        "gmres": bem_out.get("gmres_stats", {}),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.close()
    return metrics


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage-1 sphere dimer local geometry explorer.")
    parser.add_argument("--spec", type=Path, default=BASE_SPEC)
    parser.add_argument("--base-config", type=Path, help="Path to a JSON config (e.g., from top_candidates).")
    parser.add_argument("--out", type=Path, default=OUT_ROOT)
    args = parser.parse_args(argv)

    base_spec = load_spec(args.spec)
    base_s2 = next(c for c in base_spec.conductors if c.get("id", 1) == 1)
    R2_base = float(base_s2.get("radius", 1.0))
    d_base = float(base_s2.get("center", [0.0, 0.0, 2.4])[2])
    z_src = float(base_spec.charges[0].get("pos", [0, 0, 1.2])[2])

    geom_R2 = [r for r in (R2_base - 0.1, R2_base, R2_base + 0.1) if 0.8 <= r <= 1.2]
    geom_d = [d for d in (d_base - 0.2, d_base, d_base + 0.2) if 2.0 <= d <= 3.0]

    if args.base_config and args.base_config.exists():
        cfg_data = json.loads(args.base_config.read_text())
        best = cfg_data[0] if isinstance(cfg_data, list) and cfg_data else cfg_data
    else:
        best = {
            "basis_types": ["sphere_kelvin_ladder", "sphere_equatorial_ring"],
            "n_max": 8,
            "reg_l1": 1e-3,
            "restarts": 0,
        }

    results = []
    for R2 in geom_R2:
        for d in geom_d:
            # ensure no overlap: centers are 0 and d, radii 1 and R2
            if d <= 1.0 + R2:
                continue
            spec_path = args.out / "specs" / f"spec_R2_{R2:.2f}_d_{d:.2f}.json"
            write_spec(base_spec, R2=R2, d=d, z_src=z_src, out_path=spec_path)
            run_dir = args.out / f"R2_{R2:.2f}_d_{d:.2f}"
            metrics = run_candidate(spec_path, best, run_dir)
            results.append({"R2": R2, "d": d, **best, **metrics, "run_dir": str(run_dir)})

    results_path = args.out / "summary.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    def key_fn(r: Dict[str, Any]) -> Tuple[float, float, int]:
        return (r.get("inner_mean_rel", 1e9), r.get("mean_rel", 1e9), r.get("n_images", 999))

    top = sorted(results, key=key_fn)[:5]
    (args.out / "top_candidates.json").write_text(json.dumps(top, indent=2), encoding="utf-8")
    print(f"Local geometry exploration complete. Results: {results_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
