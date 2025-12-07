#!/usr/bin/env python3
from __future__ import annotations

"""
Stage-1 sphere dimer random/modes-enriched explorer (Kelvin ladder + rings).

Runs a small grid / random sweep of discovery hyperparameters for the baseline
inside spec and evaluates candidates against an oracle-grade BEM on a diagnostic
grid (surfaces + axis + gap belt).
"""

import argparse
import json
import math
import random
import time
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
OUT_ROOT = Path("runs/stage1_sphere_dimer/random_explorer")


# ---------------------- Geometry / sampling helpers ---------------------- #


def load_spec(path: Path) -> CanonicalSpec:
    return CanonicalSpec.from_json(json.loads(path.read_text()))


def sample_sphere_surface(center: Sequence[float], radius: float, n_theta: int = 18, n_phi: int = 36) -> np.ndarray:
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


def sample_axis_points(z_min: float, z_max: float, n: int, exclude: List[Tuple[float, float]]) -> np.ndarray:
    zs = np.linspace(z_min, z_max, n)
    mask = np.ones_like(zs, dtype=bool)
    for zc, r in exclude:
        mask &= np.abs(zs - zc) > r
    zs = zs[mask]
    return np.stack([np.zeros_like(zs), np.zeros_like(zs), zs], axis=1)


def sample_gap_belt(R: float, z_mid: float, n_r: int = 8, n_z: int = 8, r_span: float = 0.6) -> np.ndarray:
    rs = np.linspace(0.0, r_span, n_r)
    zs = np.linspace(z_mid - 0.4, z_mid + 0.4, n_z)
    pts = []
    for r in rs:
        for z in zs:
            pts.append([r, 0.0, z])
            pts.append([-r, 0.0, z])
    return np.asarray(pts, dtype=np.float64)


def eval_image_system(system: ImageSystem, pts: np.ndarray, device: torch.device, dtype: torch.dtype) -> np.ndarray:
    with torch.no_grad():
        P = torch.as_tensor(pts, device=device, dtype=dtype)
        V = system.potential(P)
    return V.detach().cpu().numpy()


def eval_bem(spec: CanonicalSpec, pts: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    cfg = make_oracle_cfg()
    out_dir = Path("runs/stage1_sphere_dimer/bem_cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    with JsonlLogger(out_dir) as logger:
        res = bem_solve(spec, cfg, logger, differentiable=False)
    solution = res["solution"]
    device = solution._device
    dtype = solution._dtype
    with torch.no_grad():
        P = torch.as_tensor(pts, device=device, dtype=dtype)
        V, _ = solution.eval_V_E_batched(P)
    return V.detach().cpu().numpy(), res


def error_stats(V_ref: np.ndarray, V_test: np.ndarray) -> Dict[str, float]:
    abs_err = np.abs(V_test - V_ref)
    rel_err = abs_err / np.maximum(np.abs(V_ref), 1e-6)
    return {
        "mean_abs": float(abs_err.mean()),
        "mean_rel": float(rel_err.mean()),
        "max_rel": float(rel_err.max()),
    }


def diagnostic_points(spec: CanonicalSpec) -> Tuple[np.ndarray, np.ndarray]:
    spheres = [c for c in spec.conductors if c.get("type") == "sphere"]
    s0, s1 = spheres[0], spheres[1]
    c0, c1 = np.asarray(s0["center"]), np.asarray(s1["center"])
    R0, R1 = float(s0["radius"]), float(s1["radius"])
    z0, z1 = c0[2], c1[2]
    d = z1 - z0
    boundary = np.concatenate(
        [
            sample_sphere_surface(c0, R0),
            sample_sphere_surface(c1, R1),
        ],
        axis=0,
    )
    axis = sample_axis_points(
        z_min=z0 - 0.5,
        z_max=z1 + 0.5,
        n=41,
        exclude=[(z0, R0 + 0.05), (z1, R1 + 0.05)],
    )
    belt_local = sample_gap_belt(R=R0, z_mid=z0 + 0.5 * d, n_r=10, n_z=10, r_span=0.6)
    # rotate belt back to global coords (already aligned on x-z plane)
    belt = belt_local
    pts = np.concatenate([boundary, axis, belt], axis=0)
    return pts, belt


# ---------------------- Discovery + evaluation ---------------------- #


def run_single(config: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    spec = load_spec(Path(config.get("spec", BASE_SPEC)))
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(out_dir)

    basis_types = config.get("basis_types", ["sphere_kelvin_ladder", "sphere_equatorial_ring"])
    n_max = int(config.get("n_max", 8))
    reg_l1 = float(config.get("reg_l1", 1e-3))
    restarts = int(config.get("restarts", 0))

    logger.info(
        "Random explorer run start.",
        basis_types=basis_types,
        n_max=n_max,
        reg_l1=reg_l1,
        restarts=restarts,
    )

    system = discover_images(
        spec=spec,
        basis_types=basis_types,
        n_max=n_max,
        reg_l1=reg_l1,
        restarts=restarts,
        logger=logger,
    )
    save_image_system(system, out_dir / "discovered_system.json", metadata={"config": config})

    # Diagnostics
    pts, belt = diagnostic_points(spec)
    V_ref, bem_out = eval_bem(spec, pts)
    V_img = eval_image_system(system, pts, device=system.device, dtype=system.dtype)
    stats_all = error_stats(V_ref, V_img)

    # Inner/gap belt stats
    n_bc = 2 * len(sample_sphere_surface([0, 0, 0], 1))  # for slicing
    belt_pts = belt
    V_ref_belt = V_ref[-belt_pts.shape[0] :]
    V_img_belt = V_img[-belt_pts.shape[0] :]
    stats_inner = error_stats(V_ref_belt, V_img_belt)

    metrics = {
        "n_images": len(system.elements),
        "mean_rel": stats_all["mean_rel"],
        "max_rel": stats_all["max_rel"],
        "inner_mean_rel": stats_inner["mean_rel"],
        "inner_max_rel": stats_inner["max_rel"],
        "bem_mesh": bem_out.get("mesh_stats", {}),
        "gmres": bem_out.get("gmres_stats", {}),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Random explorer run done.", **metrics)
    logger.close()
    return metrics


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage-1 sphere dimer random explorer.")
    parser.add_argument("--spec", type=Path, default=BASE_SPEC)
    parser.add_argument("--out", type=Path, default=OUT_ROOT)
    parser.add_argument("--num", type=int, default=6, help="Number of random configs to run.")
    args = parser.parse_args(argv)

    base_config = {
        "spec": str(args.spec),
    }

    n_max_choices = [6, 8, 10, 12]
    reg_choices = [1e-2, 1e-3, 1e-4]
    basis_choices = [
        ["sphere_kelvin_ladder"],
        ["sphere_kelvin_ladder", "sphere_equatorial_ring"],
    ]

    results = []
    for i in range(args.num):
        cfg = dict(base_config)
        cfg["n_max"] = random.choice(n_max_choices)
        cfg["reg_l1"] = random.choice(reg_choices)
        cfg["basis_types"] = random.choice(basis_choices)
        run_dir = args.out / f"run_{i:03d}"
        metrics = run_single(cfg, run_dir)
        results.append({**cfg, **metrics, "run_dir": str(run_dir)})

    results_path = args.out / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Select top candidates by inner_mean_rel then mean_rel then n_images
    def key_fn(r: Dict[str, Any]) -> Tuple[float, float, int]:
        return (r.get("inner_mean_rel", 1e9), r.get("mean_rel", 1e9), r.get("n_images", 999))

    top = sorted(results, key=key_fn)[:5]
    (args.out / "top_candidates.json").write_text(json.dumps(top, indent=2), encoding="utf-8")
    print(f"Saved results to {results_path} and top_candidates.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
