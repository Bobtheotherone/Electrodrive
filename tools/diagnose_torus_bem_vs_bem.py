#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from electrodrive.core.bem import bem_solve, BEMSolution
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.config import BEMConfig
from electrodrive.utils.logging import JsonlLogger


@dataclass
class TorusParams:
    R: float  # major radius
    r: float  # minor radius


def load_spec(path: Path, z_charge: float | None = None) -> Tuple[CanonicalSpec, TorusParams]:
    raw = json.loads(path.read_text())
    if z_charge is not None:
        for ch in raw.get("charges", []):
            if ch.get("type") == "point":
                ch["pos"] = [0.0, 0.0, float(z_charge)]
    spec = CanonicalSpec.from_json(raw)
    tor = next(c for c in spec.conductors if c.get("type") == "torus")
    params = TorusParams(R=float(tor["major_radius"]), r=float(tor["minor_radius"]))
    return spec, params


def make_cfg(use_near_matvec: bool, fp64: bool = True, max_refine: int = 3) -> BEMConfig:
    cfg = BEMConfig()
    cfg.fp64 = fp64
    cfg.use_gpu = torch.cuda.is_available()
    cfg.max_refine_passes = max_refine
    cfg.use_near_quadrature = True
    cfg.use_near_quadrature_matvec = bool(use_near_matvec)
    cfg.near_quadrature_order = 2
    cfg.near_quadrature_distance_factor = 2.0
    cfg.gmres_tol = 1e-9 if use_near_matvec else cfg.gmres_tol
    cfg.initial_h = 0.25
    cfg.vram_autotune = False
    return cfg


def run_bem(spec: CanonicalSpec, cfg: BEMConfig, out_dir: Path) -> BEMSolution:
    out_dir.mkdir(parents=True, exist_ok=True)
    with JsonlLogger(out_dir) as logger:
        res = bem_solve(spec, cfg, logger, differentiable=False)
    if "error" in res:
        raise RuntimeError(res["error"])
    return res["solution"]


def sample_plane(R: float, r: float, n_r: int = 81, n_z: int = 81) -> np.ndarray:
    xs = np.linspace(0.0, 1.6 * R, n_r)
    zs = np.linspace(-0.6 * r, 0.6 * r, n_z)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    pts = np.stack([X, np.zeros_like(X), Z], axis=-1).reshape(-1, 3)
    # Keep points outside the conductor with a small buffer
    inside = (np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2) - R) ** 2 + pts[:, 2] ** 2 < (r + 0.01) ** 2
    return pts[~inside]


def eval_solution(sol: BEMSolution, pts: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        P = torch.as_tensor(pts, device=sol._device, dtype=sol._dtype)
        V, _ = sol.eval_V_E_batched(P)
    return V.detach().cpu().numpy()


def error_stats(V_ref: np.ndarray, V_test: np.ndarray) -> Dict[str, float]:
    abs_err = np.abs(V_test - V_ref)
    denom = np.maximum(np.abs(V_ref), 1e-6)
    rel_err = abs_err / denom
    return {
        "mean_abs": float(abs_err.mean()),
        "mean_rel": float(rel_err.mean()),
        "max_rel": float(rel_err.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare torus BEM configs (ref vs cheap).")
    parser.add_argument("--spec", type=Path, default=Path("specs/torus_axis_point_mid.json"))
    parser.add_argument("--z", type=float, default=None, help="Override charge z position.")
    parser.add_argument("--out", type=Path, default=Path("runs/diagnostics/torus_bem_compare"))
    args = parser.parse_args()

    spec, params = load_spec(args.spec, z_charge=args.z)
    out_root: Path = args.out
    ref_cfg = make_cfg(use_near_matvec=True)
    cheap_cfg = make_cfg(use_near_matvec=False)

    sol_ref = run_bem(spec, ref_cfg, out_root / "ref")
    sol_cheap = run_bem(spec, cheap_cfg, out_root / "cheap")

    pts = sample_plane(params.R, params.r)
    V_ref = eval_solution(sol_ref, pts)
    V_cheap = eval_solution(sol_cheap, pts)
    stats_full = error_stats(V_ref, V_cheap)

    # Focused inner-rim belt
    mask_inner = (np.abs(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2) - (params.R - params.r)) < 0.1) & (
        np.abs(pts[:, 2]) < 0.2
    )
    inner_stats = error_stats(V_ref[mask_inner], V_cheap[mask_inner])

    out = {
        "spec": str(args.spec),
        "z_charge": args.z,
        "n_points_full": int(V_ref.shape[0]),
        "n_points_inner": int(mask_inner.sum()),
        "stats_full": stats_full,
        "stats_inner": inner_stats,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    def _fmt(d: Dict[str, float]) -> str:
        return f"mean_abs={d['mean_abs']:.3g}, mean_rel={d['mean_rel']:.3g}, max_rel={d['max_rel']:.3g}"

    print(f"Full domain ({out['n_points_full']} pts): {_fmt(stats_full)}")
    print(f"Inner rim ({out['n_points_inner']} pts): {_fmt(inner_stats)}")


if __name__ == "__main__":
    main()
