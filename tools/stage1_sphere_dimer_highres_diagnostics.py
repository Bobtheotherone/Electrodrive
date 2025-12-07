#!/usr/bin/env python3
from __future__ import annotations

"""
High-resolution diagnostics for Stage-1 sphere dimer candidates.

Evaluates a discovered image system vs oracle BEM on dense boundary/axis/gap grids.
Includes a stub backend flag for future FMM support.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch

from electrodrive.images.io import load_image_system
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.logging import JsonlLogger
from tools.stage1_sphere_dimer_bem_probe import make_oracle_cfg
from electrodrive.core.bem import bem_solve


def load_spec(path: Path) -> CanonicalSpec:
    return CanonicalSpec.from_json(json.loads(path.read_text()))


def sample_surface(center: Sequence[float], radius: float, n_theta: int, n_phi: int) -> np.ndarray:
    thetas = np.linspace(0.0, math.pi, n_theta)
    phis = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    pts = []
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


def sample_axis(z_min: float, z_max: float, n: int, exclude: Tuple[Tuple[float, float], ...]) -> np.ndarray:
    zs = np.linspace(z_min, z_max, n)
    mask = np.ones_like(zs, dtype=bool)
    for zc, r in exclude:
        mask &= np.abs(zs - zc) > r
    zs = zs[mask]
    return np.stack([np.zeros_like(zs), np.zeros_like(zs), zs], axis=1)


def sample_gap_grid(z_mid: float, r_max: float, n_r: int = 24, n_z: int = 24) -> np.ndarray:
    rs = np.linspace(0.0, r_max, n_r)
    zs = np.linspace(z_mid - 0.6, z_mid + 0.6, n_z)
    pts = []
    for r in rs:
        for z in zs:
            pts.append([r, 0.0, z])
            pts.append([-r, 0.0, z])
    return np.asarray(pts, dtype=np.float64)


def eval_bem(spec: CanonicalSpec, pts: np.ndarray, backend: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    if backend != "bem":
        raise NotImplementedError("FMM backend not implemented yet for high-res diagnostics.")
    cfg = make_oracle_cfg()
    with JsonlLogger(Path("runs/stage1_sphere_dimer/highres_logs")) as logger:
        res = bem_solve(spec, cfg, logger, differentiable=False)
    sol = res["solution"]
    with torch.no_grad():
        P = torch.as_tensor(pts, device=sol._device, dtype=sol._dtype)
        V, _ = sol.eval_V_E_batched(P)
    return V.detach().cpu().numpy(), res


def eval_image(system_path: Path, pts: np.ndarray, device: str | torch.device, dtype: torch.dtype) -> np.ndarray:
    sys = load_image_system(system_path, device=device, dtype=dtype)
    with torch.no_grad():
        P = torch.as_tensor(pts, device=sys.device, dtype=sys.dtype)
        V = sys.potential(P)
    return V.detach().cpu().numpy()


def error_stats(V_ref: np.ndarray, V_test: np.ndarray) -> Dict[str, float]:
    abs_err = np.abs(V_test - V_ref)
    rel_err = abs_err / np.maximum(np.abs(V_ref), 1e-6)
    return {
        "mean_rel": float(rel_err.mean()),
        "max_rel": float(rel_err.max()),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="High-res diagnostics for Stage-1 sphere dimer.")
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--system", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs/stage1_sphere_dimer/highres"))
    p.add_argument("--backend", choices=["bem", "fmm"], default="bem")
    args = p.parse_args(argv)

    spec = load_spec(args.spec)
    spheres = [c for c in spec.conductors if c.get("type") == "sphere"]
    s0, s1 = spheres[0], spheres[1]
    c0, c1 = np.asarray(s0["center"]), np.asarray(s1["center"])
    R0, R1 = float(s0["radius"]), float(s1["radius"])
    z0, z1 = c0[2], c1[2]
    z_mid = 0.5 * (z0 + z1)

    pts_boundary = np.concatenate(
        [sample_surface(c0, R0, 32, 64), sample_surface(c1, R1, 32, 64)],
        axis=0,
    )
    pts_axis = sample_axis(z_min=z0 - 0.5, z_max=z1 + 0.5, n=81, exclude=((z0, R0 + 0.01), (z1, R1 + 0.01)))
    pts_gap = sample_gap_grid(z_mid=z_mid, r_max=0.8, n_r=24, n_z=24)
    pts = np.concatenate([pts_boundary, pts_axis, pts_gap], axis=0)

    V_ref, bem_out = eval_bem(spec, pts, backend=args.backend)
    V_img = eval_image(args.system, pts, device="cpu", dtype=torch.float64)

    stats_all = error_stats(V_ref, V_img)
    stats_boundary = error_stats(V_ref[: pts_boundary.shape[0]], V_img[: pts_boundary.shape[0]])
    stats_axis = error_stats(
        V_ref[pts_boundary.shape[0] : pts_boundary.shape[0] + pts_axis.shape[0]],
        V_img[pts_boundary.shape[0] : pts_boundary.shape[0] + pts_axis.shape[0]],
    )
    stats_gap = error_stats(V_ref[-pts_gap.shape[0] :], V_img[-pts_gap.shape[0] :])

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "diagnostics.npz", pts=pts, V_ref=V_ref, V_img=V_img)
    summary = {
        "spec": str(args.spec),
        "system": str(args.system),
        "backend": args.backend,
        "stats_all": stats_all,
        "stats_boundary": stats_boundary,
        "stats_axis": stats_axis,
        "stats_gap": stats_gap,
        "mesh_stats": bem_out.get("mesh_stats", {}),
        "gmres_stats": bem_out.get("gmres_stats", {}),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("High-res diagnostics summary:", summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
