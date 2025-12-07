#!/usr/bin/env python
"""
Sweep plane BEM configurations (patch size, h, device) and compare against
analytic image-charge potentials on shared collocation points.

Outputs a JSON file under experiments/_agent_outputs/bem_plane_h_patch_scan*.json
capturing mesh stats, GMRES stats, and boundary/interior relative errors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from tests.test_bem_quadrature import (
    _build_plane_spec,
    BEM_TEST_ORACLE_CONFIG,
)

from electrodrive.core.bem import bem_solve
from electrodrive.learn.collocation import (
    _evaluate_oracle_on_points,
    _sample_points_for_spec,
    _solve_analytic,
)
from electrodrive.utils.config import BEMConfig, EPS_0


def _stats(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {"min": math.nan, "max": math.nan, "mean": math.nan, "std": math.nan}
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
    }


def _rel_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.abs(a) + np.abs(b) + 1e-9
    return np.abs(a - b) / denom


def _rel_stats(err: np.ndarray) -> Dict[str, float]:
    err = np.asarray(err, dtype=np.float64)
    if err.size == 0:
        return {"max": math.nan, "p95": math.nan, "mean": math.nan}
    return {
        "max": float(np.nanmax(err)),
        "p95": float(np.nanpercentile(err, 95)),
        "mean": float(np.nanmean(err)),
    }


def _best_scale(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.dot(a.ravel(), b.ravel()))
    den = float(np.dot(b.ravel(), b.ravel()))
    if den == 0.0:
        return 0.0
    return num / den


def _as_list_floats(val: Sequence[str]) -> List[float]:
    out: List[float] = []
    for v in val:
        for tok in v.split(","):
            tok = tok.strip()
            if not tok:
                continue
            out.append(float(tok))
    return out


def _as_list_bools(val: Sequence[str]) -> List[bool]:
    out: List[bool] = []
    for v in val:
        for tok in v.split(","):
            tok = tok.strip().lower()
            if tok in ("1", "true", "t", "yes", "y"):
                out.append(True)
            elif tok in ("0", "false", "f", "no", "n"):
                out.append(False)
            elif tok:
                raise ValueError(f"Unrecognised boolean token '{tok}'")
    return out


class _SilentLogger:
    """Minimal logger that suppresses stdout spam during sweeps."""

    def info(self, msg: str, **_: Any) -> None:
        return

    def warning(self, msg: str, **_: Any) -> None:
        return

    def error(self, msg: str, **_: Any) -> None:
        return

    def debug(self, msg: str, **_: Any) -> None:
        return


def run_case(
    patch_scale: float,
    initial_h: float,
    use_gpu: bool,
    *,
    min_passes: int,
    max_passes: int,
    n_points: int,
    ratio_boundary: float,
    seed: int,
    line_z: Sequence[float],
) -> Dict[str, Any]:
    # Environment flag controls plane patch size.
    os.environ["EDE_BEM_PLANE_PATCH_SCALE"] = str(patch_scale)

    spec = _build_plane_spec()
    cfg = BEMConfig()
    for k, v in BEM_TEST_ORACLE_CONFIG.items():
        try:
            setattr(cfg, k, v)
        except Exception:
            continue
    cfg.fp64 = True
    cfg.use_gpu = bool(use_gpu)
    cfg.initial_h = float(initial_h)
    cfg.min_refine_passes = int(min_passes)
    cfg.max_refine_passes = int(max_passes)
    # Keep near-field quad at evaluation on; matvec near-quad remains off.
    cfg.use_near_quadrature = True
    cfg.use_near_quadrature_matvec = False

    logger = _SilentLogger()
    bem_out = bem_solve(spec, cfg, logger)
    if not isinstance(bem_out, dict) or "solution" not in bem_out:
        raise RuntimeError(f"BEM solve failed for patch_scale={patch_scale}, h={initial_h}, use_gpu={use_gpu}")

    sol = bem_out["solution"]
    mesh_stats = bem_out.get("mesh_stats", {})
    gmres_stats = bem_out.get("gmres_stats", {})
    history_raw = bem_out.get("refinement_history", [])
    history = [
        {
            "pass": int(p.get("pass", i + 1)),
            "h": float(p.get("h", math.nan)),
            "bc_resid_linf": float(p.get("bc_resid_linf", math.nan)),
            "gmres_resid": float((p.get("gmres_info") or {}).get("resid", math.nan)),
            "gmres_iters": int((p.get("gmres_info") or {}).get("iters", -1)),
        }
        for i, p in enumerate(history_raw)
    ]

    rng = np.random.default_rng(seed)
    points_np, is_boundary_t = _sample_points_for_spec(
        spec,
        n_points=n_points,
        ratio_boundary=ratio_boundary,
        rng=rng,
    )
    is_boundary_np = is_boundary_t.cpu().numpy().astype(bool)

    analytic_solution = _solve_analytic(spec)
    if analytic_solution is None:
        raise RuntimeError("Analytic shortcut unavailable for plane spec")

    device = torch.device("cpu")
    dtype = torch.float64

    Va_t = _evaluate_oracle_on_points(
        spec,
        analytic_solution,
        "plane",
        points_np,
        device=device,
        dtype=dtype,
    )
    Va_np = Va_t.detach().cpu().numpy()

    pts_device = torch.tensor(points_np, device=sol._device, dtype=sol._dtype)
    Vb_t, _ = sol.eval_V_E_batched(pts_device)
    Vb_np = Vb_t.detach().cpu().numpy()
    Vb_scaled_np = Vb_np * float(EPS_0)

    mask = np.isfinite(Va_np) & np.isfinite(Vb_scaled_np)
    err_full = _rel_err(Va_np, Vb_scaled_np)
    err_full[~mask] = np.nan
    err_all = err_full[np.isfinite(err_full)]
    err_bnd = err_full[is_boundary_np & np.isfinite(err_full)]
    err_int = err_full[(~is_boundary_np) & np.isfinite(err_full)]

    scale_best = _best_scale(Va_np[mask], Vb_scaled_np[mask]) if mask.any() else math.nan

    # Optional vertical line scan to spot spatial bias.
    line_pts = np.array([[0.0, 0.0, float(z)] for z in line_z], dtype=np.float64)
    Va_line_t = _evaluate_oracle_on_points(
        spec,
        analytic_solution,
        "plane",
        line_pts,
        device=device,
        dtype=dtype,
    )
    Va_line = Va_line_t.detach().cpu().numpy()
    line_dev = torch.tensor(line_pts, device=sol._device, dtype=sol._dtype)
    Vb_line_t, _ = sol.eval_V_E_batched(line_dev)
    Vb_line = Vb_line_t.detach().cpu().numpy()
    Vb_line_scaled = Vb_line * float(EPS_0)
    line_err = _rel_err(Va_line, Vb_line_scaled)

    return {
        "patch_scale": patch_scale,
        "initial_h": initial_h,
        "use_gpu": use_gpu,
        "min_refine_passes": min_passes,
        "max_refine_passes": max_passes,
        "near_quad_enabled": bool(getattr(sol, "_near_quad_enabled", False)),
        "mesh_stats": mesh_stats,
        "gmres_stats": gmres_stats,
        "refinement_history": history,
        "bc_resid_linf": float(mesh_stats.get("bc_residual_linf", math.nan)),
        "collocation": {
            "n_points": n_points,
            "ratio_boundary": ratio_boundary,
            "mask_finite_frac": float(mask.mean() if mask.size else 0.0),
            "rel_err_overall": _rel_stats(err_all),
            "rel_err_boundary": _rel_stats(err_bnd),
            "rel_err_interior": _rel_stats(err_int),
            "best_scale_factor_for_bem": scale_best,
            "Va_stats": _stats(Va_np),
            "Vb_scaled_stats": _stats(Vb_scaled_np),
        },
        "line_scan": [
            {
                "z": float(z),
                "Va": float(Va_line[i]),
                "Vb_scaled": float(Vb_line_scaled[i]),
                "rel_err": float(line_err[i]),
            }
            for i, z in enumerate(line_z)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plane BEM h/patch scan.")
    parser.add_argument("--patch-scales", nargs="+", default=["1.0", "2.0", "4.0"])
    parser.add_argument("--hs", nargs="+", default=["0.2", "0.1", "0.05"])
    parser.add_argument("--use-gpu", nargs="+", default=["true", "false"])
    parser.add_argument("--min-refine-passes", type=int, default=1)
    parser.add_argument("--max-refine-passes", type=int, default=1)
    parser.add_argument("--n-points", type=int, default=256)
    parser.add_argument("--ratio-boundary", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--line-z",
        nargs="+",
        default=["0.1", "0.2", "0.5", "1.0", "2.0"],
        help="z values (m) for vertical line scan at x=y=0",
    )
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    patch_scales = _as_list_floats(args.patch_scales)
    hs = _as_list_floats(args.hs)
    use_gpu_list = _as_list_bools(args.use_gpu)
    line_z = _as_list_floats(args.line_z)

    results: List[Dict[str, Any]] = []
    for ps in patch_scales:
        for h in hs:
            for use_gpu in use_gpu_list:
                print(f"[scan] patch_scale={ps}, h={h}, use_gpu={use_gpu}")
                res = run_case(
                    patch_scale=ps,
                    initial_h=h,
                    use_gpu=use_gpu,
                    min_passes=args.min_refine_passes,
                    max_passes=args.max_refine_passes,
                    n_points=args.n_points,
                    ratio_boundary=args.ratio_boundary,
                    seed=args.seed,
                    line_z=line_z,
                )
                results.append(res)

    out_dir = Path("experiments") / "_agent_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_path = out_dir / f"bem_plane_h_patch_scan{tag}.json"
    payload = {
        "args": {
            "patch_scales": patch_scales,
            "hs": hs,
            "use_gpu": use_gpu_list,
            "min_refine_passes": args.min_refine_passes,
            "max_refine_passes": args.max_refine_passes,
            "n_points": args.n_points,
            "ratio_boundary": args.ratio_boundary,
            "seed": args.seed,
            "line_z": line_z,
            "tag": args.tag,
        },
        "results": results,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[scan] Wrote {out_path}")


if __name__ == "__main__":
    main()
