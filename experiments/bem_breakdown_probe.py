#!/usr/bin/env python
"""
Break down analytic vs BEM behaviour on the test geometries.

For each geometry, this script:
- Solves BEM with the same config used in tests (BEM_TEST_ORACLE_CONFIG).
- Samples the exact collocation points the tests use (n_points, ratio_boundary).
- Evaluates analytic potentials and BEM potentials on those points.
- Compares total potential error (with EPS_0 scaling), boundary vs interior.
- Compares induced potential from the BEM evaluator vs a simple centroid-lumped
  kernel (K_E * sigma_j * A_j / |x - C_j|) to spot operator/evaluator mismatch.
- Logs BC residuals and refinement history to help separate solve vs evaluation.

Results are written to experiments/_agent_outputs/bem_breakdown_probe_*.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from tests.test_bem_quadrature import (
    _build_plane_spec,
    _build_sphere_spec,
    _build_parallel_planes_spec,
    BEM_TEST_ORACLE_CONFIG,
)

import electrodrive.core.bem as bem_core
from electrodrive.core.bem import bem_solve
from electrodrive.fmm3d.logging_utils import ConsoleLogger
from electrodrive.learn.collocation import (
    _evaluate_oracle_on_points,
    _sample_points_for_spec,
    _solve_analytic,
)
from electrodrive.utils.config import BEMConfig, EPS_0, K_E


SPEC_BUILDERS = {
    "plane": _build_plane_spec,
    "sphere": _build_sphere_spec,
    "parallel_planes": _build_parallel_planes_spec,
}


def _as_dict_cfg(cfg: BEMConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in dir(cfg):
        if k.startswith("_"):
            continue
        try:
            v = getattr(cfg, k)
        except Exception:
            continue
        if callable(v):
            continue
        try:
            json.dumps(v)
        except TypeError:
            continue
        out[k] = v
    return out


def _make_cfg() -> BEMConfig:
    cfg = BEMConfig()
    for k, v in BEM_TEST_ORACLE_CONFIG.items():
        try:
            setattr(cfg, k, v)
        except Exception:
            continue
    # Mirror test expectations: fp64, near-field eval on (if CPU), no matvec near-quad.
    cfg.fp64 = True
    return cfg


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
    """Best scalar s minimizing ||a - s b||_2; returns 0 if b is all zeros."""
    num = float(np.dot(a.ravel(), b.ravel()))
    den = float(np.dot(b.ravel(), b.ravel()))
    if den == 0.0:
        return 0.0
    return num / den


def _lumped_induced(targets: np.ndarray, centroids: np.ndarray, areas: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Centroid-lumped induced potential (no near/self correction)."""
    res = np.zeros(targets.shape[0], dtype=np.float64)
    for i, x in enumerate(targets):
        r = np.linalg.norm(centroids - x[None, :], axis=1)
        r = np.maximum(r, 1e-12)
        res[i] = np.sum(K_E * sigma * areas / r)
    return res


def run_probe(geom: str, n_points: int, ratio_boundary: float, seed: int) -> Dict[str, Any]:
    if geom not in SPEC_BUILDERS:
        raise ValueError(f"Unknown geometry '{geom}'")

    spec = SPEC_BUILDERS[geom]()
    cfg = _make_cfg()
    logger = ConsoleLogger()

    bem_out = bem_solve(spec, cfg, logger)
    if not isinstance(bem_out, dict) or "solution" not in bem_out:
        raise RuntimeError(f"BEM solve failed for geometry '{geom}'")

    sol = bem_out["solution"]
    mesh_stats = bem_out.get("mesh_stats", {})
    gmres_stats = bem_out.get("gmres_stats", {})
    hist_raw = bem_out.get("refinement_history", [])
    history = [
        {
            "pass": int(p.get("pass", i + 1)),
            "h": float(p.get("h", math.nan)),
            "bc_resid_linf": float(p.get("bc_resid_linf", math.nan)),
            "gmres_resid": float((p.get("gmres_info") or {}).get("resid", math.nan)),
            "gmres_iters": int((p.get("gmres_info") or {}).get("iters", -1)),
        }
        for i, p in enumerate(hist_raw)
    ]

    rng = np.random.default_rng(seed)
    points_np, is_boundary_t = _sample_points_for_spec(
        spec, n_points=n_points, ratio_boundary=ratio_boundary, rng=rng
    )
    is_boundary_np = is_boundary_t.cpu().numpy().astype(bool)

    device = torch.device("cpu")
    dtype = torch.float64

    analytic_solution = _solve_analytic(spec)
    if analytic_solution is None:
        raise RuntimeError(f"No analytic shortcut available for geometry '{geom}'")

    Va_t = _evaluate_oracle_on_points(
        spec,
        analytic_solution,
        geom,
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

    # Induced potential split: BEM evaluator vs simple centroid lumping.
    V_free_t = bem_core._free_space_at_points(spec, pts_device, sol._dtype, sol._device)
    V_free_np = V_free_t.detach().cpu().numpy()
    V_ind_np = Vb_np - V_free_np

    C_np = sol._C.detach().cpu().numpy()
    A_np = sol._A.detach().cpu().numpy()
    S_np = sol._S.detach().cpu().numpy()
    V_ind_lumped_np = _lumped_induced(points_np, C_np, A_np, S_np)
    err_lumped = _rel_err(V_ind_np, V_ind_lumped_np)

    scale_best = _best_scale(Va_np[mask], Vb_scaled_np[mask])

    samples: List[Dict[str, Any]] = []
    n_samp = min(12, points_np.shape[0])
    for idx in range(n_samp):
        samples.append(
            {
                "idx": idx,
                "point": [float(x) for x in points_np[idx]],
                "is_boundary": bool(is_boundary_np[idx]),
                "V_analytic": float(Va_np[idx]),
                "V_bem_scaled": float(Vb_scaled_np[idx]),
                "V_bem_raw": float(Vb_np[idx]),
                "V_ind": float(V_ind_np[idx]),
                "V_ind_lumped": float(V_ind_lumped_np[idx]),
                "rel_err_total": float(err_full[idx]) if idx < err_full.size else math.nan,
                "rel_err_induced_lumped": float(err_lumped[idx] if idx < err_lumped.size else math.nan),
            }
        )

    return {
        "geometry": geom,
        "bem_config": _as_dict_cfg(cfg),
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
        "induced_probe": {
            "rel_err_lumped_vs_eval": _rel_stats(err_lumped),
            "V_ind_stats": _stats(V_ind_np),
            "V_ind_lumped_stats": _stats(V_ind_lumped_np),
        },
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Breakdown probe for BEM vs analytic.")
    parser.add_argument(
        "--geometry",
        nargs="+",
        default=list(SPEC_BUILDERS.keys()),
        choices=list(SPEC_BUILDERS.keys()),
        help="Geometries to probe.",
    )
    parser.add_argument("--n-points", type=int, default=256, dest="n_points")
    parser.add_argument("--ratio-boundary", type=float, default=0.5, dest="ratio_boundary")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    results: List[Dict[str, Any]] = []
    for geom in args.geometry:
        print(f"[probe] Running geometry={geom}")
        res = run_probe(geom, n_points=args.n_points, ratio_boundary=args.ratio_boundary, seed=args.seed)
        results.append(res)

    out_dir = Path("experiments") / "_agent_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_path = out_dir / f"bem_breakdown_probe{tag}.json"
    payload = {
        "args": {
            "geometry": args.geometry,
            "n_points": args.n_points,
            "ratio_boundary": args.ratio_boundary,
            "seed": args.seed,
            "tag": args.tag,
        },
        "results": results,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[probe] Wrote {out_path}")


if __name__ == "__main__":
    main()
