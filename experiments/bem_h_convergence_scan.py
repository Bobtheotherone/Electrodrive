#!/usr/bin/env python
"""
h-convergence scan for analytic-vs-BEM error on plane, sphere, and parallel planes.

For each geometry and h value, this script:
  * runs a single-pass BEM solve with initial_h = h
  * builds matched analytic collocation points on CPU
  * evaluates BEM on those points
  * computes rel_err splits (boundary / interior / near-surface / near-charge)
  * records basic BEM stats (n_panels, bc_resid_linf, gmres iters/resid)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from electrodrive.core.bem import bem_solve
from electrodrive.fmm3d.logging_utils import ConsoleLogger
from electrodrive.learn.collocation import make_collocation_batch_for_spec
from electrodrive.utils.config import BEMConfig, EPS_0

from tests.test_bem_quadrature import (
    _build_parallel_planes_spec,
    _build_plane_spec,
    _build_sphere_spec,
    BEM_TEST_ORACLE_CONFIG,
)


def _build_cfg_for_h(h: float) -> BEMConfig:
    cfg = BEMConfig()
    base = deepcopy(BEM_TEST_ORACLE_CONFIG)
    for k, v in base.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.initial_h = h
    cfg.min_refine_passes = 1
    cfg.max_refine_passes = 1
    cfg.refine_factor = 1.0
    cfg.gmres_tol = base.get("gmres_tol", getattr(cfg, "gmres_tol", 5e-8))
    # Explicitly disable the preconditioner for this scan.
    setattr(cfg, "use_precond", False)
    return cfg


def _geom_meta(spec, geom: str) -> Dict[str, torch.Tensor]:
    meta: Dict[str, torch.Tensor] = {}
    if geom == "plane":
        z = float(spec.conductors[0].get("z", 0.0))
        meta["plane_z"] = torch.tensor(z, dtype=torch.float64)
    elif geom == "parallel_planes":
        zs = [float(c.get("z", 0.0)) for c in spec.conductors]
        meta["plane_zs"] = torch.tensor(zs, dtype=torch.float64)
    elif geom == "sphere":
        r = float(spec.conductors[0].get("radius", 1.0))
        center = torch.tensor(spec.conductors[0].get("center", [0.0, 0.0, 0.0]), dtype=torch.float64)
        meta["radius"] = torch.tensor(r, dtype=torch.float64)
        meta["center"] = center
    if spec.charges:
        pos = spec.charges[0].get("pos") or spec.charges[0].get("position")
        if pos is not None:
            meta["charge_pos"] = torch.tensor(pos, dtype=torch.float64)
    return meta


def _classify_masks(points: torch.Tensor, geom: str, meta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    near_surface = torch.zeros(points.shape[0], dtype=torch.bool)
    if geom == "plane":
        z0 = meta["plane_z"]
        near_surface = torch.abs(points[:, 2] - z0) < 0.1
    elif geom == "parallel_planes":
        zs = meta["plane_zs"]
        near_surface = torch.min(torch.abs(points[:, 2, None] - zs[None, :]), dim=1).values < 0.1
    elif geom == "sphere":
        center = meta["center"]
        radius = meta["radius"]
        r = torch.linalg.norm(points - center[None, :], dim=1)
        near_surface = torch.abs(r - radius) < 0.1 * radius

    near_charge = torch.zeros(points.shape[0], dtype=torch.bool)
    if "charge_pos" in meta:
        charge_pos = meta["charge_pos"]
        dist = torch.linalg.norm(points - charge_pos[None, :], dim=1)
        near_charge = dist < 0.25

    interior = ~(near_surface | near_charge)
    return {"near_surface": near_surface, "near_charge": near_charge, "interior": interior}


def _max_or_nan(t: torch.Tensor) -> float:
    return float(t.max().item()) if t.numel() > 0 else float("nan")


def _compute_rel_errors(
    batch_a: Dict[str, torch.Tensor],
    Vb_scaled: torch.Tensor,
    geom: str,
    spec,
) -> Dict[str, any]:
    mask = batch_a["mask_finite"] & torch.isfinite(Vb_scaled)
    assert mask.any(), "No finite collocation points to compare."

    Va = batch_a["V_gt"][mask]
    Vb = Vb_scaled[mask]

    denom = torch.abs(Va) + torch.abs(Vb) + torch.tensor(1e-9, dtype=Va.dtype)
    rel = torch.abs(Va - Vb) / denom

    is_bnd = batch_a["is_boundary"][mask]
    points = batch_a["X"][mask]

    rel_boundary = rel[is_bnd]
    rel_interior = rel[~is_bnd]

    meta = _geom_meta(spec, geom)
    cls_masks = _classify_masks(points, geom, meta)

    sample_points = {
        "near_surface": points[cls_masks["near_surface"]][:5].tolist(),
        "near_charge": points[cls_masks["near_charge"]][:5].tolist(),
        "interior": points[cls_masks["interior"]][:5].tolist(),
    }

    return {
        "rel_err_overall_max": _max_or_nan(rel),
        "rel_err_boundary_max": _max_or_nan(rel_boundary),
        "rel_err_interior_max": _max_or_nan(rel_interior),
        "rel_err_max_near_surface": _max_or_nan(rel[cls_masks["near_surface"]]),
        "rel_err_max_near_charge": _max_or_nan(rel[cls_masks["near_charge"]]),
        "rel_err_max_interior": _max_or_nan(rel[cls_masks["interior"]]),
        "sample_points": sample_points,
    }


def _run_one_geom(
    geom: str,
    h_values: List[float],
    n_points: int,
    ratio_boundary: float,
    seed: int,
) -> Dict[str, any]:
    if geom == "plane":
        builder = _build_plane_spec
    elif geom == "sphere":
        builder = _build_sphere_spec
    elif geom == "parallel_planes":
        builder = _build_parallel_planes_spec
    else:
        raise ValueError(f"Unknown geometry {geom}")

    geom_results: Dict[str, List[float]] = {
        "h_values": [],
        "n_panels": [],
        "bc_resid_linf": [],
        "gmres_iters": [],
        "gmres_resid": [],
        "rel_err_overall_max": [],
        "rel_err_boundary_max": [],
        "rel_err_interior_max": [],
        "rel_err_max_near_surface": [],
        "rel_err_max_near_charge": [],
        "rel_err_max_interior": [],
    }
    sample_points_record: Dict[str, List[List[List[float]]]] = {
        "near_surface": [],
        "near_charge": [],
        "interior": [],
    }

    for h in h_values:
        spec = builder()
        cfg = _build_cfg_for_h(h)
        logger = ConsoleLogger()
        bem_out = bem_solve(spec, cfg, logger)
        if "error" in bem_out:
            raise RuntimeError(f"BEM solve failed for geom={geom}, h={h}: {bem_out['error']}")

        mesh_stats = bem_out.get("mesh_stats", {})
        gmres_stats = bem_out.get("gmres_stats", {})

        rng1 = np.random.default_rng(seed)
        batch_a = make_collocation_batch_for_spec(
            spec=spec,
            n_points=n_points,
            ratio_boundary=ratio_boundary,
            supervision_mode="analytic",
            device=torch.device("cpu"),
            dtype=torch.float64,
            rng=rng1,
            geom_type=geom,
        )

        solution = bem_out.get("solution", None)
        if solution is None:
            raise RuntimeError("BEM solution missing from bem_solve output.")
        # Evaluate BEM at analytic points, keeping device/dtype from the solution.
        sol_device = getattr(solution, "_device", torch.device("cpu"))
        sol_dtype = getattr(solution, "_dtype", torch.float64)
        X = batch_a["X"].to(device=sol_device, dtype=sol_dtype)
        with torch.no_grad():
            Vb, _ = solution.eval_V_E_batched(X)
        Vb = Vb.detach().cpu().to(dtype=torch.float64) * EPS_0  # scale to reduced units

        rel_stats = _compute_rel_errors(batch_a, Vb, geom, spec)

        geom_results["h_values"].append(h)
        geom_results["n_panels"].append(mesh_stats.get("n_panels"))
        geom_results["bc_resid_linf"].append(mesh_stats.get("bc_residual_linf"))
        geom_results["gmres_iters"].append(gmres_stats.get("iters"))
        geom_results["gmres_resid"].append(gmres_stats.get("resid"))
        geom_results["rel_err_overall_max"].append(rel_stats["rel_err_overall_max"])
        geom_results["rel_err_boundary_max"].append(rel_stats["rel_err_boundary_max"])
        geom_results["rel_err_interior_max"].append(rel_stats["rel_err_interior_max"])
        geom_results["rel_err_max_near_surface"].append(rel_stats["rel_err_max_near_surface"])
        geom_results["rel_err_max_near_charge"].append(rel_stats["rel_err_max_near_charge"])
        geom_results["rel_err_max_interior"].append(rel_stats["rel_err_max_interior"])
        for k in sample_points_record.keys():
            sample_points_record[k].append(rel_stats["sample_points"].get(k, []))

    return {
        "geometry": geom,
        **geom_results,
        "sample_points": sample_points_record,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--geom", choices=["plane", "sphere", "parallel_planes", "all"], default="all")
    p.add_argument(
        "--h-values",
        nargs="+",
        type=float,
        default=[0.3, 0.2, 0.15, 0.1],
    )
    p.add_argument("--n-points", type=int, default=256)
    p.add_argument("--ratio-boundary", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/_agent_outputs/bem_h_convergence_scan.json"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    geoms = ["plane", "sphere", "parallel_planes"] if args.geom == "all" else [args.geom]
    results = []
    for g in geoms:
        print(f"[run] geom={g} h_values={args.h_values}")
        res = _run_one_geom(
            geom=g,
            h_values=args.h_values,
            n_points=args.n_points,
            ratio_boundary=args.ratio_boundary,
            seed=args.seed,
        )
        results.append(res)

    args_dict = vars(args).copy()
    args_dict["out"] = str(args.out)
    payload = {"args": args_dict, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
