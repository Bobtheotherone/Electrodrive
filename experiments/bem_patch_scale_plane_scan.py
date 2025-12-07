#!/usr/bin/env python
"""
Patch-scale scan for plane and parallel-planes geometries.

For each patch scale s, sets EDE_BEM_PLANE_PATCH_SCALE=s, runs a single-pass
BEM solve, evaluates BEM vs analytic on matched collocation points, and logs
rel_err splits plus basic BEM stats (n_panels, bc_resid_linf, patch_L).
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from electrodrive.core.bem import bem_solve
from electrodrive.fmm3d.logging_utils import ConsoleLogger
from electrodrive.learn.collocation import make_collocation_batch_for_spec
from electrodrive.utils.config import BEMConfig, EPS_0

from tests.test_bem_quadrature import (
    _build_parallel_planes_spec,
    _build_plane_spec,
    BEM_TEST_ORACLE_CONFIG,
)


def _build_cfg(h: float) -> BEMConfig:
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

    near_charge = torch.zeros(points.shape[0], dtype=torch.bool)
    if "charge_pos" in meta:
        charge_pos = meta["charge_pos"]
        dist = torch.linalg.norm(points - charge_pos[None, :], dim=1)
        near_charge = dist < 0.25

    interior = ~(near_surface | near_charge)
    return {"near_surface": near_surface, "near_charge": near_charge, "interior": interior}


def _max_or_nan(t: torch.Tensor) -> float:
    return float(t.max().item()) if t.numel() > 0 else float("nan")


def _compute_rel_errors(batch_a, Vb_scaled, geom: str, spec) -> Dict[str, any]:
    mask = batch_a["mask_finite"] & torch.isfinite(Vb_scaled)
    assert mask.any(), "No finite collocation points to compare."
    Va = batch_a["V_gt"][mask]
    Vb = Vb_scaled[mask]
    denom = torch.abs(Va) + torch.abs(Vb) + torch.tensor(1e-9, dtype=Va.dtype)
    rel = torch.abs(Va - Vb) / denom
    is_bnd = batch_a["is_boundary"][mask]
    points = batch_a["X"][mask]

    meta = _geom_meta(spec, geom)
    cls_masks = _classify_masks(points, geom, meta)

    return {
        "rel_err_overall_max": _max_or_nan(rel),
        "rel_err_boundary_max": _max_or_nan(rel[is_bnd]),
        "rel_err_interior_max": _max_or_nan(rel[~is_bnd]),
        "rel_err_max_near_surface": _max_or_nan(rel[cls_masks["near_surface"]]),
        "rel_err_max_near_charge": _max_or_nan(rel[cls_masks["near_charge"]]),
        "rel_err_max_interior": _max_or_nan(rel[cls_masks["interior"]]),
    }


def _run_one_geom(
    geom: str,
    patch_scales: List[float],
    h: float,
    n_points: int,
    ratio_boundary: float,
    seed: int,
) -> Dict[str, any]:
    if geom == "plane":
        builder = _build_plane_spec
    elif geom == "parallel_planes":
        builder = _build_parallel_planes_spec
    else:
        raise ValueError(f"Unsupported geometry {geom}")

    out: Dict[str, List[float]] = {
        "patch_scales": [],
        "n_panels": [],
        "patch_L": [],
        "bc_resid_linf": [],
        "gmres_iters": [],
        "gmres_resid": [],
        "rel_err_overall_max": [],
        "rel_err_boundary_max": [],
        "rel_err_interior_max": [],
        "rel_err_max_near_surface": [],
        "rel_err_max_interior": [],
    }

    for s in patch_scales:
        spec = builder()
        cfg = _build_cfg(h)
        logger = ConsoleLogger()

        old_scale = os.environ.get("EDE_BEM_PLANE_PATCH_SCALE")
        os.environ["EDE_BEM_PLANE_PATCH_SCALE"] = str(s)
        try:
            bem_out = bem_solve(spec, cfg, logger)
        finally:
            if old_scale is None:
                os.environ.pop("EDE_BEM_PLANE_PATCH_SCALE", None)
            else:
                os.environ["EDE_BEM_PLANE_PATCH_SCALE"] = old_scale

        if "error" in bem_out:
            raise RuntimeError(f"BEM solve failed for geom={geom}, patch_scale={s}: {bem_out['error']}")

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
        sol_device = getattr(solution, "_device", torch.device("cpu"))
        sol_dtype = getattr(solution, "_dtype", torch.float64)
        X = batch_a["X"].to(device=sol_device, dtype=sol_dtype)
        with torch.no_grad():
            Vb, _ = solution.eval_V_E_batched(X)
        Vb = Vb.detach().cpu().to(dtype=torch.float64) * EPS_0

        rel_stats = _compute_rel_errors(batch_a, Vb, geom, spec)

        out["patch_scales"].append(s)
        out["n_panels"].append(mesh_stats.get("n_panels"))
        out["patch_L"].append(mesh_stats.get("patch_L"))
        out["bc_resid_linf"].append(mesh_stats.get("bc_residual_linf"))
        out["gmres_iters"].append(gmres_stats.get("iters"))
        out["gmres_resid"].append(gmres_stats.get("resid"))
        out["rel_err_overall_max"].append(rel_stats["rel_err_overall_max"])
        out["rel_err_boundary_max"].append(rel_stats["rel_err_boundary_max"])
        out["rel_err_interior_max"].append(rel_stats["rel_err_interior_max"])
        out["rel_err_max_near_surface"].append(rel_stats["rel_err_max_near_surface"])
        out["rel_err_max_interior"].append(rel_stats["rel_err_max_interior"])

    return {"geometry": geom, **out}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--geom", choices=["plane", "parallel_planes", "both"], default="both")
    p.add_argument(
        "--patch-scales",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 1.5, 2.0],
    )
    p.add_argument("--h", type=float, default=0.2)
    p.add_argument("--n-points", type=int, default=256)
    p.add_argument("--ratio-boundary", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/_agent_outputs/bem_patch_scale_plane_scan.json"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    geoms = ["plane", "parallel_planes"] if args.geom == "both" else [args.geom]

    results = []
    for g in geoms:
        print(f"[run] geom={g} patch_scales={args.patch_scales}")
        res = _run_one_geom(
            geom=g,
            patch_scales=args.patch_scales,
            h=args.h,
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
