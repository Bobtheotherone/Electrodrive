"""
Grand Challenge experiment runner for torus + parallel-plane specs.

Runs discovery with experimental bases/solver options and logs Stage-4-like
metrics to JSON files under runs/.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np

from electrodrive.images.search import discover_images, ImageSystem
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.learn.collocation import make_collocation_batch_for_spec, get_oracle_solution


def _sparsity90(weights: torch.Tensor) -> int:
    w_abs = torch.abs(weights.detach()).cpu()
    if w_abs.numel() == 0:
        return 0
    total = float(w_abs.sum())
    if total <= 0.0:
        return 0
    sorted_vals, _ = torch.sort(w_abs, descending=True)
    cumsum = torch.cumsum(sorted_vals, dim=0)
    thresh = 0.9 * total
    idx = torch.nonzero(cumsum >= thresh, as_tuple=False)
    if idx.numel() == 0:
        return int(w_abs.numel())
    return int(idx[0].item() + 1)


def _axis_points(spec: CanonicalSpec, n: int = 200) -> torch.Tensor:
    bbox = getattr(spec, "domain", {}).get("bbox", None)
    if bbox and len(bbox) == 2:
        z_min = float(bbox[0][2])
        z_max = float(bbox[1][2])
    else:
        z_min, z_max = -2.0, 2.0
    z = torch.linspace(z_min, z_max, n)
    zeros = torch.zeros_like(z)
    return torch.stack([zeros, zeros, z], dim=1)


@dataclass
class EvalResult:
    metrics: Dict[str, float]
    type_counts: Dict[str, int]


def evaluate_system(
    spec: CanonicalSpec,
    system: ImageSystem,
    n_eval: int = 2000,
    ratio_boundary: float = 0.7,
    belts: Optional[List[tuple[float, float]]] = None,
) -> EvalResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Make sure weights live on the eval device.
    system.weights = system.weights.to(device=device, dtype=dtype)

    batch = make_collocation_batch_for_spec(
        spec=spec,
        n_points=n_eval,
        ratio_boundary=ratio_boundary,
        supervision_mode="auto",
        device=device,
        dtype=dtype,
    )
    X = batch["X"]
    V_gt = batch["V_gt"]
    is_boundary = batch.get("is_boundary", torch.zeros(X.shape[0], device=device, dtype=torch.bool))
    mask_finite = batch.get("mask_finite", None)
    if mask_finite is not None and mask_finite.shape == (X.shape[0],):
        mask = mask_finite.to(device=device) & torch.isfinite(V_gt)
    else:
        mask = torch.isfinite(V_gt)

    X = X[mask]
    V_gt = V_gt[mask]
    is_boundary = is_boundary[mask]

    if X.numel() == 0:
        return EvalResult(metrics={}, type_counts={})

    V_pred = system.potential(X)
    diff = torch.abs(V_pred - V_gt)
    rel = diff / (torch.abs(V_gt) + 1e-12)

    boundary_mask = is_boundary.bool()
    interior_mask = ~boundary_mask

    belt_mask = torch.zeros_like(boundary_mask)
    if belts:
        r = torch.linalg.norm(X[:, :2], dim=1)
        z = X[:, 2]
        for r_t, z_t in belts:
            belt_mask |= (torch.abs(r - r_t) <= 0.05 * abs(r_t) + 1e-6) & (
                torch.abs(z - z_t) <= 0.1 * (abs(z_t) + 1e-6)
            )

    def _safe_mean(t: torch.Tensor) -> float:
        return float(t.mean().item()) if t.numel() > 0 else float("nan")

    metrics: Dict[str, float] = {}
    metrics["boundary_mae"] = _safe_mean(diff[boundary_mask])
    metrics["offaxis_mae"] = _safe_mean(diff[interior_mask])
    metrics["offaxis_rel"] = _safe_mean(rel[interior_mask])
    metrics["offaxis_belt_rel"] = _safe_mean(rel[belt_mask]) if belt_mask.any() else float("nan")

    # Axis metrics via oracle evaluation for stability.
    try:
        sol = get_oracle_solution(spec, mode="auto", bem_cfg={})  # type: ignore[arg-type]
        axis_pts = _axis_points(spec, n=256).to(device=device, dtype=dtype)
        if sol is not None and hasattr(sol, "eval_V_E_batched"):
            V_axis_gt, _ = sol.eval_V_E_batched(axis_pts)  # type: ignore[attr-defined]
        elif sol is not None and hasattr(sol, "eval"):
            V_axis_gt = sol.eval(axis_pts)  # type: ignore[attr-defined]
        else:
            V_axis_gt = None
        if V_axis_gt is not None:
            V_axis_pred = system.potential(axis_pts)
            axis_diff = torch.abs(V_axis_pred - V_axis_gt)
            axis_rel = axis_diff / (torch.abs(V_axis_gt) + 1e-12)
            metrics["axis_mae"] = _safe_mean(axis_diff)
            metrics["axis_rel"] = _safe_mean(axis_rel)
    except Exception:
        metrics["axis_mae"] = float("nan")
        metrics["axis_rel"] = float("nan")

    metrics["n_images"] = len(system.elements)
    metrics["sparsity90"] = _sparsity90(system.weights)
    type_counts = Counter(elem.type for elem in system.elements)
    metrics["type_counts"] = dict(type_counts)

    return EvalResult(metrics=metrics, type_counts=dict(type_counts))


def run_single(
    spec: CanonicalSpec,
    basis_types: List[str],
    n_max: int,
    reg_l1: float,
    restarts: int,
    per_type_reg: Optional[Dict[str, float]] = None,
    boundary_weight: Optional[float] = None,
    two_stage: bool = False,
    belts: Optional[List[tuple[float, float]]] = None,
) -> EvalResult:
    class _NullLogger:
        def info(self, *args, **kwargs):
            pass
        def warning(self, *args, **kwargs):
            pass
        def error(self, *args, **kwargs):
            pass

    logger = _NullLogger()
    system = discover_images(
        spec=spec,
        basis_types=basis_types,
        n_max=n_max,
        reg_l1=reg_l1,
        restarts=restarts,
        logger=logger,
        per_type_reg=per_type_reg,
        boundary_weight=boundary_weight,
        two_stage=two_stage,
    )
    return evaluate_system(spec, system, belts=belts)


def save_metrics(path: Path, metrics: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runs_root = root / "runs"

    specs = {
        "parallel": CanonicalSpec.from_json(json.load(open(root / "specs" / "parallel_planes_point.json"))),
        "torus_thin": CanonicalSpec.from_json(json.load(open(root / "specs" / "torus_axis_point_thin.json"))),
        "torus_mid": CanonicalSpec.from_json(json.load(open(root / "specs" / "torus_axis_point_mid.json"))),
    }

    results_parallel: List[Dict[str, object]] = []
    plane_grid = [
        (["point"], 4, 1e-2, 0, None, None),
        (["mirror_stack"], 8, 1e-3, 0, None, 0.7),
        (["point", "mirror_stack"], 8, 1e-3, 0, {"mirror_stack": 5e-4, "point": 1e-3}, 0.7),
        (["point", "mirror_stack"], 12, 3e-3, 0, {"mirror_stack": 1e-3, "point": 3e-3}, 0.9),
    ]
    for basis, n_max, reg, restarts, per_reg, bw in plane_grid:
        tag = f"parallel_{'_'.join(basis)}_n{n_max}_reg{reg}_bw{bw}"
        t0 = time.time()
        eval_res = run_single(
            specs["parallel"],
            basis,
            n_max=n_max,
            reg_l1=reg,
            restarts=restarts,
            per_type_reg=per_reg,
            boundary_weight=bw,
            two_stage=False,
        )
        rec: Dict[str, object] = {
            "spec": "parallel_planes_point",
            "run": tag,
            "basis_types": basis,
            "n_max": n_max,
            "reg_l1": reg,
            "boundary_weight": bw,
            "restarts": restarts,
            "metrics": eval_res.metrics,
            "type_counts": eval_res.type_counts,
            "elapsed_s": time.time() - t0,
        }
        results_parallel.append(rec)

    save_metrics(runs_root / "parallel_planes" / "stage4_metrics_grandchallenge.json", results_parallel)

    torus_results: List[Dict[str, object]] = []
    torus_grid = [
        (["point"], 8, 1e-2, 0, None, 0.5, False),
        (["point", "ring"], 12, 3e-3, 0, None, 0.5, False),
        (
            ["point", "poloidal_ring", "ring_ladder_inner"],
            12,
            3e-3,
            1,
            {"point": 4e-3, "poloidal_ring": 2e-3, "ring_ladder_inner": 2e-3},
            0.8,
            True,
        ),
        (
            ["point", "poloidal_ring", "ring_ladder_inner", "ring_ladder_outer", "toroidal_mode_cluster"],
            16,
            1e-3,
            1,
            {"point": 2e-3, "poloidal_ring": 8e-4, "ring_ladder_inner": 8e-4, "ring_ladder_outer": 8e-4, "toroidal_mode_cluster": 1e-3},
            0.8,
            True,
        ),
    ]

    for spec_key in ("torus_thin", "torus_mid"):
        for basis, n_max, reg, restarts, per_reg, bw, two_stage in torus_grid:
            tag = f"{spec_key}_{'_'.join(basis)}_n{n_max}_reg{reg}_bw{bw}_ts{two_stage}"
            t0 = time.time()
            eval_res = run_single(
                specs[spec_key],
                basis,
                n_max=n_max,
                reg_l1=reg,
                restarts=restarts,
                per_type_reg=per_reg,
                boundary_weight=bw,
                two_stage=two_stage,
            )
            rec: Dict[str, object] = {
                "spec": spec_key,
                "run": tag,
                "basis_types": basis,
                "n_max": n_max,
                "reg_l1": reg,
                "boundary_weight": bw,
                "restarts": restarts,
                "two_stage": two_stage,
                "metrics": eval_res.metrics,
                "type_counts": eval_res.type_counts,
                "elapsed_s": time.time() - t0,
            }
            torus_results.append(rec)

    save_metrics(runs_root / "torus" / "stage4_metrics_grandchallenge.json", torus_results)

    # Modes-enriched grids (family-aware)
    torus_mode_results: List[Dict[str, object]] = []
    belts_default = {
        "torus_thin": [(0.0, 0.0)],  # will be overridden below
        "torus_mid": [(0.0, 0.0)],
    }
    for spec_key in ("torus_thin", "torus_mid"):
        spec_obj = specs[spec_key]
        # derive R,a for belts
        torus = next(c for c in spec_obj.conductors if c.get("type") in ("torus", "toroid"))
        R = float(torus.get("major_radius", torus.get("radius", 1.0)))
        a = float(torus.get("minor_radius", 0.25 * R))
        belts = [
            (R - 0.5 * a, -0.2 * a),
            (R - 0.5 * a, 0.0),
            (R - 0.5 * a, 0.2 * a),
            (R, 0.0),
            (R + 0.5 * a, -0.2 * a),
            (R + 0.5 * a, 0.0),
            (R + 0.5 * a, 0.2 * a),
            (R + a, 0.0),
        ]
        belts_default[spec_key] = belts

    torus_mode_grid = []
    for spec_key in ("torus_thin", "torus_mid"):
        torus_mode_grid.extend(
            [
                (spec_key, ["point", "toroidal_eigen_mode_boundary"], 12, 1e-3, {"point": 2e-3, "toroidal_eigen_mode_boundary": 8e-4}, 0.8, False),
                (spec_key, ["point", "toroidal_eigen_mode_offaxis"], 12, 1e-3, {"point": 3e-3, "toroidal_eigen_mode_offaxis": 8e-4}, 0.5, False),
                (spec_key, ["point", "poloidal_ring", "toroidal_eigen_mode_boundary", "ring_ladder_inner"], 16, 1e-3, {"point": 3e-3, "poloidal_ring": 1e-3, "ring_ladder_inner": 1e-3, "toroidal_eigen_mode_boundary": 8e-4}, 0.8, True),
                (spec_key, ["point", "poloidal_ring", "toroidal_eigen_mode_offaxis", "ring_ladder_inner"], 16, 1e-3, {"point": 4e-3, "poloidal_ring": 1e-3, "ring_ladder_inner": 1e-3, "toroidal_eigen_mode_offaxis": 8e-4}, 0.5, True),
            ]
        )

    for spec_key, basis, n_max, reg, per_reg, bw, two_stage in torus_mode_grid:
        tag = f"{spec_key}_{'_'.join(basis)}_n{n_max}_reg{reg}_bw{bw}_ts{two_stage}"
        t0 = time.time()
        eval_res = run_single(
            specs[spec_key],
            basis,
            n_max=n_max,
            reg_l1=reg,
            restarts=1,
            per_type_reg=per_reg,
            boundary_weight=bw,
            two_stage=two_stage,
            belts=belts_default[spec_key],
        )
        rec: Dict[str, object] = {
            "spec": spec_key,
            "run": tag,
            "basis_types": basis,
            "n_max": n_max,
            "reg_l1": reg,
            "boundary_weight": bw,
            "restarts": 1,
            "two_stage": two_stage,
            "per_type_reg": per_reg,
            "metrics": eval_res.metrics,
            "type_counts": eval_res.type_counts,
            "elapsed_s": time.time() - t0,
        }
        torus_mode_results.append(rec)

    save_metrics(runs_root / "torus" / "stage4_metrics_modes_families.json", torus_mode_results)

    # Robust evaluation of top mode-family runs (best boundary / best off-axis-belt per spec)
    robust_results: List[Dict[str, object]] = []
    for spec_key in ("torus_thin", "torus_mid"):
        subset = [r for r in torus_mode_results if r.get("spec") == spec_key]
        if not subset:
            continue
        best_boundary = min(subset, key=lambda r: r["metrics"]["boundary_mae"])
        boundary_floor = best_boundary["metrics"]["boundary_mae"] * 3.0
        filt = [r for r in subset if r["metrics"]["boundary_mae"] <= boundary_floor]
        best_off = min(filt, key=lambda r: r["metrics"].get("offaxis_belt_rel", math.inf))
        for rec in {best_boundary["run"]: best_boundary, best_off["run"]: best_off}.values():
            system = discover_images(
                spec=specs[spec_key],
                basis_types=rec["basis_types"],
                n_max=rec["n_max"],
                reg_l1=rec["reg_l1"],
                restarts=rec["restarts"],
                logger=type("L", (), {"info": lambda *a, **k: None, "warning": lambda *a, **k: None, "error": lambda *a, **k: None})(),
                per_type_reg=rec.get("per_type_reg") or None,
                boundary_weight=rec.get("boundary_weight"),
                two_stage=rec.get("two_stage", False),
            )
            eval_res = evaluate_system(
                specs[spec_key],
                system,
                n_eval=6000,
                ratio_boundary=0.8,
                belts=belts_default[spec_key],
            )
            robust_results.append(
                {
                    "spec": spec_key,
                    "run": rec["run"],
                    "basis_types": rec["basis_types"],
                    "metrics": eval_res.metrics,
                    "type_counts": eval_res.type_counts,
                }
            )

    save_metrics(runs_root / "torus" / "stage4_metrics_modes_families_robust.json", robust_results)

    # Phase 4: inner-rim localized primitives
    print("[inner-rim] building belts and grid...")
    inner_rim_results: List[Dict[str, object]] = []
    belts_inner: Dict[str, List[tuple[float, float]]] = {}
    for spec_key in ("torus_thin", "torus_mid"):
        spec_obj = specs[spec_key]
        torus = next(c for c in spec_obj.conductors if c.get("type") in ("torus", "toroid"))
        R = float(torus.get("major_radius", torus.get("radius", 1.0)))
        a = float(torus.get("minor_radius", 0.25 * R))
        r_vals = [R - a, R - 0.75 * a, R - 0.5 * a]
        z_vals = [0.0, 0.2 * a, -0.2 * a]
        belts = []
        for r in r_vals:
            for z in z_vals:
                belts.append((r, z))
        belts_inner[spec_key] = belts

    inner_rim_grid = []
    for spec_key in ("torus_thin", "torus_mid"):
        inner_rim_grid.extend(
            [
                (spec_key, ["point", "toroidal_eigen_mode_boundary"], 12, 1e-3, {"point": 2e-3, "toroidal_eigen_mode_boundary": 8e-4}, 0.8, False),
                (spec_key, ["point", "poloidal_ring", "toroidal_eigen_mode_boundary", "ring_ladder_inner"], 16, 1e-3, {"point": 3e-3, "poloidal_ring": 1e-3, "ring_ladder_inner": 1e-3, "toroidal_eigen_mode_boundary": 8e-4}, 0.8, True),
                (spec_key, ["point", "toroidal_eigen_mode_boundary", "inner_rim_arc"], 12, 8e-4, {"point": 2e-3, "toroidal_eigen_mode_boundary": 8e-4, "inner_rim_arc": 8e-4}, 0.8, False),
                (spec_key, ["point", "toroidal_eigen_mode_boundary", "inner_rim_ribbon"], 12, 8e-4, {"point": 2e-3, "toroidal_eigen_mode_boundary": 8e-4, "inner_rim_ribbon": 8e-4}, 0.8, False),
                (spec_key, ["point", "toroidal_eigen_mode_boundary", "inner_rim_arc", "inner_rim_ribbon", "inner_patch_ring"], 16, 8e-4, {"point": 3e-3, "toroidal_eigen_mode_boundary": 8e-4, "inner_rim_arc": 8e-4, "inner_rim_ribbon": 8e-4, "inner_patch_ring": 1e-3}, 0.8, True),
                (spec_key, ["point", "inner_rim_arc", "inner_rim_ribbon", "inner_patch_ring"], 12, 1e-3, {"point": 4e-3, "inner_rim_arc": 1e-3, "inner_rim_ribbon": 1e-3, "inner_patch_ring": 1e-3}, 0.7, False),
            ]
        )

    print(f"[inner-rim] grid size: {len(inner_rim_grid)}")
    for spec_key, basis, n_max, reg, per_reg, bw, two_stage in inner_rim_grid:
        tag = f"{spec_key}_{'_'.join(basis)}_n{n_max}_reg{reg}_bw{bw}_ts{two_stage}_inner"
        t0 = time.time()
        print(f"[inner-rim] running {tag}")
        eval_res = run_single(
            specs[spec_key],
            basis,
            n_max=n_max,
            reg_l1=reg,
            restarts=1,
            per_type_reg=per_reg,
            boundary_weight=bw,
            two_stage=two_stage,
            belts=belts_inner[spec_key],
        )
        rec: Dict[str, object] = {
            "spec": spec_key,
            "run": tag,
            "basis_types": basis,
            "n_max": n_max,
            "reg_l1": reg,
            "boundary_weight": bw,
            "restarts": 1,
            "two_stage": two_stage,
            "per_type_reg": per_reg,
            "metrics": eval_res.metrics,
            "type_counts": eval_res.type_counts,
            "elapsed_s": time.time() - t0,
        }
        inner_rim_results.append(rec)

    print("[inner-rim] saving primary metrics")
    save_metrics(runs_root / "torus" / "stage4_metrics_inner_rim.json", inner_rim_results)

    # Robust inner-rim evaluation: best boundary and best belt under boundary filter
    inner_rim_robust: List[Dict[str, object]] = []
    for spec_key in ("torus_thin", "torus_mid"):
        subset = [r for r in inner_rim_results if r.get("spec") == spec_key]
        if not subset:
            continue
        best_boundary = min(subset, key=lambda r: r["metrics"]["boundary_mae"])
        boundary_floor = best_boundary["metrics"]["boundary_mae"] * 3.0
        filt = [r for r in subset if r["metrics"]["boundary_mae"] <= boundary_floor]
        best_off = min(filt, key=lambda r: r["metrics"].get("offaxis_belt_rel", math.inf))
        for rec in {best_boundary["run"]: best_boundary, best_off["run"]: best_off}.values():
            system = discover_images(
                spec=specs[spec_key],
                basis_types=rec["basis_types"],
                n_max=rec["n_max"],
                reg_l1=rec["reg_l1"],
                restarts=rec["restarts"],
                logger=type("L", (), {"info": lambda *a, **k: None, "warning": lambda *a, **k: None, "error": lambda *a, **k: None})(),
                per_type_reg=rec.get("per_type_reg") or None,
                boundary_weight=rec.get("boundary_weight"),
                two_stage=rec.get("two_stage", False),
            )
            eval_res = evaluate_system(
                specs[spec_key],
                system,
                n_eval=6000,
                ratio_boundary=0.8,
                belts=belts_inner[spec_key],
            )
            inner_rim_robust.append(
                {
                    "spec": spec_key,
                    "run": rec["run"],
                    "basis_types": rec["basis_types"],
                    "metrics": eval_res.metrics,
                    "type_counts": eval_res.type_counts,
                }
            )

    print("[inner-rim] saving robust metrics")
    save_metrics(runs_root / "torus" / "stage4_metrics_inner_rim_robust.json", inner_rim_robust)

    # Refined inner-rim arcs (rich_inner_rim) for near-contact thin torus
    print("[inner-rim] running refined thin rich_inner_rim grid...")
    inner_rim_refined: List[Dict[str, object]] = []
    refined_grid = [
        (
            "torus_thin",
            ["point", "toroidal_eigen_mode_boundary", "inner_rim_arc", "rich_inner_rim"],
            12,
            8e-4,
            {"point": 2e-3, "toroidal_eigen_mode_boundary": 8e-4, "inner_rim_arc": 8e-4},
            0.8,
            False,
        )
    ]
    for spec_key, basis, n_max, reg, per_reg, bw, two_stage in refined_grid:
        tag = f"{spec_key}_{'_'.join(basis)}_n{n_max}_reg{reg}_bw{bw}_ts{two_stage}_rich_inner"
        t0 = time.time()
        eval_res = run_single(
            specs[spec_key],
            basis,
            n_max=n_max,
            reg_l1=reg,
            restarts=1,
            per_type_reg=per_reg,
            boundary_weight=bw,
            two_stage=two_stage,
            belts=belts_inner[spec_key],
        )
        inner_rim_refined.append(
            {
                "spec": spec_key,
                "run": tag,
                "basis_types": basis,
                "n_max": n_max,
                "reg_l1": reg,
                "boundary_weight": bw,
                "restarts": 1,
                "two_stage": two_stage,
                "per_type_reg": per_reg,
                "metrics": eval_res.metrics,
                "type_counts": eval_res.type_counts,
                "elapsed_s": time.time() - t0,
            }
        )
    save_metrics(runs_root / "torus" / "stage4_metrics_inner_rim_refined.json", inner_rim_refined)



if __name__ == "__main__":
    main()
