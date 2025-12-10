#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import datetime
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Optional

import numpy as np
import torch

from electrodrive.images.search import ImageSystem, AugLagrangeConfig, discover_images
from electrodrive.images.basis import generate_candidate_basis
from electrodrive.images.learned_solver import LISTALayer
from electrodrive.orchestration.spec_registry import stage0_sphere_external_path
from electrodrive.images.weight_modes import (
    compute_weight_modes,
    export_weight_mode_bundle,
    fit_symbolic_modes,
    fit_quality_ok,
    load_symbolic_fits,
    load_svd_bundle,
    predict_weights_from_modes,
    render_summary,
    spectral_gap_ok,
)
from electrodrive.utils.logging import JsonlLogger
from tools.stage0_sphere_bem_vs_analytic import (
    SampleConfig,
    analytic_solution_for_spec,
    error_stats,
    free_space_potential_at_points,
    load_base_spec,
    make_spec_with_z,
    sample_axis_points,
    sample_sphere_surface,
    sphere_from_spec,
    eval_analytic_batch,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "runs/stage0_sphere_axis_moi"


def _parse_basis_arg(raw_basis: Sequence[str] | str) -> List[str]:
    if isinstance(raw_basis, str):
        parts = [p.strip() for p in raw_basis.split(",") if p.strip()]
    else:
        parts: List[str] = []
        for entry in raw_basis:
            parts.extend([p.strip() for p in str(entry).split(",") if p.strip()])
    return parts if parts else ["axis_point"]


def _parse_optional_bool(val: Optional[str]) -> Optional[bool]:
    if val is None:
        return None
    v = str(val).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def _safe_tag(tag: Optional[str]) -> Optional[str]:
    if tag is None:
        return None
    clean = tag.strip().replace(" ", "_").replace("/", "_")
    return clean if clean else None


def _config_label(
    basis_types: Sequence[str],
    solver: str,
    operator_mode: Optional[bool],
    aug_boundary: bool,
    boundary_weight: Optional[float],
    experiment_tag: Optional[str],
) -> str:
    basis_label = "+".join(basis_types) if basis_types else "axis_point"
    solver_label = solver.lower().strip() or "ista"
    op_label = "op" if operator_mode else "dense" if operator_mode is False else "auto"
    bc_label = "AL" if aug_boundary else "BW" if boundary_weight else "mix"
    parts = [p for p in [experiment_tag, basis_label, solver_label, op_label, bc_label] if p]
    return "-".join(parts)


def _safe_ratio(num: float, den: float) -> float:
    try:
        if not math.isfinite(num) or not math.isfinite(den) or den == 0.0:
            return float("nan")
        return float(num / den)
    except Exception:
        return float("nan")


def _aggregate_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}

    def _collect(key: str) -> List[float]:
        vals: List[float] = []
        for r in results:
            if key in r:
                try:
                    vals.append(float(r[key]))
                except Exception:
                    continue
        return vals

    def _nanmean(vals: List[float]) -> float:
        arr = np.asarray(vals, dtype=np.float64)
        return float(np.nanmean(arr)) if arr.size else float("nan")

    def _nanmax(vals: List[float]) -> float:
        arr = np.asarray(vals, dtype=np.float64)
        return float(np.nanmax(arr)) if arr.size else float("nan")

    return {
        "z0_grid": [float(r.get("z0", float("nan"))) for r in results],
        "bc_abs_mean_avg": _nanmean(_collect("bc_abs_mean")),
        "bc_abs_max_max": _nanmax(_collect("bc_abs_max")),
        "axis_abs_mean_avg": _nanmean(_collect("axis_abs_mean")),
        "axis_abs_max_max": _nanmax(_collect("axis_abs_max")),
        "avg_n_images": _nanmean(_collect("n_images")),
        "max_n_images": _nanmax(_collect("n_images")),
        "avg_weight_norm_l1": _nanmean(_collect("weight_norm_l1")),
        "avg_weight_norm_l2": _nanmean(_collect("weight_norm_l2")),
    }


def _eval_system(system: ImageSystem, pts: Sequence[Sequence[float]], dtype: torch.dtype) -> torch.Tensor:
    device = system.device if hasattr(system, "device") else torch.device("cpu")
    P = torch.as_tensor(pts, device=device, dtype=dtype)
    with torch.no_grad():
        return system.potential(P)


def _kelvin_prediction(z0: float, center_z: float, radius: float) -> Dict[str, float]:
    s = abs(z0 - center_z)
    if s <= 0.0:
        return {"q": float("nan"), "z": float("nan")}
    q_img = -radius / s
    z_img = center_z + (radius * radius / (s * s)) * (z0 - center_z)
    return {"q": float(q_img), "z": float(z_img)}


def _dominant_image(system: ImageSystem) -> Dict[str, float]:
    if not system.elements or system.weights.numel() == 0:
        return {"q": float("nan"), "z": float("nan")}
    weights = system.weights.detach().cpu()
    idx = int(torch.argmax(torch.abs(weights)).item())
    elem = system.elements[idx]
    pos = elem.params.get("position", None)
    if pos is None:
        return {"q": float(weights[idx].item()), "z": float("nan")}
    pos_np = pos.detach().cpu().numpy().reshape(-1)
    return {"q": float(weights[idx].item()), "z": float(pos_np[2])}


def _boundary_error_with_free_space(
    V_ref: np.ndarray, V_test: np.ndarray, V_free: np.ndarray
) -> Dict[str, float]:
    bc_abs = np.abs(V_test - V_ref)
    scale = np.maximum(np.abs(V_free), 1e-9)
    return {
        "mae": float(np.mean(bc_abs)),
        "max": float(np.max(bc_abs)),
        "rel_mean": float(np.mean(bc_abs / scale)),
        "rel_max": float(np.max(bc_abs / scale)),
    }


def run_single(
    base_spec,
    z0: float,
    n_max: int,
    reg_l1: float,
    restarts: int,
    sample_cfg: SampleConfig,
    out_dir: Path,
    basis_types: Sequence[str],
    adaptive_rounds: int,
    solver: str,
    lista_checkpoint: Path | None,
    weight_prior: Optional[np.ndarray] = None,
    lambda_weight_prior: float = 0.0,
    weight_prior_label: Optional[str] = None,
    operator_mode: Optional[bool] = None,
    boundary_weight: Optional[float] = None,
    aug_boundary: bool = False,
    n_collocation: Optional[int] = None,
    ratio_boundary: Optional[float] = None,
    model_checkpoint: Optional[Path] = None,
    experiment_label: Optional[str] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = make_spec_with_z(base_spec, z0)
    analytic = analytic_solution_for_spec(spec)
    sphere = sphere_from_spec(spec)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    lista_model = None
    solver_mode = solver.strip().lower()
    if solver_mode == "lista":
        n_cand = max(1, n_max * 4)
        candidates = generate_candidate_basis(
            spec,
            basis_types=list(basis_types),
            n_candidates=n_cand,
            device=device,
            dtype=dtype,
        )
        K = max(1, len(candidates))
        lista_model = LISTALayer(K=K, n_steps=5, dense_threshold=512, init_theta=1e-4)
        if lista_checkpoint is not None:
            if lista_checkpoint.exists():
                try:
                    state = torch.load(lista_checkpoint, map_location=device)
                    lista_model.load_state_dict(state)
                except Exception as exc:
                    print(f"[WARN] Failed to load LISTA checkpoint: {exc}")
            else:
                print(f"[WARN] LISTA checkpoint not found: {lista_checkpoint}")
        try:
            lista_model = lista_model.to(device=device, dtype=dtype).eval()
        except Exception:
            pass

    aug_cfg = AugLagrangeConfig() if aug_boundary else None
    model_ckpt_str = str(model_checkpoint) if model_checkpoint is not None else None

    with JsonlLogger(out_dir) as logger:
        system = discover_images(
            spec=spec,
            basis_types=list(basis_types),
            n_max=n_max,
            reg_l1=reg_l1,
            restarts=restarts,
            logger=logger,
            solver=solver_mode,
            lista_model=lista_model,
            adaptive_collocation_rounds=adaptive_rounds,
            weight_prior=weight_prior,
            lambda_weight_prior=lambda_weight_prior,
            weight_prior_label=weight_prior_label,
            operator_mode=operator_mode,
            boundary_weight=boundary_weight,
            aug_lagrange=aug_cfg,
            n_points_override=n_collocation,
            ratio_boundary_override=ratio_boundary,
            model_checkpoint=model_ckpt_str,
            subtract_physical_potential=True,
        )

        dtype = torch.float64
        bc_pts = sample_sphere_surface(sphere.center, sphere.radius, sample_cfg.n_theta, sample_cfg.n_phi)
        axis_pts = sample_axis_points(
            sphere.center,
            sphere.radius,
            sample_cfg.n_axis,
            sample_cfg.axis_span,
            z0,
            exclude_tol=sample_cfg.axis_exclude_tol,
            exclude_inside_radius=sphere.radius + sample_cfg.axis_exclude_tol,
        )
        if axis_pts.shape[0] == 0:
            # Fallback: one point safely outside the sphere on the axis
            axis_pts = np.array(
                [[sphere.center[0], sphere.center[1], sphere.center[2] + 1.5 * sphere.radius]],
                dtype=np.float64,
            )

        V_an_bc = eval_analytic_batch(analytic, bc_pts)
        V_an_axis = eval_analytic_batch(analytic, axis_pts)

        V_img_bc = _eval_system(system, bc_pts, dtype=dtype).detach().cpu().numpy()
        V_img_axis = _eval_system(system, axis_pts, dtype=dtype).detach().cpu().numpy()

        V_free_bc = free_space_potential_at_points(spec, bc_pts)
        V_free_axis = free_space_potential_at_points(spec, axis_pts)
        V_total_bc = V_img_bc + V_free_bc
        V_total_axis = V_img_axis + V_free_axis
        bc_err = _boundary_error_with_free_space(V_an_bc, V_total_bc, V_free_bc)
        axis_err = error_stats(V_an_axis, V_total_axis)

        kelvin = _kelvin_prediction(z0, sphere.center[2], sphere.radius)
        dominant = _dominant_image(system)

        weights_np = system.weights.detach().cpu().numpy().reshape(-1)
        weight_norm_l1 = float(np.sum(np.abs(weights_np))) if weights_np.size else 0.0
        weight_norm_l2 = float(np.linalg.norm(weights_np)) if weights_np.size else 0.0
        weight_norm_inf = float(np.max(np.abs(weights_np))) if weights_np.size else 0.0

        def _rel(a: float, b: float) -> float:
            denom = max(1.0, abs(b))
            return abs(a - b) / denom

        def _serialize_position(elem: Any) -> Any:
            pos = elem.params.get("position", None)
            if pos is None:
                return None
            try:
                return [float(x) for x in pos.detach().cpu().view(-1).tolist()]
            except Exception:
                return None

        metrics: Dict[str, Any] = {
            "z0": float(z0),
            "n_max": int(n_max),
            "reg_l1": float(reg_l1),
            "restarts": int(restarts),
            "basis_types": list(basis_types),
            "bc_abs_mean": bc_err["mae"],
            "bc_abs_max": bc_err["max"],
            "bc_rel_mean": bc_err["rel_mean"],
            "bc_rel_max": bc_err["rel_max"],
            "axis_abs_mean": axis_err["mae"],
            "axis_abs_max": axis_err["max"],
            "axis_rel_mean": axis_err["rel_mean"],
            "axis_rel_max": axis_err["rel_max"],
            "kelvin_q": kelvin["q"],
            "kelvin_z": kelvin["z"],
            "disc_q": dominant["q"],
            "disc_z": dominant["z"],
            "rel_err_q": _rel(dominant["q"], kelvin["q"]) if not math.isnan(kelvin["q"]) else float("nan"),
            "rel_err_z": _rel(dominant["z"], kelvin["z"]) if not math.isnan(kelvin["z"]) else float("nan"),
            "n_images": len(system.elements),
            "weights": system.weights.detach().cpu().tolist(),
            "adaptive_rounds": int(adaptive_rounds),
            "weight_mode_prior": bool(weight_prior is not None),
            "lambda_weight_mode": float(lambda_weight_prior),
            "operator_mode": operator_mode,
            "solver": solver_mode,
            "boundary_weight": boundary_weight,
            "aug_boundary": aug_boundary,
            "n_collocation_override": n_collocation,
            "ratio_boundary_override": ratio_boundary,
            "model_checkpoint": model_ckpt_str,
            "lista_checkpoint": str(lista_checkpoint) if lista_checkpoint is not None else None,
            "weight_norm_l1": weight_norm_l1,
            "weight_norm_l2": weight_norm_l2,
            "weight_norm_inf": weight_norm_inf,
            "image_positions": [_serialize_position(e) for e in system.elements],
            "axis_samples": int(axis_pts.shape[0]),
            "experiment_label": experiment_label,
        }
        metrics["bc_mae"] = metrics["bc_abs_mean"]
        metrics["bc_max"] = metrics["bc_abs_max"]
        metrics["axis_mae"] = metrics["axis_abs_mean"]
        metrics["axis_max"] = metrics["axis_abs_max"]

        logger.info("sphere_axis_moi", **metrics)

        sys_json = {
            "elements": [elem.serialize() for elem in system.elements],
            "weights": system.weights.detach().cpu().tolist(),
        }
        (out_dir / "discovered_system.json").write_text(json.dumps(sys_json, indent=2), encoding="utf-8")
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics


def run_sweep(
    z_values: Iterable[float],
    n_max_list: Iterable[int],
    reg_l1: float,
    restarts: int,
    sample_cfg: SampleConfig,
    out_root: Path = OUT_ROOT,
    basis_types: Sequence[str] = ("sphere_kelvin_ladder",),
    adaptive_rounds: int = 2,
    solver: str = "ista",
    lista_checkpoint: Path | None = None,
    operator_mode: Optional[bool] = None,
    boundary_weight: Optional[float] = None,
    aug_boundary: bool = False,
    n_collocation: Optional[int] = None,
    ratio_boundary: Optional[float] = None,
    model_checkpoint: Optional[Path] = None,
    experiment_tag: Optional[str] = None,
    max_rank: int = 3,
    max_poly_degree: int = 4,
    mode_bundle: Optional[Dict[str, Any]] = None,
    mode_fits: Optional[List[Dict[str, Any]]] = None,
    lambda_weight_mode: float = 0.0,
    spectral_gap_thresh: float = 0.1,
    rel_rmse_thresh: float = 0.2,
    vault: bool = False,
    vault_slug: Optional[str] = None,
) -> List[Dict[str, Any]]:
    base_spec = load_base_spec()
    sphere = sphere_from_spec(base_spec)
    adaptive_rounds = max(1, int(adaptive_rounds))
    basis_list = list(basis_types)
    tag_clean = _safe_tag(experiment_tag)
    config_id = _config_label(basis_list, solver, operator_mode, aug_boundary, boundary_weight, tag_clean)
    out_base = out_root.parent / f"{out_root.name}_{tag_clean}" if tag_clean else out_root
    config_root = out_base / config_id
    config_root.mkdir(parents=True, exist_ok=True)
    z_list = [float(z) for z in z_values]
    n_list = [int(n) for n in n_max_list]

    controller_active = (
        mode_bundle is not None
        and mode_fits is not None
        and spectral_gap_ok(mode_bundle.get("S", []), rank=max_rank, thresh=spectral_gap_thresh)
        and fit_quality_ok(mode_fits, rel_rmse_tol=rel_rmse_thresh)
    )
    weights_by_nmax: Dict[int, List[tuple[float, torch.Tensor]]] = {}
    results: List[Dict[str, Any]] = []

    for z0 in z_list:
        for n_max in n_list:
            run_dir = config_root / f"z{str(z0).replace('.', 'p')}_n{n_max}"
            w_prior = None
            if controller_active:
                w_prior = predict_weights_from_modes(float(z0), mode_bundle, mode_fits, max_rank=max_rank)  # type: ignore[arg-type]
            metrics = run_single(
                base_spec,
                float(z0),
                int(n_max),
                reg_l1,
                restarts,
                sample_cfg,
                run_dir,
                basis_list,
                adaptive_rounds,
                solver,
                lista_checkpoint,
                weight_prior=w_prior,
                lambda_weight_prior=lambda_weight_mode,
                weight_prior_label="weight_modes" if w_prior is not None else None,
                operator_mode=operator_mode,
                boundary_weight=boundary_weight,
                aug_boundary=aug_boundary,
                n_collocation=n_collocation,
                ratio_boundary=ratio_boundary,
                model_checkpoint=model_checkpoint,
                experiment_label=config_id,
            )
            metrics["run_dir"] = str(run_dir)
            results.append(metrics)
            w_tensor = torch.as_tensor(metrics.get("weights", []), device="cpu", dtype=torch.float32).view(-1)
            weights_by_nmax.setdefault(int(n_max), []).append((float(z0), w_tensor))

    summary = {
        "config_label": config_id,
        "config": {
            "basis_types": basis_list,
            "solver": solver,
            "operator_mode": operator_mode,
            "aug_boundary": aug_boundary,
            "boundary_weight": boundary_weight,
            "adaptive_rounds": adaptive_rounds,
            "n_collocation_override": n_collocation,
            "ratio_boundary_override": ratio_boundary,
            "reg_l1": reg_l1,
            "restarts": restarts,
            "n_max_list": n_list,
            "z_values": z_list,
            "lista_checkpoint": str(lista_checkpoint) if lista_checkpoint is not None else None,
            "model_checkpoint": str(model_checkpoint) if model_checkpoint is not None else None,
            "experiment_tag": tag_clean,
            "out_dir": str(config_root),
        },
        "results": results,
        "aggregates": _aggregate_results(results),
    }
    summary_path = config_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    symbolic_outputs: List[Dict[str, Any]] = []
    for n_val, pairs in weights_by_nmax.items():
        pairs_sorted = sorted(pairs, key=lambda kv: kv[0])
        z_grid = [p[0] for p in pairs_sorted]
        weight_list = [p[1] for p in pairs_sorted]
        bundle = compute_weight_modes(weight_list, z_grid, max_rank=max_rank)
        fits = fit_symbolic_modes(z_grid, bundle.mode_curves, max_rank=max_rank, max_poly_degree=max_poly_degree)
        sym_dir = config_root / f"symbolic_n{n_val}"
        export_weight_mode_bundle(
            sym_dir,
            bundle,
            fits,
            extra_metrics={
                "n_max": int(n_val),
                "reg_l1": reg_l1,
                "basis_types": basis_list,
                "z_grid": z_grid,
            },
        )
        geometry_desc = f"Stage-0 grounded sphere (radius={sphere.radius:.3g}, center_z={sphere.center[2]:.3g})"
        research_wishlist = [
            "Relate leading mode to Kelvin inversion law and quantify deviations near the surface.",
            "Prove spectral gap along axis using boundary integral operator symmetry.",
            "Map rational fit poles/zeros to candidate ladder positions for controller design.",
        ]
        summary_text = render_summary(
            label=f"Stage-0 axis sweep weight modes (n_max={n_val})",
            geometry=geometry_desc,
            basis=basis_list,
            z_grid=z_grid,
            bundle=bundle,
            fits=fits,
            research_wishlist=research_wishlist,
        )
        (sym_dir / "summary.md").write_text(summary_text, encoding="utf-8")
        symbolic_outputs.append({"n_max": int(n_val), "dir": str(sym_dir)})

        if vault or vault_slug:
            slug = vault_slug or f"stage0_axis_weight_modes_n{n_val}_{ts}"
            vault_dir = Path("the_vault") / slug
            vault_dir.mkdir(parents=True, exist_ok=True)
            for fname in ["weights_vs_axis.npy", "svd_modes.npy", "symbolic_fits.json", "metrics.json", "summary.md", "summary.json"]:
                src = sym_dir / fname
                if src.exists():
                    shutil.copy(src, vault_dir / fname)
        try:
            shutil.copy(stage0_sphere_external_path(), vault_dir / "spec_stage0.json")
        except Exception:
            pass

    if symbolic_outputs:
        summary["symbolic_outputs"] = symbolic_outputs
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Convenience mirror: surface key artifacts at the user-provided out_root
    if config_root != out_root:
        try:
            out_root.mkdir(parents=True, exist_ok=True)
            shutil.copy(summary_path, out_root / "summary.json")
            for sym_entry in symbolic_outputs:
                src_dir = Path(sym_entry.get("dir", ""))
                if not src_dir.exists():
                    continue
                dst_dir = out_root / src_dir.name
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        except Exception:
            # Mirroring should never break the main sweep; continue silently.
            pass

    return results


def run_parity_sweep(
    reg_l1: float,
    restarts: int,
    sample_cfg: SampleConfig,
    out_root: Path,
    basis_types: Sequence[str],
    adaptive_rounds: int,
    solver: str,
    lista_checkpoint: Path | None,
    boundary_weight: Optional[float],
    aug_boundary: bool,
    n_collocation: Optional[int],
    ratio_boundary: Optional[float],
    model_checkpoint: Optional[Path],
    lambda_weight_mode: float,
    experiment_tag: Optional[str],
) -> Path:
    """Run a fixed dense/operator parity sweep for Stage-0 (anomaly tracker)."""
    base_spec = load_base_spec()
    z_grid = [1.25, 1.5, 2.0]
    n_grid = [1, 3]
    parity_dir = (out_root / "parity_check").resolve()
    parity_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for z0 in z_grid:
        for n_max in n_grid:
            metrics_dense = run_single(
                base_spec,
                float(z0),
                int(n_max),
                reg_l1,
                restarts,
                sample_cfg,
                parity_dir / f"z{str(z0).replace('.', 'p')}_n{n_max}_dense",
                basis_types,
                adaptive_rounds,
                solver,
                lista_checkpoint,
                weight_prior=None,
                lambda_weight_prior=lambda_weight_mode,
                weight_prior_label=None,
                operator_mode=False,
                boundary_weight=boundary_weight,
                aug_boundary=aug_boundary,
                n_collocation=n_collocation,
                ratio_boundary=ratio_boundary,
                model_checkpoint=model_checkpoint,
                experiment_label=f"{experiment_tag or 'parity'}_dense",
            )
            metrics_op = run_single(
                base_spec,
                float(z0),
                int(n_max),
                reg_l1,
                restarts,
                sample_cfg,
                parity_dir / f"z{str(z0).replace('.', 'p')}_n{n_max}_op",
                basis_types,
                adaptive_rounds,
                solver,
                lista_checkpoint,
                weight_prior=None,
                lambda_weight_prior=lambda_weight_mode,
                weight_prior_label=None,
                operator_mode=True,
                boundary_weight=boundary_weight,
                aug_boundary=aug_boundary,
                n_collocation=n_collocation,
                ratio_boundary=ratio_boundary,
                model_checkpoint=model_checkpoint,
                experiment_label=f"{experiment_tag or 'parity'}_op",
            )
            rows.append(
                {
                    "z0": float(z0),
                    "n_max": int(n_max),
                    "bc_rel_mean_dense": metrics_dense.get("bc_rel_mean"),
                    "bc_rel_mean_op": metrics_op.get("bc_rel_mean"),
                    "axis_rel_mean_dense": metrics_dense.get("axis_rel_mean"),
                    "axis_rel_mean_op": metrics_op.get("axis_rel_mean"),
                    "weight_norm_l2_dense": metrics_dense.get("weight_norm_l2"),
                    "weight_norm_l2_op": metrics_op.get("weight_norm_l2"),
                    "bc_rel_mean_ratio": _safe_ratio(metrics_op.get("bc_rel_mean", float("nan")), metrics_dense.get("bc_rel_mean", float("nan"))),
                    "axis_rel_mean_ratio": _safe_ratio(metrics_op.get("axis_rel_mean", float("nan")), metrics_dense.get("axis_rel_mean", float("nan"))),
                }
            )

    parity_summary = {
        "z_grid": z_grid,
        "n_max_list": n_grid,
        "rows": rows,
        "parity_factor_target": 10.0,
    }
    summary_path = parity_dir / "parity_summary.json"
    summary_path.write_text(json.dumps(parity_summary, indent=2), encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-0 MOI discovery for grounded sphere.")
    parser.add_argument(
        "--z",
        nargs="+",
        type=float,
        default=[1.25, 1.5, 2.0],
        help="Axis source positions to sweep.",
    )
    parser.add_argument(
        "--nmax",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Candidate image counts to try.",
    )
    parser.add_argument(
        "--reg-l1",
        type=float,
        default=1e-4,
        help="L1 regularization weight for discovery.",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=3,
        help="Number of random restarts for discovery.",
    )
    parser.add_argument(
        "--basis",
        nargs="+",
        default=["axis_point"],
        help='Basis types (comma-separated within or space-separated across args). Default "axis_point"; try "sphere_kelvin_ladder".',
    )
    parser.add_argument(
        "--theta",
        type=int,
        default=24,
        help="Number of polar samples on the sphere surface.",
    )
    parser.add_argument(
        "--phi",
        type=int,
        default=48,
        help="Number of azimuthal samples on the sphere surface.",
    )
    parser.add_argument(
        "--axis-n",
        type=int,
        default=128,
        help="Number of axis samples for diagnostics.",
    )
    parser.add_argument(
        "--axis-span",
        type=float,
        default=None,
        help="Half-span of axis sampling window (defaults to max(2.5R, |z0|+R)).",
    )
    parser.add_argument(
        "--axis-exclude-tol",
        type=float,
        default=1e-4,
        help="Exclusion window around the source on the axis.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_ROOT,
        help="Output root directory (config label appended automatically).",
    )
    parser.add_argument(
        "--experiment-tag",
        type=str,
        default=None,
        help="Optional tag to group runs (appends to output root and config label).",
    )
    parser.add_argument(
        "--parity-sweep",
        action="store_true",
        help="Also run a fixed dense/operator parity sweep (z0 in {1.25,1.5,2.0}, n_max in {1,3}).",
    )
    parser.add_argument(
        "--adaptive-collocation-rounds",
        type=int,
        default=2,
        help="Number of oracle collocation rounds (1 disables adaptive refinement).",
    )
    parser.add_argument(
        "--collocation-points",
        type=int,
        default=None,
        help="Override collocation point count (None uses default from discover_images).",
    )
    parser.add_argument(
        "--ratio-boundary",
        type=float,
        default=None,
        help="Override boundary/interior ratio for collocation (None uses default).",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["ista", "lista"],
        default="ista",
        help="Sparse solver to use for discovery.",
    )
    parser.add_argument(
        "--operator-mode",
        type=str,
        default=None,
        help="Force operator-mode assembly (true/false). Defaults to auto/discover_images behavior.",
    )
    parser.add_argument(
        "--lista-checkpoint",
        type=Path,
        default=None,
        help="Optional path to a LISTA checkpoint to load.",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint for learned components used inside discover_images.",
    )
    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=None,
        help="Optional boundary weight for penalty/ratio modes (ignored when --aug-boundary is set).",
    )
    parser.add_argument(
        "--aug-boundary",
        action="store_true",
        help="Enable augmented-Lagrangian boundary enforcement (overrides boundary-weight).",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=3,
        help="Top singular modes to keep for symbolic fits and controller.",
    )
    parser.add_argument(
        "--max-poly-degree",
        type=int,
        default=4,
        help="Max polynomial degree when fitting mode curves.",
    )
    parser.add_argument(
        "--use-weight-modes",
        action="store_true",
        help="Enable weight-mode controller if mode-dir contains fits.",
    )
    parser.add_argument(
        "--mode-dir",
        type=Path,
        default=None,
        help="Directory with svd_modes.npy and symbolic_fits.json to seed controller.",
    )
    parser.add_argument(
        "--lambda-weight-mode",
        type=float,
        default=0.0,
        help="Quadratic prior strength pulling weights toward controller prediction.",
    )
    parser.add_argument(
        "--spectral-gap-thresh",
        type=float,
        default=0.1,
        help="Spectral gap threshold to trust controller modes.",
    )
    parser.add_argument(
        "--rel-rmse-thresh",
        type=float,
        default=0.2,
        help="Max relative RMSE allowed for mode fits when enabling controller.",
    )
    parser.add_argument(
        "--vault",
        action="store_true",
        help="Copy symbolic artifacts into the_vault for audit.",
    )
    parser.add_argument(
        "--vault-slug",
        type=str,
        default=None,
        help="Optional the_vault folder name (timestamped if omitted).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    basis_types = _parse_basis_arg(args.basis)
    operator_mode = _parse_optional_bool(args.operator_mode)
    if args.operator_mode is not None and operator_mode is None:
        print(f"[WARN] Unrecognized --operator-mode value '{args.operator_mode}'; using auto/default.")
    sample_cfg = SampleConfig(
        n_theta=args.theta,
        n_phi=args.phi,
        n_axis=args.axis_n,
        axis_span=args.axis_span,
        axis_exclude_tol=args.axis_exclude_tol,
    )
    controller_bundle = None
    controller_fits = None
    if args.use_weight_modes and args.mode_dir is not None:
        controller_bundle = load_svd_bundle(args.mode_dir / "svd_modes.npy")
        _, controller_fits_loaded = load_symbolic_fits(args.mode_dir / "symbolic_fits.json")
        if controller_bundle and controller_fits_loaded:
            if spectral_gap_ok(controller_bundle.get("S", []), rank=args.max_rank, thresh=args.spectral_gap_thresh) and fit_quality_ok(
                controller_fits_loaded, rel_rmse_tol=args.rel_rmse_thresh
            ):
                controller_fits = controller_fits_loaded
                print(f"[controller] Loaded Stage-0 weight modes from {args.mode_dir}")
            else:
                print("[controller] Skipping controller due to spectral gap or fit quality.")

    run_sweep(
        z_values=args.z,
        n_max_list=args.nmax,
        reg_l1=args.reg_l1,
        restarts=args.restarts,
        sample_cfg=sample_cfg,
        out_root=args.out,
        basis_types=basis_types,
        adaptive_rounds=args.adaptive_collocation_rounds,
        solver=args.solver,
        lista_checkpoint=args.lista_checkpoint,
        operator_mode=operator_mode,
        boundary_weight=args.boundary_weight,
        aug_boundary=args.aug_boundary,
        n_collocation=args.collocation_points,
        ratio_boundary=args.ratio_boundary,
        model_checkpoint=args.model_checkpoint,
        experiment_tag=args.experiment_tag,
        max_rank=args.max_rank,
        max_poly_degree=args.max_poly_degree,
        mode_bundle=controller_bundle,
        mode_fits=controller_fits,
        lambda_weight_mode=args.lambda_weight_mode,
        spectral_gap_thresh=args.spectral_gap_thresh,
        rel_rmse_thresh=args.rel_rmse_thresh,
        vault=args.vault,
        vault_slug=args.vault_slug,
    )
    if args.parity_sweep:
        parity_path = run_parity_sweep(
            reg_l1=args.reg_l1,
            restarts=args.restarts,
            sample_cfg=sample_cfg,
            out_root=args.out,
            basis_types=basis_types,
            adaptive_rounds=args.adaptive_collocation_rounds,
            solver=args.solver,
            lista_checkpoint=args.lista_checkpoint,
            boundary_weight=args.boundary_weight,
            aug_boundary=args.aug_boundary,
            n_collocation=args.collocation_points,
            ratio_boundary=args.ratio_boundary,
            model_checkpoint=args.model_checkpoint,
            lambda_weight_mode=args.lambda_weight_mode,
            experiment_tag=args.experiment_tag,
        )
        print(f"[parity] dense/operator summary written to {parity_path}")


if __name__ == "__main__":
    main()
