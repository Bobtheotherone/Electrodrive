#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch

from electrodrive.images.search import ImageSystem, discover_images
from electrodrive.utils.logging import JsonlLogger
from tools.stage0_sphere_bem_vs_analytic import (
    SampleConfig,
    analytic_solution_for_spec,
    error_stats,
    load_base_spec,
    make_spec_with_z,
    sample_axis_points,
    sample_sphere_surface,
    sphere_from_spec,
    eval_analytic_batch,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "runs/stage0/sphere_moi"


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


def run_single(
    base_spec,
    z0: float,
    n_max: int,
    reg_l1: float,
    restarts: int,
    sample_cfg: SampleConfig,
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = make_spec_with_z(base_spec, z0)
    analytic = analytic_solution_for_spec(spec)
    sphere = sphere_from_spec(spec)

    with JsonlLogger(out_dir) as logger:
        system = discover_images(
            spec=spec,
            basis_types=["axis_point"],
            n_max=n_max,
            reg_l1=reg_l1,
            restarts=restarts,
            logger=logger,
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
        )
        if axis_pts.shape[0] == 0:
            axis_pts = bc_pts[:1]

        V_an_bc = eval_analytic_batch(analytic, bc_pts)
        V_an_axis = eval_analytic_batch(analytic, axis_pts)

        V_img_bc = _eval_system(system, bc_pts, dtype=dtype).detach().cpu().numpy()
        V_img_axis = _eval_system(system, axis_pts, dtype=dtype).detach().cpu().numpy()

        bc_err = error_stats(V_an_bc, V_img_bc)
        axis_err = error_stats(V_an_axis, V_img_axis)

        kelvin = _kelvin_prediction(z0, sphere.center[2], sphere.radius)
        dominant = _dominant_image(system)

        def _rel(a: float, b: float) -> float:
            denom = max(1.0, abs(b))
            return abs(a - b) / denom

        metrics: Dict[str, Any] = {
            "z0": float(z0),
            "n_max": int(n_max),
            "reg_l1": float(reg_l1),
            "restarts": int(restarts),
            "bc_rel_mean": bc_err["rel_mean"],
            "bc_rel_max": bc_err["rel_max"],
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
        }

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
) -> List[Dict[str, Any]]:
    base_spec = load_base_spec()
    results: List[Dict[str, Any]] = []
    for z0 in z_values:
        for n_max in n_max_list:
            run_dir = out_root / f"z{str(z0).replace('.', 'p')}_n{n_max}"
            metrics = run_single(
                base_spec,
                float(z0),
                int(n_max),
                reg_l1,
                restarts,
                sample_cfg,
                run_dir,
            )
            results.append(metrics)
    summary = {"results": results}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return results


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
        "--out",
        type=Path,
        default=OUT_ROOT,
        help="Output root directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    sample_cfg = SampleConfig()
    run_sweep(
        z_values=args.z,
        n_max_list=args.nmax,
        reg_l1=args.reg_l1,
        restarts=args.restarts,
        sample_cfg=sample_cfg,
        out_root=args.out,
    )


if __name__ == "__main__":
    main()
