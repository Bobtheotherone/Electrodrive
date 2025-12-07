#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from electrodrive.core.bem import BEMSolution, bem_solve
from electrodrive.core.bem_kernel import bem_potential_targets
from electrodrive.core.images import potential_sphere_grounded
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.config import BEMConfig, K_E
from electrodrive.utils.logging import JsonlLogger

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = (REPO_ROOT / "runs/stage0/sphere_bem_calibration").resolve()
SPEC_PATH = REPO_ROOT / "specs/sphere_axis_point_external.json"


@dataclass
class SphereSetup:
    center: Tuple[float, float, float]
    radius: float


@dataclass
class SampleConfig:
    """Sampling controls for diagnostics."""

    n_theta: int = 24
    n_phi: int = 48
    n_axis: int = 128
    axis_span: float | None = None
    axis_exclude_tol: float = 1e-4


def load_base_spec(path: Path = SPEC_PATH) -> CanonicalSpec:
    raw = json.loads(path.read_text())
    return CanonicalSpec.from_json(raw)


def make_spec_with_z(spec: CanonicalSpec, z0: float) -> CanonicalSpec:
    data = spec.to_json()
    for ch in data.get("charges", []):
        if ch.get("type") == "point":
            ch["pos"] = [0.0, 0.0, float(z0)]
    return CanonicalSpec.from_json(data)


def sphere_from_spec(spec: CanonicalSpec) -> SphereSetup:
    sph = next(c for c in spec.conductors if c.get("type") == "sphere")
    center = tuple(map(float, sph.get("center", [0.0, 0.0, 0.0])))
    radius = float(sph.get("radius", 1.0))
    return SphereSetup(center=center, radius=radius)


def analytic_solution_for_spec(spec: CanonicalSpec):
    ch = next(c for c in spec.charges if c.get("type") == "point")
    q = float(ch["q"])
    r0 = tuple(map(float, ch["pos"]))
    sph = sphere_from_spec(spec)
    return potential_sphere_grounded(q, r0, sph.center, sph.radius)


def bem_solution_for_spec(
    spec: CanonicalSpec,
    logger: JsonlLogger,
    cfg_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = BEMConfig()
    cfg.fp64 = True
    cfg.use_gpu = torch.cuda.is_available()
    cfg.max_refine_passes = 3
    cfg.use_near_quadrature = True
    cfg.near_quadrature_order = 2  # supported orders: 1 or 2
    cfg.use_near_quadrature_matvec = True
    cfg.near_quadrature_distance_factor = 2.0
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    try:
        cfg.gmres_tol = min(float(getattr(cfg, "gmres_tol", 5e-8)), 1e-9)
    except Exception:
        cfg.gmres_tol = 1e-9
    out = bem_solve(spec, cfg, logger, differentiable=False)
    if "error" in out:
        raise RuntimeError(out["error"])
    return out


def sample_sphere_surface(
    center: Sequence[float],
    radius: float,
    n_theta: int,
    n_phi: int,
) -> np.ndarray:
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


def sample_axis_points(
    center: Sequence[float],
    radius: float,
    n_axis: int,
    span: float | None,
    z_charge: float,
    exclude_tol: float = 1e-4,
    exclude_inside_radius: float | None = None,
) -> np.ndarray:
    axis_span = float(span) if span is not None else max(2.5 * radius, abs(z_charge) + radius)
    zs = np.linspace(center[2] - axis_span, center[2] + axis_span, n_axis)
    mask = np.abs(zs - z_charge) > exclude_tol
    zs = zs[mask]
    if exclude_inside_radius is not None:
        mask_out = np.abs(zs - center[2]) >= exclude_inside_radius
        zs = zs[mask_out]
    return np.stack([np.full_like(zs, center[0]), np.full_like(zs, center[1]), zs], axis=1)


def eval_analytic_batch(analytic, pts: np.ndarray) -> np.ndarray:
    return np.asarray([analytic.eval(tuple(p)) for p in pts], dtype=np.float64)


def eval_bem_batch(solution: BEMSolution, pts: np.ndarray) -> np.ndarray:
    device = solution._device
    dtype = solution._dtype
    P = torch.as_tensor(pts, device=device, dtype=dtype)
    with torch.no_grad():
        V, _ = solution.eval_V_E_batched(P)
    return V.detach().cpu().numpy()


def error_stats(V_ref: np.ndarray, V_test: np.ndarray) -> Dict[str, float]:
    abs_err = np.abs(V_test - V_ref)
    rel_err = abs_err / np.maximum(np.abs(V_ref), 1e-12)
    return {
        "mae": float(np.mean(abs_err)),
        "max": float(np.max(abs_err)),
        "rel_mean": float(np.mean(rel_err)),
        "rel_max": float(np.max(rel_err)),
    }


def energy_from_image_meta(meta: Dict[str, Any]) -> float:
    try:
        q = float(meta.get("charge", 1.0))
        q_img = float(meta["image_charge"])
        r0 = np.asarray(meta["r0"], dtype=np.float64)
        r_img = np.asarray(meta["image_pos"], dtype=np.float64)
    except Exception:
        return float("nan")
    dist = float(np.linalg.norm(r0 - r_img))
    if dist <= 0.0:
        return float("nan")
    return 0.5 * q * (K_E * q_img / dist)


def energy_from_sigma(solution: BEMSolution, spec: CanonicalSpec) -> float:
    charges = [c for c in spec.charges if c.get("type") == "point"]
    if not charges:
        return float("nan")
    device = solution._device
    dtype = solution._dtype
    targets = torch.tensor([c["pos"] for c in charges], device=device, dtype=dtype)
    with torch.no_grad():
        V_ind = bem_potential_targets(
            targets=targets,
            src_centroids=solution._C,
            areas=solution._A,
            sigma=solution._S,
            tile_size=int(getattr(solution, "_tile", 4096)),
        )
    energy = 0.0
    for ch, phi_ind in zip(charges, V_ind.tolist()):
        energy += 0.5 * float(ch.get("q", 0.0)) * float(phi_ind)
    return float(energy)


def free_space_potential_at_points(spec: CanonicalSpec, pts: np.ndarray) -> np.ndarray:
    vals = np.zeros(pts.shape[0], dtype=np.float64)
    for ch in spec.charges:
        if ch.get("type") != "point":
            continue
        try:
            q = float(ch.get("q", 0.0))
            pos = np.asarray(ch.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
        except Exception:
            continue
        r = np.linalg.norm(pts - pos[None, :], axis=1)
        vals += K_E * q / np.maximum(r, 1e-12)
    return vals


def run_single_z(
    base_spec: CanonicalSpec,
    z0: float,
    out_dir: Path,
    sample_cfg: SampleConfig,
    bem_cfg_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with JsonlLogger(out_dir) as logger:
        spec = make_spec_with_z(base_spec, z0)
        analytic = analytic_solution_for_spec(spec)
        sphere = sphere_from_spec(spec)
        bem_out = bem_solution_for_spec(spec, logger, cfg_overrides=bem_cfg_overrides)
        solution: BEMSolution = bem_out["solution"]

        boundary_pts = sample_sphere_surface(
            sphere.center, sphere.radius, sample_cfg.n_theta, sample_cfg.n_phi
        )
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
            axis_pts = np.array(
                [
                    [
                        sphere.center[0],
                        sphere.center[1],
                        sphere.center[2] + sphere.radius * 1.5,
                    ]
                ]
            )

        V_an_bc = eval_analytic_batch(analytic, boundary_pts)
        V_bem_bc = eval_bem_batch(solution, boundary_pts)
        V_an_axis = eval_analytic_batch(analytic, axis_pts)
        V_bem_axis = eval_bem_batch(solution, axis_pts)

        V_free_bc = free_space_potential_at_points(spec, boundary_pts)
        bc_abs = np.abs(V_bem_bc - V_an_bc)
        bc_scale = np.maximum(np.abs(V_an_bc), np.abs(V_free_bc))
        bc_scale = np.maximum(bc_scale, 1e-6)
        bc_err = {
            "mae": float(np.mean(bc_abs)),
            "max": float(np.max(bc_abs)),
            "rel_mean": float(np.mean(bc_abs / bc_scale)),
            "rel_max": float(np.max(bc_abs / bc_scale)),
        }
        axis_err = error_stats(V_an_axis, V_bem_axis)

        energy_true = energy_from_image_meta(analytic.meta)
        energy_bem = energy_from_sigma(solution, spec)
        denom = max(1.0, abs(energy_true), abs(energy_bem))
        energy_rel = abs(energy_bem - energy_true) / denom

        mesh_stats = bem_out.get("mesh_stats", {})
        gmres_stats = bem_out.get("gmres_stats", {})

        metrics: Dict[str, Any] = {
            "z0": float(z0),
            "bc_mae": bc_err["mae"],
            "bc_max": bc_err["max"],
            "bc_rel_mean": bc_err["rel_mean"],
            "bc_rel_max": bc_err["rel_max"],
            "axis_mae": axis_err["mae"],
            "axis_max": axis_err["max"],
            "axis_rel_mean": axis_err["rel_mean"],
            "axis_rel_max": axis_err["rel_max"],
            "energy_analytic": energy_true,
            "energy_bem": energy_bem,
            "energy_rel_err": float(energy_rel),
            "n_panels": int(mesh_stats.get("n_panels", 0)),
            "device": str(solution._device),
            "dtype": str(solution._dtype),
            "mesh_stats": mesh_stats,
            "gmres_stats": gmres_stats,
        }

        logger.info("sphere_bem_vs_analytic", **metrics)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics


def run_calibration(
    z_values: Iterable[float],
    out_root: Path = OUT_ROOT,
    sample_cfg: SampleConfig | None = None,
    bem_cfg_overrides: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    cfg = sample_cfg or SampleConfig()
    base_spec = load_base_spec()
    out_root.mkdir(parents=True, exist_ok=True)
    z_list = list(z_values)
    results: List[Dict[str, Any]] = []
    for z0 in z_list:
        safe = str(z0).replace(".", "p")
        run_dir = out_root / f"z_{safe}"
        metrics = run_single_z(base_spec, float(z0), run_dir, cfg, bem_cfg_overrides)
        results.append(metrics)
    summary = {"z_values": z_list, "results": results}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-0 sphere BEM vs analytic calibration.")
    parser.add_argument(
        "--z",
        nargs="+",
        type=float,
        default=[1.1, 1.25, 1.5, 2.0, 3.0],
        help="List of external source z positions (absolute distance on axis).",
    )
    parser.add_argument(
        "--include-internal",
        action="store_true",
        help="Also sweep internal source positions.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_ROOT,
        help="Output directory for metrics and logs.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    z_values = list(args.z)
    if args.include_internal:
        z_values_internal = [0.1, 0.3, 0.5, 0.8]
        z_values.extend(z_values_internal)
    sample_cfg = SampleConfig(n_theta=args.theta, n_phi=args.phi, n_axis=args.axis_n)
    run_calibration(z_values=z_values, out_root=args.out, sample_cfg=sample_cfg)


if __name__ == "__main__":
    main()
