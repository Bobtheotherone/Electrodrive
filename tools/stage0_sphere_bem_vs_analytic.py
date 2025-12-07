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
from electrodrive.orchestration.spec_registry import stage0_sphere_external_path
from electrodrive.utils.config import BEMConfig, K_E
from electrodrive.utils.logging import JsonlLogger

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = (REPO_ROOT / "runs/stage0/sphere_bem_calibration").resolve()
FNO_EXPORT_ROOT = (REPO_ROOT / "runs/stage0_sphere_bem_vs_analytic").resolve()
SPEC_PATH = stage0_sphere_external_path()


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


@dataclass
class FNOExportConfig:
    enabled: bool = False
    out_root: Path = FNO_EXPORT_ROOT
    n_theta: int = 64
    n_phi: int = 128


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
    float_precision: str = "fp64",
) -> Dict[str, Any]:
    cfg = BEMConfig()
    cfg.fp64 = str(float_precision).lower() != "fp32"
    cfg.use_gpu = torch.cuda.is_available()

    # Strong defaults mirroring the Stage-0 calibration and oracle config.
    cfg.max_refine_passes = max(3, int(getattr(cfg, "max_refine_passes", 3)))
    if hasattr(cfg, "min_refine_passes"):
        setattr(cfg, "min_refine_passes", max(1, int(getattr(cfg, "min_refine_passes", 1))))

    cfg.use_near_quadrature = True
    cfg.near_quadrature_order = 2  # supported orders: 1 or 2
    cfg.use_near_quadrature_matvec = True
    cfg.near_quadrature_distance_factor = 2.0

    try:
        cfg.gmres_tol = min(float(getattr(cfg, "gmres_tol", 5e-8)), 1e-9)
    except Exception:
        cfg.gmres_tol = 1e-9

    if cfg_overrides:
        for k, v in cfg_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

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
    """
    Sample points along the symmetry axis, excluding a small neighborhood
    around the source and (optionally) points inside the conductor.
    """
    cz = float(center[2])
    axis_span = float(span) if span is not None else max(2.5 * radius, abs(z_charge) + radius)
    zs = np.linspace(cz - axis_span, cz + axis_span, n_axis, dtype=np.float64)

    # Exclude a small window around the source location to avoid 1/r spikes.
    mask = np.abs(zs - float(z_charge)) >= float(exclude_tol)
    zs = zs[mask]

    # Optionally exclude points inside the sphere radius.
    if exclude_inside_radius is not None:
        mask_out = np.abs(zs - cz) >= float(exclude_inside_radius)
        zs = zs[mask_out]

    if zs.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    return np.stack(
        [
            np.full_like(zs, float(center[0])),
            np.full_like(zs, float(center[1])),
            zs,
        ],
        axis=1,
    )


def derive_boundary_grid(n_boundary: int, default_theta: int, default_phi: int) -> Tuple[int, int]:
    """
    Convert a target number of boundary samples into a (theta, phi) grid.

    Keeps the original defaults if the request is invalid.
    """
    try:
        n = int(n_boundary)
    except Exception:
        return default_theta, default_phi
    if n <= 0:
        return default_theta, default_phi
    n_theta = max(4, int(round(math.sqrt(n / 2.0))))
    n_phi = max(8, int(math.ceil(n / max(1, n_theta))))
    return n_theta, n_phi


def resolve_sample_config(args: argparse.Namespace) -> SampleConfig:
    n_theta, n_phi = args.theta, args.phi
    if getattr(args, "n_boundary", None) is not None:
        n_theta, n_phi = derive_boundary_grid(args.n_boundary, args.theta, args.phi)
    n_axis = args.axis_n
    if getattr(args, "n_interior", None) is not None and args.n_interior is not None:
        n_axis = int(args.n_interior)
    return SampleConfig(n_theta=n_theta, n_phi=n_phi, n_axis=n_axis)


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
    """
    Compute energy via induced potential at the real charges:

        U = 0.5 * sum_k q_k * phi_induced(x_k)
    """
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
        q = float(ch.get("q", 0.0))
        energy += 0.5 * q * float(phi_ind)
    return float(energy)


def free_space_potential_at_points(spec: CanonicalSpec, pts: np.ndarray) -> np.ndarray:
    """
    Free-space potential from all point charges in `spec` evaluated at `pts`.
    """
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


def export_fno_boundary_grid(
    solution: BEMSolution,
    analytic,
    sphere: SphereSetup,
    spec: CanonicalSpec,
    fno_cfg: FNOExportConfig,
    run_name: str,
    float_precision: str,
) -> Path:
    """
    Export spherical boundary potentials on a fixed grid for FNO training.
    """
    out_dir = (fno_cfg.out_root / run_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = np.float32 if float_precision == "fp32" else np.float64
    theta = np.linspace(0.0, math.pi, fno_cfg.n_theta, dtype=dtype)
    phi = np.linspace(0.0, 2.0 * math.pi, fno_cfg.n_phi, endpoint=False, dtype=dtype)
    theta_g, phi_g = np.meshgrid(theta, phi, indexing="ij")

    s = np.sin(theta_g)
    x = s * np.cos(phi_g)
    y = s * np.sin(phi_g)
    z = np.cos(theta_g)
    pts = np.stack([x, y, z], axis=-1) * float(sphere.radius)
    pts = pts + np.asarray(sphere.center, dtype=dtype)
    pts_flat = pts.reshape(-1, 3)

    V_bem = eval_bem_batch(solution, pts_flat).reshape(theta_g.shape)
    V_analytic = eval_analytic_batch(analytic, pts_flat).reshape(theta_g.shape)
    V_bem = V_bem.astype(dtype, copy=False)
    V_analytic = V_analytic.astype(dtype, copy=False)

    np.save(out_dir / "theta.npy", theta_g.astype(dtype, copy=False))
    np.save(out_dir / "phi.npy", phi_g.astype(dtype, copy=False))
    np.save(out_dir / "V.npy", V_bem)
    np.save(out_dir / "V_bem.npy", V_bem)
    np.save(out_dir / "V_analytic.npy", V_analytic)

    charge = next((c for c in spec.charges if c.get("type") == "point"), None)
    q = float(charge.get("q", float("nan"))) if charge else float("nan")
    z0_abs = float(charge.get("pos", [0.0, 0.0, 0.0])[2]) if charge else float("nan")
    z0_rel = z0_abs - float(sphere.center[2]) if charge else float("nan")
    meta = {
        "run_name": run_name,
        "q": q,
        "z0_abs": z0_abs,
        "z0_rel": z0_rel,
        "radius": float(sphere.radius),
        "center": list(map(float, sphere.center)),
        "n_theta": int(fno_cfg.n_theta),
        "n_phi": int(fno_cfg.n_phi),
        "float_precision": float_precision,
        "bem_dtype": str(getattr(solution, "_dtype", "")),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def run_single_z(
    base_spec: CanonicalSpec,
    z0: float,
    out_dir: Path,
    sample_cfg: SampleConfig,
    bem_cfg_overrides: Dict[str, Any] | None = None,
    float_precision: str = "fp64",
    fno_export: FNOExportConfig | None = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with JsonlLogger(out_dir) as logger:
        spec = make_spec_with_z(base_spec, z0)
        analytic = analytic_solution_for_spec(spec)
        sphere = sphere_from_spec(spec)
        bem_out = bem_solution_for_spec(
            spec,
            logger,
            cfg_overrides=bem_cfg_overrides,
            float_precision=float_precision,
        )
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
                ],
                dtype=np.float64,
            )

        V_an_bc = eval_analytic_batch(analytic, boundary_pts)
        V_bem_bc = eval_bem_batch(solution, boundary_pts)
        V_an_axis = eval_analytic_batch(analytic, axis_pts)
        V_bem_axis = eval_bem_batch(solution, axis_pts)

        # Boundary error: scale by free-space potential, not by |V_an| ~ 0.
        V_free_bc = free_space_potential_at_points(spec, boundary_pts)
        bc_abs = np.abs(V_bem_bc - V_an_bc)
        bc_scale = np.maximum(np.abs(V_free_bc), 1e-9)
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

        if fno_export and fno_export.enabled:
            run_name = out_dir.name or f"z_{str(z0).replace('.', 'p')}"
            fno_dir = export_fno_boundary_grid(
                solution,
                analytic,
                sphere,
                spec,
                fno_export,
                run_name,
                float_precision=float_precision,
            )
            metrics["fno_grid_dir"] = str(fno_dir)

        logger.info("sphere_bem_vs_analytic", **metrics)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics


def run_calibration(
    z_values: Iterable[float],
    out_root: Path = OUT_ROOT,
    sample_cfg: SampleConfig | None = None,
    bem_cfg_overrides: Dict[str, Any] | None = None,
    float_precision: str = "fp64",
    fno_export: FNOExportConfig | None = None,
) -> List[Dict[str, Any]]:
    cfg = sample_cfg or SampleConfig()
    base_spec = load_base_spec()
    out_root.mkdir(parents=True, exist_ok=True)
    z_list = list(z_values)
    results: List[Dict[str, Any]] = []
    for z0 in z_list:
        safe = str(z0).replace(".", "p")
        run_dir = out_root / f"z_{safe}"
        metrics = run_single_z(
            base_spec,
            float(z0),
            run_dir,
            cfg,
            bem_cfg_overrides,
            float_precision=float_precision,
            fno_export=fno_export,
        )
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
        "--n-boundary",
        type=int,
        default=None,
        help="Target number of boundary samples; overrides --theta/--phi with a derived grid.",
    )
    parser.add_argument(
        "--axis-n",
        type=int,
        default=128,
        help="Number of axis samples for diagnostics.",
    )
    parser.add_argument(
        "--n-interior",
        type=int,
        default=None,
        help="Override number of interior (axis) samples; maps to --axis-n when set.",
    )
    parser.add_argument(
        "--float-precision",
        choices=("fp64", "fp32"),
        default="fp64",
        help="Floating-point precision for the BEM solve (default: fp64).",
    )
    parser.add_argument(
        "--export-fno-grid",
        action="store_true",
        help="If set, export spherical boundary potentials on a fixed grid for FNO training.",
    )
    parser.add_argument(
        "--fno-theta",
        type=int,
        default=64,
        help="Theta resolution for --export-fno-grid output.",
    )
    parser.add_argument(
        "--fno-phi",
        type=int,
        default=128,
        help="Phi resolution for --export-fno-grid output.",
    )
    parser.add_argument(
        "--fno-out",
        type=Path,
        default=FNO_EXPORT_ROOT,
        help="Output directory root for --export-fno-grid artifacts.",
    )
    parser.add_argument(
        "--disable-near-matvec",
        action="store_true",
        help="If set, turns off use_near_quadrature_matvec in the BEM solve (diagnostic).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    z_values = list(args.z)
    if args.include_internal:
        z_values_internal = [0.1, 0.3, 0.5, 0.8]
        z_values.extend(z_values_internal)
    sample_cfg = resolve_sample_config(args)
    bem_overrides: Dict[str, Any] = {}
    if args.disable_near_matvec:
        bem_overrides["use_near_quadrature_matvec"] = False
    fno_cfg = FNOExportConfig(
        enabled=bool(args.export_fno_grid),
        out_root=args.fno_out,
        n_theta=args.fno_theta,
        n_phi=args.fno_phi,
    )
    run_calibration(
        z_values=z_values,
        out_root=args.out,
        sample_cfg=sample_cfg,
        bem_cfg_overrides=bem_overrides or None,
        float_precision=args.float_precision,
        fno_export=fno_cfg if args.export_fno_grid else None,
    )


if __name__ == "__main__":
    main()
