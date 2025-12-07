"""
Axis sweep for the fixed mid-torus 2-ring + 4-point geometry.

Keeps geometry from mid_bem_highres_trial02 fixed and re-solves only weights
as the source charge moves along the symmetry axis. Runs Stage-4 style metrics,
high-res BEM diagnostics, and an optional FMM matvec sanity check.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from electrodrive.images.io import load_image_system, save_image_system
from electrodrive.images.search import assemble_basis_matrix, solve_l1_ista, ImageSystem
from electrodrive.learn.collocation import make_collocation_batch_for_spec, get_oracle_solution
from electrodrive.orchestration.parser import CanonicalSpec
from tools.mid_torus_bem_fmm_refine import (
    RingParams,
    PointParams,
    load_seed_params,
    build_elements,
    belts_inner,
    highres_bem_diag,
)
from tools.run_grandchallenge_experiments import evaluate_system
from electrodrive.core.bem_kernel import bem_matvec_gpu, DEFAULT_SINGLE_LAYER_KERNEL

try:
    from electrodrive.fmm3d.bem_fmm import make_laplace_fmm_backend
except Exception:
    make_laplace_fmm_backend = None


class _NullLogger:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


logger = _NullLogger()


def parse_z_list(z_str: str) -> List[float]:
    return [float(s) for s in z_str.split(",") if s.strip()]


def move_charge(spec: CanonicalSpec, z: float) -> CanonicalSpec:
    spec_z = copy.deepcopy(spec)
    for ch in spec_z.charges:
        if ch.get("type") == "point":
            ch["pos"] = [0.0, 0.0, float(z)]
    return spec_z


def solve_fixed_geometry(
    spec: CanonicalSpec,
    elements: Sequence,
    reg_l1: float,
    per_type_reg: Dict[str, float],
    boundary_weight: float,
    n_points: int,
    ratio_boundary: float,
    device: torch.device,
    dtype: torch.dtype,
) -> ImageSystem:
    batch = make_collocation_batch_for_spec(
        spec=spec,
        n_points=n_points,
        ratio_boundary=ratio_boundary,
        supervision_mode="auto",
        device=device,
        dtype=dtype,
    )
    X = batch["X"]
    V_gt = batch["V_gt"]
    mask_finite = batch.get("mask_finite")
    if mask_finite is not None and mask_finite.shape == (X.shape[0],):
        mask = mask_finite.to(device=device) & torch.isfinite(V_gt)
    else:
        mask = torch.isfinite(V_gt)
    X = X[mask]
    V_gt = V_gt[mask]
    is_boundary = batch.get("is_boundary", torch.zeros(X.shape[0], device=device, dtype=torch.bool))[mask]

    A = assemble_basis_matrix(elements, X)
    target = V_gt
    if boundary_weight is not None and is_boundary is not None and is_boundary.shape == (X.shape[0],):
        alpha = float(max(0.0, min(1.0, boundary_weight)))
        beta = 1.0 - alpha
        row_weights = torch.where(
            is_boundary.to(device=device),
            torch.full_like(is_boundary, alpha, dtype=dtype),
            torch.full_like(is_boundary, beta, dtype=dtype),
        )
        rw_sqrt = torch.sqrt(row_weights).view(-1, 1)
        A = A * rw_sqrt
        target = target * rw_sqrt.view(-1)

    reg_vec = torch.tensor([float(per_type_reg.get(e.type, reg_l1)) for e in elements], device=device, dtype=dtype)
    weights, _ = solve_l1_ista(A, target, reg_l1=reg_l1, logger=logger, per_elem_reg=reg_vec)
    return ImageSystem(list(elements), weights)


@dataclass
class AxisSweepResult:
    z: float
    stage_metrics: Dict[str, float]
    bem_metrics: Dict[str, float]
    n_images: int
    type_counts: Dict[str, int]
    system_path: Path
    npz_path: Path
    reg_l1: float
    boundary_weight: float


def try_fmm_matvec(spec: CanonicalSpec, bem_cfg: Dict[str, object]) -> Tuple[bool, Dict[str, float]]:
    if make_laplace_fmm_backend is None:
        return False, {"error": "FMM backend unavailable"}
    try:
        sol = get_oracle_solution(spec, mode="bem", bem_cfg=bem_cfg)
        if sol is None:
            return False, {"error": "BEM oracle unavailable"}
        centroids = sol._C.detach().to(device="cpu", dtype=torch.float64)  # type: ignore[attr-defined]
        areas = sol._A.detach().to(device="cpu", dtype=torch.float64)  # type: ignore[attr-defined]
        sigma = sol._S.detach().to(device="cpu", dtype=torch.float64)  # type: ignore[attr-defined]
        tile_size = int(getattr(sol, "_tile", 512))  # type: ignore[attr-defined]
        fmm = make_laplace_fmm_backend(
            src_centroids=centroids,
            areas=areas,
            max_leaf_size=64,
            theta=0.6,
            use_dipole=True,
            logger=None,
        )
        V_ref = bem_matvec_gpu(
            sigma=sigma,
            src_centroids=centroids,
            areas=areas,
            tile_size=min(tile_size, 2048),
            self_integrals=None,
            logger=None,
            use_keops=False,
            kernel=DEFAULT_SINGLE_LAYER_KERNEL,
            backend="torch_tiled",
        )
        V_fmm = bem_matvec_gpu(
            sigma=sigma,
            src_centroids=centroids,
            areas=areas,
            tile_size=min(tile_size, 2048),
            self_integrals=None,
            logger=None,
            use_keops=False,
            kernel=DEFAULT_SINGLE_LAYER_KERNEL,
            backend="external",
            matvec_impl=fmm.matvec,
        )
        diff = torch.linalg.norm(V_ref - V_fmm)
        denom = torch.linalg.norm(V_ref).clamp_min(1e-12)
        rel = float((diff / denom).item())
        return True, {"rel_l2_err": rel, "n_panels": int(centroids.shape[0])}
    except Exception as exc:
        return False, {"error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Axis sweep for fixed 2-ring + 4-point geometry on mid torus.")
    parser.add_argument("--z-list", type=str, default="0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--n-colloc", type=int, default=3072)
    parser.add_argument("--ratio-boundary", type=float, default=0.8)
    parser.add_argument("--reg-l1", type=float, default=4e-4)
    parser.add_argument("--point-reg-mult", type=float, default=4.0)
    parser.add_argument("--boundary-weight", type=float, default=0.9)
    parser.add_argument("--nr", type=int, default=220)
    parser.add_argument("--nz", type=int, default=220)
    parser.add_argument("--out-metrics", type=Path, default=Path("runs/torus/stage4_metrics_mid_axis_sweep_stage4.json"))
    parser.add_argument("--out-bem-metrics", type=Path, default=Path("runs/torus/stage4_metrics_mid_axis_sweep_bem.json"))
    parser.add_argument("--discovered-root", type=Path, default=Path("runs/torus/discovered"))
    parser.add_argument("--diagnostics-root", type=Path, default=Path("runs/torus/diagnostics"))
    parser.add_argument("--geometry-path", type=Path, default=Path("runs/torus/discovered/mid_bem_highres_trial02/discovered_system.json"))
    parser.add_argument("--fmm-check", action="store_true", help="Attempt FMM matvec sanity check for z=0.7")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    spec = CanonicalSpec.from_json(json.load(open(root / "specs" / "torus_axis_point_mid.json")))
    torus = next(c for c in spec.conductors if c.get("type") in ("torus", "toroid"))
    center = torch.tensor(torus.get("center", [0.0, 0.0, 0.0]), device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32)
    device = center.device
    dtype = torch.float32

    rings_seed, pts_seed = load_seed_params(root / args.geometry_path)
    belts = belts_inner(spec)

    bem_cfg = {
        "use_gpu": True,
        "fp64": True,
        "initial_h": 0.15,
        "max_refine_passes": 5,
        "min_refine_passes": 2,
        "gmres_tol": 1e-8,
        "target_bc_inf_norm": 1e-8,
        "use_near_quadrature": True,
        "use_near_quadrature_matvec": False,
        "tile_mem_divisor": 2.5,
        "target_vram_fraction": 0.9,
        "logger": logger,
    }

    z_vals = parse_z_list(args.z_list)
    stage_metrics: List[Dict[str, object]] = []
    bem_metrics: List[Dict[str, object]] = []

    for z in z_vals:
        spec_z = move_charge(spec, z)
        elems = build_elements(rings_seed, pts_seed, center=center, device=device, dtype=dtype)
        per_type_reg = {"poloidal_ring": args.reg_l1, "point": args.reg_l1 * args.point_reg_mult}
        system = solve_fixed_geometry(
            spec=spec_z,
            elements=elems,
            reg_l1=args.reg_l1,
            per_type_reg=per_type_reg,
            boundary_weight=args.boundary_weight,
            n_points=args.n_colloc,
            ratio_boundary=args.ratio_boundary,
            device=device,
            dtype=dtype,
        )
        # Stage-style metrics
        stage_res = evaluate_system(spec_z, system, n_eval=args.n_colloc, ratio_boundary=args.ratio_boundary, belts=belts)
        sys_dir = root / args.discovered_root / f"mid_axis_sweep_z{z:.2f}"
        save_image_system(system, sys_dir / "discovered_system.json", metadata={"z": z, "reg_l1": args.reg_l1, "boundary_weight": args.boundary_weight})

        stage_rec = {
            "z": z,
            "metrics": stage_res.metrics,
            "n_images": len(system.elements),
            "type_counts": stage_res.type_counts,
            "reg_l1": args.reg_l1,
            "boundary_weight": args.boundary_weight,
            "system_path": str(sys_dir / "discovered_system.json"),
        }
        stage_metrics.append(stage_rec)

        npz_path = root / args.diagnostics_root / f"mid_axis_sweep_z{z:.2f}.npz"
        try:
            bem_stats = highres_bem_diag(spec_z, system, npz_path, nr=args.nr, nz=args.nz, bem_cfg=bem_cfg)
        except Exception as exc:
            bem_stats = {"error": str(exc)}
        bem_rec = {
            "z": z,
            "metrics": bem_stats,
            "n_images": len(system.elements),
            "type_counts": stage_res.type_counts,
            "reg_l1": args.reg_l1,
            "boundary_weight": args.boundary_weight,
            "system_path": str(sys_dir / "discovered_system.json"),
            "npz_path": str(npz_path),
        }
        bem_metrics.append(bem_rec)
        print(f"[axis_sweep] z={z:.2f} stage_metrics={stage_res.metrics} bem_metrics={bem_stats}")

    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_bem_metrics.parent.mkdir(parents=True, exist_ok=True)
    json.dump(stage_metrics, open(root / args.out_metrics, "w"), indent=2)
    json.dump(bem_metrics, open(root / args.out_bem_metrics, "w"), indent=2)
    print(f"Saved stage metrics to {root / args.out_metrics}")
    print(f"Saved BEM metrics to {root / args.out_bem_metrics}")

    if args.fmm_check:
        ok, stats = try_fmm_matvec(move_charge(spec, 0.7), bem_cfg)
        print(f"[FMM-check] ok={ok} stats={stats}")


if __name__ == "__main__":
    main()
