"""
Mid-torus local geometry explorer around the 2-ring + 4-point seed.

Workflow:
- Load the mid-torus spec and the seed discovered system (2 poloidal rings + 4 points).
- Randomly perturb ring radii/delta_r/orders and point cylindrical coords (rho, phi, z),
  optionally adding 1–2 extra points near the torus surface.
- For each perturbed geometry, solve for weights with fixed geometry using the ISTA
  solver from electrodrive.images.search, then evaluate Stage4-style metrics.
- Rank interesting trials; optionally run BEM diagnostics for the top few.
Results are written to runs/torus/stage4_metrics_mid_local_geometry.json and diagnostics
under runs/torus/diagnostics/.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from electrodrive.images.basis import PoloidalRingBasis, PointChargeBasis, ImageBasisElement
from electrodrive.images.io import load_image_system, save_image_system
from electrodrive.images.search import (
    assemble_basis_matrix,
    solve_l1_ista,
    get_collocation_data,
    ImageSystem,
)
from electrodrive.learn.collocation import get_oracle_solution
from electrodrive.orchestration.parser import CanonicalSpec
from tools.run_grandchallenge_experiments import evaluate_system


class _NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


logger = _NullLogger()


@dataclass
class RingParams:
    radius: float
    delta_r: float
    order: int


@dataclass
class PointParams:
    rho: float
    phi: float
    z: float


@dataclass
class TrialRecord:
    run: str
    params: Dict[str, object]
    basis_types: List[str]
    n_images: int
    metrics: Dict[str, float]
    type_counts: Dict[str, int]
    reg_l1: float
    boundary_weight: float
    per_type_reg: Dict[str, float]


def load_seed_params(seed_path: Path) -> Tuple[List[RingParams], List[PointParams]]:
    system = load_image_system(seed_path)
    rings: List[RingParams] = []
    points: List[PointParams] = []
    for elem, w in zip(system.elements, system.weights.tolist()):
        if elem.type == "poloidal_ring":
            p = elem.params
            radius = float(p["radius"])
            delta_r = float(p.get("delta_r", 0.1))
            order = int(p.get("order", 0))
            rings.append(RingParams(radius, delta_r, order))
        elif elem.type == "point":
            pos = torch.as_tensor(elem.params["position"]).view(-1).cpu().numpy()
            rho = float(np.linalg.norm(pos[:2]))
            phi = float(math.atan2(pos[1], pos[0]))
            points.append(PointParams(rho, phi, float(pos[2])))
    # Keep deterministic ordering: sort rings by order, points by phi.
    rings.sort(key=lambda r: r.order)
    points.sort(key=lambda p: p.phi)
    return rings, points


def belts_inner(spec: CanonicalSpec) -> List[Tuple[float, float]]:
    torus = next(c for c in spec.conductors if c.get("type") in ("torus", "toroid"))
    R = float(torus.get("major_radius", torus.get("radius", 1.0)))
    a = float(torus.get("minor_radius", 0.25 * R))
    belts: List[Tuple[float, float]] = []
    for r in (R - a, R - 0.75 * a, R - 0.5 * a, R, R + 0.5 * a):
        for z in (0.0, 0.2 * a, -0.2 * a):
            belts.append((r, z))
    return belts


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def perturb_seed(
    rings: Sequence[RingParams],
    points: Sequence[PointParams],
    R: float,
    a: float,
    jitter_scale: float = 0.05,
    allow_extra_points: bool = True,
) -> Tuple[List[RingParams], List[PointParams]]:
    rng = random.random

    def jitter(val: float, width: float) -> float:
        return val + (random.uniform(-1.0, 1.0) * width)

    pert_rings: List[RingParams] = []
    for rp in rings:
        radius = clamp(jitter(rp.radius, 0.07), 0.9 * R, 1.1 * R)
        delta_r = clamp(jitter(rp.delta_r, 0.5 * rp.delta_r), 0.2 * rp.delta_r, 1.5 * rp.delta_r)
        order = rp.order
        if rng() < 0.15:
            order = 0 if order == 2 else 2
        pert_rings.append(RingParams(radius=radius, delta_r=delta_r, order=order))

    rho_lo, rho_hi = R - 0.8 * a, R + 0.8 * a
    z_lo, z_hi = -0.4 * a, 0.4 * a
    pert_points: List[PointParams] = []
    for pp in points:
        rho = clamp(pp.rho + random.uniform(-jitter_scale, jitter_scale), rho_lo, rho_hi)
        phi = pp.phi + math.radians(random.uniform(-15.0, 15.0))
        z = clamp(pp.z + random.uniform(-jitter_scale, jitter_scale), z_lo, z_hi)
        pert_points.append(PointParams(rho=rho, phi=phi, z=z))

    if allow_extra_points and rng() < 0.25:
        base = random.choice(pert_points)
        rho = clamp(base.rho + random.uniform(-2 * jitter_scale, 2 * jitter_scale), rho_lo, rho_hi)
        phi = base.phi + math.radians(random.uniform(-20.0, 20.0))
        z = clamp(base.z + random.uniform(-2 * jitter_scale, 2 * jitter_scale), z_lo, z_hi)
        pert_points.append(PointParams(rho=rho, phi=phi, z=z))

    return pert_rings, pert_points


def build_basis_from_params(
    rings: Sequence[RingParams],
    points: Sequence[PointParams],
    center: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> List[ImageBasisElement]:
    elems: List[ImageBasisElement] = []
    for rp in rings:
        elems.append(
            PoloidalRingBasis(
                {
                    "center": center,
                    "radius": torch.tensor(rp.radius, device=device, dtype=dtype),
                    "delta_r": torch.tensor(rp.delta_r, device=device, dtype=dtype),
                    "order": torch.tensor(int(rp.order), device=device),
                    "n_quad": torch.tensor(128, device=device),
                }
            )
        )
    for pp in points:
        x = pp.rho * math.cos(pp.phi)
        y = pp.rho * math.sin(pp.phi)
        pos = torch.tensor([x, y, pp.z], device=device, dtype=dtype) + center
        elems.append(PointChargeBasis({"position": pos}))
    return elems


def solve_fixed_geometry(
    spec: CanonicalSpec,
    elements: List[ImageBasisElement],
    reg_l1: float,
    per_type_reg: Dict[str, float],
    boundary_weight: float,
    device: torch.device,
    dtype: torch.dtype,
) -> ImageSystem:
    colloc = get_collocation_data(spec, logger=type("L", (), {"info": lambda *a, **k: None, "warning": lambda *a, **k: None, "error": lambda *a, **k: None})(), device=device, dtype=dtype, return_is_boundary=True)
    if len(colloc) == 3:
        X, V, is_boundary = colloc  # type: ignore[misc]
    else:
        X, V = colloc  # type: ignore[misc]
        is_boundary = None
    if X.shape[0] == 0:
        return ImageSystem([], torch.zeros(0, device=device, dtype=dtype))

    A = assemble_basis_matrix(elements, X)
    target = V

    if boundary_weight is not None and is_boundary is not None and is_boundary.shape == (X.shape[0],):
        alpha = float(max(0.0, min(1.0, boundary_weight)))
        beta = 1.0 - alpha
        is_boundary = is_boundary.to(device=device)
        row_weights = torch.where(
            is_boundary,
            torch.full_like(is_boundary, alpha, dtype=dtype),
            torch.full_like(is_boundary, beta, dtype=dtype),
        )
        rw_sqrt = torch.sqrt(row_weights).view(-1, 1)
        A = A * rw_sqrt
        target = target * rw_sqrt.view(-1)

    reg_vec = torch.tensor([float(per_type_reg.get(e.type, reg_l1)) for e in elements], device=device, dtype=dtype)
    weights, _ = solve_l1_ista(A, target, reg_l1=reg_l1, logger=logger, per_elem_reg=reg_vec)
    return ImageSystem(elements, weights)


def diag_stats(path: Path, R: float, a: float) -> Dict[str, float]:
    data = np.load(path)
    r = data["r"]
    z = data["z"]
    rr, zz = np.meshgrid(r, z, indexing="ij")
    abs_err = data["abs_err"]
    rel_err = data["rel_err"]
    mask_inner = (rr >= R - a) & (rr <= R - 0.4 * a) & (np.abs(zz) <= 0.3 * a)
    out: Dict[str, float] = {
        "max_rel": float(rel_err.max()),
        "mean_rel": float(rel_err.mean()),
    }
    if mask_inner.any():
        out["inner_mean_abs"] = float(abs_err[mask_inner].mean())
        out["inner_mean_rel"] = float(rel_err[mask_inner].mean())
    return out


def run_bem_diag(
    spec: CanonicalSpec,
    system: ImageSystem,
    out_npz: Path,
    nr: int = 200,
    nz: int = 200,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    torus = next(c for c in spec.conductors if c.get("type") in ("torus", "toroid"))
    R = float(torus.get("major_radius", torus.get("radius", 1.0)))
    a = float(torus.get("minor_radius", 0.25 * R))
    r_min, r_max = max(1e-6, R - 1.5 * a), R + 1.5 * a
    z_min, z_max = -1.5 * a, 1.5 * a
    r_arr = np.linspace(r_min, r_max, nr)
    z_arr = np.linspace(z_min, z_max, nz)
    rr, zz = np.meshgrid(r_arr, z_arr, indexing="ij")
    xx = rr
    yy = np.zeros_like(rr)
    pts_np = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    pts = torch.tensor(pts_np, device=device, dtype=dtype)

    with torch.no_grad():
        V_img = system.potential(pts).view(nr, nz).cpu().numpy()
        sol = get_oracle_solution(spec, mode="bem", bem_cfg={})  # type: ignore[arg-type]
        if sol is None:
            raise RuntimeError("BEM oracle unavailable")
        if hasattr(sol, "eval_V_E_batched"):
            V_bem, _ = sol.eval_V_E_batched(pts)  # type: ignore[attr-defined]
        else:
            V_bem = sol.eval(pts)  # type: ignore[attr-defined]
        V_bem = V_bem.view(nr, nz).cpu().numpy()
    abs_err = np.abs(V_img - V_bem)
    rel_err = abs_err / (np.abs(V_bem) + 1e-12)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, r=r_arr, z=z_arr, V_img=V_img, V_bem=V_bem, abs_err=abs_err, rel_err=rel_err)
    return diag_stats(out_npz, R=R, a=a)


def main() -> None:
    ap = argparse.ArgumentParser(description="Local geometry explorer around mid 2-ring+4-point seed.")
    ap.add_argument("--trials", type=int, default=24, help="Number of perturbation trials.")
    ap.add_argument("--bem-top", type=int, default=3, help="Max candidates to BEM-check.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    spec = CanonicalSpec.from_json(json.load(open(root / "specs" / "torus_axis_point_mid.json")))
    seed_path = root / "runs/torus/discovered/mid_random_seed5500358/discovered_system.json"
    baseline_path = root / "runs/torus/discovered/mid_baseline_eigen/discovered_system.json"
    rings_seed, points_seed = load_seed_params(seed_path)
    torus = next(c for c in spec.conductors if c.get("type") in ("torus", "toroid"))
    R = float(torus.get("major_radius", torus.get("radius", 1.0)))
    a = float(torus.get("minor_radius", 0.25 * R))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    center = torch.tensor(torus.get("center", [0.0, 0.0, 0.0]), device=device, dtype=dtype)
    belts = belts_inner(spec)

    records: List[TrialRecord] = []
    for idx in range(args.trials):
        rng_seed = random.randint(0, 10_000_000)
        random.seed(rng_seed)
        pr_rings, pr_points = perturb_seed(rings_seed, points_seed, R=R, a=a, allow_extra_points=True)
        elems = build_basis_from_params(pr_rings, pr_points, center=center, device=device, dtype=dtype)
        reg_l1 = random.choice([2e-4, 3e-4, 4e-4, 6e-4, 8e-4])
        bw = random.choice([0.85, 0.9, 0.95])
        per_type_reg = {
            "poloidal_ring": reg_l1 * random.uniform(0.8, 1.2),
            "point": reg_l1 * random.uniform(3.0, 5.0),
        }
        system = solve_fixed_geometry(spec, elems, reg_l1=reg_l1, per_type_reg=per_type_reg, boundary_weight=bw, device=device, dtype=dtype)
        eval_res = evaluate_system(spec, system, belts=belts)
        tag = f"mid_local_seed5500358_trial{idx:03d}"
        rec = TrialRecord(
            run=tag,
            params={
                "rings": [r.__dict__ for r in pr_rings],
                "points": [p.__dict__ for p in pr_points],
                "seed": rng_seed,
            },
            basis_types=["poloidal_ring", "point"],
            n_images=len(system.elements),
            metrics=eval_res.metrics,
            type_counts=eval_res.type_counts,
            reg_l1=reg_l1,
            boundary_weight=bw,
            per_type_reg=per_type_reg,
        )
        records.append(rec)
        print(f"[trial {idx:03d}] b_mae={eval_res.metrics.get('boundary_mae')} off_rel={eval_res.metrics.get('offaxis_rel')} n={len(system.elements)}")

    out_path = root / "runs/torus/stage4_metrics_mid_local_geometry.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump([r.__dict__ for r in records], out_path.open("w"), indent=2)
    print(f"Saved {len(records)} trials to {out_path}")

    # Rank promising by boundary_mae then offaxis_rel
    records_sorted = sorted(records, key=lambda r: (r.metrics.get("boundary_mae", math.inf), r.metrics.get("offaxis_rel", math.inf)))
    top = records_sorted[: args.bem_top]
    print("[top for BEM]")
    for r in top:
        print(r.run, r.metrics)

    # BEM diagnostics on top candidates
    diag_out: Dict[str, Dict[str, float]] = {}
    for r in top:
        elems = build_basis_from_params(
            [RingParams(**rp) for rp in r.params["rings"]],  # type: ignore[arg-type]
            [PointParams(**pp) for pp in r.params["points"]],  # type: ignore[arg-type]
            center=center,
            device=device,
            dtype=dtype,
        )
        system = solve_fixed_geometry(spec, elems, reg_l1=r.reg_l1, per_type_reg=r.per_type_reg, boundary_weight=r.boundary_weight, device=device, dtype=dtype)
        diag_path = root / f"runs/torus/diagnostics/{r.run}.npz"
        stats = run_bem_diag(spec, system, diag_path)
        diag_out[r.run] = stats
        # Persist discovered system
        disc_path = root / f"runs/torus/discovered/{r.run}/discovered_system.json"
        save_image_system(system, disc_path, metadata={"run": r.run, "reg_l1": r.reg_l1, "boundary_weight": r.boundary_weight, "per_type_reg": r.per_type_reg})
        print(f"[BEM] {r.run} stats {stats} saved {diag_path}")

    # Save combined diag stats
    json.dump(diag_out, open(root / "runs/torus/diagnostics/mid_local_geometry_diag_stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
