"""
High-accuracy BEM/FMM refinement for mid-torus 2-ring + 4-point systems.

Runs a small high-res BEM refinement around the current best geometry (trial003),
attempts FMM diagnostics (if available), and logs metrics/diagnostics.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch

from electrodrive.images.basis import PoloidalRingBasis, PointChargeBasis, ImageBasisElement
from electrodrive.images.io import load_image_system, save_image_system
from electrodrive.images.search import assemble_basis_matrix, solve_l1_ista, ImageSystem
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


def load_seed_params(path: Path) -> Tuple[List[RingParams], List[PointParams]]:
    sys = load_image_system(path)
    rings: List[RingParams] = []
    pts: List[PointParams] = []
    for elem in sys.elements:
        if elem.type == "poloidal_ring":
            p = elem.params
            rings.append(
                RingParams(
                    radius=float(p["radius"]),
                    delta_r=float(p.get("delta_r", 0.0)),
                    order=int(p.get("order", 0)),
                )
            )
        elif elem.type == "point":
            pos = torch.as_tensor(elem.params["position"]).view(-1).cpu().numpy()
            rho = float(np.linalg.norm(pos[:2]))
            phi = float(math.atan2(pos[1], pos[0]))
            pts.append(PointParams(rho=rho, phi=phi, z=float(pos[2])))
    rings.sort(key=lambda r: r.order)
    pts.sort(key=lambda p: p.phi)
    return rings, pts


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


def perturb(
    rings: Sequence[RingParams],
    pts: Sequence[PointParams],
    R: float,
    a: float,
    jitter_scale: float,
    extra_point_prob: float = 0.1,
) -> Tuple[List[RingParams], List[PointParams]]:
    rho_lo, rho_hi = R - 0.8 * a, R + 0.8 * a
    z_lo, z_hi = -0.4 * a, 0.4 * a
    rng = random.random
    pr: List[RingParams] = []
    for r in rings:
        radius = clamp(r.radius + random.uniform(-jitter_scale, jitter_scale), 0.9 * R, 1.1 * R)
        delta_r = clamp(r.delta_r + random.uniform(-0.5 * jitter_scale, 0.5 * jitter_scale), 0.2 * r.delta_r, 1.5 * r.delta_r)
        order = r.order if rng() > 0.1 else (0 if r.order == 2 else 2)
        pr.append(RingParams(radius=radius, delta_r=delta_r, order=order))
    pp: List[PointParams] = []
    for p in pts:
        rho = clamp(p.rho + random.uniform(-jitter_scale, jitter_scale), rho_lo, rho_hi)
        phi = p.phi + math.radians(random.uniform(-10.0, 10.0))
        z = clamp(p.z + random.uniform(-jitter_scale, jitter_scale), z_lo, z_hi)
        pp.append(PointParams(rho=rho, phi=phi, z=z))
    if rng() < extra_point_prob:
        base = random.choice(pp)
        rho = clamp(base.rho + random.uniform(-2 * jitter_scale, 2 * jitter_scale), rho_lo, rho_hi)
        phi = base.phi + math.radians(random.uniform(-20.0, 20.0))
        z = clamp(base.z + random.uniform(-2 * jitter_scale, 2 * jitter_scale), z_lo, z_hi)
        pp.append(PointParams(rho=rho, phi=phi, z=z))
    return pr, pp


def build_elements(
    rings: Sequence[RingParams],
    pts: Sequence[PointParams],
    center: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> List[ImageBasisElement]:
    elems: List[ImageBasisElement] = []
    for r in rings:
        elems.append(
            PoloidalRingBasis(
                {
                    "center": center,
                    "radius": torch.tensor(r.radius, device=device, dtype=dtype),
                    "delta_r": torch.tensor(r.delta_r, device=device, dtype=dtype),
                    "order": torch.tensor(int(r.order), device=device),
                    "n_quad": torch.tensor(128, device=device),
                }
            )
        )
    for p in pts:
        x = p.rho * math.cos(p.phi)
        y = p.rho * math.sin(p.phi)
        pos = torch.tensor([x, y, p.z], device=device, dtype=dtype) + center
        elems.append(PointChargeBasis({"position": pos}))
    return elems


def solve_weights_fixed_geometry(
    spec: CanonicalSpec,
    elements: List[ImageBasisElement],
    reg_l1: float,
    per_type_reg: Dict[str, float],
    boundary_weight: float,
    belts: Optional[List[Tuple[float, float]]],
    device: torch.device,
    dtype: torch.dtype,
) -> ImageSystem:
    # Build collocation via evaluate_system helper (but we need dictionary for ISTA)
    # Reuse collocation from evaluate_system: it already uses make_collocation_batch_for_spec.
    from electrodrive.learn.collocation import make_collocation_batch_for_spec

    batch = make_collocation_batch_for_spec(
        spec=spec,
        n_points=2048,
        ratio_boundary=0.8,
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
    return ImageSystem(elements, weights)


def highres_bem_diag(
    spec: CanonicalSpec,
    system: ImageSystem,
    out_npz: Path,
    nr: int = 220,
    nz: int = 220,
    bem_cfg: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    torus = next(c for c in spec.conductors if c.get("type") in ("torus", "toroid"))
    R = float(torus.get("major_radius", torus.get("radius", 1.0)))
    a = float(torus.get("minor_radius", 0.25 * R))
    r_min, r_max = max(1e-6, R - 1.5 * a), R + 1.5 * a
    z_min, z_max = -1.5 * a, 1.5 * a
    r = np.linspace(r_min, r_max, nr)
    z = np.linspace(z_min, z_max, nz)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    pts_np = np.stack([rr, np.zeros_like(rr), zz], axis=-1).reshape(-1, 3)
    pts = torch.tensor(pts_np, device=device, dtype=dtype)

    with torch.no_grad():
        V_img = system.potential(pts).view(nr, nz).cpu().numpy()
        sol = get_oracle_solution(spec, mode="bem", bem_cfg=bem_cfg or {})
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
    np.savez_compressed(out_npz, r=r, z=z, V_img=V_img, V_bem=V_bem, abs_err=abs_err, rel_err=rel_err)
    mask_inner = (rr >= R - a) & (rr <= R - 0.4 * a) & (np.abs(zz) <= 0.3 * a)
    stats = {
        "max_rel": float(rel_err.max()),
        "mean_rel": float(rel_err.mean()),
        "inner_mean_abs": float(abs_err[mask_inner].mean()),
        "inner_mean_rel": float(rel_err[mask_inner].mean()),
    }
    return stats


def attempt_fmm_diag(spec: CanonicalSpec, system: ImageSystem, out_npz: Path, nr: int = 220, nz: int = 220) -> Tuple[bool, Dict[str, float]]:
    """Attempt FMM oracle; return (ok, stats)."""
    try:
        sol = get_oracle_solution(spec, mode="fmm", bem_cfg={})  # type: ignore[arg-type]
    except Exception as exc:
        return False, {"error": str(exc)}
    if sol is None:
        return False, {"error": "FMM oracle unavailable"}
    # Reuse BEM grid eval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    torus = next(c for c in spec.conductors if c.get("type") in ("torus", "toroid"))
    R = float(torus.get("major_radius", torus.get("radius", 1.0)))
    a = float(torus.get("minor_radius", 0.25 * R))
    r_min, r_max = max(1e-6, R - 1.5 * a), R + 1.5 * a
    z_min, z_max = -1.5 * a, 1.5 * a
    r = np.linspace(r_min, r_max, nr)
    z = np.linspace(z_min, z_max, nz)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    pts_np = np.stack([rr, np.zeros_like(rr), zz], axis=-1).reshape(-1, 3)
    pts = torch.tensor(pts_np, device=device, dtype=dtype)
    with torch.no_grad():
        V_img = system.potential(pts).view(nr, nz).cpu().numpy()
        if hasattr(sol, "eval_V_E_batched"):
            V_ref, _ = sol.eval_V_E_batched(pts)  # type: ignore[attr-defined]
        else:
            V_ref = sol.eval(pts)  # type: ignore[attr-defined]
        V_ref = V_ref.view(nr, nz).cpu().numpy()
    abs_err = np.abs(V_img - V_ref)
    rel_err = abs_err / (np.abs(V_ref) + 1e-12)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, r=r, z=z, V_img=V_img, V_ref=V_ref, abs_err=abs_err, rel_err=rel_err)
    mask_inner = (rr >= R - a) & (rr <= R - 0.4 * a) & (np.abs(zz) <= 0.3 * a)
    stats = {
        "max_rel": float(rel_err.max()),
        "mean_rel": float(rel_err.mean()),
        "inner_mean_abs": float(abs_err[mask_inner].mean()),
        "inner_mean_rel": float(rel_err[mask_inner].mean()),
    }
    return True, stats


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = CanonicalSpec.from_json(json.load(open(root / "specs" / "torus_axis_point_mid.json")))
    torus = next(c for c in spec.conductors if c.get("type") in ("torus", "toroid"))
    R = float(torus.get("major_radius", torus.get("radius", 1.0)))
    a = float(torus.get("minor_radius", 0.25 * R))
    center = torch.tensor(torus.get("center", [0.0, 0.0, 0.0]), device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32)
    device = center.device
    dtype = center.dtype

    seed_path = root / "runs/torus/discovered/mid_local_seed5500358_trial003/discovered_system.json"
    rings_seed, pts_seed = load_seed_params(seed_path)
    belts = belts_inner(spec)

    # High-res BEM config
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
    }

    trials = []
    n_trials = 12
    jitter = 0.02
    for i in range(n_trials):
        random.seed(10_000 + i)
        pr, pp = perturb(rings_seed, pts_seed, R=R, a=a, jitter_scale=jitter, extra_point_prob=0.05)
        elems = build_elements(pr, pp, center=center, device=device, dtype=dtype)
        reg_l1 = random.choice([2e-4, 3e-4, 4e-4, 6e-4])
        bw = random.choice([0.9, 0.95])
        per_type_reg = {"poloidal_ring": reg_l1 * 1.0, "point": reg_l1 * 4.0}
        system = solve_weights_fixed_geometry(spec, elems, reg_l1=reg_l1, per_type_reg=per_type_reg, boundary_weight=bw, belts=belts, device=device, dtype=dtype)
        tag = f"mid_bem_highres_trial{i:02d}"
        npz_path = root / "runs/torus/diagnostics" / f"{tag}.npz"
        try:
            stats = highres_bem_diag(spec, system, npz_path, bem_cfg=bem_cfg)
        except Exception as exc:  # fallback to standard BEM if highres fails
            stats = {"error": str(exc)}
        type_counts: Dict[str, int] = {}
        for e in system.elements:
            type_counts[e.type] = type_counts.get(e.type, 0) + 1
        rec = {
            "run": tag,
            "rings": [r.__dict__ for r in pr],
            "points": [p.__dict__ for p in pp],
            "reg_l1": reg_l1,
            "boundary_weight": bw,
            "metrics": stats,
            "n_images": len(system.elements),
            "type_counts": type_counts,
        }
        trials.append(rec)
        # Persist discovered system
        save_image_system(system, root / f"runs/torus/discovered/{tag}/discovered_system.json", metadata={"reg_l1": reg_l1, "boundary_weight": bw})
        print(f"[BEM trial {i}] stats {stats}")

    out_metrics = root / "runs/torus/stage4_metrics_mid_bem_highres_local.json"
    json.dump(trials, out_metrics.open("w"), indent=2)
    print(f"Saved {len(trials)} highres BEM trials to {out_metrics}")

    # Attempt FMM diag on baseline/seed/trial003 and best BEM trial
    baseline = load_image_system(root / "runs/torus/discovered/mid_baseline_eigen/discovered_system.json", device=device, dtype=dtype)
    seed_sys = load_image_system(root / "runs/torus/discovered/mid_random_seed5500358/discovered_system.json", device=device, dtype=dtype)
    trial003 = load_image_system(root / "runs/torus/discovered/mid_local_seed5500358_trial003/discovered_system.json", device=device, dtype=dtype)

    candidates_fmm = {
        "baseline": baseline,
        "seed": seed_sys,
        "trial003": trial003,
    }
    # pick best BEM trial by mean_rel if available
    best_bem = None
    best_val = math.inf
    for rec in trials:
        m = rec["metrics"]
        if "mean_rel" in m and m["mean_rel"] < best_val:
            best_val = m["mean_rel"]
            best_bem = rec["run"]
    if best_bem:
        best_sys = load_image_system(root / f"runs/torus/discovered/{best_bem}/discovered_system.json", device=device, dtype=dtype)
        candidates_fmm["best_bem"] = best_sys

    fmm_stats: Dict[str, Dict[str, float]] = {}
    for name, sys in candidates_fmm.items():
        ok, stats = attempt_fmm_diag(spec, sys, root / f"runs/torus/diagnostics/mid_{name}_fmm.npz")
        stats["ok"] = ok
        fmm_stats[name] = stats
        print(f"[FMM] {name} stats {stats}")

    json.dump(fmm_stats, open(root / "runs/torus/diagnostics/mid_fmm_stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
