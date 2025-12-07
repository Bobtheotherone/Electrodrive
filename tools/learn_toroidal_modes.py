"""
Learn data-driven toroidal eigenmodes from BEM snapshots and fit them with
finite image bases (poloidal rings, ladders, mode clusters).

Outputs JSON with learned composite modes that can be used by
ToroidalEigenModeBasis.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from electrodrive.images.basis import (
    PoloidalRingBasis,
    RingLadderBasis,
    ToroidalModeClusterBasis,
    ImageBasisElement,
    build_dictionary,
)
from electrodrive.learn.collocation import make_collocation_batch_for_spec
from electrodrive.learn.collocation import get_oracle_solution
from electrodrive.orchestration.parser import CanonicalSpec


def _torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_collocation(
    spec: CanonicalSpec,
    n_points: int,
    ratio_boundary: float,
    seed: int,
    belts: List[Tuple[float, float]],
    belt_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    batch = make_collocation_batch_for_spec(
        spec=spec,
        n_points=n_points,
        ratio_boundary=ratio_boundary,
        supervision_mode="auto",
        device=_torch_device(),
        dtype=torch.float32,
        rng=rng,
    )
    X = batch["X"]
    is_boundary = batch["is_boundary"]
    mask_finite = batch.get("mask_finite")
    if mask_finite is not None and mask_finite.shape == (X.shape[0],):
        mask = mask_finite.to(device=X.device) & torch.isfinite(batch["V_gt"])
    else:
        mask = torch.isfinite(batch["V_gt"])
    X = X[mask]
    is_boundary = is_boundary[mask]

    # Build an off-axis belt mask for weighting later.
    belt_mask = torch.zeros(X.shape[0], dtype=torch.bool, device=X.device)
    if belts:
        for r_target, z_target in belts:
            r = torch.linalg.norm(X[:, :2], dim=1)
            z = X[:, 2]
            belt_mask |= (torch.abs(r - r_target) <= 0.05 * abs(r_target) + 1e-6) & (
                torch.abs(z - z_target) <= 0.1 * (abs(z_target) + 1e-6)
            )

    # Per-row weights (start at 1.0)
    row_weights = torch.ones(X.shape[0], device=X.device)
    row_weights = row_weights + belt_mask.to(dtype=row_weights.dtype) * (belt_weight - 1.0)

    return X, is_boundary, row_weights


def _eval_bem_potential(spec: CanonicalSpec, points: torch.Tensor) -> torch.Tensor:
    sol = get_oracle_solution(spec, mode="bem", bem_cfg={})  # type: ignore[arg-type]
    if sol is None:
        raise RuntimeError("BEM solution unavailable for spec.")
    if hasattr(sol, "eval_V_E_batched"):
        V, _ = sol.eval_V_E_batched(points)  # type: ignore[attr-defined]
        return V
    if hasattr(sol, "eval"):
        return sol.eval(points)  # type: ignore[attr-defined]
    raise RuntimeError("BEM solution has no eval interface.")


def _torus_geom(spec: CanonicalSpec) -> Tuple[torch.Tensor, float, float]:
    torus = None
    for c in spec.conductors:
        if c.get("type") in ("torus", "toroid"):
            torus = c
            break
    if torus is None:
        raise ValueError("No torus conductor found in spec.")
    center = torch.tensor(torus.get("center", [0.0, 0.0, 0.0]), dtype=torch.float32)
    R = float(torus.get("major_radius", torus.get("radius", 1.0)))
    a = float(torus.get("minor_radius", 0.25 * R))
    return center, R, a


def _build_primitives(center: torch.Tensor, R: float, a: float) -> List[ImageBasisElement]:
    device = _torch_device()
    dtype = torch.float32
    primitives: List[ImageBasisElement] = []
    # Poloidal orders 0/1/2
    for order in (0, 1, 2):
        primitives.append(
            PoloidalRingBasis(
                {
                    "center": center.to(device=device, dtype=dtype),
                    "radius": torch.tensor(R, device=device, dtype=dtype),
                    "delta_r": torch.tensor(0.5 * a, device=device, dtype=dtype),
                    "order": torch.tensor(order, device=device),
                    "n_quad": torch.tensor(128, device=device),
                }
            )
        )
    # Ladders
    primitives.append(
        RingLadderBasis(
            {
                "center": center.to(device=device, dtype=dtype),
                "radius": torch.tensor(R, device=device, dtype=dtype),
                "minor_radius": torch.tensor(a, device=device, dtype=dtype),
                "variant": torch.tensor(0, device=device),
                "n_quad": torch.tensor(96, device=device),
            }
        )
    )
    primitives.append(
        RingLadderBasis(
            {
                "center": center.to(device=device, dtype=dtype),
                "radius": torch.tensor(R, device=device, dtype=dtype),
                "minor_radius": torch.tensor(a, device=device, dtype=dtype),
                "variant": torch.tensor(1, device=device),
                "n_quad": torch.tensor(96, device=device),
            }
        )
    )
    # Mode clusters m=0/1/2
    for m in (0, 1, 2):
        primitives.append(
            ToroidalModeClusterBasis(
                {
                    "center": center.to(device=device, dtype=dtype),
                    "major_radius": torch.tensor(R, device=device, dtype=dtype),
                    "minor_radius": torch.tensor(a, device=device, dtype=dtype),
                    "mode_m": torch.tensor(m, device=device),
                    "n_phi": torch.tensor(16, device=device),
                    "radial_offset": torch.tensor(0.5 * a, device=device, dtype=dtype),
                }
            )
        )
    return primitives


def _fit_mode(
    primitives: Sequence[ImageBasisElement],
    points: torch.Tensor,
    is_boundary: torch.Tensor,
    target: torch.Tensor,
    reg: float = 1e-8,
    boundary_weight: float = 0.8,
    topk: int = 8,
    row_weights: torch.Tensor | None = None,
) -> List[Tuple[float, ImageBasisElement]]:
    device = points.device
    dtype = points.dtype
    A = build_dictionary(primitives, points, device=device, dtype=dtype)
    # Boundary weighting
    alpha = boundary_weight
    beta = 1.0 - alpha
    base_w = torch.where(
        is_boundary.to(device=device),
        torch.tensor(alpha, device=device, dtype=dtype),
        torch.tensor(beta, device=device, dtype=dtype),
    )
    if row_weights is not None:
        base_w = base_w * row_weights.to(device=device, dtype=dtype)
    W = torch.sqrt(base_w).view(-1, 1)
    A_w = A * W
    b_w = target * W.view(-1)
    ATA = A_w.T @ A_w + reg * torch.eye(A_w.shape[1], device=device, dtype=dtype)
    ATb = A_w.T @ b_w
    try:
        coeffs = torch.linalg.solve(ATA, ATb)
    except Exception:
        coeffs = torch.linalg.pinv(ATA) @ ATb
    # Keep top contributors
    abs_coeff = torch.abs(coeffs)
    keep = torch.topk(abs_coeff, k=min(topk, coeffs.numel()), largest=True)
    idxs = keep.indices.tolist()
    result: List[Tuple[float, ImageBasisElement]] = []
    for i in idxs:
        c = float(coeffs[i].detach().cpu())
        if abs(c) < 1e-8:
            continue
        result.append((c, primitives[i]))
    return result


@dataclass
class LearnedMode:
    index: int
    components: List[Dict]


def learn_modes(
    spec: CanonicalSpec,
    charge_positions: List[Sequence[float]],
    n_points: int,
    ratio_boundary: float,
    n_modes: int,
    reg: float,
    boundary_weight: float,
    topk_components: int,
    belt_weight: float,
    belts: List[Tuple[float, float]],
    family: str,
) -> Dict:
    device = _torch_device()
    X, is_boundary, row_weights = _make_collocation(
        spec,
        n_points=n_points,
        ratio_boundary=ratio_boundary,
        seed=12345,
        belts=belts,
        belt_weight=belt_weight,
    )
    X = X.to(device=device)
    is_boundary = is_boundary.to(device=device)
    row_weights = row_weights.to(device=device)

    # Snapshot potentials for each charge position.
    snapshots = []
    for pos in charge_positions:
        spec_mod = deepcopy(spec)
        for ch in spec_mod.charges:
            if ch.get("type") == "point":
                ch["pos"] = list(pos)
        V = _eval_bem_potential(spec_mod, X)
        snapshots.append(V.detach().to(device=device, dtype=torch.float32))

    V_mat = torch.stack(snapshots, dim=1)  # [N, M]
    # Center for stability.
    V_mean = V_mat.mean(dim=1, keepdim=True)
    V_center = V_mat - V_mean

    # SVD on CPU for robustness.
    U, S, _ = torch.linalg.svd(V_center.cpu(), full_matrices=False)
    U = U.to(device=device)
    S = S.to(device=device)

    center, R, a = _torus_geom(spec)
    primitives = _build_primitives(center, R, a)

    learned: List[LearnedMode] = []
    for k in range(min(n_modes, U.shape[1])):
        mode_vec = U[:, k] * S[k]
        comps = _fit_mode(
            primitives,
            X,
            is_boundary,
            mode_vec.to(device=device, dtype=torch.float32),
            reg=reg,
            boundary_weight=boundary_weight,
            topk=topk_components,
            row_weights=row_weights,
        )
        comp_serialized: List[Dict] = []
        for coeff, elem in comps:
            comp_serialized.append(
                {
                    "coeff": coeff,
                    "elem": elem.serialize(),
                }
            )
        learned.append(LearnedMode(index=k, components=comp_serialized))

    meta = {
        "spec": getattr(spec, "name", "torus"),
        "n_points": n_points,
        "ratio_boundary": ratio_boundary,
        "charge_positions": [list(p) for p in charge_positions],
        "n_modes": n_modes,
        "reg": reg,
        "boundary_weight": boundary_weight,
        "topk_components": topk_components,
        "belt_weight": belt_weight,
        "belts": belts,
        "family": family,
    }
    return {
        "meta": meta,
        "modes": [asdict(m) for m in learned],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn toroidal eigenmodes from BEM snapshots.")
    parser.add_argument("--spec", type=str, default="specs/torus_axis_point_mid.json")
    parser.add_argument("--out", type=str, default="runs/torus/toroidal_eigenmodes_mid.json")
    parser.add_argument("--family", type=str, default="boundary", choices=["boundary", "offaxis"])
    parser.add_argument("--n-points", type=int, default=6000)
    parser.add_argument("--ratio-boundary", type=float, default=0.8)
    parser.add_argument("--n-modes", type=int, default=6)
    parser.add_argument("--reg", type=float, default=1e-6)
    parser.add_argument("--boundary-weight", type=float, default=0.8)
    parser.add_argument("--offaxis-weight", type=float, default=1.0)
    parser.add_argument("--topk-components", type=int, default=8)
    args = parser.parse_args()

    spec = CanonicalSpec.from_json(json.load(open(args.spec)))
    center, R, a = _torus_geom(spec)
    # Source positions: on-axis sweep, near-surface, off-axis at two phis, inner/outer probes.
    charge_positions: List[List[float]] = []
    z_vals = [0.4, 0.6, 0.8, max(0.1, 0.3 * a), max(0.2, 0.5 * a)]
    for z in z_vals:
        charge_positions.append([0.0, 0.0, z])
    # Off-axis small rho at two phis.
    for rho in (0.1, 0.2):
        for phi in (0.0, 0.5 * torch.pi):
            charge_positions.append([rho * float(torch.cos(torch.tensor(phi))), rho * float(torch.sin(torch.tensor(phi))), 0.6])
    # Inner rim probe (slightly inside outer surface) and outer side probe.
    inner_r = max(1e-3, R - a + 0.25 * a)
    outer_r = R + a + 0.25 * a
    charge_positions.append([inner_r, 0.0, 0.0])
    charge_positions.append([0.0, inner_r, 0.0])
    charge_positions.append([outer_r, 0.0, 0.0])
    charge_positions.append([0.0, outer_r, 0.0])

    # Off-axis belts for weighting (r, z)
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

    learned = learn_modes(
        spec,
        charge_positions=charge_positions,
        n_points=args.n_points,
        ratio_boundary=args.ratio_boundary,
        n_modes=args.n_modes,
        reg=args.reg,
        boundary_weight=args.boundary_weight,
        topk_components=args.topk_components,
        belt_weight=args.offaxis_weight if args.family == "offaxis" else 1.0,
        belts=belts,
        family=args.family,
    )

    # Adjust output path to include family if not provided.
    out_path = Path(args.out)
    if out_path.name.endswith(".json") and "eigenmodes" in out_path.name and args.family not in out_path.name:
        stem = out_path.stem
        out_path = out_path.with_name(stem + f"_{args.family}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(learned, f, indent=2)
    print(f"Wrote learned modes to {out_path}")


if __name__ == "__main__":
    main()
