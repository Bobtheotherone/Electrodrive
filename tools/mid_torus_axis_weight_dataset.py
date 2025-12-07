"""
Build a dense weight dataset for the fixed mid-torus 2-ring + 4-point geometry.

Reuses the geometry from mid_bem_highres_trial02, solves only for weights for a
list of axis source positions, and saves:
- discovered_system.json per z
- a consolidated JSON with weights, ordering metadata, and Stage-style metrics.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch

from electrodrive.images.io import save_image_system
from electrodrive.orchestration.parser import CanonicalSpec
from tools.mid_torus_bem_fmm_refine import (
    RingParams,
    PointParams,
    load_seed_params,
    build_elements,
    belts_inner,
)
from tools.mid_torus_axis_sweep import move_charge, solve_fixed_geometry, _NullLogger
from tools.run_grandchallenge_experiments import evaluate_system


def _default_z_list() -> List[float]:
    return [round(0.40 + 0.05 * i, 2) for i in range(11)]  # 0.40 ... 0.90


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense axis weight dataset for fixed 2-ring + 4-point geometry.")
    parser.add_argument("--z-list", type=str, default=",".join(f"{z:.2f}" for z in _default_z_list()))
    parser.add_argument("--geometry-path", type=Path, default=Path("runs/torus/discovered/mid_bem_highres_trial02/discovered_system.json"))
    parser.add_argument("--out-json", type=Path, default=Path("runs/torus/mid_axis_weights.json"))
    parser.add_argument("--discovered-root", type=Path, default=Path("runs/torus/discovered"))
    parser.add_argument("--n-colloc", type=int, default=4096)
    parser.add_argument("--ratio-boundary", type=float, default=0.8)
    parser.add_argument("--reg-l1", type=float, default=4e-4)
    parser.add_argument("--point-reg-mult", type=float, default=4.0)
    parser.add_argument("--boundary-weight", type=float, default=0.9)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    spec = CanonicalSpec.from_json(json.load(open(root / "specs" / "torus_axis_point_mid.json")))
    torus = next(c for c in spec.conductors if c.get("type") in ("torus", "toroid"))
    center = torch.tensor(torus.get("center", [0.0, 0.0, 0.0]), device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32)
    device = center.device
    dtype = torch.float32

    rings_seed, pts_seed = load_seed_params(root / args.geometry_path)
    belts = belts_inner(spec)
    per_type_reg = {"poloidal_ring": args.reg_l1, "point": args.reg_l1 * args.point_reg_mult}

    z_vals = [float(z) for z in args.z_list.split(",") if z.strip()]
    records: List[Dict[str, object]] = []

    for z in z_vals:
        spec_z = move_charge(spec, z)
        elems = build_elements(rings_seed, pts_seed, center=center, device=device, dtype=dtype)
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
        stage = evaluate_system(spec_z, system, n_eval=args.n_colloc, ratio_boundary=args.ratio_boundary, belts=belts)
        sys_dir = root / args.discovered_root / f"mid_axis_sweep_z{z:.2f}"
        save_image_system(system, sys_dir / "discovered_system.json", metadata={"z": z, "reg_l1": args.reg_l1, "boundary_weight": args.boundary_weight})
        rec = {
            "z": z,
            "weights": system.weights.detach().cpu().tolist(),
            "metrics": stage.metrics,
            "n_images": len(system.elements),
            "type_counts": stage.type_counts,
            "system_path": str(sys_dir / "discovered_system.json"),
            "reg_l1": args.reg_l1,
            "boundary_weight": args.boundary_weight,
            "n_colloc": args.n_colloc,
            "ratio_boundary": args.ratio_boundary,
        }
        records.append(rec)
        print(f"[dataset] z={z:.2f} weights={rec['weights']}")

    meta = {
        "geometry_path": str(args.geometry_path),
        "rings": [asdict(r) for r in rings_seed],
        "points": [asdict(p) for p in pts_seed],
        "basis_order": ["ring1", "ring2", "pt1", "pt2", "pt3", "pt4"],
        "reg_l1": args.reg_l1,
        "point_reg_mult": args.point_reg_mult,
        "boundary_weight": args.boundary_weight,
        "n_colloc": args.n_colloc,
        "ratio_boundary": args.ratio_boundary,
    }
    out = {"meta": meta, "records": records}
    out_path = root / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"Saved weight dataset to {out_path}")


if __name__ == "__main__":
    main()
