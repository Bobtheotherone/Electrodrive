"""
SVD and low-rank diagnostics for the mid-torus axis weight dataset.

Loads runs/torus/mid_axis_weights.json, computes singular values, reconstructs
rank-r weight families, and runs BEM diagnostics for selected z positions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from electrodrive.images.search import ImageSystem
from electrodrive.orchestration.parser import CanonicalSpec
from tools.mid_torus_bem_fmm_refine import (
    load_seed_params,
    build_elements,
    highres_bem_diag,
)
from tools.mid_torus_axis_sweep import move_charge


def reconstruct_weights(U: np.ndarray, S: np.ndarray, VT: np.ndarray, rank: int) -> np.ndarray:
    Ur = U[:, :rank]
    Sr = S[:rank]
    VTr = VT[:rank, :]
    return (Ur * Sr.reshape(1, -1)) @ VTr


def main() -> None:
    parser = argparse.ArgumentParser(description="SVD analysis and BEM truncation diagnostics for axis weights.")
    parser.add_argument("--dataset", type=Path, default=Path("runs/torus/mid_axis_weights.json"))
    parser.add_argument("--svd-out", type=Path, default=Path("runs/torus/mid_axis_weight_svd.json"))
    parser.add_argument("--metrics-out", type=Path, default=Path("runs/torus/stage4_metrics_mid_axis_weight_svd_bem.json"))
    parser.add_argument("--diagnostics-root", type=Path, default=Path("runs/torus/diagnostics"))
    parser.add_argument("--ranks", type=str, default="2,3")
    parser.add_argument("--z-bem", type=str, default="0.40,0.60,0.70,0.90")
    parser.add_argument("--nr", type=int, default=200)
    parser.add_argument("--nz", type=int, default=200)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data = json.load(open(root / args.dataset))
    meta = data["meta"]
    records = data["records"]
    z_vals = [float(rec["z"]) for rec in records]
    W = np.stack([rec["weights"] for rec in records], axis=1)  # shape (6, M)

    U, S, VT = np.linalg.svd(W, full_matrices=False)
    sigma_rel = (S / S[0]).tolist()
    rank_thresh = {}
    for tol in (1e-1, 1e-2, 1e-3):
        rank_thresh[str(tol)] = int(np.sum(S / S[0] > tol))
    svd_info = {
        "singular_values": S.tolist(),
        "singular_values_rel": sigma_rel,
        "rank_thresholds": rank_thresh,
        "z_values": z_vals,
    }
    Path(root / args.svd_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(svd_info, open(root / args.svd_out, "w"), indent=2)
    print("Singular values (normalized):", sigma_rel)
    print("Rank thresholds:", rank_thresh)

    # Prep geometry
    spec = CanonicalSpec.from_json(json.load(open(root / "specs" / "torus_axis_point_mid.json")))
    rings_seed, pts_seed = load_seed_params(root / meta["geometry_path"])
    torus = next(c for c in spec.conductors if c.get("type") in ("torus", "toroid"))
    center = torch.tensor(torus.get("center", [0.0, 0.0, 0.0]), device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32)
    device = center.device
    dtype = torch.float32

    ranks = [int(r) for r in args.ranks.split(",") if r.strip()]
    z_eval = [float(z) for z in args.z_bem.split(",") if z.strip()]

    metrics: List[Dict[str, object]] = []
    bem_cfg = {
        "use_gpu": True,
        "fp64": True,
        "initial_h": 0.15,
        "max_refine_passes": 4,
        "min_refine_passes": 2,
        "gmres_tol": 1e-8,
        "target_bc_inf_norm": 1e-8,
        "use_near_quadrature": True,
        "use_near_quadrature_matvec": False,
        "tile_mem_divisor": 2.5,
        "target_vram_fraction": 0.9,
    }

    # Map z -> weights (full) for quick lookup
    weights_full: Dict[float, Sequence[float]] = {float(rec["z"]): rec["weights"] for rec in records}
    for z in z_eval:
        spec_z = move_charge(spec, z)
        elems = build_elements(rings_seed, pts_seed, center=center, device=device, dtype=dtype)
        # Full weights
        w_full = torch.tensor(weights_full[z], device=device, dtype=dtype)
        sys_full = ImageSystem(elems, w_full)
        npz_full = root / args.diagnostics_root / f"mid_axis_weight_rankfull_z{z:.2f}.npz"
        stats_full = highres_bem_diag(spec_z, sys_full, npz_full, nr=args.nr, nz=args.nz, bem_cfg=bem_cfg)
        rec_full = {
            "z": z,
            "rank": "full",
            "metrics": stats_full,
            "npz_path": str(npz_full),
        }
        metrics.append(rec_full)
        print(f"[BEM full] z={z:.2f} stats={stats_full}")

        # Low-rank reconstructions
        for r in ranks:
            Wr = reconstruct_weights(U, S, VT, r)
            w_r = torch.tensor(Wr[:, z_vals.index(z)], device=device, dtype=dtype)
            sys_r = ImageSystem(elems, w_r)
            npz_r = root / args.diagnostics_root / f"mid_axis_weight_rank{r}_z{z:.2f}.npz"
            stats_r = highres_bem_diag(spec_z, sys_r, npz_r, nr=args.nr, nz=args.nz, bem_cfg=bem_cfg)
            rec_r = {
                "z": z,
                "rank": r,
                "metrics": stats_r,
                "npz_path": str(npz_r),
            }
            metrics.append(rec_r)
            print(f"[BEM rank {r}] z={z:.2f} stats={stats_r}")

    Path(root / args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(root / args.metrics_out, "w"), indent=2)
    print(f"Saved BEM diagnostics to {root / args.metrics_out}")


if __name__ == "__main__":
    main()
