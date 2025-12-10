from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from electrodrive.images.io import load_image_system
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.images.search import get_collocation_data
from electrodrive.utils.logging import JsonlLogger


class _StubLogger(JsonlLogger):
    def __init__(self):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def compute_metrics(V_pred: torch.Tensor, V_gt: torch.Tensor, is_boundary: torch.Tensor) -> dict:
    err = torch.abs(V_pred - V_gt)
    stats = {}

    # Absolute metrics
    stats["mae"] = float(err.mean().item()) if err.numel() else float("nan")
    stats["max"] = float(err.max().item()) if err.numel() else float("nan")
    if bool(is_boundary.any()):
        err_b = err[is_boundary]
        stats["boundary_mae"] = float(err_b.mean().item())
        stats["boundary_max"] = float(err_b.max().item())
    else:
        stats["boundary_mae"] = float("nan")
        stats["boundary_max"] = float("nan")

    # Relative metrics scaled by mean |V_gt|
    V_scale = torch.mean(torch.abs(V_gt)).clamp_min(1e-12)
    stats["rel_mae"] = float((err.mean() / V_scale).item()) if err.numel() else float("nan")
    stats["rel_max"] = float((err.max() / V_scale).item()) if err.numel() else float("nan")

    if bool(is_boundary.any()):
        V_scale_b = torch.mean(torch.abs(V_gt[is_boundary])).clamp_min(1e-12)
        stats["rel_boundary_mae"] = float((err_b.mean() / V_scale_b).item())
        stats["rel_boundary_max"] = float((err_b.max() / V_scale_b).item())
    else:
        stats["rel_boundary_mae"] = float("nan")
        stats["rel_boundary_max"] = float("nan")

    return stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Gate 1 checker for three-layer runs (handles subtract-physical semantics)."
    )
    ap.add_argument("--run", required=True, help="Path to discovered_system.json")
    ap.add_argument("--spec", required=True, help="Path to CanonicalSpec JSON")
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--ratio-boundary", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--subtract-physical",
        action="store_true",
        help="Evaluate induced-only targets and compare directly; otherwise compare full field.",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    spec = CanonicalSpec.from_json(json.loads(Path(args.spec).read_text()))
    system = load_image_system(Path(args.run))

    rng = np.random.default_rng(args.seed)
    X, V_gt, is_b = get_collocation_data(
        spec,
        logger=_StubLogger(),
        device=device,
        dtype=dtype,
        return_is_boundary=True,
        rng=rng,
        n_points_override=args.n_points,
        ratio_override=args.ratio_boundary,
        subtract_physical_potential=args.subtract_physical,
    )

    with torch.no_grad():
        V_sys = system.potential(X)
        if not args.subtract_physical:
            # Optionally reconstruct full field by adding physical source if needed.
            pass
    stats = compute_metrics(V_sys, V_gt, is_b)
    stats.update(
        {
            "device": str(device),
            "dtype": str(dtype),
            "n_points": int(X.shape[0]),
            "ratio_boundary": args.ratio_boundary,
            "subtract_physical": bool(args.subtract_physical),
        }
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
