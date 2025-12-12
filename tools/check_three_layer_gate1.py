from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

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


def _find_repo_root(start: Path) -> Path:
    """Walk parents to locate the repo root (marked by .git)."""
    for p in [start, *start.parents]:
        if (p / ".git").exists():
            return p
    return start


def _resolve_spec_path(spec_arg: str | None, manifest: Dict[str, Any], run_path: Path) -> Path:
    """Resolve spec path robustly (handles absolute, repo-relative, manifest-provided)."""
    candidates = []
    if spec_arg:
        candidates.append(Path(spec_arg))
    manifest_spec = manifest.get("spec_path")
    if manifest_spec:
        candidates.append(Path(manifest_spec))

    repo_root = _find_repo_root(Path(__file__).resolve())
    for cand in candidates:
        # Absolute path
        if cand.is_absolute() and cand.exists():
            return cand
        # Try as provided
        if cand.exists():
            return cand
        # Try relative to repo root
        alt = repo_root / cand
        if alt.exists():
            return alt
    if not candidates:
        raise FileNotFoundError("No spec path provided or available in manifest.")
    raise FileNotFoundError(f"Could not resolve spec path from candidates: {candidates}")


def _load_manifest(manifest_path: Path) -> Dict[str, Any]:
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _update_manifest(manifest_path: Path, gate1_metrics: Dict[str, Any], gate1_status: str) -> None:
    manifest = _load_manifest(manifest_path)
    manifest["gate1_metrics"] = gate1_metrics
    manifest["gate1_status"] = gate1_status
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Gate 1 checker for three-layer runs (handles subtract-physical semantics)."
    )
    ap.add_argument("--run", required=True, help="Path to discovered_system.json")
    ap.add_argument(
        "--spec",
        required=False,
        default=None,
        help="Path to CanonicalSpec JSON (defaults to manifest spec_path when omitted)",
    )
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--ratio-boundary", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--subtract-physical",
        action="store_true",
        help="Evaluate induced-only targets and compare directly; otherwise compare full field.",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest path to update (defaults to run_dir/discovery_manifest.json).",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    run_path = Path(args.run)
    manifest_path = args.manifest or (run_path.parent / "discovery_manifest.json")
    manifest = _load_manifest(manifest_path)
    spec_path = _resolve_spec_path(args.spec, manifest, run_path)

    spec = CanonicalSpec.from_json(json.loads(Path(spec_path).read_text()))
    system = load_image_system(run_path)

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
    # Gate 1 decision: require numeric_status ok, conditioning not ill-conditioned, and reasonable residuals.
    numeric_status = manifest.get("numeric_status", "ok")
    condition_status = manifest.get("condition_status", None)
    gate1_status = "pass"
    if numeric_status != "ok":
        gate1_status = "fail"
    if condition_status == "ill_conditioned":
        gate1_status = "fail"
    if not math.isfinite(stats["rel_mae"]) or not math.isfinite(stats["rel_max"]):
        gate1_status = "fail"
    if stats["rel_mae"] > 1e-3 or stats["rel_max"] > 1e-2:
        gate1_status = "fail"

    if manifest_path:
        _update_manifest(manifest_path, stats, gate1_status)

    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
