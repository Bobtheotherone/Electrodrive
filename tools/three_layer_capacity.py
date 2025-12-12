from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
import math

import numpy as np
import torch

from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.images.basis import generate_candidate_basis, BasisOperator
from electrodrive.images.search import get_collocation_data
from electrodrive.utils.logging import JsonlLogger


class _StubLogger(JsonlLogger):
    def __init__(self):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _find_repo_root(start: Path) -> Path:
    """Walk upward to locate repo root (identified by .git)."""
    for p in [start, *start.parents]:
        if (p / ".git").exists():
            return p
    return start


def _resolve_spec_path(spec_arg: str | None) -> Path:
    """Resolve spec path robustly (absolute or repo-relative)."""
    if spec_arg is None:
        raise FileNotFoundError("No spec path provided.")
    cand = Path(spec_arg)
    if cand.is_absolute() and cand.exists():
        return cand
    if cand.exists():
        return cand
    repo_root = _find_repo_root(Path(__file__).resolve())
    alt = repo_root / cand
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Could not resolve spec path: {spec_arg}")


def build_spec(eps2: float, h: float) -> CanonicalSpec:
    return CanonicalSpec.from_json(
        {
            "domain": {"bbox": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]},
            "conductors": [],
            "dielectrics": [
                {"name": "region1", "epsilon": 1.0, "z_min": 0.0, "z_max": 5.0},
                {"name": "slab", "epsilon": float(eps2), "z_min": -float(h), "z_max": 0.0},
                {"name": "region3", "epsilon": 1.0, "z_min": -5.0, "z_max": -float(h)},
            ],
            "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.2]}],
            "BCs": "dielectric_interfaces",
            "symmetry": ["rot_z"],
        }
    )


def assemble_dense_operator(spec: CanonicalSpec, basis_types: list[str], n_candidates: int, points: torch.Tensor) -> BasisOperator:
    candidates = generate_candidate_basis(
        spec,
        basis_types=basis_types,
        n_candidates=n_candidates,
        device=points.device,
        dtype=points.dtype,
        rng=torch.Generator(device=points.device).manual_seed(0),
    )
    return BasisOperator(candidates, points=points, device=points.device, dtype=points.dtype)


def family_norms(op: BasisOperator, dense: torch.Tensor) -> dict[str, float]:
    fam_sums = defaultdict(float)
    fam_counts = defaultdict(int)
    for j, elem in enumerate(op.elements):
        fam = getattr(elem, "_group_info", {}).get("family_name", elem.type)
        norm = float(torch.linalg.norm(dense[:, j]).item())
        fam_sums[fam] += norm
        fam_counts[fam] += 1
    return {k: fam_sums[k] / max(1, fam_counts[k]) for k in fam_sums}


def apply_family_scaling(op: BasisOperator, dense: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    norms = family_norms(op, dense)
    if not norms:
        return dense, {}
    target = np.median(list(norms.values()))
    scales = {fam: (target / val if val > 0 else 1.0) for fam, val in norms.items()}
    dense_scaled = dense.clone()
    for j, elem in enumerate(op.elements):
        fam = getattr(elem, "_group_info", {}).get("family_name", elem.type)
        s = scales.get(fam, 1.0)
        dense_scaled[:, j] *= s
    return dense_scaled, scales


def lstsq_metrics(A: torch.Tensor, V: torch.Tensor, is_boundary: torch.Tensor) -> dict:
    sol = torch.linalg.lstsq(A, V).solution
    V_fit = A @ sol
    err = torch.abs(V_fit - V)
    stats = {
        "mae": float(err.mean().item()),
        "max": float(err.max().item()),
    }
    if bool(is_boundary.any()):
        err_b = err[is_boundary]
        stats["boundary_mae"] = float(err_b.mean().item())
        stats["boundary_max"] = float(err_b.max().item())
    else:
        stats["boundary_mae"] = float("nan")
        stats["boundary_max"] = float("nan")
    # Relative metrics scaled by mean |V|
    V_scale = torch.mean(torch.abs(V)).clamp_min(1e-12)
    stats["rel_mae"] = float((err.mean() / V_scale).item())
    stats["rel_max"] = float((err.max() / V_scale).item())
    if bool(is_boundary.any()):
        V_b = V[is_boundary]
        err_b = err[is_boundary]
        V_scale_b = torch.mean(torch.abs(V_b)).clamp_min(1e-12)
        stats["rel_boundary_mae"] = float((err_b.mean() / V_scale_b).item())
        stats["rel_boundary_max"] = float((err_b.max() / V_scale_b).item())
    else:
        stats["rel_boundary_mae"] = float("nan")
        stats["rel_boundary_max"] = float("nan")
    stats["n_cols"] = int(A.shape[1])
    stats["n_rows"] = int(A.shape[0])
    col_norms = torch.linalg.norm(A, dim=0)
    stats["col_norm_min"] = float(col_norms.min().item())
    stats["col_norm_max"] = float(col_norms.max().item())
    try:
        svals = torch.linalg.svdvals(A)
        stats["cond_est"] = float((svals.max() / svals.min().clamp_min(1e-12)).item())
    except Exception:
        stats["cond_est"] = float("nan")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Dense LS capacity diagnostics for three-layer slabs.")
    ap.add_argument("--spec", type=str, default=None, help="Path to spec JSON; if omitted, build from eps2/h.")
    ap.add_argument("--eps2", type=float, default=4.0, help="Middle layer permittivity when spec not provided.")
    ap.add_argument("--h", type=float, default=0.3, help="Slab thickness when spec not provided.")
    ap.add_argument("--basis", type=str, default="axis_point,three_layer_images")
    ap.add_argument("--n-candidates", type=int, default=32)
    ap.add_argument("--n-points", type=int, default=512)
    ap.add_argument("--ratio-boundary", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--family-scale", action="store_true", help="Apply per-family column scaling.")
    ap.add_argument("--manifest", type=Path, default=None, help="Optional manifest path to write condition_status.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_env = str(os.getenv("EDE_IMAGES_DTYPE", "float32")).lower()
    if dtype_env in {"float64", "double"}:
        dtype = torch.float64
    else:
        dtype = torch.float32

    if args.spec:
        spec_path = _resolve_spec_path(args.spec)
        spec = CanonicalSpec.from_json(json.loads(Path(spec_path).read_text()))
    else:
        spec = build_spec(args.eps2, args.h)

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
        subtract_physical_potential=False,
    )

    basis_types = [b.strip() for b in args.basis.split(",") if b.strip()]
    op = assemble_dense_operator(spec, basis_types, args.n_candidates, X)
    A, inv_norms = op.normalized_dense(X)

    stats_base = lstsq_metrics(A, V_gt, is_b)
    output = {"base": stats_base}
    condition_status = "ok"
    cond_est = stats_base.get("cond_est", float("nan"))
    if cond_est and math.isfinite(cond_est) and cond_est > 1e7:
        condition_status = "ill_conditioned"

    if args.family_scale:
        dense_raw = op.to_dense(max_entries=None, targets=X)
        if dense_raw is None:
            raise RuntimeError("Dense dictionary could not be materialized.")
        A_scaled_raw, scales = apply_family_scaling(op, dense_raw)
        A_scaled = A_scaled_raw * inv_norms
        stats_scaled = lstsq_metrics(A_scaled, V_gt, is_b)
        output["scaled"] = stats_scaled
        output["scales"] = scales
        cond_est = stats_scaled.get("cond_est", cond_est)
        if cond_est and math.isfinite(cond_est) and cond_est > 1e7:
            condition_status = "ill_conditioned"

    if args.manifest:
        try:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest.exists() else {}
        except Exception:
            manifest = {}
        manifest["condition_status"] = condition_status
        manifest["cond_est"] = cond_est
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
