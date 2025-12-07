"""
Mid-torus stability-aware summary of Stage4 metrics and BEM diagnostics.

Loads mid-related records from existing Stage4 metric JSONs and pairs them
with BEM diagnostic NPZ files (when present) to print a compact table with
collocation and BEM metrics plus a qualitative stability label.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_json(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def diag_stats(path: Path, R: float = 1.0, a: float = 0.35) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    try:
        data = np.load(path)
        r = data["r"]
        z = data["z"]
        rr, zz = np.meshgrid(r, z, indexing="ij")
        abs_err = data["abs_err"]
        rel_err = data["rel_err"]
        mask_inner = (rr >= R - a) & (rr <= R - 0.4 * a) & (np.abs(zz) <= 0.3 * a)
        stats = {
            "max_abs": float(np.max(abs_err)),
            "max_rel": float(np.max(rel_err)),
            "mean_rel": float(np.mean(rel_err)),
        }
        if mask_inner.any():
            stats.update(
                {
                    "inner_max_abs": float(np.max(abs_err[mask_inner])),
                    "inner_mean_abs": float(np.mean(abs_err[mask_inner])),
                    "inner_mean_rel": float(np.mean(rel_err[mask_inner])),
                }
            )
        return stats
    except Exception:
        return None


def label_run(
    run: str,
    diag: Optional[Dict[str, float]],
    baseline_inner_rel: Optional[float],
    baseline_run: str,
) -> str:
    if run == baseline_run:
        return "baseline"
    if diag is None:
        return "no_diag"
    max_rel = diag.get("max_rel", float("inf"))
    inner_rel = diag.get("inner_mean_rel", float("inf"))
    if max_rel > 1e6:
        return "unstable"
    if baseline_inner_rel is None:
        return "unknown"
    if inner_rel > 1.1 * baseline_inner_rel:
        return "negative"
    if inner_rel < baseline_inner_rel:
        return "better"
    return "neutral"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runs_root = root / "runs" / "torus"

    stage4_files = [
        runs_root / "stage4_metrics_modes_families.json",
        runs_root / "stage4_metrics_inner_rim.json",
        runs_root / "stage4_metrics_mid_ribbon_patch.json",
    ]

    # Map run names to diagnostic NPZ files (extend as new diagnostics are created).
    diag_map = {
        "torus_mid_point_toroidal_eigen_mode_boundary_n12_reg0.001_bw0.8_tsFalse_inner": runs_root
        / "diagnostics"
        / "mid_baseline_eigen.npz",
        "torus_mid_point_toroidal_eigen_mode_boundary_inner_rim_arc_n12_reg0.0008_bw0.8_tsFalse_inner": runs_root
        / "diagnostics"
        / "mid_hybrid_arc.npz",
        "mid_hybrid_ribbon_best": runs_root / "diagnostics" / "mid_hybrid_ribbon_best.npz",
        "torus_mid_point_toroidal_eigen_mode_boundary_inner_patch_ring_n12_reg0.001_bw0.75_tsFalse_midpatch": runs_root
        / "diagnostics"
        / "mid_hybrid_patch_best.npz",
        "torus_mid_point_toroidal_eigen_mode_boundary_inner_rim_ribbon_n12_reg0.001_bw0.8_tsFalse_midpatch": runs_root
        / "diagnostics"
        / "mid_hybrid_ribbon_best.npz",
    }

    records: List[dict] = []
    for path in stage4_files:
        for rec in load_json(path):
            if rec.get("spec") != "torus_mid":
                continue
            records.append(rec)

    # Precompute baseline inner_mean_rel for relative comparison.
    baseline_run = "torus_mid_point_toroidal_eigen_mode_boundary_n12_reg0.001_bw0.8_tsFalse_inner"
    baseline_diag = diag_stats(diag_map.get(baseline_run, Path()))
    baseline_inner_rel = baseline_diag.get("inner_mean_rel") if baseline_diag else None

    def fmt(val: float) -> str:
        if val is None or np.isnan(val):
            return "nan"
        if abs(val) >= 1e6:
            return f"{val:.3e}"
        return f"{val:.4f}"

    header = [
        "run",
        "basis",
        "n_img",
        "b_mae",
        "off_rel",
        "belt_rel",
        "diag_inner_rel",
        "diag_mean_rel",
        "diag_max_rel",
        "label",
    ]
    print("\t".join(header))

    for rec in records:
        run = rec.get("run", "")
        m = rec.get("metrics", {})
        basis = ",".join(rec.get("basis_types", []))
        n_img = m.get("n_images") or rec.get("type_counts", {}).get("n_images")
        b_mae = m.get("boundary_mae")
        off_rel = m.get("offaxis_rel")
        belt_rel = m.get("offaxis_belt_rel")

        diag = diag_stats(diag_map.get(run, Path())) if run in diag_map else None
        lab = label_run(run, diag, baseline_inner_rel, baseline_run)

        row = [
            run,
            basis,
            str(n_img or ""),
            fmt(b_mae) if b_mae is not None else "nan",
            fmt(off_rel) if off_rel is not None else "nan",
            fmt(belt_rel) if belt_rel is not None else "nan",
            fmt(diag.get("inner_mean_rel")) if diag else "nan",
            fmt(diag.get("mean_rel")) if diag else "nan",
            fmt(diag.get("max_rel")) if diag else "nan",
            lab,
        ]
        print("\t".join(row))


if __name__ == "__main__":
    main()
