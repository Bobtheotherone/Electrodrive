"""
Plot r–z error fields from NPZ diagnostics for mid-torus experiments with consistent styling.

Generates:
- Baseline vs best (abs/rel) maps with shared limits and torus overlay.
- Full vs rank-2 comparison at selected z (0.60 and 0.90) with shared limits and overlay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from electrodrive.orchestration.parser import CanonicalSpec


def _load_npz(path: Path):
    data = np.load(path)
    return data["r"], data["z"], data.get("abs_err"), data.get("rel_err")


def _torus_outline(spec: CanonicalSpec):
    torus = next(c for c in spec.conductors if c.get("type") in ("torus", "toroid"))
    R = float(torus.get("major_radius", torus.get("radius", 1.0)))
    a = float(torus.get("minor_radius", 0.25 * R))
    theta = np.linspace(0, 2 * np.pi, 400)
    r = R + a * np.cos(theta)
    z = a * np.sin(theta)
    return r, z


def _bounds(*fields):
    arr = np.concatenate([f.ravel() for f in fields if f is not None])
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    return 0.0, np.nanpercentile(arr, 99)


def _plot_field(r, z, field, title: str, out_path: Path, overlay_r, overlay_z, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        field,
        origin="lower",
        aspect="auto",
        extent=[z.min(), z.max(), r.min(), r.max()],
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    ax.plot(overlay_z, overlay_r, color="white", linewidth=1.2, alpha=0.9)
    ax.set_xlabel("z")
    ax.set_ylabel("r")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="error")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mid-torus error maps from NPZ diagnostics.")
    parser.add_argument("--baseline", type=Path, default=Path("runs/torus/diagnostics/mid_baseline_eigen.npz"))
    parser.add_argument("--best", type=Path, default=Path("runs/torus/diagnostics/mid_bem_highres_trial02.npz"))
    parser.add_argument("--rankfull_z060", type=Path, default=Path("runs/torus/diagnostics/mid_axis_weight_rankfull_z0.60.npz"))
    parser.add_argument("--rank2_z060", type=Path, default=Path("runs/torus/diagnostics/mid_axis_weight_rank2_z0.60.npz"))
    parser.add_argument("--rankfull_z090", type=Path, default=Path("runs/torus/diagnostics/mid_axis_weight_rankfull_z0.90.npz"))
    parser.add_argument("--rank2_z090", type=Path, default=Path("runs/torus/diagnostics/mid_axis_weight_rank2_z0.90.npz"))
    parser.add_argument("--spec", type=Path, default=Path("specs/torus_axis_point_mid.json"))
    parser.add_argument("--outdir", type=Path, default=Path("staging/figures/paper"))
    args = parser.parse_args()

    spec = CanonicalSpec.from_json(json.load(open(args.spec)))
    tor_r, tor_z = _torus_outline(spec)

    # Baseline vs best (z=0.7)
    r, z, abs_b, rel_b = _load_npz(args.baseline)
    _, _, abs_best, rel_best = _load_npz(args.best)
    vmin_abs, vmax_abs = _bounds(abs_b, abs_best)
    vmin_rel, vmax_rel = _bounds(rel_b, rel_best)
    _plot_field(r, z, abs_b, "Baseline abs err (eigen-only)", args.outdir / "fig2a_baseline_abs.png", tor_r, tor_z, vmin_abs, vmax_abs)
    _plot_field(r, z, rel_b, "Baseline rel err", args.outdir / "fig2b_baseline_rel.png", tor_r, tor_z, vmin_rel, vmax_rel)
    _plot_field(r, z, abs_best, "Best (2R+4P) abs err", args.outdir / "fig2c_best_abs.png", tor_r, tor_z, vmin_abs, vmax_abs)
    _plot_field(r, z, rel_best, "Best (2R+4P) rel err", args.outdir / "fig2d_best_rel.png", tor_r, tor_z, vmin_rel, vmax_rel)

    # Full vs rank2 at z=0.60
    r, z, abs_full60, rel_full60 = _load_npz(args.rankfull_z060)
    _, _, abs_r2_60, rel_r2_60 = _load_npz(args.rank2_z060)
    vmin_abs60, vmax_abs60 = _bounds(abs_full60, abs_r2_60)
    vmin_rel60, vmax_rel60 = _bounds(rel_full60, rel_r2_60)
    _plot_field(r, z, abs_full60, "Full (r=6) abs err z=0.60", args.outdir / "fig6a_full_abs_z0.60.png", tor_r, tor_z, vmin_abs60, vmax_abs60)
    _plot_field(r, z, rel_full60, "Full (r=6) rel err z=0.60", args.outdir / "fig6b_full_rel_z0.60.png", tor_r, tor_z, vmin_rel60, vmax_rel60)
    _plot_field(r, z, abs_r2_60, "Rank-2 abs err z=0.60", args.outdir / "fig6c_rank2_abs_z0.60.png", tor_r, tor_z, vmin_abs60, vmax_abs60)
    _plot_field(r, z, rel_r2_60, "Rank-2 rel err z=0.60", args.outdir / "fig6d_rank2_rel_z0.60.png", tor_r, tor_z, vmin_rel60, vmax_rel60)

    # Full vs rank2 at z=0.90 (failure mode)
    r, z, abs_full90, rel_full90 = _load_npz(args.rankfull_z090)
    _, _, abs_r2_90, rel_r2_90 = _load_npz(args.rank2_z090)
    vmin_abs90, vmax_abs90 = _bounds(abs_full90, abs_r2_90)
    vmin_rel90, vmax_rel90 = _bounds(rel_full90, rel_r2_90)
    _plot_field(r, z, abs_full90, "Full (r=6) abs err z=0.90", args.outdir / "fig7a_full_abs_z0.90.png", tor_r, tor_z, vmin_abs90, vmax_abs90)
    _plot_field(r, z, rel_full90, "Full (r=6) rel err z=0.90", args.outdir / "fig7b_full_rel_z0.90.png", tor_r, tor_z, vmin_rel90, vmax_rel90)
    _plot_field(r, z, abs_r2_90, "Rank-2 abs err z=0.90", args.outdir / "fig7c_rank2_abs_z0.90.png", tor_r, tor_z, vmin_abs90, vmax_abs90)
    _plot_field(r, z, rel_r2_90, "Rank-2 rel err z=0.90", args.outdir / "fig7d_rank2_rel_z0.90.png", tor_r, tor_z, vmin_rel90, vmax_rel90)


if __name__ == "__main__":
    main()
