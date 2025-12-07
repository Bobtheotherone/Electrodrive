"""
Plot weights vs z (rings and points) and optional mode coefficients from SVD.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_weights(z, W, basis_order, out_path: Path):
    z = np.array(z)
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    # Rings (assumed first two entries)
    axes[0].plot(z, W[0], "-o", label=basis_order[0])
    axes[0].plot(z, W[1], "-o", label=basis_order[1])
    axes[0].set_ylabel("weight")
    axes[0].set_title("Ring weights vs z")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    # Points
    for i in range(2, W.shape[0]):
        axes[1].plot(z, W[i], "-o", label=basis_order[i])
    axes[1].set_xlabel("z (axis)")
    axes[1].set_ylabel("weight")
    axes[1].set_title("Point weights vs z")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_mode_coeffs(z, U, S, Vt, r_max: int, out_path: Path):
    z = np.array(z)
    coeffs = np.diag(S) @ Vt  # shape (6, M)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(min(r_max, coeffs.shape[0])):
        ax.plot(z, coeffs[i], "-o", label=f"mode {i+1}")
    ax.set_xlabel("z (axis)")
    ax.set_ylabel("mode coefficient")
    ax.set_title("Axis weight mode coefficients vs z")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot axis weight behavior vs z.")
    ap.add_argument("--weights-json", type=Path, default=Path("runs/torus/mid_axis_weights.json"))
    ap.add_argument("--svd-json", type=Path, default=Path("runs/torus/mid_axis_weight_svd.json"))
    ap.add_argument("--outdir", type=Path, default=Path("staging/figures/paper"))
    args = ap.parse_args()

    data = json.load(open(args.weights_json))
    basis_order = data["meta"]["basis_order"]
    z_vals = [float(rec["z"]) for rec in data["records"]]
    W = np.stack([rec["weights"] for rec in data["records"]], axis=1)  # (6, M)

    plot_weights(z_vals, W, basis_order, args.outdir / "fig9_weights_vs_z.png")

    try:
        svd = json.load(open(args.svd_json))
        U = np.array(svd["U"]) if "U" in svd else None
    except Exception:
        U = None
    if U is None:
        # reconstruct from numpy SVD for mode coefficients
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
    else:
        S = np.array(svd["singular_values"])
        Vt = np.array(svd["VT"])
    plot_mode_coeffs(z_vals, U, S, Vt, r_max=3, out_path=args.outdir / "fig9b_mode_coeffs_vs_z.png")


if __name__ == "__main__":
    main()
