"""
Histogram of relative error for the best system (mid_bem_highres_trial02).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=Path("runs/torus/diagnostics/mid_bem_highres_trial02.npz"))
    parser.add_argument("--out", type=Path, default=Path("staging/figures/paper/fig8_error_histogram_z0.70.png"))
    args = parser.parse_args()

    data = np.load(args.npz)
    rel = data["rel_err"].ravel()
    rel = rel[np.isfinite(rel)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rel, bins=100, log=True, color="steelblue", alpha=0.8)
    ax.set_xlabel("relative error")
    ax.set_ylabel("count (log)")
    ax.set_title("Error distribution (best system z≈0.70)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
