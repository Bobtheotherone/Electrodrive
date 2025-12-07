"""
Plot singular values (relative) of the axis weight matrix.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    data = json.load(open("runs/torus/mid_axis_weight_svd.json"))
    sigma_rel = data["singular_values_rel"]
    out_path = Path("staging/figures/paper/fig4_singular_values.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.semilogy(range(1, len(sigma_rel) + 1), sigma_rel, "-o")
    ax.axhline(0.1, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0.01, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("mode index")
    ax.set_ylabel("sigma_i / sigma_1")
    ax.set_title("Axis weight singular values")
    ax.grid(True, which="both", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
