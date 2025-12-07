"""
Plot axis sweep BEM metrics vs z for mid-torus fixed-geometry runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    metrics_path = Path("runs/torus/stage4_metrics_mid_axis_sweep_bem.json")
    out_path = Path("staging/figures/paper/fig3_axis_sweep_errors_vs_z.png")
    data = json.load(open(metrics_path))
    z = [float(rec["z"]) for rec in data]
    mean_rel = [rec["metrics"].get("mean_rel", float("nan")) for rec in data]
    inner_rel = [rec["metrics"].get("inner_mean_rel", float("nan")) for rec in data]
    max_rel = [rec["metrics"].get("max_rel", float("nan")) for rec in data]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(z, mean_rel, "-o", label="mean_rel")
    ax.plot(z, inner_rel, "-s", label="inner_mean_rel")
    ax.set_xlabel("z (axis)")
    ax.set_ylabel("relative error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax2 = ax.twinx()
    ax2.plot(z, max_rel, "--", color="gray", alpha=0.5, label="max_rel")
    ax2.set_ylabel("max_rel")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
