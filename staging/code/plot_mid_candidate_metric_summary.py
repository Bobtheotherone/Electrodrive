"""
Plot and export summary metrics for baseline→seed→trial003→trial02.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    data = json.load(open("runs/torus/mid_candidate_metrics_combined.json"))
    order = ["baseline", "seed", "trial003", "bem_highres_trial02"]
    labels = ["baseline", "seed (2R+4P)", "trial003", "trial02 (hi-res)"]

    rows = []
    for key, label in zip(order, labels):
        m = data[key]
        rows.append(
            {
                "name": label,
                "n_images": {"baseline": 12, "seed": 6, "trial003": 6, "bem_highres_trial02": 6}[key],
                "mean_rel": m["mean_rel"],
                "inner_mean_rel": m["inner_mean_rel"],
                "max_rel": m["max_rel"],
            }
        )
    df = pd.DataFrame(rows)

    out_csv = Path("staging/tables/mid_candidate_metrics_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Plot mean_rel and inner_mean_rel bars
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    axes[0].bar(df["name"], df["mean_rel"], color="steelblue")
    axes[0].set_title("mean_rel")
    axes[0].set_ylabel("relative error")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(df["name"], df["inner_mean_rel"], color="darkorange")
    axes[1].set_title("inner_mean_rel")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("Mid-torus candidate metrics (BEM)", fontsize=12)
    fig.tight_layout()
    out_fig = Path("staging/figures/paper/fig10_candidate_metrics_summary.png")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
