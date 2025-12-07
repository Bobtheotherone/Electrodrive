"""
Plot mean_rel and inner_mean_rel for full vs rank-2/3 truncated weights across selected z.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    metrics_path = Path("runs/torus/stage4_metrics_mid_axis_weight_svd_bem.json")
    data = json.load(open(metrics_path))
    z_vals = sorted({float(rec["z"]) for rec in data})
    ranks = ["full", 2, 3]

    def collect(metric_name: str):
        res = {r: [] for r in ranks}
        for z in z_vals:
            recs = [r for r in data if float(r["z"]) == z]
            for r in ranks:
                key = "full" if r == "full" else r
                rec = next((x for x in recs if x["rank"] == key), None)
                res[r].append(rec["metrics"].get(metric_name, float("nan")) if rec else float("nan"))
        return res

    mean_rel = collect("mean_rel")
    inner_rel = collect("inner_mean_rel")

    outdir = Path("staging/figures/paper")
    outdir.mkdir(parents=True, exist_ok=True)

    def plot_metric(vals: dict, title: str, fname: str):
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in ranks:
            ax.plot(z_vals, vals[r], "-o", label=f"rank {r}")
        ax.set_xlabel("z (axis)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=200)
        plt.close(fig)

    plot_metric(mean_rel, "mean_rel", "fig5_rank_truncation_mean_rel.png")
    plot_metric(inner_rel, "inner_mean_rel", "fig5_rank_truncation_inner_mean_rel.png")


if __name__ == "__main__":
    main()
