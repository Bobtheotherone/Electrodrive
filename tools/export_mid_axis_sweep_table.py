"""
Export axis sweep BEM metrics to CSV.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    data = json.load(open("runs/torus/stage4_metrics_mid_axis_sweep_bem.json"))
    rows = []
    for rec in data:
        m = rec["metrics"]
        rows.append(
            {
                "z": rec["z"],
                "mean_rel": m.get("mean_rel"),
                "inner_mean_rel": m.get("inner_mean_rel"),
                "max_rel": m.get("max_rel"),
            }
        )
    df = pd.DataFrame(rows)
    out_csv = Path("staging/tables/mid_axis_sweep_bem.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
