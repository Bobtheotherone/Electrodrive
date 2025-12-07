import json
import math
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = SANDBOX_ROOT / "experiments" / "results"
INPUT_FILE = RESULTS_DIR / "l2l_sweep.jsonl"
OUTPUT_CSV = RESULTS_DIR / "l2l_sweep_summary.csv"


def load_records(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def mean_ignore_nan(vals: List[float]) -> float:
    clean = [v for v in vals if v is not None and not math.isnan(v)]
    if not clean:
        return math.nan
    return sum(clean) / len(clean)


def aggregate(records: List[Dict]) -> Tuple[List[Dict], int]:
    grouped: Dict[Tuple[int, float], List[Dict]] = defaultdict(list)
    p_max = 0
    for r in records:
        p = int(r.get("parameters", {}).get("p", -1))
        tnorm = float(r.get("parameters", {}).get("translation_norm_over_scale", math.nan))
        grouped[(p, tnorm)].append(r)
        p_max = max(p_max, p)

    summary_rows: List[Dict] = []
    for (p, tnorm), runs in sorted(grouped.items(), key=lambda k: (k[0][0], k[0][1])):
        rel_errs = []
        max_errs = []
        per_order_rel: Dict[int, List[float]] = defaultdict(list)
        per_order_bias: Dict[int, List[float]] = defaultdict(list)
        for r in runs:
            metrics = r.get("metrics", {})
            rel_errs.append(metrics.get("rel_l2_error"))
            max_errs.append(metrics.get("max_abs_error"))
            spectral = metrics.get("spectral", {})
            per_order = spectral.get("per_order_rel_error", {})
            bias = spectral.get("bias_factor_alpha", {})
            for l_str, v in per_order.items():
                try:
                    l = int(l_str)
                except Exception:
                    continue
                per_order_rel[l].append(v)
            for l_str, v in bias.items():
                try:
                    l = int(l_str)
                except Exception:
                    continue
                per_order_bias[l].append(v)

        row: Dict[str, float] = {
            "p": p,
            "translation_norm_over_scale": tnorm,
            "count": len(runs),
            "mean_rel_l2_error": mean_ignore_nan(rel_errs),
            "max_rel_l2_error": max(max_errs) if max_errs else math.nan,
        }
        for l in range(p_max + 1):
            row[f"mean_rel_l2_l{l}"] = mean_ignore_nan(per_order_rel.get(l, []))
            row[f"mean_bias_alpha_l{l}"] = mean_ignore_nan(per_order_bias.get(l, []))
        summary_rows.append(row)

    return summary_rows, p_max


def write_csv(rows: List[Dict], p_max: int, path: Path) -> None:
    base_fields = ["p", "translation_norm_over_scale", "count", "mean_rel_l2_error", "max_rel_l2_error"]
    per_order_fields: List[str] = []
    for l in range(p_max + 1):
        per_order_fields.append(f"mean_rel_l2_l{l}")
        per_order_fields.append(f"mean_bias_alpha_l{l}")
    fieldnames = base_fields + per_order_fields

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    records = load_records(INPUT_FILE)
    rows, p_max = aggregate(records)
    write_csv(rows, p_max, OUTPUT_CSV)


if __name__ == "__main__":
    main()
