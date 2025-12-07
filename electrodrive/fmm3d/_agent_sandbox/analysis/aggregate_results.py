import json
import csv
from pathlib import Path
from typing import List, Dict


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = SANDBOX_ROOT / "experiments" / "results"
OUTPUT_CSV = RESULTS_DIR / "summary.csv"


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def aggregate_runs(records: List[Dict]) -> Dict:
    if not records:
        return {
            "count": 0,
            "mean_rel_l2_error": None,
            "max_rel_l2_error": None,
        }
    rel_errs = [r.get("metrics", {}).get("rel_l2_error") for r in records]
    rel_errs = [x for x in rel_errs if x is not None]
    count = len(records)
    mean_err = sum(rel_errs) / len(rel_errs) if rel_errs else None
    max_err = max(rel_errs) if rel_errs else None
    return {
        "count": count,
        "mean_rel_l2_error": mean_err,
        "max_rel_l2_error": max_err,
    }


def main() -> None:
    summaries: List[Dict] = []
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        records = load_jsonl(path)
        agg = aggregate_runs(records)
        summaries.append(
            {
                "file": path.name,
                **agg,
            }
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "count", "mean_rel_l2_error", "max_rel_l2_error"])
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


if __name__ == "__main__":
    main()
