from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import json


NONFINITE_STRINGS = {"NaN", "Infinity", "-Infinity"}


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _is_nonfinite(value: object) -> Tuple[bool, str]:
    if isinstance(value, float):
        return (not math.isfinite(value), "float")
    if isinstance(value, int):
        return (False, "int")
    if isinstance(value, str):
        if value in NONFINITE_STRINGS:
            return (True, "string")
        return (False, "string")
    return (False, type(value).__name__)


def scan(run_dir: Path, max_files: int) -> None:
    cert_dir = run_dir / "artifacts" / "certificates"
    summaries = sorted(cert_dir.glob("*_summary.json"))[:max_files]
    if not summaries:
        raise SystemExit(f"No summary JSONs found in {cert_dir}")

    fields = [
        "proxy_gateD_variance",
        "proxy_gateD_rel_change",
        "proxy_gateB_max_d_jump",
        "proxy_score",
    ]

    counts = {field: 0 for field in fields}
    string_counts = {field: 0 for field in fields}
    sample = []

    for path in summaries:
        metrics = _load_json(path).get("metrics", {})
        for field in fields:
            value = metrics.get(field)
            nonfinite, kind = _is_nonfinite(value)
            if nonfinite:
                counts[field] += 1
                if kind == "string":
                    string_counts[field] += 1
                if len(sample) < 5:
                    sample.append((str(path), field, value, kind))

    print(f"scanned={len(summaries)} summaries")
    for field in fields:
        print(
            f"{field}: nonfinite={counts[field]} string_nonfinite={string_counts[field]}"
        )
    if sample:
        print("samples:")
        for entry in sample:
            print(entry)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-files", type=int, default=12)
    args = parser.parse_args()

    scan(Path(args.run_dir), args.max_files)


if __name__ == "__main__":
    main()
