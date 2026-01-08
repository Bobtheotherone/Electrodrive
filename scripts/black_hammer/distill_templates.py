#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from electrodrive.verify.utils import sha256_json


def _iter_records(path: Path) -> Iterable[Tuple[Dict[str, Any], str]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield rec, f"{path.name}:{idx}"
        return
    if path.suffix == ".json":
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if isinstance(rec, dict):
            yield rec, path.name


def _gather_candidates(input_dir: Path) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for path in sorted(input_dir.rglob("*.json")):
        for rec, tag in _iter_records(path):
            if "program" not in rec and "elements" not in rec:
                continue
            rec = dict(rec)
            rec["_source"] = str(path)
            rec["_tag"] = tag
            candidates.append(rec)
    for path in sorted(input_dir.rglob("*.jsonl")):
        for rec, tag in _iter_records(path):
            if "program" not in rec and "elements" not in rec:
                continue
            rec = dict(rec)
            rec["_source"] = str(path)
            rec["_tag"] = tag
            candidates.append(rec)
    return candidates


def _program_fingerprint(record: Dict[str, Any]) -> Tuple[str, Any]:
    program = record.get("program")
    if program is None:
        elements = record.get("elements", [])
        program = [{"type": elem.get("type")} for elem in elements]
    return sha256_json(program), program


def _write_template_csv(path: Path, entries: List[Dict[str, Any]]) -> None:
    fieldnames = ["candidate_id", "source", "program_hash", "weights", "params", "metrics"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "candidate_id": entry["candidate_id"],
                    "source": entry["source"],
                    "program_hash": entry["program_hash"],
                    "weights": json.dumps(entry.get("weights"), ensure_ascii=True),
                    "params": json.dumps(entry.get("params"), ensure_ascii=True),
                    "metrics": json.dumps(entry.get("metrics"), ensure_ascii=True),
                }
            )


def _write_template_npz(path: Path, entries: List[Dict[str, Any]]) -> None:
    np.savez(
        path,
        candidate_id=np.array([e["candidate_id"] for e in entries], dtype=object),
        weights=np.array([e.get("weights") for e in entries], dtype=object),
        params=np.array([e.get("params") for e in entries], dtype=object),
        metrics=np.array([e.get("metrics") for e in entries], dtype=object),
    )


def distill_templates(input_dir: Path, output_dir: Path) -> Dict[str, Any]:
    candidates = _gather_candidates(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters: Dict[str, List[Dict[str, Any]]] = {}
    for rec in candidates:
        fingerprint, program = _program_fingerprint(rec)
        entry = {
            "candidate_id": rec.get("_tag", ""),
            "source": rec.get("_source", ""),
            "program": program,
            "program_hash": fingerprint,
            "weights": rec.get("weights"),
            "params": rec.get("elements") or rec.get("params") or {},
            "metrics": rec.get("metrics", {}),
        }
        clusters.setdefault(fingerprint, []).append(entry)

    summaries: List[Dict[str, Any]] = []
    for fingerprint in sorted(clusters.keys()):
        entries = clusters[fingerprint]
        program = entries[0]["program"] if entries else []
        summaries.append(
            {
                "fingerprint": fingerprint,
                "count": len(entries),
                "program": program,
            }
        )
        stem = f"template_{fingerprint}"
        _write_template_csv(output_dir / f"{stem}.csv", entries)
        _write_template_npz(output_dir / f"{stem}.npz", entries)

    clusters_path = output_dir / "clusters.json"
    clusters_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    return {"total_candidates": len(candidates), "total_templates": len(summaries), "clusters": summaries}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cluster GFDSL templates by structural fingerprint.")
    parser.add_argument("input_dir", type=Path, help="Directory containing run artifacts.")
    parser.add_argument("output_dir", type=Path, help="Output directory for distilled templates.")
    args = parser.parse_args(argv)

    summary = distill_templates(args.input_dir, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
