#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


_GEN_RANK_RE = re.compile(r"gen(\d+)_rank(\d+)_verifier")


@dataclass
class CandidateEntry:
    verifier_dir: Path
    summary_path: Path
    status_a: str
    status_b: str
    status_c: str
    status_d: str
    gen: int
    rank: int


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _gate_status(gates: Dict[str, object], gate: str) -> str:
    payload = gates.get(gate, {})
    if isinstance(payload, dict):
        return str(payload.get("status", "")).lower()
    return ""


def _iter_certificates(run_dir: Path) -> Iterable[Path]:
    pattern = "artifacts/certificates/*_verifier/discovery_certificate.json"
    yield from sorted(run_dir.glob(pattern))


def _parse_gen_rank(verifier_dir: Path) -> Tuple[int, int]:
    match = _GEN_RANK_RE.search(verifier_dir.name)
    if not match:
        return (999999, 999999)
    return (int(match.group(1)), int(match.group(2)))


def _rank_key(entry: CandidateEntry) -> Tuple[int, int, int]:
    status_order = {"pass": 0, "borderline": 1, "fail": 2, "": 3}
    d_rank = status_order.get(entry.status_d, 3)
    return (d_rank, entry.gen, entry.rank)


def collect_bc_passers(run_dir: Path) -> List[CandidateEntry]:
    entries: List[CandidateEntry] = []
    for cert_path in _iter_certificates(run_dir):
        cert = _load_json(cert_path)
        gates = cert.get("gates", {})
        if not isinstance(gates, dict):
            continue
        status_b = _gate_status(gates, "B")
        status_c = _gate_status(gates, "C")
        if status_b != "pass" or status_c != "pass":
            continue
        status_a = _gate_status(gates, "A")
        status_d = _gate_status(gates, "D")
        verifier_dir = cert_path.parent
        gen, rank = _parse_gen_rank(verifier_dir)
        summary_path = verifier_dir.parent / f"gen{gen:03d}_rank{rank}_summary.json"
        entries.append(
            CandidateEntry(
                verifier_dir=verifier_dir,
                summary_path=summary_path,
                status_a=status_a,
                status_b=status_b,
                status_c=status_c,
                status_d=status_d,
                gen=gen,
                rank=rank,
            )
        )
    entries.sort(key=_rank_key)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    entries = collect_bc_passers(run_dir)
    top = entries[: args.limit]
    print(f"run_dir={run_dir}")
    print(f"bc_passers_total={len(entries)} showing={len(top)}")
    for idx, entry in enumerate(top, start=1):
        print(
            f"{idx:02d} verifier_dir={entry.verifier_dir} "
            f"summary={entry.summary_path} "
            f"A={entry.status_a} B={entry.status_b} C={entry.status_c} D={entry.status_d}"
        )


if __name__ == "__main__":
    main()
