from __future__ import annotations

import json
from pathlib import Path

import pytest


def _merge():
    from electrodrive.researched.ingest import merge as m
    assert hasattr(m, "merge_event_files"), "merge.merge_event_files(events_path, evidence_path) required"
    return m


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_merge_and_dedup(tmp_path: Path):
    m = _merge()

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    ev = run_dir / "events.jsonl"
    legacy = run_dir / "evidence_log.jsonl"

    shared = {"ts": "2025-12-12T10:15:30Z", "level": "info", "msg": "gmres_iter", "iter": 1, "resid": 1e-3}
    _write_jsonl(ev, [shared, {"ts": "2025-12-12T10:15:31Z", "level": "info", "msg": "gmres_iter", "iter": 2, "resid": 9e-4}])
    _write_jsonl(legacy, [shared])  # duplicate on purpose

    merged = list(m.merge_event_files(events_path=ev, evidence_path=legacy))
    assert len(merged) == 2, "Duplicate record should be deduped"
    assert merged[0]["iter"] == 1
    assert merged[1]["iter"] == 2
