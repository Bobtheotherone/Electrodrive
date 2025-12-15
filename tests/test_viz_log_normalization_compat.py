from __future__ import annotations

import json
from pathlib import Path

from electrodrive.utils.log_normalize import iter_merged_events, normalize_record


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_merge_and_normalize(tmp_path: Path):
    run = tmp_path / "run"
    ev = run / "events.jsonl"
    legacy = run / "evidence_log.jsonl"

    shared = {"ts": "2025-12-12T10:15:30Z", "level": "info", "msg": "gmres_iter", "k": 1, "resid_precond": 1e-3}
    ev_rows = [shared, {"ts": "2025-12-12T10:15:31Z", "event": "gmres_iter", "step": 2, "resid_true": 9e-4}]
    _write_jsonl(ev, ev_rows)
    _write_jsonl(legacy, [shared])

    merged = list(iter_merged_events(run))
    assert len(merged) == 2  # duplicate deduped

    first = merged[0]
    assert first["event"] == "gmres_iter"
    assert first["iter"] == 1
    assert first["resid_precond"] == 1e-3
    assert first["resid"] == 1e-3

    second = merged[1]
    assert second["iter"] == 2
    assert second["resid_true"] == 9e-4
    assert second["resid"] == 9e-4
    assert isinstance(second["t"], float)


def test_message_json_parsing():
    embedded = json.dumps({"event": "train_step", "step": 5, "loss": 0.2})
    rec = {"message": embedded, "ts": "2025-12-12T10:15:30Z"}
    norm = normalize_record(rec)
    assert norm["event"] == "train_step"
    assert norm["iter"] == 5
    assert norm["fields"]["loss"] == 0.2
