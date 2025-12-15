from __future__ import annotations

import json
from pathlib import Path

from electrodrive.viz.ai_solve import _extract_solver_trace, _load_events


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_ai_solve_trace_handles_msg_and_filename_drift(tmp_path: Path):
    run = tmp_path / "run"
    events = run / "events.jsonl"
    evidence = run / "evidence_log.jsonl"

    shared = {"ts": "2025-12-12T10:15:30Z", "msg": "gmres_iter", "k": 1, "resid_precond": 1e-3}
    ev_rows = [
        shared,
        {"ts": "2025-12-12T10:15:31Z", "event": "gmres_iter", "step": 2, "resid_true": 9e-4},
    ]
    _write_jsonl(events, ev_rows)
    # duplicate in legacy name
    _write_jsonl(evidence, [shared])

    loaded = _load_events(run)
    iters, resids, tile_info = _extract_solver_trace(loaded)

    assert iters == [1, 2]
    assert resids[0] == 1e-3
    assert resids[1] == 9e-4
    assert tile_info == {}
