from __future__ import annotations

import json
from pathlib import Path

from electrodrive.utils.log_normalize import iter_merged_events
from electrodrive.viz.iter_viz import _parse_iter_event


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_iter_viz_parser_handles_msg_event_and_residual_variants(tmp_path: Path):
    run = tmp_path / "run"
    ev = run / "events.jsonl"
    _write_jsonl(
        ev,
        [
            {"ts": "2025-12-12T10:15:30Z", "msg": "gmres_iter", "iter": 3, "resid_precond": 5e-5},
            {"ts": "2025-12-12T10:15:31Z", "event": "gmres_progress", "k": 4, "resid_true": 4e-5},
        ],
    )

    merged = list(iter_merged_events(run))
    samples = [_parse_iter_event(m) for m in merged]
    samples = [s for s in samples if s is not None]

    assert [s.iter for s in samples] == [3, 4]
    assert [round(s.resid, 6) for s in samples] == [5e-05, 4e-05]
