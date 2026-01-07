import json
from pathlib import Path

from electrodrive.experiments.preflight import RunCounters, write_first_offender, write_preflight_report


def test_preflight_report_writer(tmp_path: Path) -> None:
    counters = RunCounters()
    counters.add("sampled_programs_total", 10)
    counters.add("compiled_ok", 7)
    counters.add("compiled_empty_basis", 2)
    counters.add("compiled_failed", 1)
    counters.add("solved_ok", 6)
    counters.add("fast_scored", 6)
    extra = {"tag": "test_run", "preflight_out": "preflight.json"}
    write_preflight_report(tmp_path, counters, extra)

    report_path = tmp_path / "preflight.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["extra"]["tag"] == "test_run"
    assert payload["counters"]["sampled_programs_total"] == 10
    assert payload["counters"]["compiled_ok"] == 7


def test_preflight_first_offender_once(tmp_path: Path) -> None:
    payload = {"reason": "first", "gen": 0}
    assert write_first_offender(tmp_path, payload)
    first = (tmp_path / "preflight_first_offender.json").read_text(encoding="utf-8")
    assert not write_first_offender(tmp_path, {"reason": "second"})
    second = (tmp_path / "preflight_first_offender.json").read_text(encoding="utf-8")
    assert first == second
