from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict


@dataclass
class RunCounters:
    sampled_programs_total: int = 0
    compiled_ok: int = 0
    compiled_empty_basis: int = 0
    compiled_failed: int = 0
    assembled_ok: int = 0
    assembled_failed: int = 0
    solved_ok: int = 0
    solved_failed: int = 0
    fast_scored: int = 0
    mid_scored: int = 0
    verified_attempted: int = 0
    verified_written: int = 0
    nonfinite_pred_count: int = 0
    nonfinite_pred_total: int = 0
    proxy_computed_count: int = 0
    proxy_score_nonfinite_sanitized: int = 0
    complex_guard_failed: int = 0
    weights_empty: int = 0
    a_train_nonfinite_count: int = 0
    a_train_total: int = 0
    a_hold_nonfinite_count: int = 0
    a_hold_total: int = 0
    v_train_nonfinite_count: int = 0
    v_train_total: int = 0
    weights_nonfinite_count: int = 0
    weights_total: int = 0
    holdout_nonfinite_candidate_count: int = 0
    dcim_baseline_nonfinite_count: int = 0
    interior_metric_nonfinite_count: int = 0
    lap_metric_nonfinite_count: int = 0
    holdout_total: int = 0
    holdout_boundary_total: int = 0
    holdout_interior_total: int = 0
    holdout_boundary_empty_count: int = 0
    holdout_interior_empty_count: int = 0
    holdout_denom_nonfinite_count: int = 0

    def add(self, key: str, n: int = 1) -> None:
        if not hasattr(self, key):
            raise KeyError(f"Unknown counter '{key}'")
        value = int(getattr(self, key))
        setattr(self, key, value + int(n))

    def as_dict(self) -> Dict[str, int]:
        return {f.name: int(getattr(self, f.name)) for f in fields(self)}


def write_preflight_report(out_dir: Path, counters: RunCounters, extra: Dict[str, Any]) -> None:
    out_name = str(extra.get("preflight_out", "preflight.json")).strip() or "preflight.json"
    payload = {
        "counters": counters.as_dict(),
        "extra": dict(extra),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_first_offender(out_dir: Path, payload: Dict[str, Any]) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preflight_first_offender.json"
    if out_path.exists():
        return False
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return True


def summarize_to_stdout(counters: RunCounters) -> None:
    c = counters
    print(
        "PREFLIGHT summary:"
        f" sampled={c.sampled_programs_total}"
        f" compiled_ok={c.compiled_ok}"
        f" empty_basis={c.compiled_empty_basis}"
        f" compiled_failed={c.compiled_failed}"
        f" solved_ok={c.solved_ok}"
        f" fast_scored={c.fast_scored}"
        f" verified_written={c.verified_written}"
    )


__all__ = [
    "RunCounters",
    "summarize_to_stdout",
    "write_first_offender",
    "write_preflight_report",
]
