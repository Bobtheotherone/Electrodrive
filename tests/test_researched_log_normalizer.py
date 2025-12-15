from __future__ import annotations

import json
from typing import Any, Dict

import pytest


def _normalizer():
    from electrodrive.researched.ingest import normalizer as n
    assert hasattr(n, "normalize_record"), "normalizer.normalize_record(rec, ...) is required"
    return n


def test_event_name_fallbacks():
    n = _normalizer()

    rec1 = {"ts": "2025-12-12T10:15:30Z", "level": "info", "msg": "gmres_iter", "iter": 1, "resid": 1e-3}
    out1 = n.normalize_record(rec1, source="events.jsonl")
    assert out1["event"] == "gmres_iter"
    assert out1["iter"] == 1
    assert abs(out1["resid"] - 1e-3) < 1e-12
    assert isinstance(out1["t"], float)

    rec2 = {"ts": "2025-12-12T10:15:30Z", "level": "info", "event": "gmres_iter", "k": 2, "resid_true": 9e-4}
    out2 = n.normalize_record(rec2, source="evidence_log.jsonl")
    assert out2["event"] == "gmres_iter"
    assert out2["iter"] == 2
    assert abs(out2["resid"] - 9e-4) < 1e-12
    assert abs(out2["resid_true"] - 9e-4) < 1e-12

    embedded = json.dumps({"event": "train_step", "step": 10, "loss": 0.12})
    rec3 = {"ts": "2025-12-12T10:15:30Z", "level": "info", "message": embedded}
    out3 = n.normalize_record(rec3, source="train_log.jsonl")
    assert out3["event"] == "train_step"
    assert out3["iter"] == 10
    assert out3["fields"].get("loss") == 0.12


def test_residual_variants_precedence():
    n = _normalizer()
    rec = {
        "ts": "2025-12-12T10:15:30Z",
        "level": "info",
        "msg": "gmres_iter",
        "iter": 3,
        "resid_precond": 1e-6,
        "resid_true": 2e-6,
    }
    out = n.normalize_record(rec, source="events.jsonl")
    assert out["resid_precond"] == 1e-6
    assert out["resid_true"] == 2e-6
    assert out["resid"] in (1e-6, 2e-6)  # spec: resid = resid OR resid_precond OR resid_true


def test_timestamp_parsing_numeric():
    n = _normalizer()
    rec = {"ts": 1765534530.0, "level": "info", "msg": "x"}
    out = n.normalize_record(rec, source="events.jsonl")
    assert out["t"] == pytest.approx(1765534530.0, rel=0, abs=1e-9)
