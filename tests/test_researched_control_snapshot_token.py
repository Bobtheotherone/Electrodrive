from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_snapshot_is_string_token(tmp_path: Path):
    """
    ResearchED should provide a function or API path that writes controls for a run.
    This test expects a helper like:
      electrodrive.researched.controls.request_snapshot(control_path: Path, token: str)
    or a generic set_controls() that accepts snapshot token.
    """
    from electrodrive.researched import controls as rc

    assert hasattr(rc, "set_controls") or hasattr(rc, "request_snapshot"), (
        "Provide rc.set_controls(...) or rc.request_snapshot(...) in ResearchED backend"
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    control_path = run_dir / "control.json"

    if hasattr(rc, "request_snapshot"):
        rc.request_snapshot(control_path=control_path, token="token-123")
    else:
        rc.set_controls(control_path=control_path, snapshot="token-123")

    data = json.loads(control_path.read_text())
    assert isinstance(data.get("snapshot"), str)
    assert data.get("snapshot") == "token-123"
    assert data.get("snapshot") is not True
