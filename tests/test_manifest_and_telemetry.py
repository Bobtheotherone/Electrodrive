from __future__ import annotations

import json
from pathlib import Path

from electrodrive.utils.telemetry import TelemetryWriter
from electrodrive.utils.manifest import build_manifest, write_manifest


def test_manifest_schema_and_phase(tmp_path: Path):
    identities = {"problem_hash": "p", "mesh_hash": "m", "kernel_hash": "k"}
    kernel = {"backend": "bem"}
    manifest = build_manifest(
        identities=identities,
        kernel=kernel,
        code_commit="abc",
        extra_env={"pre_solve": True},
        run_id="R",
    )
    path = write_manifest(tmp_path / "manifest.json", manifest)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["run_id"] == "R"
    assert "device" in data
    assert "tf32" in data["device"]
    assert "tf32_effective_mode" in data["device"]
    assert data["env"].get("pre_solve") is True


def test_telemetry_writer_creates_and_appends(tmp_path: Path, monkeypatch):
    path = tmp_path / "trace.csv"
    headers = ["ts", "iter"]
    monkeypatch.setenv("EDE_SOLVER_TRACE_MAX_ROWS", "2")
    tw = TelemetryWriter(path, headers)
    tw.append_row({"ts": 1.0, "iter": 0})
    tw.append_row({"ts": 2.0, "iter": 1})
    tw.append_row({"ts": 3.0, "iter": 2})
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 3
