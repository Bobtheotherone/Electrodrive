from __future__ import annotations

import json
from pathlib import Path

import pytest


def _contract_module():
    """
    Adapt these imports to actual module names.
    Expect a function that initializes a run directory and writes:
      - manifest.json (status=running)
      - command.txt
      - artifacts/ and plots/
      - optionally report.html placeholder
      - optional bridging for events.jsonl <-> evidence_log.jsonl
    """
    from electrodrive.researched.contracts import run_dir as rd
    return rd


def test_create_run_dir_contract(tmp_path: Path):
    rd = _contract_module()

    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    # Expect one of these APIs to exist; implement a thin wrapper if needed.
    if hasattr(rd, "create_run_dir"):
        run_dir = rd.create_run_dir(
            runs_root=runs_root,
            workflow="solve",
            inputs={"spec_path": "spec.json"},
            command=["echo", "hello"],
        )
    elif hasattr(rd, "init_run_dir"):
        run_dir = rd.init_run_dir(
            runs_root=runs_root,
            workflow="solve",
            inputs={"spec_path": "spec.json"},
            command=["echo", "hello"],
        )
    else:
        raise AssertionError("Need create_run_dir() or init_run_dir() in contracts/run_dir.py")

    run_dir = Path(run_dir)
    assert run_dir.exists()

    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists(), "manifest.json must be written at run start"
    manifest = json.loads(manifest_path.read_text())

    assert manifest.get("workflow") == "solve"
    assert manifest.get("status") in ("running", "success", "error", "killed")

    assert (run_dir / "command.txt").exists()
    assert (run_dir / "artifacts").exists()
    assert (run_dir / "plots").exists()

    # Optional but recommended by spec:
    # report.html should exist or be created by report service later
    # If you create it at end, relax this assertion.
    # assert (run_dir / "report.html").exists()

    # Bridging policy: ResearchED must ingest both names.
    # If only one file exists, it should still work; bridging may create both.
    ev = run_dir / "events.jsonl"
    legacy = run_dir / "evidence_log.jsonl"
    assert ev.exists() or legacy.exists(), "At least one log stream file should exist/be reserved"
