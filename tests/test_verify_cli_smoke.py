import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for verify CLI smoke test")


def _latest_run_dir(base: Path) -> Path:
    runs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not runs:
        raise RuntimeError("No run directory created by verify CLI")
    return runs[-1]


def test_verify_cli_smoke(tmp_path: Path) -> None:
    _skip_if_no_cuda()
    spec = {
        "domain": "halfspace",
        "BCs": "Dirichlet",
        "conductors": [{"type": "plane", "z": 0.0, "potential": 0.0}],
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 1.0]}],
        "dielectrics": [],
    }
    points = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.2], [0.0, 0.0, 0.5]]
    spec_path = tmp_path / "spec.json"
    points_path = tmp_path / "points.json"
    out_root = tmp_path / "runs"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    points_path.write_text(json.dumps(points), encoding="utf-8")

    cmd = [
        sys.executable,
        "tools/verify_discovery.py",
        "--spec",
        str(spec_path),
        "--points",
        str(points_path),
        "--oracle",
        "F0",
        "--outdir",
        str(out_root),
    ]
    subprocess.run(cmd, check=True)

    run_dir = _latest_run_dir(out_root)
    cert_path = run_dir / "discovery_certificate.json"
    assert cert_path.exists()
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    assert cert.get("oracle_runs")
    assert "B" in cert.get("gates", {})
    assert "E" in cert.get("gates", {})
