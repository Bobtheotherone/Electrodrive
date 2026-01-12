import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Stage 10 Gate B triage smoke test"
)


def test_stage10_triage_gateB_smoke(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_triage_smoke"
    cert_dir = run_dir / "artifacts" / "certificates"
    cert_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "seed": 0,
        "run": {"generations": 1, "tag": "triage_smoke"},
        "spec": {
            "family": "plane",
            "source_height_range": [0.5, 0.5],
            "domain_scale": 1.0,
        },
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    summary_path = cert_dir / "gen000_rank0_summary.json"
    summary_payload = {
        "generation": 0,
        "rank": 0,
        "spec_digest": "stub",
        "elements": [
            {"type": "point", "params": {"position": [0.0, 0.0, 1.0]}},
        ],
        "weights": [1.0],
        "metrics": {},
        "verification": {"path": str(cert_dir / "gen000_rank0_verifier")},
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    out_root = tmp_path / "audit"
    cmd = [
        sys.executable,
        "tools/stage10/triage_gateB.py",
        str(run_dir),
        "--out-root",
        str(out_root),
        "--top-n",
        "1",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    out_dir = out_root / run_dir.name / "triage"
    json_path = out_dir / "gateB_triage.json"
    md_path = out_dir / "gateB_triage.md"
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    assert candidates
    modes = candidates[0].get("modes", {})
    assert set(modes.keys()) == {
        "candidate_only",
        "candidate_plus_reference",
        "candidate_minus_reference",
    }
    for mode in modes.values():
        assert "dirichlet_max_err" in mode
        assert "interface_max_v_jump" in mode
        assert "interface_max_d_jump" in mode
