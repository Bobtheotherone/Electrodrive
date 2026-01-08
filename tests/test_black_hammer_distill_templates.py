import json
from pathlib import Path

import numpy as np

from electrodrive.verify.utils import sha256_json
from scripts.black_hammer.distill_templates import distill_templates


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def test_distill_templates_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True, exist_ok=True)

    program = [{"type": "add_primitive", "family_name": "baseline"}]
    record_a = {
        "program": program,
        "elements": [{"type": "point", "params": {"position": [0.0, 0.0, 0.0]}}],
        "weights": [1.0],
        "metrics": {"score": 1.0},
    }
    record_b = {
        "program": program,
        "elements": [{"type": "point", "params": {"position": [1.0, 0.0, 0.0]}}],
        "weights": [0.5],
        "metrics": {"score": 0.5},
    }

    _write_json(input_dir / "candidate.json", record_a)
    (input_dir / "candidates.jsonl").write_text(
        json.dumps(record_b, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    summary = distill_templates(input_dir, output_dir)
    fingerprint = sha256_json(program)
    cluster_path = output_dir / "clusters.json"
    csv_path = output_dir / f"template_{fingerprint}.csv"
    npz_path = output_dir / f"template_{fingerprint}.npz"

    assert summary["total_candidates"] == 2
    assert cluster_path.exists()
    assert csv_path.exists()
    assert npz_path.exists()

    clusters = json.loads(cluster_path.read_text(encoding="utf-8"))
    assert clusters[0]["fingerprint"] == fingerprint
    assert clusters[0]["count"] == 2

    data = np.load(npz_path, allow_pickle=True)
    assert data["weights"].shape[0] == 2
    assert data["params"].shape[0] == 2
