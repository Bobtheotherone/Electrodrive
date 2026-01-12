import json
import uuid
from pathlib import Path

import pytest
import torch
import yaml

from electrodrive.experiments.preflight import RunCounters
from electrodrive.experiments.run_discovery import run_discovery
from electrodrive.verify.certificate import DiscoveryCertificate
from electrodrive.verify.verifier import Verifier


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Stage 10 run logging smoke test"
)


def _write_stage10_smoke_config(path: Path, tag: str, generations: int) -> None:
    cfg = {
        "seed": 0,
        "run": {
            "tag": tag,
            "generations": generations,
            "population_B": 2,
            "topK_fast": 2,
            "topk_mid": 1,
            "topk_final": 1,
            "preflight_mode": "lite",
            "allow_not_ready": True,
            "diversity_guard": False,
        },
        "points": {
            "bc_train": 16,
            "bc_holdout": 16,
            "interior_train": 16,
            "interior_holdout": 16,
        },
        "model": {
            "dtype": "bf16",
            "compile": False,
            "use_tf32": True,
            "geom_encoder": {"name": "simple", "hidden_dim": 8, "layers": 2},
            "structure_policy": {
                "name": "gflownet",
                "max_steps": 6,
                "min_length_for_stop": 4,
            },
            "param_sampler": {"name": "none"},
        },
        "solver": {
            "name": "differentiable_lasso",
            "max_iters": 8,
            "tol": 1e-6,
            "reg_l1": 1e-3,
        },
        "oracle": {
            "fast": {"name": "fast", "dtype": "fp32"},
            "mid": {"name": "mid", "dtype": "fp32"},
        },
        "spec": {
            "family": "plane",
            "source_height_range": [0.2, 0.3],
            "domain_scale": 1.0,
        },
        "paths": {
            "gfn_checkpoint": "artifacts/step10_gfn_flow_smoke/gfn_ckpt.pt",
        },
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def test_stage10_run_log_and_preflight_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tag = f"stage10_preflight_smoke_{uuid.uuid4().hex[:8]}"
    config_path = tmp_path / "stage10_smoke.yaml"
    generations = 1
    _write_stage10_smoke_config(config_path, tag, generations)

    def _fast_verify(
        self: Verifier,
        candidate: dict,
        spec: dict,
        plan=None,
        *,
        points=None,
        outdir=None,
    ) -> DiscoveryCertificate:
        return DiscoveryCertificate(
            spec_digest="stub",
            candidate_digest="stub",
            git_sha="stub",
            final_status="fail",
        )

    monkeypatch.setattr(Verifier, "run", _fast_verify)

    exit_code = run_discovery(config_path)
    assert exit_code in {0, 2, 3}

    run_dirs = sorted(Path("runs").glob(f"*_{tag}"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    run_log = run_dir / "run.log"
    assert run_log.is_file()
    run_log_text = run_log.read_text(encoding="utf-8")
    assert "RUN START" in run_log_text
    assert "PREFLIGHT heartbeat initialized" in run_log_text

    preflight_path = run_dir / "preflight.json"
    assert preflight_path.is_file()
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    counters = preflight.get("counters", {})
    assert counters
    assert set(counters.keys()) == set(RunCounters().as_dict().keys())
    assert all(isinstance(val, int) for val in counters.values())

    extra = preflight.get("extra", {})
    per_gen = extra.get("per_gen", [])
    assert isinstance(per_gen, list)
    assert len(per_gen) == generations
    for entry in per_gen:
        assert isinstance(entry, dict)
        assert isinstance(entry.get("gen"), int)
        entry_counters = entry.get("counters", {})
        assert set(entry_counters.keys()) == set(RunCounters().as_dict().keys())
