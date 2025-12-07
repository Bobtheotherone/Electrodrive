from __future__ import annotations

import argparse
from pathlib import Path

from electrodrive.images.training import BilevelTrainConfig, train_stage0
from electrodrive.learn import mo3_train


def test_train_stage0_smoke(tmp_path):
    out_dir = tmp_path / "mo3_test_smoke"
    cfg = BilevelTrainConfig(
        out_dir=out_dir,
        max_steps=1,
        batch_size=1,
        n_candidates_static=4,
        n_candidates_learned=0,
        stage0_geoms=["plane"],
        n_points_train=32,
        n_points_val=32,
        device="cpu",
        dtype="float32",
    )

    metrics = train_stage0(cfg)

    for key in ("loss", "err_int", "err_bc", "sparsity"):
        assert key in metrics

    ckpts = list(out_dir.glob("checkpoint_step*.pt"))
    assert ckpts, "Expected a checkpoint file to be written."


def test_mo3_train_run_from_namespace_smoke(tmp_path):
    out_dir = tmp_path / "mo3_cli_smoke"
    parser = mo3_train.build_parser()
    args = parser.parse_args(
        [
            "--out",
            str(out_dir),
            "--steps",
            "0",
            "--batch-size",
            "1",
            "--n-static",
            "4",
            "--n-learned",
            "0",
            "--n-points",
            "16",
            "--n-points-val",
            "16",
            "--ratio-boundary",
            "0.5",
            "--ratio-boundary-val",
            "0.5",
            "--lista-steps",
            "2",
            "--device",
            "cpu",
            "--dtype",
            "float32",
        ]
    )

    ret = mo3_train.run_from_namespace(args)
    assert ret == 0
    assert out_dir.exists()
    assert any(out_dir.iterdir()), "Expected outputs (e.g., checkpoint/log) in out_dir."


def test_train_stage0_uses_get_collocation_data(monkeypatch, tmp_path):
    calls = {"count": 0}

    import electrodrive.images.training as training_mod
    from electrodrive.images import search as search_mod

    real_get = training_mod.get_collocation_data

    def counting_get_collocation_data(*args, **kwargs):
        calls["count"] += 1
        return real_get(*args, **kwargs)

    monkeypatch.setattr(training_mod, "get_collocation_data", counting_get_collocation_data)
    monkeypatch.setattr(
        training_mod,
        "MLPBasisGenerator",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Generator should not be constructed")),
    )
    monkeypatch.setattr(
        search_mod,
        "build_adaptive_collocation",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Adaptive collocation should not be used")),
    )

    out_dir = tmp_path / "mo3_colloc_check"
    cfg = BilevelTrainConfig(
        out_dir=out_dir,
        max_steps=1,
        batch_size=1,
        n_candidates_static=4,
        n_candidates_learned=0,
        stage0_geoms=["plane"],
        n_points_train=8,
        n_points_val=8,
        device="cpu",
        dtype="float32",
    )

    train_stage0(cfg)

    assert calls["count"] >= 2, "Expected train and val collocation sampling."
