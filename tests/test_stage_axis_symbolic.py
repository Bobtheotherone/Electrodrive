import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.stage0_sphere_axis_moi import SampleConfig, run_sweep as run_stage0_sweep
import tools.stage1_sphere_dimer_axis_sweep_svd as stage1_mod
from electrodrive.orchestration.spec_registry import stage1_sphere_dimer_inside_path


@pytest.fixture(autouse=True)
def _force_cpu_env(monkeypatch, tmp_path):
    monkeypatch.setenv("EDE_DEVICE", "cpu")
    monkeypatch.setenv("EDE_IMAGES_N_POINTS", "32")
    monkeypatch.setenv("EDE_IMAGES_ADAPTIVE_ROUNDS", "1")
    # keep cache dirs per test to avoid collisions
    monkeypatch.chdir(tmp_path)
    yield


def test_stage0_symbolic_outputs(tmp_path):
    out_root = tmp_path / "stage0_symbolic"
    sample_cfg = SampleConfig(n_theta=6, n_phi=8, n_axis=6, axis_span=2.0, axis_exclude_tol=1e-3)
    run_stage0_sweep(
        z_values=[1.25, 1.5],
        n_max_list=[1],
        reg_l1=1e-4,
        restarts=0,
        sample_cfg=sample_cfg,
        out_root=out_root,
        basis_types=["sphere_kelvin_ladder"],
        adaptive_rounds=1,
    )

    sym_dir = out_root / "symbolic_n1"
    required = [
        "weights_vs_axis.npy",
        "svd_modes.npy",
        "symbolic_fits.json",
        "metrics.json",
        "summary.md",
        "summary.json",
    ]
    for fname in required:
        path = sym_dir / fname if fname != "summary.json" else out_root / fname
        assert path.exists(), f"Missing {fname}"

    # Controller gating check: zero singulars should disable controller.
    from electrodrive.images.weight_modes import spectral_gap_ok

    assert not spectral_gap_ok([0.0, 0.0], rank=1, thresh=0.1)


def test_stage1_symbolic_outputs(tmp_path, monkeypatch):
    out_root = tmp_path / "stage1_symbolic"
    spec_path = stage1_sphere_dimer_inside_path()
    # Stub discovery to keep runtime tiny.
    class _StubElem:
        def serialize(self):
            return {"type": "stub", "params": {}}

    dummy_system = type(
        "DummySystem",
        (),
        {"weights": torch.tensor([0.1, -0.2]), "elements": [_StubElem(), _StubElem()]},
    )()
    monkeypatch.setattr(stage1_mod, "discover_images", lambda *args, **kwargs: dummy_system)
    monkeypatch.setattr(stage1_mod, "save_image_system", lambda system, path, metadata=None: path.write_text("{}", encoding="utf-8"))

    args = [
        "--spec",
        str(spec_path),
        "--basis",
        "sphere_kelvin_ladder",
        "--nmax",
        "1",
        "--reg-l1",
        "1e-2",
        "--restarts",
        "0",
        "--z",
        "0.7",
        "--out",
        str(out_root),
        "--max-rank",
        "2",
        "--max-poly-degree",
        "3",
    ]
    # Keep run small and CPU-only.
    monkeypatch.setenv("EDE_IMAGES_N_POINTS", "64")
    monkeypatch.setenv("EDE_IMAGES_ADAPTIVE_ROUNDS", "1")
    stage1_mod.main(args)

    required = [
        out_root / "weights_vs_axis.npy",
        out_root / "svd_modes.npy",
        out_root / "symbolic_fits.json",
        out_root / "metrics.json",
        out_root / "summary.md",
        out_root / "summary.json",
        out_root / "weights_svd.npz",
    ]
    for path in required:
        assert path.exists(), f"Missing {path.name}"

    data = np.load(out_root / "weights_vs_axis.npy")
    assert data.shape[1] == 1  # one column per z value


def test_stage1_controller_gating_skips_bad_prior(tmp_path, monkeypatch):
    out_root = tmp_path / "stage1_skip_controller"
    spec_path = stage1_sphere_dimer_inside_path()

    class _StubElem:
        def serialize(self):
            return {"type": "stub", "params": {}}

    dummy_system = type(
        "DummySystem",
        (),
        {"weights": torch.tensor([0.05]), "elements": [_StubElem()]},
    )()
    monkeypatch.setattr(stage1_mod, "discover_images", lambda *args, **kwargs: dummy_system)
    monkeypatch.setattr(stage1_mod, "save_image_system", lambda system, path, metadata=None: path.write_text("{}", encoding="utf-8"))

    mode_dir = tmp_path / "bad_prior"
    mode_dir.mkdir(parents=True, exist_ok=True)
    np.save(mode_dir / "svd_modes.npy", {"U": np.zeros((1, 1)), "S": np.zeros((1,)), "VT": np.zeros((1, 1)), "z_grid": np.array([0.7]), "mode_curves": np.zeros((1, 1)), "sigma_norm": np.zeros((1,)), "effective_rank": {"eps_1e-1": 0, "eps_1e-2": 0}}, allow_pickle=True)
    mode_dir.joinpath("symbolic_fits.json").write_text(
        json.dumps({"z_grid": [0.7], "fits": [{"mode": 0, "method": "poly", "coefficients": {"poly_coeffs": [0.0]}, "rmse": 1.0, "mae": 1.0, "max_abs": 1.0, "rel_rmse": 0.9}]})
    )

    args = [
        "--spec",
        str(spec_path),
        "--basis",
        "sphere_kelvin_ladder",
        "--nmax",
        "1",
        "--reg-l1",
        "1e-2",
        "--restarts",
        "0",
        "--z",
        "0.7",
        "--out",
        str(out_root),
        "--use-weight-modes",
        "--mode-dir",
        str(mode_dir),
        "--spectral-gap-thresh",
        "0.1",
        "--rel-rmse-thresh",
        "0.2",
    ]
    monkeypatch.setenv("EDE_IMAGES_N_POINTS", "32")
    monkeypatch.setenv("EDE_IMAGES_ADAPTIVE_ROUNDS", "1")
    stage1_mod.main(args)

    summary = json.loads((out_root / "summary.json").read_text())
    assert summary.get("controller_used") is False
