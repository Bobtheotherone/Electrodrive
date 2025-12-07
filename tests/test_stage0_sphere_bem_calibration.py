from __future__ import annotations

from pathlib import Path

from tools.stage0_sphere_bem_vs_analytic import SampleConfig, run_calibration


def test_stage0_sphere_bem_vs_analytic_smoke(tmp_path: Path):
    sample_cfg = SampleConfig(
        n_theta=6,
        n_phi=12,
        n_axis=24,
        axis_exclude_tol=5e-4,
    )
    results = run_calibration(
        z_values=[1.25],
        out_root=tmp_path / "runs",
        sample_cfg=sample_cfg,
        bem_cfg_overrides={
            "max_refine_passes": 2,
            "initial_h": 0.25,
            "use_gpu": False,
            "vram_autotune": False,
        },
    )
    assert results, "Calibration produced no results"
    metrics = results[0]
    assert metrics["bc_rel_mean"] < 5e-2
    assert metrics["axis_rel_mean"] < 5e-2
    assert metrics["energy_rel_err"] < 5e-2
