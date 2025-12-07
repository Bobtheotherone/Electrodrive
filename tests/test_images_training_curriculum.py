from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from electrodrive.images import training


def test_stage0_sampler_respects_q_range_plane() -> None:
    cfg = training.BilevelTrainConfig(
        out_dir=Path("runs/dummy"),
        stage=0,
        stage0_geoms=["plane"],
        ranges=training.Stage0Ranges(
            plane_z=(0.7, 0.7),
            sphere_external=(1.2, 1.2),
            sphere_internal=(0.3, 0.3),
            q=(0.25, 0.25),
        ),
    )
    rng = np.random.default_rng(0)
    base_plane = training._load_spec("plane_point.json")
    base_sphere_ext = training.load_stage0_sphere_external()
    base_sphere_int = training._load_spec("sphere_axis_point_internal.json")

    label, spec, z = training._sample_stage0_task(
        cfg,
        rng,
        base_plane=base_plane,
        base_sphere_ext=base_sphere_ext,
        base_sphere_int=base_sphere_int,
    )

    assert label == "plane"
    assert pytest.approx(0.7) == z
    assert pytest.approx(0.25) == spec.charges[0]["q"]
    assert pytest.approx(0.7) == spec.charges[0]["pos"][2]


def test_stage1_sampler_stays_in_gap_and_scales_q() -> None:
    cfg = training.BilevelTrainConfig(
        out_dir=Path("runs/dummy"),
        stage=1,
        ranges_stage1=training.Stage1Ranges(charge_frac=(0.2, 0.8), gap_margin=0.01, q=(0.9, 1.1)),
    )
    rng = np.random.default_rng(1)
    stage1_specs = training._load_stage1_specs(include_variants=False)

    label, spec, z = training._sample_stage1_task(
        cfg,
        rng,
        stage1_specs=stage1_specs,
    )

    spheres = sorted(
        [c for c in spec.conductors if c.get("type") == "sphere"],
        key=lambda c: float(c.get("center", [0.0, 0.0, 0.0])[2]),
    )
    assert len(spheres) >= 2

    z_min = float(spheres[0].get("center", [0.0, 0.0, 0.0])[2]) + float(spheres[0].get("radius", 1.0)) + cfg.ranges_stage1.gap_margin
    z_max = float(spheres[1].get("center", [0.0, 0.0, 0.0])[2]) - float(spheres[1].get("radius", 1.0)) - cfg.ranges_stage1.gap_margin

    assert label == "sphere_dimer"
    assert z_min <= z <= z_max
    assert cfg.ranges_stage1.q[0] <= abs(spec.charges[0]["q"]) <= cfg.ranges_stage1.q[1]
