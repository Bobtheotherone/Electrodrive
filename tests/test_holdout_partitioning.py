import math

import torch

from electrodrive.experiments import run_discovery as rd


def test_holdout_partition_flags_empty_interior() -> None:
    flags = rd._holdout_partition_flags(n_boundary=16, n_interior=0)
    assert "holdout_interior_empty" in flags
    assert "holdout_boundary_empty" not in flags


def test_mean_abs_float64_prevents_overflow() -> None:
    vals = torch.full((1024,), 1.0e38, dtype=torch.float32)
    mean_abs = rd._mean_abs(vals)
    assert math.isfinite(mean_abs)
    assert mean_abs > 0.0
