import math

import torch

from electrodrive.experiments import run_discovery as rd


def test_finite_helpers_float_and_tensor() -> None:
    fail = 1e30
    assert rd._finite(1.0)
    assert not rd._finite(float("nan"))
    assert not rd._finite(float("inf"))
    assert rd._finite(torch.tensor(1.0))
    assert not rd._finite(torch.tensor(float("nan")))

    assert rd._finite_or_fail(1.0, fail) == 1.0
    assert rd._finite_or_fail(float("nan"), fail) == fail
    assert rd._finite_or_fail(torch.tensor(float("inf")), fail) == fail


def test_tensor_all_finite_helper() -> None:
    assert rd._tensor_all_finite(torch.tensor([1.0, 2.0]))
    assert not rd._tensor_all_finite(torch.tensor([float("nan")]))


def test_sanitize_metric_block_replaces_nonfinite() -> None:
    ok, sanitized, nonfinite = rd._sanitize_metric_block(
        {"mean_in": float("nan"), "rel_lap": 0.25},
        fail_value=1e30,
    )
    assert not ok
    assert "mean_in" in nonfinite
    assert math.isfinite(sanitized["mean_in"])
    assert sanitized["mean_in"] == 1e30
    assert sanitized["rel_lap"] == 0.25


def test_dcim_used_as_best_rejects_nonfinite_holdout() -> None:
    metrics = {"is_dcim_block_baseline": True, "holdout_nonfinite": True}
    assert not rd._dcim_used_as_best(metrics, None)
    metrics = {"is_dcim_block_baseline": True, "holdout_nonfinite": False}
    assert rd._dcim_used_as_best(metrics, None)
