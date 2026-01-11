import math

import torch

from electrodrive.experiments.run_discovery import _resample_invalid_oracle_targets


def test_resample_invalid_oracle_targets_recovers() -> None:
    points = torch.tensor(
        [[0.0, 0.0, -1.0], [0.1, 0.0, -2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    values = torch.tensor([1.0e20, float("nan"), 1.0], dtype=torch.float32)

    def sample_fn(n: int, seed: int) -> torch.Tensor:
        _ = seed
        return torch.stack(
            [
                torch.zeros(n, dtype=torch.float32),
                torch.zeros(n, dtype=torch.float32),
                torch.ones(n, dtype=torch.float32),
            ],
            dim=1,
        )

    def eval_fn(pts: torch.Tensor) -> torch.Tensor:
        z = pts[:, 2]
        return torch.where(z < 0.5, torch.full_like(z, 1.0e20), torch.ones_like(z))

    pts_out, vals_out, nonfinite_count, extreme_count = _resample_invalid_oracle_targets(
        points=points,
        values=values,
        sample_fn=sample_fn,
        eval_fn=eval_fn,
        max_abs=1.0e10,
        max_attempts=2,
        seed=123,
    )

    assert pts_out.shape == points.shape
    assert vals_out.shape == values.shape
    assert nonfinite_count == 1
    assert extreme_count == 1
    assert torch.isfinite(vals_out).all()
    assert math.isfinite(float(torch.max(torch.abs(vals_out)).item()))
    assert float(torch.max(torch.abs(vals_out)).item()) <= 1.0e10
