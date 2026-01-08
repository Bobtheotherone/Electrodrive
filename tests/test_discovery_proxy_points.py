import torch

from electrodrive.experiments.run_discovery import build_proxy_stability_points


def test_build_proxy_stability_points_mix() -> None:
    bc_hold = torch.tensor(
        [
            [0.0, 0.0, 0.01],
            [0.0, 0.0, 0.02],
            [0.0, 0.0, 0.03],
            [0.0, 0.0, 0.04],
        ],
        dtype=torch.float32,
    )
    interior_hold = torch.tensor(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.6],
            [0.0, 0.0, 0.7],
            [0.0, 0.0, 0.8],
        ],
        dtype=torch.float32,
    )
    out = build_proxy_stability_points(bc_hold, interior_hold, 6)
    assert out.shape == (6, 3)
    assert torch.equal(out[:3], bc_hold[:3])
    assert torch.equal(out[3:], interior_hold[:3])


def test_build_proxy_stability_points_fallback() -> None:
    bc_hold = torch.zeros((0, 3), dtype=torch.float32)
    interior_hold = torch.tensor(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.6],
            [0.0, 0.0, 0.7],
        ],
        dtype=torch.float32,
    )
    out = build_proxy_stability_points(bc_hold, interior_hold, 2)
    assert out.shape == (2, 3)
    assert torch.equal(out, interior_hold[:2])
