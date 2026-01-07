import torch

from electrodrive.experiments.run_discovery import _fast_weights


def test_fast_weights_stability_with_scaling() -> None:
    torch.manual_seed(0)
    n = 256
    k = 16
    A = torch.randn(n, k, dtype=torch.float32)
    w_true = torch.randn(k, dtype=torch.float32)
    b = A @ w_true

    scale = 1e12
    A_scaled = A * scale
    b_scaled = b * scale

    w_est = _fast_weights(A_scaled, b_scaled, reg=1e-6, normalize=True)
    assert w_est.numel() == k
    assert torch.isfinite(w_est).all()
    assert torch.allclose(w_est, w_true, rtol=1e-2, atol=1e-2)
