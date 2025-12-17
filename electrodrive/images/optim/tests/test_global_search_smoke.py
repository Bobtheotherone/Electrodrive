import torch
import pytest

from electrodrive.images.optim import OuterSolveConfig
from electrodrive.images.optim import search as global_search


class _ToyObjective:
    def __init__(self, A_base: torch.Tensor, A_dir: torch.Tensor, X: torch.Tensor, g: torch.Tensor) -> None:
        self.A_base = A_base
        self.A_dir = A_dir
        self.X = X
        self.g = g

    def build_dictionary(self, theta: torch.Tensor):
        scale = theta.view(1, 1)
        A = self.A_base + scale * self.A_dir
        return A, self.X, self.g, {"A": A}

    def loss(self, theta: torch.Tensor, w: torch.Tensor, metadata: dict):
        A = metadata.get("A", self.A_base)
        pred = A.matmul(w)
        return torch.mean((pred - self.g) ** 2)

    def constraints(self, theta: torch.Tensor):
        return None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for global search smoke test")
def test_global_search_smoke():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float32

    n, k = 24, 12
    A_base = torch.randn(n, k, device=device, dtype=dtype)
    A_dir = 0.1 * torch.randn(n, k, device=device, dtype=dtype)
    w_true = torch.randn(k, device=device, dtype=dtype)
    theta_true = torch.tensor([0.3], device=device, dtype=dtype)
    g = (A_base + theta_true.view(1, 1) * A_dir).matmul(w_true)
    X = torch.empty((0, 3), device=device, dtype=dtype)

    objective = _ToyObjective(A_base, A_dir, X, g)
    solve_cfg = OuterSolveConfig(solver="implicit_lasso", reg_l1=1e-3, max_iter=100, tol=1e-5)

    theta_init = torch.zeros(1, device=device, dtype=dtype)
    theta_best, report = global_search(
        theta_init,
        objective,
        solve_cfg,
        method="multistart",
        budget=3,
        seed=1,
    )
    assert theta_best.shape == theta_init.shape
    assert isinstance(report.best_loss, float)
    assert report.evaluations >= 1
