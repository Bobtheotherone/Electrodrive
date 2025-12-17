import torch
import pytest

from electrodrive.images.optim import OuterSolveConfig, evaluate_bilevel_objective, optimize_theta_adam


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bilevel outer optimization test")
def test_outer_bilevel_grad_flow():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float32

    n, k = 32, 16
    A_base = torch.randn(n, k, device=device, dtype=dtype)
    A_dir = 0.2 * torch.randn(n, k, device=device, dtype=dtype)
    w_true = torch.zeros(k, device=device, dtype=dtype)
    w_true[:4] = torch.randn(4, device=device, dtype=dtype)
    theta_true = torch.tensor([0.7], device=device, dtype=dtype)
    g = (A_base + theta_true.view(1, 1) * A_dir).matmul(w_true)
    X = torch.empty((0, 3), device=device, dtype=dtype)

    objective = _ToyObjective(A_base, A_dir, X, g)
    solve_cfg = OuterSolveConfig(solver="implicit_lasso", reg_l1=1e-3, max_iter=200, tol=1e-5)

    theta_init = torch.tensor([0.0], device=device, dtype=dtype, requires_grad=True)
    loss0, _, _ = evaluate_bilevel_objective(theta_init, objective, solve_cfg)
    loss0.backward()
    assert theta_init.grad is not None
    assert float(theta_init.grad.abs().sum().item()) > 0.0

    theta_seed = theta_init.detach()
    result = optimize_theta_adam(
        theta_seed,
        objective,
        solve_cfg,
        steps=30,
        lr=5e-2,
        seed=0,
        restarts=1,
    )
    assert result.loss < float(loss0.detach().cpu())
