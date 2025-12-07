import torch

from electrodrive.images.search import (
    AugLagrangeConfig,
    JsonlLogger,
    solve_l1_augmented_lagrangian,
    solve_l1_ista,
)


class _StubLogger(JsonlLogger):
    """JsonlLogger writes to disk; keep a temp dir per test."""

    def __init__(self) -> None:
        super().__init__("_tmp_test_logs")

    def close(self) -> None:  # pragma: no cover - allow GC cleanup without errors
        try:
            super().close()
        except Exception:
            pass


# Keep the AL loop extremely light for tests.
_AL_CFG = AugLagrangeConfig(max_outer=1, rho0=1.0, rho_growth=10.0, rho_max=1e3, base_tol=1e-5)


def test_weight_prior_pulls_solution_toward_target():
    torch.manual_seed(0)
    A = torch.tensor([[1.0, 0.0], [0.0, 0.5]])
    g = torch.tensor([1.0, -0.5])
    reg = 1e-3
    logger = _StubLogger()

    w_base, _ = solve_l1_ista(A, g, reg, logger, max_iter=200, tol=1e-6)

    w_prior = torch.tensor([0.0, 0.0])
    w_prior_sol, _ = solve_l1_ista(
        A,
        g,
        reg,
        logger,
        max_iter=200,
        tol=1e-6,
        weight_prior=w_prior,
        lambda_weight_prior=0.05,
    )

    dist_base = torch.linalg.norm(w_base - w_prior)
    dist_prior = torch.linalg.norm(w_prior_sol - w_prior)
    assert float(dist_prior) < float(dist_base)


def test_weight_prior_disabled_matches_baseline():
    torch.manual_seed(1)
    A = torch.tensor([[1.0, 0.2], [0.1, 0.9]])
    g = torch.tensor([0.3, -0.7])
    reg = 1e-3
    logger = _StubLogger()

    w_base, _ = solve_l1_ista(A, g, reg, logger, max_iter=150, tol=1e-6)
    w_zero_prior, _ = solve_l1_ista(
        A,
        g,
        reg,
        logger,
        max_iter=150,
        tol=1e-6,
        weight_prior=torch.zeros_like(w_base),
        lambda_weight_prior=0.0,
    )

    assert torch.allclose(w_base, w_zero_prior, atol=1e-6, rtol=1e-6)


def test_augmented_lagrangian_respects_weight_prior():
    torch.manual_seed(2)
    A_base = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    g = torch.tensor([0.5, -1.0])
    boundary_mask = torch.tensor([True, False])
    logger = _StubLogger()

    def make_weighted_dict(row_w: torch.Tensor) -> torch.Tensor:
        # Row weighting in the same shape as the collocation targets.
        weights = torch.sqrt(row_w).view(-1, 1)
        return weights * A_base

    def predict_unweighted(w: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        out = A_base @ w
        return out if mask is None else out[mask]

    collocation = torch.zeros((2, 3))  # unused for dense path

    w_base, *_ = solve_l1_augmented_lagrangian(
        g,
        boundary_mask,
        make_weighted_dict=make_weighted_dict,
        predict_unweighted=predict_unweighted,
        collocation=collocation,
        reg_l1=1e-3,
        logger=logger,
        cfg=_AL_CFG,
    )

    w_prior = torch.tensor([0.25, 0.0])
    w_prior_sol, *_ = solve_l1_augmented_lagrangian(
        g,
        boundary_mask,
        make_weighted_dict=make_weighted_dict,
        predict_unweighted=predict_unweighted,
        collocation=collocation,
        reg_l1=1e-3,
        logger=logger,
        cfg=_AL_CFG,
        weight_prior=w_prior,
        lambda_weight_prior=0.1,
    )

    dist_base = torch.linalg.norm(w_base - w_prior)
    dist_prior = torch.linalg.norm(w_prior_sol - w_prior)
    assert float(dist_prior) < float(dist_base)
