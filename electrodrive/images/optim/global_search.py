from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

import math
import torch

from .outer_nonlinear import (
    BilevelObjective,
    OuterSolveConfig,
    ParameterConstraint,
    evaluate_bilevel_objective,
    optimize_theta_adam,
    optimize_theta_lbfgs,
)


class BatchBilevelObjective(Protocol):
    def batch_evaluate(
        self,
        thetas: torch.Tensor,
        solve_cfg: OuterSolveConfig,
        *,
        seed: Optional[int] = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        ...


@dataclass
class GlobalSearchReport:
    method: str
    best_loss: float
    best_stats: dict[str, Any] = field(default_factory=dict)
    history: list[float] = field(default_factory=list)
    evaluations: int = 0


class _NullLogger:
    def info(self, *args: Any, **kwargs: Any) -> None:
        return None

    def warning(self, *args: Any, **kwargs: Any) -> None:
        return None

    def error(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


def _default_rng(seed: Optional[int], device: torch.device) -> torch.Generator:
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(int(seed))
    return gen


def _evaluate_candidates(
    thetas: torch.Tensor,
    objective: BilevelObjective,
    solve_cfg: OuterSolveConfig,
    *,
    param_constraints: Optional[Sequence[ParameterConstraint]] = None,
    logger: Optional[Any] = None,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if logger is None:
        logger = _NullLogger()

    if hasattr(objective, "batch_evaluate"):
        return objective.batch_evaluate(thetas, solve_cfg, seed=seed)

    losses: list[torch.Tensor] = []
    stats_list: list[dict[str, Any]] = []
    for theta in thetas:
        with torch.no_grad():
            loss, stats, _ = evaluate_bilevel_objective(
                theta,
                objective,
                solve_cfg,
                param_constraints=param_constraints,
                logger=logger,
                track_timing=True,
                full_stats=True,
            )
        losses.append(loss.detach())
        stats_list.append(stats)
    return torch.stack(losses), stats_list


def cmaes_search(
    initial_theta: torch.Tensor,
    objective: BilevelObjective,
    solve_cfg: OuterSolveConfig,
    *,
    budget: int,
    seed: Optional[int] = None,
    popsize: Optional[int] = None,
    sigma0: Optional[float] = None,
    param_constraints: Optional[Sequence[ParameterConstraint]] = None,
    logger: Optional[Any] = None,
) -> tuple[torch.Tensor, GlobalSearchReport]:
    device = initial_theta.device
    theta_shape = initial_theta.shape
    dim = int(initial_theta.numel())
    if popsize is None:
        popsize = max(4, 4 + int(3 * math.log(max(2, dim))))
    popsize = int(popsize)
    mu = max(1, popsize // 2)
    weights = torch.linspace(mu, 1, steps=mu, device=device, dtype=initial_theta.dtype)
    weights = weights / weights.sum()

    rng = _default_rng(seed, device)
    mean = initial_theta.detach().reshape(-1)
    sigma = float(sigma0) if sigma0 is not None else 0.3
    diag_cov = torch.ones(dim, device=device, dtype=initial_theta.dtype)

    best_loss = float("inf")
    best_theta = initial_theta.detach().clone()
    best_stats: dict[str, Any] = {}
    history: list[float] = []
    evals = 0

    while evals < int(budget):
        batch = min(popsize, int(budget) - evals)
        z = torch.randn(batch, dim, device=device, generator=rng, dtype=initial_theta.dtype)
        candidates = mean + sigma * z * torch.sqrt(diag_cov)
        candidates = candidates.view(batch, *theta_shape)
        losses, stats_list = _evaluate_candidates(
            candidates,
            objective,
            solve_cfg,
            param_constraints=param_constraints,
            logger=logger,
            seed=seed,
        )
        evals += batch

        prev_best = best_loss
        loss_vals = losses.detach().to(device=device, dtype=initial_theta.dtype)
        loss_min = float(loss_vals.min().item())
        history.append(loss_min)
        best_idx = int(torch.argmin(loss_vals).item())
        if float(loss_vals[best_idx].item()) < best_loss:
            best_loss = float(loss_vals[best_idx].item())
            best_theta = candidates[best_idx].detach().clone()
            best_stats = stats_list[best_idx]

        idx = torch.argsort(loss_vals)[:mu]
        elite = candidates[idx].reshape(mu, dim)
        mean = torch.sum(weights.view(-1, 1) * elite, dim=0)
        diff = elite - mean
        diag_cov = torch.sum(weights.view(-1, 1) * diff * diff, dim=0).clamp_min(1e-12)

        if loss_min < prev_best:
            sigma = max(1e-4, sigma * 0.9)
        else:
            sigma = min(2.0, sigma * 1.1)

    report = GlobalSearchReport(method="cmaes", best_loss=best_loss, best_stats=best_stats, history=history, evaluations=evals)
    return best_theta, report


def basinhop_search(
    initial_theta: torch.Tensor,
    objective: BilevelObjective,
    solve_cfg: OuterSolveConfig,
    *,
    budget: int,
    seed: Optional[int] = None,
    hop_scale: float = 0.1,
    temperature: float = 1.0,
    local_max_iter: int = 20,
    param_constraints: Optional[Sequence[ParameterConstraint]] = None,
    logger: Optional[Any] = None,
) -> tuple[torch.Tensor, GlobalSearchReport]:
    device = initial_theta.device
    rng = _default_rng(seed, device)

    with torch.no_grad():
        base_loss, base_stats, _ = evaluate_bilevel_objective(
            initial_theta,
            objective,
            solve_cfg,
            param_constraints=param_constraints,
            logger=logger,
            track_timing=True,
            full_stats=True,
        )
    current_theta = initial_theta.detach().clone()
    current_loss = float(base_loss.detach().item())
    best_theta = current_theta.detach().clone()
    best_loss = current_loss
    best_stats = base_stats
    history = [current_loss]
    evals = 1

    hops = max(1, int(budget))
    for _ in range(hops):
        perturb = hop_scale * torch.randn(current_theta.shape, device=device, dtype=current_theta.dtype, generator=rng)
        trial = current_theta + perturb
        local = optimize_theta_lbfgs(
            trial,
            objective,
            solve_cfg,
            max_iter=local_max_iter,
            seed=seed,
            restarts=1,
            param_constraints=param_constraints,
            logger=logger,
            track_timing=True,
        )
        evals += 1
        loss_val = float(local.loss)
        accept = loss_val < current_loss
        if not accept and temperature > 0.0:
            delta = loss_val - current_loss
            accept_prob = math.exp(-delta / max(1e-12, temperature))
            accept = bool(torch.rand((), device=device, generator=rng).item() < accept_prob)
        if accept:
            current_theta = local.theta.detach().clone()
            current_loss = loss_val
        if loss_val < best_loss:
            best_loss = loss_val
            best_theta = local.theta.detach().clone()
            best_stats = local.stats
        history.append(current_loss)

    report = GlobalSearchReport(method="basinhop", best_loss=best_loss, best_stats=best_stats, history=history, evaluations=evals)
    return best_theta, report


def multistart_search(
    initial_theta: torch.Tensor,
    objective: BilevelObjective,
    solve_cfg: OuterSolveConfig,
    *,
    budget: int,
    seed: Optional[int] = None,
    restarts: int = 4,
    perturb_scale: float = 0.1,
    local_steps: int = 30,
    param_constraints: Optional[Sequence[ParameterConstraint]] = None,
    logger: Optional[Any] = None,
) -> tuple[torch.Tensor, GlobalSearchReport]:
    device = initial_theta.device
    rng = _default_rng(seed, device)
    best_loss = float("inf")
    best_theta = initial_theta.detach().clone()
    best_stats: dict[str, Any] = {}
    history: list[float] = []
    evals = 0

    n_restarts = max(1, min(int(restarts), int(budget)))
    for i in range(n_restarts):
        if i == 0:
            theta0 = initial_theta.detach().clone()
        else:
            theta0 = initial_theta.detach().clone() + perturb_scale * torch.randn(
                initial_theta.shape, device=device, dtype=initial_theta.dtype, generator=rng
            )
        local = optimize_theta_adam(
            theta0,
            objective,
            solve_cfg,
            steps=local_steps,
            lr=1e-2,
            seed=seed,
            restarts=1,
            param_constraints=param_constraints,
            logger=logger,
            track_timing=True,
        )
        evals += 1
        history.append(float(local.loss))
        if float(local.loss) < best_loss:
            best_loss = float(local.loss)
            best_theta = local.theta.detach().clone()
            best_stats = local.stats

    report = GlobalSearchReport(method="multistart", best_loss=best_loss, best_stats=best_stats, history=history, evaluations=evals)
    return best_theta, report


def search(
    initial_theta: torch.Tensor,
    objective: BilevelObjective,
    solve_cfg: OuterSolveConfig,
    *,
    method: str,
    budget: int,
    seed: Optional[int] = None,
    param_constraints: Optional[Sequence[ParameterConstraint]] = None,
    logger: Optional[Any] = None,
) -> tuple[torch.Tensor, GlobalSearchReport]:
    method_norm = (method or "none").strip().lower()
    if method_norm == "cmaes":
        return cmaes_search(
            initial_theta,
            objective,
            solve_cfg,
            budget=budget,
            seed=seed,
            param_constraints=param_constraints,
            logger=logger,
        )
    if method_norm in {"basinhop", "basin_hop", "basin"}:
        return basinhop_search(
            initial_theta,
            objective,
            solve_cfg,
            budget=budget,
            seed=seed,
            param_constraints=param_constraints,
            logger=logger,
        )
    if method_norm in {"multistart", "restarts", "restart"}:
        return multistart_search(
            initial_theta,
            objective,
            solve_cfg,
            budget=budget,
            seed=seed,
            param_constraints=param_constraints,
            logger=logger,
        )
    report = GlobalSearchReport(method="none", best_loss=float("nan"), best_stats={}, history=[], evaluations=0)
    return initial_theta.detach().clone(), report
