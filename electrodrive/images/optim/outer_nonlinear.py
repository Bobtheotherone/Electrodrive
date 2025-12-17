from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence
import torch

from electrodrive.images.operator import BasisOperator
from .core import ADMMConfig, ConstraintSpec, DTypePolicy
from .diagnostics import CudaTimer, collect_solver_stats


class BilevelObjective(Protocol):
    def build_dictionary(
        self, theta: torch.Tensor
    ) -> tuple[torch.Tensor | BasisOperator, torch.Tensor, torch.Tensor, dict[str, Any]]:
        ...

    def loss(self, theta: torch.Tensor, w: torch.Tensor, metadata: dict[str, Any]) -> torch.Tensor:
        ...

    def constraints(self, theta: torch.Tensor) -> Optional[Sequence["ConstraintTerm"]]:
        ...


@dataclass
class ConstraintTerm:
    name: str
    residual: torch.Tensor
    kind: str = "eq"  # "eq" or "ineq"
    weight: float = 1.0


@dataclass
class ParameterConstraint:
    name: str
    kind: str  # forbidden_region | conjugate_pair | symmetry
    indices: torch.Tensor
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AugmentedLagrangianConfig:
    rho0: float = 10.0
    rho_growth: float = 10.0
    rho_max: float = 1e6
    max_outer: int = 3
    tol: float = 1e-6


@dataclass
class AugmentedLagrangianState:
    rho: float
    multipliers: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class OuterSolveConfig:
    solver: str = "implicit_lasso"
    reg_l1: float = 1e-3
    max_iter: int = 200
    tol: float = 1e-6
    lambda_group: float | torch.Tensor = 0.0
    group_ids: Optional[torch.Tensor] = None
    weight_prior: Optional[torch.Tensor] = None
    lambda_weight_prior: float | torch.Tensor = 0.0
    normalize_columns: bool = True
    constraints: Optional[list[ConstraintSpec]] = None
    constraint_mode: str = "none"
    admm_cfg: Optional[ADMMConfig] = None
    dtype_policy: Optional[DTypePolicy] = None


@dataclass
class OuterOptimizationResult:
    theta: torch.Tensor
    loss: float
    history: list[float]
    stats: dict[str, Any] = field(default_factory=dict)


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


def _gather_theta(theta: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    idx = indices.to(device=theta.device, dtype=torch.long).view(-1)
    if theta.ndim == 1:
        return theta.index_select(0, idx)
    return theta.index_select(0, idx)


def _pair_indices(indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    idx = indices.to(dtype=torch.long).view(-1)
    if idx.numel() % 2 != 0:
        raise ValueError("Pair constraints require an even number of indices.")
    pairs = idx.view(-1, 2)
    return pairs[:, 0], pairs[:, 1]


def _constraint_forbidden_region(theta: torch.Tensor, constraint: ParameterConstraint) -> ConstraintTerm:
    params = constraint.params or {}
    sel = _gather_theta(theta, constraint.indices)
    dims = params.get("dims")
    if dims is not None:
        sel = sel[..., torch.as_tensor(dims, device=sel.device, dtype=torch.long)]
    if "center" in params and "radius" in params:
        center = torch.as_tensor(params["center"], device=sel.device, dtype=sel.dtype)
        radius = torch.as_tensor(params["radius"], device=sel.device, dtype=sel.dtype)
        while center.ndim < sel.ndim:
            center = center.unsqueeze(0)
        dist = torch.linalg.norm(sel - center, dim=-1)
        residual = radius - dist
    elif "min" in params and "max" in params:
        minv = torch.as_tensor(params["min"], device=sel.device, dtype=sel.dtype)
        maxv = torch.as_tensor(params["max"], device=sel.device, dtype=sel.dtype)
        while minv.ndim < sel.ndim:
            minv = minv.unsqueeze(0)
            maxv = maxv.unsqueeze(0)
        inside_margin = torch.minimum(sel - minv, maxv - sel)
        residual = inside_margin.min(dim=-1).values
    else:
        raise ValueError("Forbidden-region constraint requires center/radius or min/max params.")
    return ConstraintTerm(name=constraint.name, residual=residual, kind="ineq", weight=constraint.weight)


def _constraint_conjugate_pair(theta: torch.Tensor, constraint: ParameterConstraint) -> ConstraintTerm:
    i_idx, j_idx = _pair_indices(constraint.indices)
    ti = _gather_theta(theta, i_idx)
    tj = _gather_theta(theta, j_idx)
    if torch.is_complex(theta):
        residual = ti - torch.conj(tj)
    else:
        if ti.ndim == 0 or (ti.ndim == 1 and ti.numel() == i_idx.numel()):
            residual = ti - tj
        else:
            if ti.shape[-1] < 2:
                raise ValueError("Conjugate-pair constraint expects last dim >= 2 for real theta.")
            real_i, imag_i = ti[..., 0], ti[..., 1]
            real_j, imag_j = tj[..., 0], tj[..., 1]
            residual = torch.stack([real_i - real_j, imag_i + imag_j], dim=-1)
    return ConstraintTerm(name=constraint.name, residual=residual, kind="eq", weight=constraint.weight)


def _apply_symmetry(theta: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    if "matrix" in params:
        mat = torch.as_tensor(params["matrix"], device=theta.device, dtype=theta.dtype)
        return theta @ mat.transpose(-1, -2)
    if "sign" in params:
        sign = torch.as_tensor(params["sign"], device=theta.device, dtype=theta.dtype)
        return theta * sign
    if "permute" in params:
        perm = list(params["permute"])
        return theta.index_select(-1, torch.as_tensor(perm, device=theta.device, dtype=torch.long))
    if "mirror_dim" in params:
        dim = int(params["mirror_dim"])
        out = theta.clone()
        out[..., dim] = -out[..., dim]
        return out
    return theta


def _constraint_symmetry(theta: torch.Tensor, constraint: ParameterConstraint) -> ConstraintTerm:
    i_idx, j_idx = _pair_indices(constraint.indices)
    ti = _gather_theta(theta, i_idx)
    tj = _gather_theta(theta, j_idx)
    tj_sym = _apply_symmetry(tj, constraint.params or {})
    residual = ti - tj_sym
    return ConstraintTerm(name=constraint.name, residual=residual, kind="eq", weight=constraint.weight)


def parameter_constraints_to_terms(
    theta: torch.Tensor, constraints: Optional[Sequence[ParameterConstraint]]
) -> list[ConstraintTerm]:
    if not constraints:
        return []
    terms: list[ConstraintTerm] = []
    for spec in constraints:
        kind = spec.kind.strip().lower()
        if kind == "forbidden_region":
            terms.append(_constraint_forbidden_region(theta, spec))
        elif kind == "conjugate_pair":
            terms.append(_constraint_conjugate_pair(theta, spec))
        elif kind == "symmetry":
            terms.append(_constraint_symmetry(theta, spec))
        else:
            raise ValueError(f"Unknown parameter constraint kind '{spec.kind}'.")
    return terms


def _evaluate_constraint_terms(
    terms: Sequence[ConstraintTerm],
    *,
    device: torch.device,
    dtype: torch.dtype,
    al_state: Optional[AugmentedLagrangianState] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not terms:
        return torch.tensor(0.0, device=device, dtype=dtype), {}

    penalty = torch.tensor(0.0, device=terms[0].residual.device, dtype=terms[0].residual.dtype)
    stats: dict[str, float] = {}

    for term in terms:
        residual = term.residual
        if term.kind == "ineq":
            residual = torch.clamp(residual, min=0.0)
        weighted = residual * float(term.weight)
        norm = torch.linalg.norm(weighted)
        stats[f"{term.name}_res"] = float(norm.detach().cpu())
        if al_state is None:
            penalty = penalty + 0.5 * torch.sum(weighted * weighted)
        else:
            lam = al_state.multipliers.get(term.name)
            if lam is None or lam.shape != weighted.shape:
                lam = torch.zeros_like(weighted)
                al_state.multipliers[term.name] = lam
            penalty = penalty + torch.sum(lam * weighted) + 0.5 * float(al_state.rho) * torch.sum(weighted * weighted)

    return penalty, stats


def _update_al_multipliers(terms: Sequence[ConstraintTerm], al_state: AugmentedLagrangianState) -> float:
    if not terms:
        return 0.0
    max_violation = 0.0
    with torch.no_grad():
        for term in terms:
            residual = term.residual
            if term.kind == "ineq":
                residual = torch.clamp(residual, min=0.0)
            weighted = residual * float(term.weight)
            max_violation = max(max_violation, float(torch.linalg.norm(weighted).item()))
            lam = al_state.multipliers.get(term.name)
            if lam is None or lam.shape != weighted.shape:
                lam = torch.zeros_like(weighted)
            al_state.multipliers[term.name] = lam + float(al_state.rho) * weighted
    return max_violation


def backtracking_line_search(
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    theta: torch.Tensor,
    direction: torch.Tensor,
    *,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    tau: float = 0.5,
    max_steps: int = 10,
) -> tuple[torch.Tensor, float]:
    with torch.no_grad():
        base_loss = loss_fn(theta)
        alpha = float(alpha0)
        grad_dot = torch.sum(direction * direction)
        for _ in range(max_steps):
            candidate = theta + alpha * direction
            cand_loss = loss_fn(candidate)
            if cand_loss <= base_loss - c1 * alpha * grad_dot:
                return candidate, float(cand_loss.detach().cpu())
            alpha *= float(tau)
    return theta, float(base_loss.detach().cpu())


def _solve_sparse_inner(
    A: torch.Tensor | BasisOperator,
    X: torch.Tensor,
    g: torch.Tensor,
    is_boundary: Optional[torch.Tensor],
    solve_cfg: OuterSolveConfig,
    *,
    constraints: Optional[list[ConstraintSpec]] = None,
    group_ids: Optional[torch.Tensor] = None,
    logger: Optional[Any] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if logger is None:
        logger = _NullLogger()
    from electrodrive.images.search import solve_sparse

    w, _, stats = solve_sparse(
        A,
        X,
        g,
        is_boundary,
        logger,
        reg_l1=float(solve_cfg.reg_l1),
        solver=str(solve_cfg.solver),
        group_ids=group_ids if group_ids is not None else solve_cfg.group_ids,
        lambda_group=solve_cfg.lambda_group,
        weight_prior=solve_cfg.weight_prior,
        lambda_weight_prior=solve_cfg.lambda_weight_prior,
        normalize_columns=bool(solve_cfg.normalize_columns),
        max_iter=int(solve_cfg.max_iter),
        tol=float(solve_cfg.tol),
        constraints=constraints if constraints is not None else solve_cfg.constraints,
        constraint_mode=str(solve_cfg.constraint_mode),
        admm_cfg=solve_cfg.admm_cfg,
        return_stats=True,
        dtype_policy=solve_cfg.dtype_policy,
    )
    return w, stats


def evaluate_bilevel_objective(
    theta: torch.Tensor,
    objective: BilevelObjective,
    solve_cfg: OuterSolveConfig,
    *,
    param_constraints: Optional[Sequence[ParameterConstraint]] = None,
    al_state: Optional[AugmentedLagrangianState] = None,
    logger: Optional[Any] = None,
    track_timing: bool = False,
    full_stats: bool = False,
) -> tuple[torch.Tensor, dict[str, Any], list[ConstraintTerm]]:
    A, X, g, metadata = objective.build_dictionary(theta)
    is_boundary = metadata.get("is_boundary") if isinstance(metadata, dict) else None
    constraints = metadata.get("constraints") if isinstance(metadata, dict) else None
    group_ids = metadata.get("group_ids") if isinstance(metadata, dict) else None

    solve_timer: Optional[CudaTimer] = None
    if track_timing and torch.cuda.is_available():
        solve_timer = CudaTimer()
        solve_timer.__enter__()
    w, inner_stats = _solve_sparse_inner(
        A,
        X,
        g,
        is_boundary,
        solve_cfg,
        constraints=constraints,
        group_ids=group_ids,
        logger=logger,
    )
    if solve_timer is not None:
        solve_timer.__exit__(None, None, None)

    base_loss = objective.loss(theta, w, metadata)

    terms = parameter_constraints_to_terms(theta, param_constraints)
    if hasattr(objective, "constraints"):
        extra = objective.constraints(theta)
        if extra:
            terms.extend(list(extra))

    penalty, c_stats = _evaluate_constraint_terms(
        terms,
        device=base_loss.device,
        dtype=base_loss.dtype,
        al_state=al_state,
    )
    total = base_loss + penalty

    stats: dict[str, Any] = {
        "loss": float(base_loss.detach().cpu()),
        "penalty": float(penalty.detach().cpu()),
    }
    stats.update(c_stats)
    stats["inner"] = inner_stats
    if solve_timer is not None:
        stats["solve_ms"] = float(solve_timer.elapsed_ms)
    if full_stats:
        try:
            stats["solver_stats"] = collect_solver_stats(
                solver=str(solve_cfg.solver),
                A=A,
                w=w,
                g=g,
                lambda_l1=float(solve_cfg.reg_l1),
                X=X,
                is_boundary=is_boundary,
                constraints=constraints or solve_cfg.constraints,
                weight_prior=solve_cfg.weight_prior,
                lambda_weight_prior=solve_cfg.lambda_weight_prior,
                admm_stats=inner_stats if isinstance(inner_stats, dict) else None,
                timing_ms={"solve": float(solve_timer.elapsed_ms)} if solve_timer is not None else None,
            )
        except Exception:
            pass
    return total, stats, terms


def optimize_theta_lbfgs(
    theta_init: torch.Tensor,
    objective: BilevelObjective,
    solve_cfg: OuterSolveConfig,
    *,
    max_iter: int = 25,
    history_size: int = 10,
    lr: float = 1.0,
    seed: Optional[int] = None,
    restarts: int = 1,
    restart_fn: Optional[Callable[[torch.Tensor, torch.Generator, int], torch.Tensor]] = None,
    param_constraints: Optional[Sequence[ParameterConstraint]] = None,
    al_cfg: Optional[AugmentedLagrangianConfig] = None,
    logger: Optional[Any] = None,
    track_timing: bool = False,
) -> OuterOptimizationResult:
    device = theta_init.device
    rng = _default_rng(seed, device)
    best_loss = float("inf")
    best_theta = theta_init.detach()
    best_stats: dict[str, Any] = {}
    history: list[float] = []

    for restart in range(max(1, int(restarts))):
        if restart == 0:
            theta = theta_init.detach().clone()
        else:
            if restart_fn is None:
                noise = 0.05 * torch.randn(theta_init.shape, device=device, dtype=theta_init.dtype, generator=rng)
                theta = theta_init.detach().clone() + noise
            else:
                theta = restart_fn(theta_init.detach(), rng, restart)
        theta = theta.to(device=device)
        theta.requires_grad_(True)

        al_state = None
        if al_cfg is not None:
            al_state = AugmentedLagrangianState(rho=float(al_cfg.rho0))

        outer_loops = max(1, int(al_cfg.max_outer)) if al_cfg is not None else 1
        for outer in range(outer_loops):
            optimizer = torch.optim.LBFGS(
                [theta],
                lr=float(lr),
                max_iter=int(max_iter),
                history_size=int(history_size),
                line_search_fn="strong_wolfe",
            )
            closure_stats: dict[str, Any] = {}

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                loss, stats, terms = evaluate_bilevel_objective(
                    theta,
                    objective,
                    solve_cfg,
                    param_constraints=param_constraints,
                    al_state=al_state,
                    logger=logger,
                    track_timing=track_timing,
                )
                closure_stats.clear()
                closure_stats.update(stats)
                loss.backward()
                return loss

            optimizer.step(closure)

            with torch.no_grad():
                loss_val, stats, terms = evaluate_bilevel_objective(
                    theta,
                    objective,
                    solve_cfg,
                    param_constraints=param_constraints,
                    al_state=al_state,
                    logger=logger,
                    track_timing=track_timing,
                )
                loss_scalar = float(loss_val.detach().cpu())
                history.append(loss_scalar)

            if al_state is not None:
                max_violation = _update_al_multipliers(terms, al_state)
                if max_violation <= float(al_cfg.tol):
                    break
                al_state.rho = min(float(al_cfg.rho_max), float(al_state.rho) * float(al_cfg.rho_growth))

            if loss_scalar < best_loss:
                best_loss = loss_scalar
                best_theta = theta.detach().clone()
                best_stats = stats

    return OuterOptimizationResult(theta=best_theta, loss=best_loss, history=history, stats=best_stats)


def optimize_theta_adam(
    theta_init: torch.Tensor,
    objective: BilevelObjective,
    solve_cfg: OuterSolveConfig,
    *,
    lr: float = 1e-2,
    steps: int = 100,
    weight_decay: float = 0.0,
    seed: Optional[int] = None,
    restarts: int = 1,
    restart_fn: Optional[Callable[[torch.Tensor, torch.Generator, int], torch.Tensor]] = None,
    param_constraints: Optional[Sequence[ParameterConstraint]] = None,
    al_cfg: Optional[AugmentedLagrangianConfig] = None,
    logger: Optional[Any] = None,
    track_timing: bool = False,
    line_search: bool = False,
) -> OuterOptimizationResult:
    device = theta_init.device
    rng = _default_rng(seed, device)
    best_loss = float("inf")
    best_theta = theta_init.detach()
    best_stats: dict[str, Any] = {}
    history: list[float] = []

    for restart in range(max(1, int(restarts))):
        if restart == 0:
            theta = theta_init.detach().clone()
        else:
            if restart_fn is None:
                noise = 0.05 * torch.randn(theta_init.shape, device=device, dtype=theta_init.dtype, generator=rng)
                theta = theta_init.detach().clone() + noise
            else:
                theta = restart_fn(theta_init.detach(), rng, restart)
        theta = theta.to(device=device)
        theta.requires_grad_(True)

        al_state = None
        if al_cfg is not None:
            al_state = AugmentedLagrangianState(rho=float(al_cfg.rho0))

        optimizer = torch.optim.Adam([
            {"params": [theta], "weight_decay": float(weight_decay)}
        ], lr=float(lr))

        outer_loops = max(1, int(al_cfg.max_outer)) if al_cfg is not None else 1
        for outer in range(outer_loops):
            for _ in range(int(steps)):
                optimizer.zero_grad(set_to_none=True)
                loss, stats, terms = evaluate_bilevel_objective(
                    theta,
                    objective,
                    solve_cfg,
                    param_constraints=param_constraints,
                    al_state=al_state,
                    logger=logger,
                    track_timing=track_timing,
                )
                loss.backward()
                if line_search:
                    direction = -theta.grad

                    def _loss_fn(candidate: torch.Tensor) -> torch.Tensor:
                        val, _, _ = evaluate_bilevel_objective(
                            candidate,
                            objective,
                            solve_cfg,
                            param_constraints=param_constraints,
                            al_state=al_state,
                            logger=logger,
                            track_timing=False,
                        )
                        return val

                    theta_new, _ = backtracking_line_search(_loss_fn, theta.detach(), direction.detach(), alpha0=lr)
                    theta.data.copy_(theta_new)
                else:
                    optimizer.step()

            with torch.no_grad():
                loss_val, stats, terms = evaluate_bilevel_objective(
                    theta,
                    objective,
                    solve_cfg,
                    param_constraints=param_constraints,
                    al_state=al_state,
                    logger=logger,
                    track_timing=track_timing,
                )
                loss_scalar = float(loss_val.detach().cpu())
                history.append(loss_scalar)

            if al_state is not None:
                max_violation = _update_al_multipliers(terms, al_state)
                if max_violation <= float(al_cfg.tol):
                    break
                al_state.rho = min(float(al_cfg.rho_max), float(al_state.rho) * float(al_cfg.rho_growth))

            if loss_scalar < best_loss:
                best_loss = loss_scalar
                best_theta = theta.detach().clone()
                best_stats = stats

    return OuterOptimizationResult(theta=best_theta, loss=best_loss, history=history, stats=best_stats)
