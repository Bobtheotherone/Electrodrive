from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import torch

from electrodrive.images.operator import BasisOperator
from .core import ADMMConfig, ConstraintOp, ConstraintSpec, DTypePolicy, SparseSolveRequest, SparseSolveResult
from .precision import resolve_forward_dtype
from .lasso_implicit import (
    _estimate_lipschitz_from_ops,
    _matvec_scaled,
    _prepare_weight_prior,
    _rmatvec_scaled,
    _support_from_weights,
)
from .grouplasso_implicit import _group_prox, _support_from_groups
from .bases.fourier_planar import PlanarFFTConstraintOp
from .bases.spherical_harmonics import SphericalHarmonicsConstraintOp
from .bases.fourier_bessel import CylindricalFourierConstraintOp


@dataclass
class _ConstraintBlock:
    name: str
    kind: str
    eps: float
    op: ConstraintOp


class WeightedConstraintOp:
    def __init__(self, op: ConstraintOp, weight: float) -> None:
        self.op = op
        self.weight = float(weight)

    def apply(self, r: torch.Tensor) -> torch.Tensor:
        return self.op.apply(r) * self.weight

    def adjoint(self, c: torch.Tensor) -> torch.Tensor:
        return self.op.adjoint(c) * self.weight


class CollocationConstraintOp:
    def __init__(
        self,
        n_rows: int,
        indices: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.n_rows = int(n_rows)
        self.indices = indices.to(dtype=torch.long)
        self.weights = weights

    def apply(self, r: torch.Tensor) -> torch.Tensor:
        r_sel = r.index_select(0, self.indices)
        if self.weights is not None:
            return r_sel * self.weights
        return r_sel

    def adjoint(self, c: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.n_rows, device=c.device, dtype=c.dtype)
        if self.weights is not None:
            out.index_add_(0, self.indices, c * self.weights)
        else:
            out.index_add_(0, self.indices, c)
        return out


def _project_constraint(kind: str, c: torch.Tensor, eps: float) -> torch.Tensor:
    if kind == "eq":
        return torch.zeros_like(c)
    if kind == "l2":
        norm = torch.linalg.norm(c)
        if float(norm) <= eps:
            return c
        if float(norm) == 0.0:
            return c
        return c * (eps / norm)
    if kind == "linf":
        return torch.clamp(c, min=-eps, max=eps)
    raise ValueError(f"Unknown constraint kind '{kind}'.")


def _constraint_violation(kind: str, c: torch.Tensor, eps: float) -> float:
    if kind == "eq":
        return float(torch.linalg.norm(c).item())
    if kind == "l2":
        norm = float(torch.linalg.norm(c).item())
        return max(0.0, norm - eps)
    if kind == "linf":
        max_abs = float(torch.max(torch.abs(c)).item()) if c.numel() else 0.0
        return max(0.0, max_abs - eps)
    return float("nan")


def _resolve_inv_norms(
    A: torch.Tensor | BasisOperator,
    X: Optional[torch.Tensor],
    normalize_columns: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(A, BasisOperator):
        col_norms = getattr(A, "col_norms", None)
        if normalize_columns:
            if col_norms is None:
                col_norms = A.estimate_col_norms(X)
                A.col_norms = col_norms
        if col_norms is None or not normalize_columns:
            col_norms = torch.ones((A.shape[1],), device=device, dtype=dtype)
        col_norms = col_norms.to(device=device, dtype=dtype).view(-1).clamp_min(1e-6)
    else:
        if normalize_columns:
            col_norms = torch.linalg.norm(A, dim=0).clamp_min(1e-6)
        else:
            col_norms = torch.ones((A.shape[1],), device=device, dtype=dtype)
    return 1.0 / col_norms


def _estimate_augmented_lipschitz(
    A: torch.Tensor | BasisOperator,
    X: Optional[torch.Tensor],
    inv_norms: torch.Tensor,
    constraint_blocks: Sequence[_ConstraintBlock],
    rho: float,
    prior_diag: Optional[torch.Tensor],
    n_rows: int,
    n_cols: int,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    if n_rows == 0 or n_cols == 0:
        return 1.0
    if not constraint_blocks and prior_diag is None:
        def _mv(v: torch.Tensor) -> torch.Tensor:
            return _matvec_scaled(A, v, inv_norms, X)

        def _rmv(r: torch.Tensor) -> torch.Tensor:
            return _rmatvec_scaled(A, r, inv_norms, X)

        return _estimate_lipschitz_from_ops(_mv, _rmv, (n_rows, n_cols), device, dtype, max_power_iters=25)

    def _apply_hessian(v: torch.Tensor) -> torch.Tensor:
        r = _matvec_scaled(A, v, inv_norms, X)
        out = _rmatvec_scaled(A, r, inv_norms, X)
        if constraint_blocks:
            for blk in constraint_blocks:
                c = blk.op.apply(r)
                adj = blk.op.adjoint(c)
                out = out + float(rho) * _rmatvec_scaled(A, adj, inv_norms, X)
        if prior_diag is not None:
            out = out + prior_diag * v
        return out

    with torch.no_grad():
        v = torch.randn(n_cols, device=device, dtype=dtype)
        v_norm = torch.linalg.norm(v)
        if float(v_norm) > 0.0:
            v = v / v_norm
        for _ in range(20):
            hv = _apply_hessian(v)
            hv_norm = torch.linalg.norm(hv)
            if float(hv_norm) < 1e-8:
                break
            v = (hv / hv_norm).detach()
        hv = _apply_hessian(v)
        L = float(torch.dot(v, hv).item())
    return L if L > 0.0 else 1.0


def _soft_threshold(x: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - thr, min=0.0)


def _compile_constraints(
    specs: Sequence[ConstraintSpec],
    *,
    device: torch.device,
    dtype: torch.dtype,
    points: Optional[torch.Tensor],
    is_boundary: Optional[torch.Tensor],
    n_rows: int,
) -> list[_ConstraintBlock]:
    blocks: list[_ConstraintBlock] = []
    for idx, spec in enumerate(specs):
        basis = (spec.basis or "collocation").strip().lower()
        kind = (spec.kind or "eq").strip().lower()
        eps = float(spec.eps or 0.0)
        params = spec.params or {}
        name = spec.name or f"constraint_{idx}"

        if basis == "collocation":
            indices = None
            if "indices" in params:
                indices = torch.as_tensor(params["indices"], device=device, dtype=torch.long)
            elif "mask" in params:
                mask = torch.as_tensor(params["mask"], device=device, dtype=torch.bool)
                indices = torch.nonzero(mask, as_tuple=False).view(-1)
            elif spec.region in {"boundary", "interior"}:
                if is_boundary is None:
                    raise ValueError("Constraint region requested but is_boundary mask is missing.")
                mask = torch.as_tensor(is_boundary, device=device, dtype=torch.bool).view(-1)
                if spec.region == "interior":
                    mask = ~mask
                indices = torch.nonzero(mask, as_tuple=False).view(-1)
            if indices is None:
                indices = torch.arange(n_rows, device=device, dtype=torch.long)
            weights = None
            if "weights" in params:
                weights = torch.as_tensor(params["weights"], device=device, dtype=dtype).view(-1)
            op: ConstraintOp = CollocationConstraintOp(n_rows=n_rows, indices=indices, weights=weights)
        elif basis in {"planar_fft", "fft_planar"}:
            if params.get("irregular_fft", False):
                raise NotImplementedError("Irregular FFT constraints are not implemented yet.")
            grid_shape = params.get("grid_shape")
            if grid_shape is None:
                grid_h = params.get("grid_h")
                grid_w = params.get("grid_w")
                if grid_h is None or grid_w is None:
                    raise ValueError("Planar FFT constraint requires grid_shape or grid_h/grid_w.")
                grid_shape = (int(grid_h), int(grid_w))
            mode_indices = params.get("mode_indices")
            mask = params.get("mask")
            fft_shift = bool(params.get("fft_shift", False))
            op = PlanarFFTConstraintOp(
                grid_shape=grid_shape,
                mode_indices=mode_indices,
                mask=mask,
                fft_shift=fft_shift,
                device=device,
                dtype=dtype,
            )
        elif basis in {"sphere_sh", "spherical_harmonics"}:
            lmax = params.get("Lmax")
            if lmax is None:
                lmax = params.get("lmax")
            if lmax is None:
                raise ValueError("Spherical harmonics constraint requires Lmax.")
            theta = params.get("theta")
            phi = params.get("phi")
            mode_indices = params.get("mode_indices")
            op = SphericalHarmonicsConstraintOp(
                points=points,
                theta=theta,
                phi=phi,
                lmax=int(lmax),
                mode_indices=mode_indices,
                device=device,
                dtype=dtype,
            )
        elif basis in {"fourier_bessel", "cylindrical_fourier"}:
            grid_shape = params.get("grid_shape")
            if grid_shape is None:
                grid_phi = params.get("grid_phi")
                grid_z = params.get("grid_z")
                if grid_phi is None or grid_z is None:
                    raise ValueError("Cylindrical constraint requires grid_shape or grid_phi/grid_z.")
                grid_shape = (int(grid_phi), int(grid_z))
            mode_indices = params.get("mode_indices")
            mask = params.get("mask")
            op = CylindricalFourierConstraintOp(
                grid_shape=grid_shape,
                mode_indices=mode_indices,
                mask=mask,
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unknown constraint basis '{basis}'.")

        if abs(float(spec.weight) - 1.0) > 1e-12:
            op = WeightedConstraintOp(op, spec.weight)
        blocks.append(_ConstraintBlock(name=name, kind=kind, eps=eps, op=op))
    return blocks


def _fista_w_update(
    A: torch.Tensor | BasisOperator,
    g: torch.Tensor,
    X: Optional[torch.Tensor],
    inv_norms: torch.Tensor,
    weight_prior: Optional[torch.Tensor],
    lambda_weight_prior: Optional[float | torch.Tensor],
    lambda_l1: float,
    group_ids: Optional[torch.Tensor],
    lambda_group: float | torch.Tensor,
    constraint_blocks: Sequence[_ConstraintBlock],
    z_list: Sequence[torch.Tensor],
    u_list: Sequence[torch.Tensor],
    rho: float,
    w_init: torch.Tensor,
    max_iter: int,
    tol: float,
    L: float,
) -> tuple[torch.Tensor, float]:
    device = g.device
    dtype = g.dtype
    n_cols = inv_norms.shape[0]
    alpha = 1.0 / max(L, 1e-6)
    thr = torch.full((n_cols,), float(lambda_l1), device=device, dtype=dtype) * inv_norms * alpha
    use_group = group_ids is not None and (float(lambda_group) != 0.0 if not torch.is_tensor(lambda_group) else True)
    lambda_group_eff = lambda_group * alpha if torch.is_tensor(lambda_group) else float(lambda_group) * alpha

    def _grad(w_norm: torch.Tensor) -> torch.Tensor:
        r = _matvec_scaled(A, w_norm, inv_norms, X) - g
        grad = _rmatvec_scaled(A, r, inv_norms, X)
        if weight_prior is not None and lambda_weight_prior is not None:
            w_phys = w_norm * inv_norms
            grad_prior = w_phys - weight_prior
            if torch.is_tensor(lambda_weight_prior):
                grad_prior = grad_prior * lambda_weight_prior
            else:
                grad_prior = grad_prior * float(lambda_weight_prior)
            grad = grad + grad_prior * inv_norms
        if constraint_blocks:
            for blk, z_i, u_i in zip(constraint_blocks, z_list, u_list):
                c = blk.op.apply(r)
                adj = blk.op.adjoint(c - z_i + u_i)
                grad = grad + float(rho) * _rmatvec_scaled(A, adj, inv_norms, X)
        return grad

    w = w_init
    y = w
    t = 1.0
    last_rel_change = float("inf")
    for _ in range(int(max_iter)):
        w_prev = w
        grad = _grad(y)
        w_next = y - alpha * grad
        if float(lambda_l1) > 0.0:
            w_next = _soft_threshold(w_next, thr)
        if use_group:
            w_next = _group_prox(w_next, group_ids, lambda_group_eff)
        t_next = 0.5 * (1.0 + torch.sqrt(torch.tensor(1.0 + 4.0 * t * t, device=device, dtype=dtype)))
        y = w_next + ((t - 1.0) / t_next) * (w_next - w)
        w = w_next
        t = float(t_next.item())
        diff = torch.linalg.norm(w - w_prev)
        denom = torch.linalg.norm(w) + 1e-9
        if float(denom) > 0.0:
            last_rel_change = float((diff / denom).item())
            if last_rel_change < tol:
                break
    return w, last_rel_change


def admm_constrained_solve(
    req: SparseSolveRequest,
    cfg: Optional[ADMMConfig] = None,
) -> SparseSolveResult:
    cfg = cfg or ADMMConfig()
    if cfg.diff_mode.strip().lower() in {"implicit", "kkt"}:
        raise NotImplementedError("Implicit KKT differentiation for ADMM is not implemented yet.")

    A = req.A
    is_operator = isinstance(A, BasisOperator)
    if is_operator:
        device = getattr(A, "device", req.g.device)
        dtype = getattr(A, "dtype", req.g.dtype)
        A_eff = A
        input_a_dtype = None
    else:
        device = A.device
        dtype = resolve_forward_dtype(req.dtype_policy, A.dtype)
        A_eff = A.to(device=device, dtype=dtype)
        input_a_dtype = A.dtype

    g = req.g.to(device=device, dtype=dtype).view(-1)
    X = req.X.to(device=device, dtype=dtype) if req.X is not None else None

    n_cols = A_eff.shape[1]
    n_rows = g.shape[0]
    inv_norms = _resolve_inv_norms(A_eff, X, req.normalize_columns, device, dtype)

    w_prior, lambda_prior_eff = _prepare_weight_prior(
        req.weight_prior, req.lambda_weight_prior, n_cols, device, dtype
    )

    warm_start_norm: Optional[torch.Tensor] = None
    if req.warm_start is not None:
        w_phys = req.warm_start.to(device=device, dtype=dtype).view(-1)
        if w_phys.numel() == 1:
            w_phys = w_phys.expand(n_cols)
        if w_phys.numel() < n_cols:
            w_phys = torch.cat(
                [w_phys, torch.zeros(n_cols - w_phys.numel(), device=device, dtype=dtype)],
                dim=0,
            )
        elif w_phys.numel() > n_cols:
            w_phys = w_phys[:n_cols]
        warm_start_norm = w_phys * (1.0 / inv_norms)

    constraint_blocks = _compile_constraints(
        req.constraints,
        device=device,
        dtype=dtype,
        points=X,
        is_boundary=req.is_boundary,
        n_rows=n_rows,
    )

    if lambda_prior_eff is None:
        prior_diag = None
    else:
        if torch.is_tensor(lambda_prior_eff):
            prior_diag = lambda_prior_eff.to(device=device, dtype=dtype) * (inv_norms * inv_norms)
        else:
            prior_diag = torch.full_like(inv_norms, float(lambda_prior_eff)) * (inv_norms * inv_norms)

    w_norm_init = warm_start_norm if warm_start_norm is not None else torch.zeros(n_cols, device=device, dtype=dtype)

    n_iters = int(cfg.unroll_steps) if int(cfg.unroll_steps) > 0 else int(cfg.max_iter)
    use_no_grad = int(cfg.unroll_steps) <= 0

    def _solve_loop() -> tuple[torch.Tensor, dict[str, Any]]:
        w_norm = w_norm_init
        z_list: list[torch.Tensor] = []
        u_list: list[torch.Tensor] = []
        if constraint_blocks:
            r0 = _matvec_scaled(A_eff, w_norm, inv_norms, X) - g
            for blk in constraint_blocks:
                c0 = blk.op.apply(r0)
                z0 = _project_constraint(blk.kind, c0, blk.eps)
                z_list.append(z0)
                u_list.append(torch.zeros_like(c0))
        else:
            z_list = []
            u_list = []

        rho = float(cfg.rho)
        L = _estimate_augmented_lipschitz(
            A_eff,
            X,
            inv_norms,
            constraint_blocks,
            rho,
            prior_diag,
            n_rows,
            n_cols,
            device,
            dtype,
        )

        primal_res = float("nan")
        dual_res = float("nan")
        for it in range(n_iters):
            w_new, rel_change = _fista_w_update(
                A_eff,
                g,
                X,
                inv_norms,
                w_prior,
                lambda_prior_eff,
                req.lambda_l1,
                req.group_ids,
                req.lambda_group,
                constraint_blocks,
                z_list,
                u_list,
                rho,
                w_norm,
                max_iter=int(cfg.w_update_iters),
                tol=float(cfg.tol),
                L=L,
            )
            w_norm = w_new

            if not constraint_blocks:
                primal_res = rel_change
                dual_res = 0.0
                if primal_res < float(cfg.tol):
                    return w_norm, {"iters": it + 1, "rel_change": rel_change, "primal_res": primal_res, "dual_res": dual_res, "rho": rho}
                continue

            r = _matvec_scaled(A_eff, w_norm, inv_norms, X) - g
            primal_terms: list[torch.Tensor] = []
            dual_terms: list[torch.Tensor] = []
            for idx_blk, blk in enumerate(constraint_blocks):
                c = blk.op.apply(r)
                z_old = z_list[idx_blk]
                z_new = _project_constraint(blk.kind, c + u_list[idx_blk], blk.eps)
                z_list[idx_blk] = z_new
                u_list[idx_blk] = u_list[idx_blk] + c - z_new
                primal_terms.append(torch.linalg.norm(c - z_new))
                dual_terms.append(torch.linalg.norm(z_new - z_old))

            primal_res = float(torch.linalg.norm(torch.stack(primal_terms)).item()) if primal_terms else 0.0
            dual_res = float(rho * torch.linalg.norm(torch.stack(dual_terms)).item()) if dual_terms else 0.0

            if rho < cfg.max_rho and cfg.rho_growth > 1.0:
                if primal_res > 10.0 * dual_res:
                    rho = min(float(cfg.max_rho), rho * float(cfg.rho_growth))
                    L = _estimate_augmented_lipschitz(
                        A_eff,
                        X,
                        inv_norms,
                        constraint_blocks,
                        rho,
                        prior_diag,
                        n_rows,
                        n_cols,
                        device,
                        dtype,
                    )
            if primal_res < float(cfg.tol) and dual_res < float(cfg.tol):
                return w_norm, {"iters": it + 1, "rel_change": rel_change, "primal_res": primal_res, "dual_res": dual_res, "rho": rho}

        return w_norm, {"iters": n_iters, "rel_change": primal_res, "primal_res": primal_res, "dual_res": dual_res, "rho": rho}

    if use_no_grad:
        with torch.no_grad():
            w_norm, stats = _solve_loop()
    else:
        w_norm, stats = _solve_loop()

    w_phys = w_norm * inv_norms
    support_idx: torch.Tensor
    if req.group_ids is not None and (float(req.lambda_group) != 0.0 if not torch.is_tensor(req.lambda_group) else True):
        support_idx, _, _ = _support_from_groups(w_norm, req.group_ids)
    else:
        support_idx, _, _ = _support_from_weights(w_phys)

    constraint_residuals: dict[str, float] = {}
    if constraint_blocks:
        r_final = _matvec_scaled(A_eff, w_norm, inv_norms, X) - g
        for blk in constraint_blocks:
            c = blk.op.apply(r_final)
            constraint_residuals[blk.name] = _constraint_violation(blk.kind, c, blk.eps)

    stats_out: dict[str, Any] = {
        "solver": "admm_constrained",
        "iters": stats.get("iters", 0),
        "rel_change": stats.get("rel_change", float("nan")),
        "primal_res": stats.get("primal_res", float("nan")),
        "dual_res": stats.get("dual_res", float("nan")),
        "rho": stats.get("rho", float(cfg.rho)),
        "dtype": str(w_phys.dtype),
    }
    if constraint_blocks:
        stats_out["constraint_residuals"] = constraint_residuals
    stats_out["support_size"] = int(support_idx.numel())

    return SparseSolveResult(
        w=w_phys.to(device=device, dtype=dtype),
        support=support_idx,
        stats=stats_out,
        aux={"input_a_dtype": input_a_dtype},
    )
