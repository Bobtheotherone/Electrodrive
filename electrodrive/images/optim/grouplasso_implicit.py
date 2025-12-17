from __future__ import annotations

from typing import Optional, Tuple
import math

import torch

from electrodrive.images.operator import BasisOperator
from .core import DTypePolicy, SparseSolveRequest, SparseSolveResult
from .diagnostics import active_set_stats, conditioning_proxy
from .precision import resolve_forward_dtype, resolve_kkt_dtype
from .lasso_implicit import (
    _estimate_lipschitz_from_ops,
    _extract_active_dense,
    _matvec_scaled,
    _prepare_weight_prior,
    _rmatvec_scaled,
)


def _group_prox(
    w: torch.Tensor,
    group_ids: torch.Tensor,
    lambda_group: float | torch.Tensor,
) -> torch.Tensor:
    if group_ids.numel() == 0:
        return w
    w_out = w.clone()
    group_ids = group_ids.to(device=w.device)
    if torch.is_tensor(lambda_group):
        lam_vec = lambda_group.to(device=w.device, dtype=w.dtype).view(-1)
        if lam_vec.numel() == 1:
            lam_vec = lam_vec.expand_as(w)
    else:
        lam_vec = None
    unique_groups = torch.unique(group_ids)
    for g_val in unique_groups:
        mask = group_ids == g_val
        if not bool(mask.any()):
            continue
        w_g = w_out[mask]
        norm_g = torch.linalg.norm(w_g)
        lam = float(lam_vec[mask].mean().item()) if lam_vec is not None else float(lambda_group)
        if float(norm_g) <= lam:
            w_out[mask] = 0.0
        else:
            shrink = (norm_g - lam) / norm_g
            w_out[mask] = shrink * w_g
    return w_out


def _support_from_groups(
    w_norm: torch.Tensor,
    group_ids: torch.Tensor,
    base_threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, float, bool]:
    if w_norm.numel() == 0:
        return torch.zeros((0,), device=w_norm.device, dtype=torch.long), 0.0, False
    group_ids = group_ids.to(device=w_norm.device)
    unique_groups = torch.unique(group_ids)
    norms = []
    for g_val in unique_groups:
        mask = group_ids == g_val
        if not bool(mask.any()):
            norms.append(torch.tensor(0.0, device=w_norm.device, dtype=w_norm.dtype))
        else:
            norms.append(torch.linalg.norm(w_norm[mask]))
    norms_t = torch.stack(norms) if norms else torch.zeros(0, device=w_norm.device, dtype=w_norm.dtype)
    max_norm = float(norms_t.max().item()) if norms_t.numel() else 0.0
    threshold = base_threshold if base_threshold is not None else max(1e-6, 1e-6 * max(1.0, max_norm))
    active_groups = norms_t > threshold
    active_set = set(
        int(g) for g, active in zip(unique_groups.tolist(), active_groups.tolist()) if active
    )
    mask = torch.zeros_like(group_ids, dtype=torch.bool)
    for g_val in active_set:
        mask |= group_ids == g_val
    indices = torch.nonzero(mask, as_tuple=False).view(-1)
    near = (norms_t > (0.5 * threshold)) & (norms_t < (2.0 * threshold)) if norms_t.numel() else torch.tensor(False)
    unstable = bool(near.any().item()) if norms_t.numel() else False
    return indices, float(threshold), unstable


def _fista_group_forward(
    A: torch.Tensor | BasisOperator,
    g: torch.Tensor,
    inv_norms: torch.Tensor,
    group_ids: torch.Tensor,
    lambda_group: float | torch.Tensor,
    weight_prior: Optional[torch.Tensor],
    lambda_weight_prior: Optional[float | torch.Tensor],
    max_iter: int,
    tol: float,
    warm_start_norm: Optional[torch.Tensor],
    X: Optional[torch.Tensor],
    L: float,
) -> Tuple[torch.Tensor, dict]:
    device = g.device
    dtype = g.dtype
    n_cols = inv_norms.shape[0]
    alpha = 1.0 / max(L, 1e-6)
    lambda_group_eff = lambda_group * alpha if torch.is_tensor(lambda_group) else float(lambda_group) * alpha
    w = warm_start_norm.clone() if warm_start_norm is not None else torch.zeros(n_cols, device=device, dtype=dtype)
    y = w.clone()
    t = 1.0
    last_rel_change = float("inf")
    converged = False
    iters_done = 0
    with torch.no_grad():
        for it in range(max_iter):
            iters_done = int(it + 1)
            w_prev = w
            r = _matvec_scaled(A, y, inv_norms, X) - g
            grad = _rmatvec_scaled(A, r, inv_norms, X)
            if weight_prior is not None and lambda_weight_prior is not None:
                w_phys = y * inv_norms
                grad_prior = w_phys - weight_prior
                if torch.is_tensor(lambda_weight_prior):
                    grad_prior = grad_prior * lambda_weight_prior
                else:
                    grad_prior = grad_prior * float(lambda_weight_prior)
                grad = grad + grad_prior * inv_norms
            w = y - alpha * grad
            w = _group_prox(w, group_ids, lambda_group_eff)
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            y = w + ((t - 1.0) / t_next) * (w - w_prev)
            diff = torch.linalg.norm(w - w_prev)
            denom = torch.linalg.norm(w) + 1e-9
            if float(denom) > 0.0:
                last_rel_change = float((diff / denom).item())
                if last_rel_change < tol:
                    converged = True
                    break
            t = t_next
    stats = {"iters": iters_done, "converged": converged, "rel_change": last_rel_change}
    return w, stats


def _fista_group_unrolled(
    A: torch.Tensor | BasisOperator,
    g: torch.Tensor,
    inv_norms: torch.Tensor,
    group_ids: torch.Tensor,
    lambda_group: float | torch.Tensor,
    weight_prior: Optional[torch.Tensor],
    lambda_weight_prior: Optional[float | torch.Tensor],
    steps: int,
    warm_start_norm: Optional[torch.Tensor],
    X: Optional[torch.Tensor],
    L: float,
) -> torch.Tensor:
    device = g.device
    dtype = g.dtype
    n_cols = inv_norms.shape[0]
    alpha = 1.0 / max(L, 1e-6)
    lambda_group_eff = lambda_group * alpha if torch.is_tensor(lambda_group) else float(lambda_group) * alpha
    w = warm_start_norm.clone() if warm_start_norm is not None else torch.zeros(n_cols, device=device, dtype=dtype)
    y = w
    t = 1.0
    for _ in range(int(steps)):
        r = _matvec_scaled(A, y, inv_norms, X) - g
        grad = _rmatvec_scaled(A, r, inv_norms, X)
        if weight_prior is not None and lambda_weight_prior is not None:
            w_phys = y * inv_norms
            grad_prior = w_phys - weight_prior
            if torch.is_tensor(lambda_weight_prior):
                grad_prior = grad_prior * lambda_weight_prior
            else:
                grad_prior = grad_prior * float(lambda_weight_prior)
            grad = grad + grad_prior * inv_norms
        w_next = y - alpha * grad
        w_next = _group_prox(w_next, group_ids, lambda_group_eff)
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        y = w_next + ((t - 1.0) / t_next) * (w_next - w)
        w = w_next
        t = t_next
    return w


class _ImplicitGroupLassoFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor | BasisOperator,
        g: torch.Tensor,
        X: Optional[torch.Tensor],
        col_norms: Optional[torch.Tensor],
        warm_start: Optional[torch.Tensor],
        weight_prior: Optional[torch.Tensor],
        lambda_group: float | torch.Tensor,
        lambda_l1: float,
        group_ids: Optional[torch.Tensor],
        lambda_weight_prior: float | torch.Tensor,
        normalize_columns: bool,
        max_iter: int,
        tol: float,
        support_threshold: Optional[float],
        kkt_damping: float,
        unroll_fallback_steps: int,
        use_unroll_if_unstable: bool,
        dtype_policy: Optional[DTypePolicy],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_operator = isinstance(A, BasisOperator)
        if is_operator:
            device = getattr(A, "device", g.device)
            dtype = getattr(A, "dtype", g.dtype)
            A_eff = A
            input_a_dtype = None
        else:
            device = A.device
            dtype = resolve_forward_dtype(dtype_policy, A.dtype)
            A_eff = A.to(device=device, dtype=dtype)
            input_a_dtype = A.dtype
        g_in_dtype = g.dtype
        g = g.to(device=device, dtype=dtype).view(-1)

        X_eff: Optional[torch.Tensor] = None
        if is_operator:
            if X is None:
                X_eff = getattr(A, "points", None)
            else:
                X_eff = X
            if X_eff is None:
                raise ValueError("Operator-mode implicit group-LASSO requires collocation points.")
            X_eff = X_eff.to(device=device, dtype=dtype).contiguous()

        if is_operator:
            n_rows, n_cols = A.shape  # type: ignore[assignment]
            if n_rows in (-1, 0):
                n_rows = int(g.shape[0])
        else:
            n_rows, n_cols = A_eff.shape

        if g.numel() == 0 or n_rows == 0 or n_cols == 0:
            w_empty = torch.zeros(n_cols, device=device, dtype=dtype)
            support = torch.zeros((0,), device=device, dtype=torch.long)
            ctx.mark_non_differentiable(support)
            return w_empty, support

        if col_norms is None:
            if is_operator:
                col_norms = getattr(A, "col_norms", None)
                if col_norms is None and normalize_columns:
                    col_norms = A.estimate_col_norms(X_eff)
                    A.col_norms = col_norms
            else:
                if normalize_columns:
                    col_norms = torch.linalg.norm(A_eff, dim=0)
        if col_norms is None or not normalize_columns:
            col_norms = torch.ones(n_cols, device=device, dtype=dtype)
        col_norms = col_norms.to(device=device, dtype=dtype).view(-1).clamp_min(1e-6)
        inv_norms = 1.0 / col_norms

        if group_ids is None:
            if is_operator:
                group_ids = getattr(A, "groups", None)
        if group_ids is None:
            group_ids = torch.arange(n_cols, device=device, dtype=torch.long)
        group_ids = group_ids.to(device=device, dtype=torch.long).view(-1)
        if group_ids.numel() != n_cols:
            raise ValueError("group_ids length does not match dictionary width.")
        if torch.is_tensor(lambda_group):
            lambda_group = lambda_group.to(device=device, dtype=dtype)

        w_prior, lambda_prior_eff = _prepare_weight_prior(
            weight_prior, lambda_weight_prior, n_cols, device, dtype
        )

        warm_start_norm: Optional[torch.Tensor] = None
        if warm_start is not None:
            w_phys = warm_start.to(device=device, dtype=dtype).view(-1)
            if w_phys.numel() == 1:
                w_phys = w_phys.expand(n_cols)
            if w_phys.numel() < n_cols:
                w_phys = torch.cat(
                    [w_phys, torch.zeros(n_cols - w_phys.numel(), device=device, dtype=dtype)],
                    dim=0,
                )
            elif w_phys.numel() > n_cols:
                w_phys = w_phys[:n_cols]
            warm_start_norm = w_phys * col_norms

        def _mv(v: torch.Tensor) -> torch.Tensor:
            return _matvec_scaled(A_eff, v, inv_norms, X_eff)

        def _rmv(r: torch.Tensor) -> torch.Tensor:
            return _rmatvec_scaled(A_eff, r, inv_norms, X_eff)

        L = _estimate_lipschitz_from_ops(_mv, _rmv, (n_rows, n_cols), device, dtype, max_power_iters=min(max_iter, 50))
        w_norm, stats = _fista_group_forward(
            A_eff,
            g,
            inv_norms,
            group_ids,
            lambda_group,
            w_prior,
            lambda_prior_eff,
            max_iter,
            tol,
            warm_start_norm,
            X_eff,
            L,
        )
        w_phys = w_norm * inv_norms
        support_idx, support_thr, unstable = _support_from_groups(w_norm, group_ids, support_threshold)

        ctx.save_for_backward(w_norm, inv_norms, g, support_idx, group_ids)
        ctx.A = A_eff
        ctx.input_a_dtype = input_a_dtype
        ctx.X = X_eff
        ctx.lambda_group = lambda_group
        ctx.lambda_l1 = float(lambda_l1)
        ctx.lambda_weight_prior = lambda_prior_eff
        ctx.normalize_columns = bool(normalize_columns)
        ctx.kkt_damping = float(kkt_damping)
        ctx.unroll_fallback_steps = int(unroll_fallback_steps)
        ctx.use_unroll_if_unstable = bool(use_unroll_if_unstable)
        ctx.dtype_policy = dtype_policy
        ctx.weight_prior = w_prior
        ctx.forward_dtype = dtype
        ctx.input_g_dtype = g_in_dtype
        ctx.support_threshold = float(support_thr)
        ctx.unstable = bool(unstable)
        ctx.L = float(L)
        ctx.stats = stats
        ctx.mark_non_differentiable(support_idx)
        return w_phys, support_idx

    @staticmethod
    def backward(ctx, grad_w: Optional[torch.Tensor], grad_support: Optional[torch.Tensor]):
        del grad_support
        if grad_w is None:
            return (None,) * 18

        w_norm, inv_norms, g, support_idx, group_ids = ctx.saved_tensors
        device = w_norm.device
        dtype = w_norm.dtype
        A = ctx.A
        X = ctx.X
        n_cols = w_norm.shape[0]
        n_rows = g.shape[0]

        grad_w = grad_w.to(device=device, dtype=dtype)
        grad_w_norm = grad_w * inv_norms

        def _unrolled_fallback():
            steps = max(int(ctx.unroll_fallback_steps), 0)
            if steps <= 0:
                return None, None
            if not (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
                return None, None
            with torch.enable_grad():
                if isinstance(A, torch.Tensor):
                    A_var = A.detach().to(device=device, dtype=dtype).requires_grad_(ctx.needs_input_grad[0])
                    g_var = g.detach().to(device=device, dtype=dtype).requires_grad_(ctx.needs_input_grad[1])
                    w_unrolled = _fista_group_unrolled(
                        A_var,
                        g_var,
                        inv_norms.detach(),
                        group_ids,
                        ctx.lambda_group,
                        ctx.weight_prior,
                        ctx.lambda_weight_prior,
                        steps,
                        w_norm.detach(),
                        X,
                        ctx.L,
                    )
                    w_phys = w_unrolled * inv_norms.detach()
                    loss = torch.dot(w_phys, grad_w.detach())
                    grads = torch.autograd.grad(
                        loss,
                        (A_var, g_var),
                        allow_unused=True,
                        retain_graph=False,
                        create_graph=False,
                    )
                    grad_A_unroll = grads[0] if ctx.needs_input_grad[0] else None
                    grad_g_unroll = grads[1] if ctx.needs_input_grad[1] else None
                    return grad_A_unroll, grad_g_unroll
                if not ctx.needs_input_grad[1]:
                    return None, None
                g_var = g.detach().to(device=device, dtype=dtype).requires_grad_(True)
                w_unrolled = _fista_group_unrolled(
                    A,
                    g_var,
                    inv_norms.detach(),
                    group_ids,
                    ctx.lambda_group,
                    ctx.weight_prior,
                    ctx.lambda_weight_prior,
                    steps,
                    w_norm.detach(),
                    X,
                    ctx.L,
                )
                w_phys = w_unrolled * inv_norms.detach()
                loss = torch.dot(w_phys, grad_w.detach())
                grad_g_unroll = torch.autograd.grad(
                    loss,
                    g_var,
                    allow_unused=True,
                    retain_graph=False,
                    create_graph=False,
                )[0]
            return None, grad_g_unroll

        if support_idx.numel() == 0:
            grad_A = torch.zeros_like(A) if isinstance(A, torch.Tensor) and ctx.needs_input_grad[0] else None
            grad_g = torch.zeros_like(g) if ctx.needs_input_grad[1] else None
            return (
                grad_A,
                grad_g.to(dtype=ctx.input_g_dtype) if grad_g is not None else None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        if ctx.use_unroll_if_unstable and ctx.unstable:
            grad_A_unroll, grad_g_unroll = _unrolled_fallback()
            if grad_g_unroll is not None:
                return (
                    grad_A_unroll,
                    grad_g_unroll.to(dtype=ctx.input_g_dtype),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )

        kkt_dtype = resolve_kkt_dtype(ctx.dtype_policy)
        support_idx = support_idx.to(device=device)
        inv_s = inv_norms.index_select(0, support_idx)
        grad_w_s = grad_w_norm.index_select(0, support_idx)
        group_ids_s = group_ids.index_select(0, support_idx)

        try:
            A_s = _extract_active_dense(
                A,
                X,
                support_idx,
                device=device,
                dtype=kkt_dtype,
                n_rows=n_rows,
            )
            A_norm_s = A_s * inv_s.to(dtype=kkt_dtype)
            H = A_norm_s.transpose(0, 1).matmul(A_norm_s)
            if ctx.lambda_weight_prior is not None:
                if torch.is_tensor(ctx.lambda_weight_prior):
                    lam_s = ctx.lambda_weight_prior.index_select(0, support_idx).to(dtype=kkt_dtype)
                else:
                    lam_s = torch.full_like(inv_s, float(ctx.lambda_weight_prior), dtype=kkt_dtype)
                H = H + torch.diag(lam_s * (inv_s.to(dtype=kkt_dtype) ** 2))

            # Group-lasso Hessian contributions.
            w_s = w_norm.index_select(0, support_idx).to(dtype=kkt_dtype)
            unique_groups = torch.unique(group_ids_s)
            for g_val in unique_groups:
                mask = group_ids_s == g_val
                if not bool(mask.any()):
                    continue
                idx = torch.nonzero(mask, as_tuple=False).view(-1)
                w_g = w_s.index_select(0, idx)
                norm_g = torch.linalg.norm(w_g).clamp_min(1e-12)
                if torch.is_tensor(ctx.lambda_group):
                    lam_g = ctx.lambda_group.index_select(0, support_idx)
                    lam_g = float(lam_g[mask].mean().item())
                else:
                    lam_g = float(ctx.lambda_group)
                if lam_g <= 0.0:
                    continue
                eye = torch.eye(idx.shape[0], device=H.device, dtype=H.dtype)
                outer = torch.outer(w_g, w_g) / (norm_g * norm_g)
                J = (eye - outer) / norm_g
                H[idx[:, None], idx[None, :]] += lam_g * J

            diag = torch.diag(H)
            diag_scale = torch.max(torch.abs(diag)).clamp_min(1.0)
            H = H + (ctx.kkt_damping * diag_scale) * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)

            try:
                L = torch.linalg.cholesky(H)
                v = torch.cholesky_solve(grad_w_s.unsqueeze(-1).to(dtype=kkt_dtype), L).squeeze(-1)
            except Exception:
                v = torch.linalg.solve(H, grad_w_s.to(dtype=kkt_dtype))

            v = v.to(dtype=dtype)
            u_full = torch.zeros(n_cols, device=device, dtype=dtype)
            u_full.index_copy_(0, support_idx, v * inv_s)

            grad_g = None
            if ctx.needs_input_grad[1]:
                if isinstance(A, BasisOperator):
                    grad_g = A.matvec(u_full, X)
                else:
                    grad_g = A.matmul(u_full)

            grad_A = None
            if isinstance(A, torch.Tensor) and ctx.needs_input_grad[0]:
                w_phys = w_norm * inv_norms
                r = A.matmul(w_phys) - g
                p = A.matmul(u_full)
                grad_A = -(
                    r.unsqueeze(1) * u_full.unsqueeze(0)
                    + p.unsqueeze(1) * w_phys.unsqueeze(0)
                )

        except Exception:
            grad_A_unroll, grad_g_unroll = _unrolled_fallback()
            grad_A = grad_A_unroll
            grad_g = grad_g_unroll

        if grad_g is not None:
            grad_g = grad_g.to(dtype=ctx.input_g_dtype)
        if grad_A is not None and ctx.input_a_dtype is not None:
            grad_A = grad_A.to(dtype=ctx.input_a_dtype)
        return (
            grad_A,
            grad_g,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def implicit_grouplasso_solve(req: SparseSolveRequest) -> SparseSolveResult:
    dtype_policy = req.dtype_policy
    if dtype_policy is None:
        dtype_policy = DTypePolicy(forward_dtype=req.g.dtype)

    w, support_idx = _ImplicitGroupLassoFn.apply(
        req.A,
        req.g,
        req.X,
        req.col_norms,
        req.warm_start,
        req.weight_prior,
        req.lambda_group,
        float(req.lambda_l1),
        req.group_ids,
        req.lambda_weight_prior,
        bool(req.normalize_columns),
        int(req.max_iter),
        float(req.tol),
        None,
        1e-6,
        8,
        True,
        dtype_policy,
    )

    stats: dict = {}
    if req.return_stats:
        stats.update(active_set_stats(w))
        stats["support_size"] = int(support_idx.numel())
        stats["dtype"] = str(w.dtype)

    aux: dict = {}
    if req.return_stats:
        try:
            if support_idx.numel() > 0:
                n_rows = req.g.numel()
                A_s = _extract_active_dense(
                    req.A,
                    req.X,
                    support_idx,
                    device=w.device,
                    dtype=resolve_kkt_dtype(dtype_policy),
                    n_rows=n_rows,
                )
                inv_norms = torch.ones(support_idx.shape[0], dtype=w.dtype, device=w.device)
                if req.normalize_columns:
                    col_norms = req.col_norms
                    if col_norms is None and isinstance(req.A, BasisOperator):
                        col_norms = getattr(req.A, "col_norms", None)
                    if col_norms is not None:
                        inv_norms = 1.0 / col_norms.to(device=w.device, dtype=w.dtype)[support_idx]
                H = (A_s * inv_norms).transpose(0, 1).matmul(A_s * inv_norms)
                aux.update(conditioning_proxy(H))
        except Exception:
            pass

    return SparseSolveResult(w=w, support=support_idx, stats=stats, aux=aux)
