from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from electrodrive.images.operator import BasisOperator


def _group_prox(
    w: torch.Tensor,
    group_ids: Optional[torch.Tensor],
    lambda_group: float | torch.Tensor,
) -> torch.Tensor:
    """Group-lasso proximal operator applied after elementwise shrinkage."""
    if group_ids is None:
        return w
    if not torch.is_tensor(lambda_group) and lambda_group <= 0.0:
        return w
    if group_ids.shape[0] != w.shape[0]:
        return w
    w_out = w.clone()
    lam_vec = None
    if torch.is_tensor(lambda_group):
        lam_vec = lambda_group.to(device=w.device, dtype=w.dtype).view(-1)
        if lam_vec.numel() == 1:
            lam_vec = lam_vec.expand_as(w)
        if lam_vec.numel() not in (0, w.shape[0]):
            return w
    group_ids = group_ids.to(device=w.device)
    unique_groups, inverse = torch.unique(
        group_ids, sorted=True, return_inverse=True
    )
    if unique_groups.numel() == 0:
        return w_out

    norms_sq = torch.zeros(
        unique_groups.numel(), device=w.device, dtype=w.dtype
    )
    norms_sq.scatter_add_(0, inverse, w_out * w_out)
    norms = torch.sqrt(norms_sq)

    if lam_vec is not None:
        lam_sum = torch.zeros(
            unique_groups.numel(), device=w.device, dtype=w.dtype
        )
        lam_sum.scatter_add_(0, inverse, lam_vec)
        counts = torch.bincount(inverse, minlength=unique_groups.numel()).to(
            device=w.device, dtype=w.dtype
        )
        lam_group = lam_sum / counts.clamp_min(1.0)
    else:
        lam_group = torch.full(
            (unique_groups.numel(),),
            float(lambda_group),  # type: ignore[arg-type]
            device=w.device,
            dtype=w.dtype,
        )

    shrink = torch.where(
        norms > 0,
        (norms - lam_group) / norms,
        torch.zeros_like(norms),
    )
    shrink = torch.clamp(shrink, min=0.0)
    w_out = w_out * shrink[inverse]
    return w_out


def _map_groups_to_indices(
    group_ids: Optional[torch.Tensor],
    K: int,
    device: torch.device,
) -> torch.Tensor:
    """Return contiguous group indices [0, G) for threshold lookup."""
    if group_ids is None or group_ids.shape[0] != K:
        return torch.zeros(K, device=device, dtype=torch.long)
    gids = group_ids.to(device=device, dtype=torch.long).view(-1)
    _, inverse = torch.unique(gids, sorted=True, return_inverse=True)
    return inverse


class LISTALayer(nn.Module):
    """
    LISTA-style unrolled sparse solver operating in column-normalised space.

    For small candidate counts (K <= dense_threshold) a dense S/B parameterisation
    is used. For larger K, we switch to diagonal S/B with optional low-rank
    corrections and per-group thresholds.
    """

    def __init__(
        self,
        K: int,
        n_steps: int,
        *,
        n_groups: Optional[int] = None,
        rank: int = 0,
        dense_threshold: int = 512,
        init_L: float = 1.0,
        init_theta: float = 1e-3,
    ) -> None:
        super().__init__()
        self.K = int(K)
        self.n_steps = int(n_steps)
        self.use_dense = self.K <= int(dense_threshold)
        self.rank = int(rank) if rank > 0 else 0

        if self.use_dense:
            eye = torch.eye(self.K)
            self.S_dense = nn.Parameter(eye * (1.0 - 1.0 / max(1e-6, init_L)))
            self.B_dense = nn.Parameter(eye * (1.0 / max(1e-6, init_L)))
            self.theta_elem = nn.Parameter(torch.full((self.K,), float(init_theta)))
            self.theta_group = None
            self.U = None
            self.V = None
        else:
            self.S_diag = nn.Parameter(
                torch.ones(self.K) * (1.0 - 1.0 / max(1e-6, init_L))
            )
            self.B_diag = nn.Parameter(torch.ones(self.K) * (1.0 / max(1e-6, init_L)))
            self.theta_group = nn.Parameter(
                torch.full((max(1, n_groups or 1),), float(init_theta))
            )
            if self.rank > 0:
                self.U = nn.Parameter(torch.zeros(self.K, self.rank))
                self.V = nn.Parameter(torch.zeros(self.K, self.rank))
            else:
                self.U = None
                self.V = None
            self.S_dense = None
            self.B_dense = None
            self.theta_elem = None

    @staticmethod
    def soft_threshold(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.clamp(torch.abs(x) - theta, min=0.0)

    def _theta_for_groups(
        self,
        group_ids: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if self.use_dense and self.theta_elem is not None:
            return self.theta_elem.to(device=device)

        if self.theta_group is None:
            return torch.zeros(self.K, device=device)

        idx = _map_groups_to_indices(group_ids, self.K, device=device)
        # Clamp to available theta entries to avoid index errors when G varies.
        idx = torch.clamp(idx, max=self.theta_group.shape[0] - 1)
        theta = self.theta_group.to(device=device)
        return theta[idx]

    def _resolve_linear_ops(
        self,
        A: torch.Tensor | BasisOperator,
        X: torch.Tensor | None,
        g: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Callable[[torch.Tensor], torch.Tensor],
        torch.Callable[[torch.Tensor], torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Return (col_norms, inv_norms, matvec, rmatvec, group_ids)."""
        if isinstance(A, BasisOperator):
            device = getattr(A, "device", g.device)
            dtype = getattr(A, "dtype", g.dtype)
            pts = X if X is not None else getattr(A, "points", None)
            if pts is None:
                raise ValueError("LISTA requires collocation points for operator A.")
            pts = pts.to(device=device, dtype=dtype).contiguous()
            col_norms = getattr(A, "col_norms", None)
            if col_norms is None:
                col_norms = A.estimate_col_norms(pts)
            col_norms = col_norms.to(device=device, dtype=dtype).clamp_min(1e-6)
            A.col_norms = col_norms
            inv_norms = 1.0 / col_norms

            def matvec(w: torch.Tensor) -> torch.Tensor:
                return A.matvec(w * inv_norms, pts)  # type: ignore[arg-type]

            def rmatvec(r_vec: torch.Tensor) -> torch.Tensor:
                return A.rmatvec(r_vec, pts) * inv_norms  # type: ignore[arg-type]

            group_ids = getattr(A, "groups", None)
            return col_norms, inv_norms, matvec, rmatvec, group_ids

        if not isinstance(A, torch.Tensor):
            raise TypeError("LISTALayer expects BasisOperator or torch.Tensor for A.")
        A_t = A.to(device=g.device, dtype=g.dtype)
        if A_t.ndim != 2:
            raise ValueError(f"A must be 2D, got shape {tuple(A_t.shape)}")
        col_norms = torch.linalg.norm(A_t, dim=0).clamp_min(1e-6)
        inv_norms = 1.0 / col_norms

        def matvec(w: torch.Tensor) -> torch.Tensor:
            return A_t @ (w * inv_norms)

        def rmatvec(r_vec: torch.Tensor) -> torch.Tensor:
            return (A_t.T @ r_vec) * inv_norms

        return col_norms, inv_norms, matvec, rmatvec, None

    def forward(
        self,
        A: torch.Tensor | BasisOperator,
        X: Optional[torch.Tensor],
        g: torch.Tensor,
        *,
        group_ids: Optional[torch.Tensor] = None,
        lambda_group: float | torch.Tensor = 0.0,
    ) -> torch.Tensor:
        """
        Run the unrolled LISTA iterations and return physical-space weights.

        Parameters
        ----------
        A : torch.Tensor or BasisOperator
            Dictionary or operator. Operator mode requires collocation points ``X``.
        X : torch.Tensor or None
            Collocation points for operator mode; ignored for dense matrices.
        g : torch.Tensor
            Target potentials.
        group_ids : torch.Tensor, optional
            Precomputed group IDs. If not provided, operator groups are used when
            available.
        lambda_group : float
            Group-sparsity strength for the group prox step.
        """
        device = g.device
        dtype = g.dtype

        col_norms, inv_norms, matvec, rmatvec, group_default = self._resolve_linear_ops(
            A, X, g
        )
        K = int(col_norms.shape[0])
        if K != self.K:
            raise ValueError(f"LISTALayer was built for K={self.K} but got K={K}")

        group_tensor = None
        if group_ids is not None:
            group_tensor = torch.as_tensor(
                group_ids, device=device, dtype=torch.long
            ).view(-1)
        elif group_default is not None:
            try:
                group_tensor = torch.as_tensor(
                    group_default, device=device, dtype=torch.long
                ).view(-1)
            except Exception:
                group_tensor = None
        theta_vec = self._theta_for_groups(group_tensor, device=device)
        group_tensor = (
            _map_groups_to_indices(group_tensor, K, device=device)
            if group_tensor is not None
            else None
        )

        lambda_vec: Optional[torch.Tensor] = None
        if torch.is_tensor(lambda_group):
            lambda_vec = lambda_group.to(device=device, dtype=dtype).view(-1)
            if lambda_vec.numel() == 1:
                lambda_vec = lambda_vec.expand(K)
            elif lambda_vec.numel() != K:
                raise ValueError(
                    f"lambda_group tensor has shape {tuple(lambda_vec.shape)}, expected ({K},)"
                )
        elif lambda_group > 0.0:
            lambda_vec = torch.full((K,), float(lambda_group), device=device, dtype=dtype)

        w = torch.zeros(K, device=device, dtype=dtype)
        for _ in range(self.n_steps):
            grad = rmatvec(matvec(w) - g)
            if self.use_dense and self.S_dense is not None and self.B_dense is not None:
                w = self.soft_threshold(
                    self.S_dense @ w - self.B_dense @ grad,
                    theta_vec,
                )
            else:
                update = self.S_diag * w - self.B_diag * grad  # type: ignore[attr-defined]
                if self.U is not None and self.V is not None:
                    update = update + self.U @ (self.V.T @ w)
                w = self.soft_threshold(update, theta_vec)
            if group_tensor is not None and lambda_vec is not None:
                w = _group_prox(w, group_tensor, lambda_vec)
            elif group_tensor is not None and lambda_group > 0.0:
                w = _group_prox(w, group_tensor, lambda_group)

        w_phys = w * inv_norms
        return w_phys

    def solve(
        self,
        A_operator: torch.Tensor | BasisOperator,
        b: torch.Tensor,
        *,
        X: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
        lambda_group: float | torch.Tensor = 0.0,
        reg_l1: Optional[float] = None,
    ) -> torch.Tensor:
        """Wrapper matching ECO contract for LISTA solves."""
        # reg_l1 included for API compatibility; thresholds already learned.
        return self.forward(
            A_operator,
            X,
            b,
            group_ids=group_ids,
            lambda_group=lambda_group,
        )


def _infer_lista_dim(state_dict: dict[str, torch.Tensor], cfg: Optional[dict[str, Any]]) -> Optional[int]:
    """Infer LISTA layer width from a checkpoint payload."""
    if isinstance(cfg, dict):
        meta = cfg.get("lista_meta") if isinstance(cfg.get("lista_meta"), dict) else None
        if isinstance(meta, dict):
            try:
                k_meta = int(meta.get("K", 0))
                if k_meta > 0:
                    return k_meta
            except Exception:
                pass
    if isinstance(cfg, dict):
        try:
            k_cfg = int(cfg.get("lista_K", 0))
            if k_cfg > 0:
                return k_cfg
        except Exception:
            pass
        try:
            n_static = int(cfg.get("n_candidates_static", 0))
            n_learned = int(cfg.get("n_candidates_learned", 0))
            if n_static + n_learned > 0:
                return n_static + n_learned
        except Exception:
            pass

    for key in ("S_dense", "B_dense", "theta_elem", "S_diag", "B_diag", "theta_group", "U", "V"):
        tensor = state_dict.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1:
            return int(tensor.shape[0])
    return None


def _infer_lista_rank(state_dict: dict[str, torch.Tensor]) -> int:
    """Infer low-rank correction dimension from checkpoint payload."""
    U = state_dict.get("U")
    if isinstance(U, torch.Tensor) and U.ndim == 2:
        return int(U.shape[1])
    return 0


def load_lista_from_checkpoint(
    checkpoint: str | Path,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Optional[LISTALayer]:
    """
    Best-effort LISTA loader that accepts module or state-dict checkpoints.

    Returns None on failure so callers can gracefully fall back to ISTA.
    """
    path = Path(checkpoint)
    map_location = device if device is not None else "cpu"
    try:
        payload = torch.load(path, map_location=map_location)
    except Exception:
        return None

    cfg = payload.get("config") if isinstance(payload, dict) else None
    meta = payload.get("lista_meta") if isinstance(payload, dict) else None

    lista_obj: Optional[LISTALayer] = None
    state_dict: Optional[dict[str, torch.Tensor]] = None

    if isinstance(payload, LISTALayer):
        lista_obj = payload
    elif isinstance(payload, dict):
        maybe_lista = payload.get("lista")
        if isinstance(maybe_lista, LISTALayer):
            lista_obj = maybe_lista
        elif isinstance(maybe_lista, dict):
            state_dict = {k: v for k, v in maybe_lista.items() if isinstance(v, torch.Tensor)}
        elif all(isinstance(v, torch.Tensor) for v in payload.values()):
            state_dict = {k: v for k, v in payload.items()}

    if lista_obj is None and state_dict is None:
        return None

    if lista_obj is None and state_dict is not None:
        K = _infer_lista_dim(state_dict, cfg)
        if K is None and isinstance(meta, dict):
            try:
                k_meta = int(meta.get("K", 0))
                if k_meta > 0:
                    K = k_meta
            except Exception:
                pass
        if K is None or K <= 0:
            return None
        steps = int(cfg.get("lista_steps", 10)) if isinstance(cfg, dict) else 10
        dense_threshold = int(cfg.get("lista_dense_threshold", max(512, K))) if isinstance(cfg, dict) else max(512, K)
        rank = _infer_lista_rank(state_dict)
        try:
            lista_obj = LISTALayer(
                K=K,
                n_steps=steps,
                rank=rank,
                dense_threshold=dense_threshold,
            )
            lista_obj.load_state_dict(state_dict, strict=False)
        except Exception:
            return None

    if lista_obj is None:
        return None

    if device is not None or dtype is not None:
        try:
            lista_obj = lista_obj.to(device=device if device is not None else None, dtype=dtype)  # type: ignore[assignment]
        except Exception:
            try:
                lista_obj = lista_obj.to(device=device if device is not None else None)  # type: ignore[assignment]
            except Exception:
                pass

    try:
        lista_obj.eval()
    except Exception:
        pass

    return lista_obj
