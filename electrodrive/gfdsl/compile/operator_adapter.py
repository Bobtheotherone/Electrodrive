"""Operator adapter for GFDSL linear contributions."""

from __future__ import annotations

from typing import Optional

import torch

from electrodrive.images.basis import BasisOperator
from electrodrive.gfdsl.compile.lower import LinearContribution
from electrodrive.gfdsl.compile.legacy_adapter import linear_contribution_to_legacy_basis


class GFDSLOperator(BasisOperator):
    """BasisOperator backed by a GFDSL ColumnEvaluator for fast matvec/rmatvec."""

    def __init__(
        self,
        contrib: LinearContribution,
        points: Optional[torch.Tensor] = None,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        row_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self._contrib = contrib
        self._evaluator = contrib.evaluator
        self._fixed_term = contrib.fixed_term
        elements = linear_contribution_to_legacy_basis(contrib)
        super().__init__(elements, points=points, device=device, dtype=dtype, row_weights=row_weights)

    @property
    def fixed_term(self):
        return self._fixed_term

    @property
    def evaluator(self):
        return self._evaluator

    def matvec(self, w: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        X = self._targets(targets)
        w = w.to(self.device, dtype=self.dtype).reshape(-1)
        if w.numel() != self._evaluator.K:
            raise ValueError(f"w must have shape [{self._evaluator.K}], got {tuple(w.shape)}")
        out = self._evaluator.matvec(w, X)
        if self._sqrt_row_weights is not None:
            out = self._sqrt_row_weights * out
        return out

    def rmatvec(self, r: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        X = self._targets(targets)
        r = r.to(self.device, dtype=self.dtype).reshape(-1)
        if r.numel() != X.shape[0]:
            raise ValueError(f"r must have shape [{X.shape[0]}], got {tuple(r.shape)}")
        if self._sqrt_row_weights is not None:
            r = r * self._sqrt_row_weights
        return self._evaluator.rmatvec(r, X)

    def estimate_col_norms(self, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        X = self._targets(targets)
        if self._sqrt_row_weights is None:
            norms = self._evaluator.estimate_col_norms(X)
        else:
            Phi = self._evaluator.eval_columns(X)
            Phi = self._sqrt_row_weights.view(-1, 1) * Phi
            norms = torch.sqrt(torch.sum(Phi * Phi, dim=0))
        self.col_norms = norms
        self._inv_col_norms = 1.0 / norms.clamp_min(1e-6)
        return norms

    def to_dense(
        self,
        max_entries: Optional[int] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        X = self._targets(targets)
        Phi = self._evaluator.eval_columns(X)
        if max_entries is not None and Phi.numel() > max_entries:
            return None
        if self._sqrt_row_weights is not None:
            Phi = self._sqrt_row_weights.view(-1, 1) * Phi
        return Phi


def linear_contribution_to_operator(
    contrib: LinearContribution,
    points: Optional[torch.Tensor] = None,
    *,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
    row_weights: Optional[torch.Tensor] = None,
) -> GFDSLOperator:
    """Convenience wrapper to build a GFDSLOperator."""
    return GFDSLOperator(
        contrib,
        points,
        device=device,
        dtype=dtype,
        row_weights=row_weights,
    )
