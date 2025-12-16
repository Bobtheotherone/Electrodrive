"""Lower GFDSL AST nodes into linear evaluation objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch

from electrodrive.gfdsl.compile.types import CoeffSlot, CompileContext


@dataclass
class LinearContribution:
    """Represents a linear contribution produced by lowering a node."""

    slots: List[CoeffSlot]
    evaluator: "ColumnEvaluator"
    fixed_term: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


class ColumnEvaluator:
    """Interface for evaluating basis columns and applying linear operators."""

    K: int

    def eval_columns(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def matvec(self, w: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def rmatvec(self, r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def estimate_col_norms(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DenseEvaluator(ColumnEvaluator):
    """Evaluator backed by an explicit dense column builder."""

    def __init__(self, K: int, dense_fn: Callable[[torch.Tensor], torch.Tensor]):
        self.K = int(K)
        self._dense_fn = dense_fn

    def eval_columns(self, X: torch.Tensor) -> torch.Tensor:
        Phi = self._dense_fn(X)
        if Phi.dim() != 2 or Phi.shape[1] != self.K:
            raise ValueError(f"DenseEvaluator expected output with K={self.K}, got {Phi.shape}")
        return Phi

    def matvec(self, w: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        Phi = self.eval_columns(X)
        w = w.to(device=Phi.device, dtype=Phi.dtype).reshape(-1)
        return Phi @ w

    def rmatvec(self, r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        Phi = self.eval_columns(X)
        r = r.to(device=Phi.device, dtype=Phi.dtype).reshape(-1)
        return Phi.transpose(0, 1) @ r

    def estimate_col_norms(self, X: torch.Tensor) -> torch.Tensor:
        Phi = self.eval_columns(X)
        return torch.sqrt(torch.sum(Phi * Phi, dim=0))


class OperatorEvaluator(ColumnEvaluator):
    """Evaluator that provides matvec/rmatvec without requiring Phi materialization."""

    def __init__(
        self,
        K: int,
        matvec_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        rmatvec_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        dense_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.K = int(K)
        self._matvec_fn = matvec_fn
        self._rmatvec_fn = rmatvec_fn
        self._dense_fn = dense_fn

    def eval_columns(self, X: torch.Tensor) -> torch.Tensor:
        if self._dense_fn is not None:
            return self._dense_fn(X)
        # Fallback reconstruction for debug/tests
        eye = torch.eye(self.K, device=X.device, dtype=X.dtype)
        cols = [self._matvec_fn(eye[:, k], X) for k in range(self.K)]
        return torch.stack(cols, dim=1)

    def matvec(self, w: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        w = w.to(device=X.device, dtype=X.dtype).reshape(-1)
        return self._matvec_fn(w, X)

    def rmatvec(self, r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        r = r.to(device=X.device, dtype=X.dtype).reshape(-1)
        return self._rmatvec_fn(r, X)

    def estimate_col_norms(self, X: torch.Tensor) -> torch.Tensor:
        Phi = self.eval_columns(X)
        return torch.sqrt(torch.sum(Phi * Phi, dim=0))


def lower_program(root: Any, ctx: CompileContext) -> LinearContribution:
    """Entry point to lower a GFDSL AST."""
    if not hasattr(root, "lower"):
        raise TypeError(f"Object of type {type(root)} cannot be lowered")
    return root.lower(ctx)
