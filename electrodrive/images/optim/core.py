from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol
import contextlib

import torch

from electrodrive.images.operator import BasisOperator


@dataclass
class ConstraintSpec:
    name: str
    kind: str
    weight: float = 1.0
    eps: float = 0.0
    region: Optional[Any] = None
    basis: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)


class ConstraintOp(Protocol):
    def apply(self, r: torch.Tensor) -> torch.Tensor:
        ...

    def adjoint(self, c: torch.Tensor) -> torch.Tensor:
        ...


@dataclass
class DTypePolicy:
    forward_dtype: torch.dtype = torch.float32
    kkt_dtype: torch.dtype = torch.float32
    certify_dtype: torch.dtype = torch.float64
    autocast: bool = True

    def __post_init__(self) -> None:
        if self.kkt_dtype in (torch.float16, torch.bfloat16):
            self.kkt_dtype = torch.float32
        if self.certify_dtype != torch.float64:
            self.certify_dtype = torch.float64

    def autocast_ctx(self, device: torch.device | str):
        if not self.autocast:
            return contextlib.nullcontext()
        dev = torch.device(device)
        if dev.type == "cuda":
            return torch.autocast("cuda", dtype=self.forward_dtype)
        if dev.type == "cpu":
            return torch.autocast("cpu", dtype=self.forward_dtype)
        return contextlib.nullcontext()


@dataclass
class SparseSolveRequest:
    A: torch.Tensor | BasisOperator
    X: Optional[torch.Tensor]
    g: torch.Tensor
    is_boundary: Optional[torch.Tensor]
    lambda_l1: float
    lambda_group: float | torch.Tensor
    group_ids: Optional[torch.Tensor]
    weight_prior: Optional[torch.Tensor]
    lambda_weight_prior: float | torch.Tensor
    normalize_columns: bool
    col_norms: Optional[torch.Tensor]
    constraints: list[ConstraintSpec] = field(default_factory=list)
    max_iter: int = 1000
    tol: float = 1e-6
    warm_start: Optional[torch.Tensor] = None
    return_stats: bool = False
    dtype_policy: Optional[DTypePolicy] = None


@dataclass
class SparseSolveResult:
    w: torch.Tensor
    support: torch.Tensor
    stats: dict[str, Any] = field(default_factory=dict)
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class ADMMConfig:
    rho: float = 1.0
    rho_growth: float = 1.0
    max_rho: float = 1e4
    max_iter: int = 200
    tol: float = 1e-4
    unroll_steps: int = 0
    w_update_iters: int = 25
    diff_mode: str = "unroll"
    verbose: bool = False
    track_residuals: bool = True
