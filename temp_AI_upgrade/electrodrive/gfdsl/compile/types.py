"""Core compile-time types for GFDSL lowering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import torch

from electrodrive.gfdsl.ast.constraints import GroupInfo


def _default_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _default_dtype() -> torch.dtype:
    return torch.float32


@dataclass
class CoeffSlot:
    """A coefficient slot emitted by lowering."""

    slot_id: str
    dim: int = 1
    regularizer: Dict[str, Any] = field(default_factory=dict)
    group_info: Optional[GroupInfo] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompileContext:
    """Context passed through validation and lowering."""

    spec: Optional[Any] = None
    device: torch.device = field(default_factory=_default_device)
    dtype: torch.dtype = field(default_factory=_default_dtype)
    eval_backend: Literal["dense", "operator", "hybrid"] = "dense"
    cache: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    def clone_with(self, **updates: Any) -> "CompileContext":
        base = {
            "spec": self.spec,
            "device": self.device,
            "dtype": self.dtype,
            "eval_backend": self.eval_backend,
            "cache": self.cache,
            "extras": dict(self.extras),
        }
        base.update(updates)
        return CompileContext(**base)

