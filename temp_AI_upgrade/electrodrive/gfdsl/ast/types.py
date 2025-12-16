"""
Parameter and transform definitions for GFDSL.

These types are intentionally lightweight and focused on serialization and
canonicalization; evaluation-time behavior will be added in later milestones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional, Tuple, Type

import torch
import torch.nn.functional as F


def _tensor_from_raw(raw: Any) -> torch.Tensor:
    """Convert incoming raw data to a tensor without forcing device or dtype."""
    if isinstance(raw, torch.Tensor):
        tensor = raw.detach().clone()
    else:
        tensor = torch.as_tensor(raw, dtype=torch.float32)
    return tensor


def _tensor_to_serializable(raw: torch.Tensor) -> Any:
    """Convert tensor to a JSON-serializable python object on CPU/float."""
    cpu_raw = raw.detach().float().cpu()
    if cpu_raw.numel() == 1:
        return cpu_raw.item()
    return cpu_raw.tolist()


@dataclass
class ParamTransform:
    """Base class for parameter transforms."""

    name: ClassVar[str] = "base"

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def to_json_dict(self) -> Dict[str, Any]:
        return {"type": self.name}

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "ParamTransform":
        transform_type = data.get("type")
        if transform_type is None:
            raise ValueError("ParamTransform JSON missing 'type'")
        if transform_type not in _TRANSFORM_REGISTRY:
            raise ValueError(f"Unknown ParamTransform type '{transform_type}'")
        transform_cls = _TRANSFORM_REGISTRY[transform_type]
        return transform_cls.from_json_dict(data)

    def canonical_config(self) -> Dict[str, Any]:
        """Return deterministic config used by canonicalization."""
        return self.to_json_dict()


@dataclass
class IdentityTransform(ParamTransform):
    name: ClassVar[str] = "identity"

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return raw

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        return value

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "IdentityTransform":
        return cls()


@dataclass
class SoftplusTransform(ParamTransform):
    name: ClassVar[str] = "softplus"
    min: float = 0.0

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw) + float(self.min)

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        shifted = value - float(self.min)
        return torch.log(torch.expm1(shifted))

    def to_json_dict(self) -> Dict[str, Any]:
        return {"type": self.name, "min": self.min}

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "SoftplusTransform":
        return cls(min=float(data.get("min", 0.0)))


@dataclass
class SigmoidRangeTransform(ParamTransform):
    name: ClassVar[str] = "sigmoid_range"
    lo: float = 0.0
    hi: float = 1.0

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        span = float(self.hi - self.lo)
        return float(self.lo) + span * torch.sigmoid(raw)

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        span = float(self.hi - self.lo)
        scaled = (value - float(self.lo)) / span
        scaled = torch.clamp(scaled, 1e-8, 1 - 1e-8)
        return torch.log(scaled / (1 - scaled))

    def to_json_dict(self) -> Dict[str, Any]:
        return {"type": self.name, "lo": self.lo, "hi": self.hi}

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "SigmoidRangeTransform":
        return cls(lo=float(data.get("lo", 0.0)), hi=float(data.get("hi", 1.0)))


@dataclass
class TanhRangeTransform(ParamTransform):
    name: ClassVar[str] = "tanh_range"
    lo: float = -1.0
    hi: float = 1.0

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        span = float(self.hi - self.lo) / 2.0
        mid = float(self.lo + self.hi) / 2.0
        return mid + span * torch.tanh(raw)

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        span = float(self.hi - self.lo) / 2.0
        mid = float(self.lo + self.hi) / 2.0
        shifted = (value - mid) / span
        shifted = torch.clamp(shifted, -1 + 1e-6, 1 - 1e-6)
        return 0.5 * torch.log((1 + shifted) / (1 - shifted))

    def to_json_dict(self) -> Dict[str, Any]:
        return {"type": self.name, "lo": self.lo, "hi": self.hi}

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "TanhRangeTransform":
        return cls(lo=float(data.get("lo", -1.0)), hi=float(data.get("hi", 1.0)))


@dataclass
class IntegerSoftRoundTransform(ParamTransform):
    name: ClassVar[str] = "integer_soft_round"
    min_value: int = 0
    max_value: int = 100

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        rounded = torch.round(raw)
        soft = raw + (rounded - raw).detach()
        return torch.clamp(soft, float(self.min_value), float(self.max_value))

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "type": self.name,
            "min_value": int(self.min_value),
            "max_value": int(self.max_value),
        }

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "IntegerSoftRoundTransform":
        return cls(
            min_value=int(data.get("min_value", 0)),
            max_value=int(data.get("max_value", 100)),
        )

    def canonical_config(self) -> Dict[str, Any]:
        return {
            "type": self.name,
            "min_value": int(self.min_value),
            "max_value": int(self.max_value),
        }


@dataclass
class ComplexPairTransform(ParamTransform):
    name: ClassVar[str] = "complex_pair"
    imag_min: float = 0.0

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.shape[-1] < 2:
            raise ValueError("ComplexPairTransform expects last dim >= 2 for (a,b_raw)")
        a = raw[..., 0]
        b_raw = raw[..., 1]
        b = F.softplus(b_raw) + float(self.imag_min)
        return torch.stack((a, b), dim=-1)

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] < 2:
            raise ValueError("ComplexPairTransform inverse expects last dim >= 2")
        a = value[..., 0]
        b = value[..., 1] - float(self.imag_min)
        b = torch.log(torch.expm1(b))
        return torch.stack((a, b), dim=-1)

    def to_json_dict(self) -> Dict[str, Any]:
        return {"type": self.name, "imag_min": self.imag_min}

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "ComplexPairTransform":
        return cls(imag_min=float(data.get("imag_min", 0.0)))


_TRANSFORM_REGISTRY: Dict[str, Type[ParamTransform]] = {
    IdentityTransform.name: IdentityTransform,
    SoftplusTransform.name: SoftplusTransform,
    SigmoidRangeTransform.name: SigmoidRangeTransform,
    TanhRangeTransform.name: TanhRangeTransform,
    IntegerSoftRoundTransform.name: IntegerSoftRoundTransform,
    ComplexPairTransform.name: ComplexPairTransform,
}


@dataclass
class Param:
    """A parameter with an associated transform."""

    raw: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    transform: ParamTransform = field(default_factory=IdentityTransform)
    trainable: bool = True
    dtype_policy: str = "work"
    bounds_hint: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        self.raw = _tensor_from_raw(self.raw)
        self.raw.requires_grad_(bool(self.trainable))

    def raw_as(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        if device is None and dtype is None:
            return self.raw
        return self.raw.to(device=device, dtype=dtype)

    def value(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        requested_dtype = dtype if dtype is not None else self.raw.dtype
        base = self.raw_as(device=device, dtype=dtype)
        work = base
        if work.dtype in (torch.float16, torch.bfloat16):
            work = work.float()
        out = self.transform.forward(work)
        if requested_dtype is not None and requested_dtype != out.dtype:
            out = out.to(dtype=requested_dtype)
        return out

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "raw": _tensor_to_serializable(self.raw),
            "transform": self.transform.to_json_dict(),
            "trainable": bool(self.trainable),
            "dtype_policy": self.dtype_policy,
            "bounds_hint": self.bounds_hint,
        }

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "Param":
        if "transform" not in data:
            raise ValueError("Param JSON missing 'transform'")
        transform = ParamTransform.from_json_dict(data["transform"])
        raw = _tensor_from_raw(data.get("raw", 0.0))
        trainable = bool(data.get("trainable", True))
        dtype_policy = data.get("dtype_policy", "work")
        bounds_hint = data.get("bounds_hint")
        return cls(
            raw=raw,
            transform=transform,
            trainable=trainable,
            dtype_policy=dtype_policy,
            bounds_hint=tuple(bounds_hint) if bounds_hint is not None else None,
        )

    def canonical_dict(self, include_raw: bool = False, quantization: float = 1e-12) -> Dict[str, Any]:
        base = {
            "transform": self.transform.canonical_config(),
            "trainable": bool(self.trainable),
            "dtype_policy": self.dtype_policy,
            "bounds_hint": self.bounds_hint,
            "shape": list(self.raw.shape),
        }
        if include_raw:
            quantized = torch.round(self.raw.detach().float().cpu() / quantization) * quantization
            base["raw"] = _tensor_to_serializable(quantized)
        return base
