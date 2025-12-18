from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import torch

from .utils import (
    device_matches,
    dtype_from_str,
    normalize_dtype,
    require_cuda,
)


def _coerce_enum(enum_cls: type[Enum], value: Any) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        for member in enum_cls:  # type: ignore[assignment]
            if value == member.value:
                return member
            if str(value).lower() == str(member.value).lower():
                return member
    raise ValueError(f"Invalid {enum_cls.__name__}: {value!r}")


class OracleQuantity(str, Enum):
    POTENTIAL = "potential"
    FIELD = "field"
    BOTH = "both"


class OracleFidelity(str, Enum):
    F0 = "F0"
    F1 = "F1"
    F2 = "F2"
    F3 = "F3"
    AUTO = "auto"


class CachePolicy(str, Enum):
    USE_CACHE = "use_cache"
    REFRESH = "refresh"
    WRITE_ONLY = "write_only"
    OFF = "off"


class TraceLevel(str, Enum):
    NONE = "none"
    MINIMAL = "minimal"
    FULL = "full"


class ErrorEstimateType(str, Enum):
    NONE = "none"
    HEURISTIC = "heuristic"
    A_POSTERIORI = "a_posteriori"
    BOUND = "bound"


class CacheStatus(str, Enum):
    HIT = "hit"
    MISS = "miss"


@dataclass(frozen=True)
class OracleErrorEstimate:
    type: ErrorEstimateType = ErrorEstimateType.NONE
    metrics: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", _coerce_enum(ErrorEstimateType, self.type))

    def to_json(self) -> Dict[str, object]:
        return {
            "type": self.type.value,
            "metrics": {str(k): float(v) for k, v in self.metrics.items()},
            "confidence": float(self.confidence),
            "notes": [str(x) for x in self.notes],
        }

    @staticmethod
    def from_json(d: Dict[str, object]) -> "OracleErrorEstimate":
        return OracleErrorEstimate(
            type=_coerce_enum(ErrorEstimateType, d.get("type", ErrorEstimateType.NONE)),
            metrics={str(k): float(v) for k, v in dict(d.get("metrics", {})).items()},
            confidence=float(d.get("confidence", 0.0)),
            notes=[str(x) for x in d.get("notes", [])],
        )


@dataclass(frozen=True)
class OracleCost:
    wall_ms: float
    cuda_ms: float
    peak_vram_mb: float

    def to_json(self) -> Dict[str, object]:
        return {
            "wall_ms": float(self.wall_ms),
            "cuda_ms": float(self.cuda_ms),
            "peak_vram_mb": float(self.peak_vram_mb),
        }

    @staticmethod
    def from_json(d: Dict[str, object]) -> "OracleCost":
        return OracleCost(
            wall_ms=float(d.get("wall_ms", 0.0)),
            cuda_ms=float(d.get("cuda_ms", 0.0)),
            peak_vram_mb=float(d.get("peak_vram_mb", 0.0)),
        )


@dataclass(frozen=True)
class OracleCacheStatus:
    status: CacheStatus
    key: Optional[str] = None
    path: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _coerce_enum(CacheStatus, self.status))

    def to_json(self) -> Dict[str, object]:
        return {
            "status": self.status.value,
            "key": self.key,
            "path": self.path,
        }

    @staticmethod
    def from_json(d: Dict[str, object]) -> "OracleCacheStatus":
        return OracleCacheStatus(
            status=_coerce_enum(CacheStatus, d.get("status", CacheStatus.MISS)),
            key=d.get("key"),
            path=d.get("path"),
        )


@dataclass(frozen=True)
class OracleProvenance:
    git_sha: str
    torch_version: str
    cuda_version: str
    device_name: str
    device: str
    dtype: str
    timestamp: str

    def to_json(self) -> Dict[str, object]:
        return {
            "git_sha": self.git_sha,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "device_name": self.device_name,
            "device": self.device,
            "dtype": self.dtype,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_json(d: Dict[str, object]) -> "OracleProvenance":
        device = str(d.get("device", "") or "cuda")
        dtype = str(d.get("dtype", "") or "float32")
        return OracleProvenance(
            git_sha=str(d.get("git_sha", "")),
            torch_version=str(d.get("torch_version", "")),
            cuda_version=str(d.get("cuda_version", "")),
            device_name=str(d.get("device_name", "")),
            device=device,
            dtype=dtype,
            timestamp=str(d.get("timestamp", "")),
        )


@dataclass(frozen=True)
class OracleQuery:
    spec: Dict[str, object]
    points: torch.Tensor
    quantity: OracleQuantity
    requested_fidelity: OracleFidelity
    device: Optional[str] = None
    dtype: Optional[str] = None
    seed: int = 0
    budget: Dict[str, object] = field(default_factory=dict)
    cache_policy: CachePolicy = CachePolicy.USE_CACHE
    trace: TraceLevel = TraceLevel.NONE

    def __post_init__(self) -> None:
        if not isinstance(self.spec, dict):
            raise TypeError("spec must be a dict")
        if not torch.is_tensor(self.points):
            raise TypeError("points must be a torch.Tensor")
        require_cuda(self.points, "points")
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must have shape [N, 3]")
        if not isinstance(self.budget, dict):
            raise TypeError("budget must be a dict")

        quantity = _coerce_enum(OracleQuantity, self.quantity)
        fidelity = _coerce_enum(OracleFidelity, self.requested_fidelity)
        cache_policy = _coerce_enum(CachePolicy, self.cache_policy)
        trace = _coerce_enum(TraceLevel, self.trace)

        if self.device is not None and not device_matches(self.device, self.points.device):
            raise ValueError("device does not match points device")
        if self.dtype is not None:
            expected_dtype = dtype_from_str(normalize_dtype(self.dtype))
            if expected_dtype != self.points.dtype:
                raise ValueError("dtype does not match points dtype")

        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "requested_fidelity", fidelity)
        object.__setattr__(self, "cache_policy", cache_policy)
        object.__setattr__(self, "trace", trace)
        object.__setattr__(self, "device", str(self.points.device))
        object.__setattr__(self, "dtype", normalize_dtype(self.points.dtype))
        object.__setattr__(self, "seed", int(self.seed))

    def to_json(self) -> Dict[str, object]:
        require_cuda(self.points, "points")
        return {
            "spec": self.spec,
            "points": self.points.detach().contiguous().cpu().tolist(),
            "quantity": self.quantity.value,
            "requested_fidelity": self.requested_fidelity.value,
            "device": self.device,
            "dtype": self.dtype,
            "seed": int(self.seed),
            "budget": self.budget,
            "cache_policy": self.cache_policy.value,
            "trace": self.trace.value,
        }

    @staticmethod
    def from_json(d: Dict[str, object]) -> "OracleQuery":
        device = str(d.get("device", "cuda"))
        dtype = dtype_from_str(str(d.get("dtype", "float32")))
        points = torch.tensor(d.get("points", []), device=device, dtype=dtype)
        return OracleQuery(
            spec=dict(d.get("spec", {})),
            points=points,
            quantity=_coerce_enum(OracleQuantity, d.get("quantity", OracleQuantity.BOTH)),
            requested_fidelity=_coerce_enum(OracleFidelity, d.get("requested_fidelity", OracleFidelity.AUTO)),
            device=device,
            dtype=normalize_dtype(dtype),
            seed=int(d.get("seed", 0)),
            budget=dict(d.get("budget", {})),
            cache_policy=_coerce_enum(CachePolicy, d.get("cache_policy", CachePolicy.USE_CACHE)),
            trace=_coerce_enum(TraceLevel, d.get("trace", TraceLevel.NONE)),
        )


@dataclass(frozen=True)
class OracleResult:
    V: Optional[torch.Tensor]
    E: Optional[torch.Tensor]
    valid_mask: torch.Tensor
    method: str
    fidelity: OracleFidelity
    config_fingerprint: str
    error_estimate: OracleErrorEstimate
    cost: OracleCost
    cache: OracleCacheStatus
    provenance: OracleProvenance

    def __post_init__(self) -> None:
        if self.V is None and self.E is None:
            raise ValueError("OracleResult must include V, E, or both")
        if not torch.is_tensor(self.valid_mask):
            raise TypeError("valid_mask must be a torch.Tensor")
        require_cuda(self.valid_mask, "valid_mask")
        if self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be a bool tensor")
        if self.valid_mask.ndim != 1:
            raise ValueError("valid_mask must be a 1D tensor")

        n = int(self.valid_mask.shape[0])
        if self.V is not None:
            if not torch.is_tensor(self.V):
                raise TypeError("V must be a torch.Tensor")
            require_cuda(self.V, "V")
            if self.V.ndim != 1 or int(self.V.shape[0]) != n:
                raise ValueError("V must have shape [N]")
        if self.E is not None:
            if not torch.is_tensor(self.E):
                raise TypeError("E must be a torch.Tensor")
            require_cuda(self.E, "E")
            if self.E.ndim != 2 or self.E.shape[1] != 3 or int(self.E.shape[0]) != n:
                raise ValueError("E must have shape [N, 3]")

        if not isinstance(self.error_estimate, OracleErrorEstimate):
            raise TypeError("error_estimate must be OracleErrorEstimate")
        if not isinstance(self.cost, OracleCost):
            raise TypeError("cost must be OracleCost")
        if not isinstance(self.cache, OracleCacheStatus):
            raise TypeError("cache must be OracleCacheStatus")
        if not isinstance(self.provenance, OracleProvenance):
            raise TypeError("provenance must be OracleProvenance")

        fidelity = _coerce_enum(OracleFidelity, self.fidelity)
        object.__setattr__(self, "fidelity", fidelity)

        device_ref = self.valid_mask.device
        if not device_matches(self.provenance.device, device_ref):
            raise ValueError("provenance.device does not match result tensors")
        if self.V is not None and self.V.device != device_ref:
            raise ValueError("V device does not match valid_mask")
        if self.E is not None and self.E.device != device_ref:
            raise ValueError("E device does not match valid_mask")

        dtype_ref = None
        if self.V is not None:
            dtype_ref = self.V.dtype
        elif self.E is not None:
            dtype_ref = self.E.dtype
        if dtype_ref is not None:
            expected_dtype = dtype_from_str(normalize_dtype(self.provenance.dtype))
            if expected_dtype != dtype_ref:
                raise ValueError("provenance.dtype does not match result tensors")

    def to_json(self) -> Dict[str, object]:
        if self.V is not None:
            require_cuda(self.V, "V")
        if self.E is not None:
            require_cuda(self.E, "E")
        require_cuda(self.valid_mask, "valid_mask")
        return {
            "V": None if self.V is None else self.V.detach().contiguous().cpu().tolist(),
            "E": None if self.E is None else self.E.detach().contiguous().cpu().tolist(),
            "valid_mask": self.valid_mask.detach().contiguous().cpu().tolist(),
            "method": self.method,
            "fidelity": self.fidelity.value,
            "config_fingerprint": self.config_fingerprint,
            "error_estimate": self.error_estimate.to_json(),
            "cost": self.cost.to_json(),
            "cache": self.cache.to_json(),
            "provenance": self.provenance.to_json(),
        }

    @staticmethod
    def from_json(d: Dict[str, object]) -> "OracleResult":
        prov = OracleProvenance.from_json(dict(d.get("provenance", {})))
        device = prov.device or "cuda"
        dtype = dtype_from_str(prov.dtype or "float32")
        V = d.get("V", None)
        V_t = None if V is None else torch.tensor(V, device=device, dtype=dtype)
        E = d.get("E", None)
        E_t = None if E is None else torch.tensor(E, device=device, dtype=dtype)
        valid_mask = torch.tensor(d.get("valid_mask", []), device=device, dtype=torch.bool)
        return OracleResult(
            V=V_t,
            E=E_t,
            valid_mask=valid_mask,
            method=str(d.get("method", "")),
            fidelity=_coerce_enum(OracleFidelity, d.get("fidelity", OracleFidelity.F0)),
            config_fingerprint=str(d.get("config_fingerprint", "")),
            error_estimate=OracleErrorEstimate.from_json(dict(d.get("error_estimate", {}))),
            cost=OracleCost.from_json(dict(d.get("cost", {}))),
            cache=OracleCacheStatus.from_json(dict(d.get("cache", {}))),
            provenance=prov,
        )
