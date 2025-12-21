"""Parameter schemas and GPU transforms for flow-based sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

import torch

from electrodrive.flows.device_guard import ensure_cuda


def _get_field(spec: Any, name: str, default: Any = None) -> Any:
    if spec is None:
        return default
    if isinstance(spec, dict):
        return spec.get(name, default)
    return getattr(spec, name, default)


def _coerce_vec3(value: Any) -> Optional[Sequence[float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return [float(v) for v in value]
        except Exception:
            return None
    return None


def _coerce_bbox(value: Any) -> Optional[Sequence[Sequence[float]]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    lo = _coerce_vec3(value[0])
    hi = _coerce_vec3(value[1])
    if lo is None or hi is None:
        return None
    return [lo, hi]


def _select_slab_layer(dielectrics: Iterable[dict]) -> Optional[dict]:
    slab = None
    best_thickness = None
    for layer in dielectrics:
        if not isinstance(layer, dict):
            continue
        z_min = layer.get("z_min")
        z_max = layer.get("z_max")
        if z_min is None or z_max is None:
            continue
        try:
            z_min_f = float(z_min)
            z_max_f = float(z_max)
        except Exception:
            continue
        thickness = abs(z_max_f - z_min_f)
        if thickness <= 0:
            continue
        if layer.get("name") == "slab":
            return {"z_min": z_min_f, "z_max": z_max_f}
        if best_thickness is None or thickness < best_thickness:
            best_thickness = thickness
            slab = {"z_min": z_min_f, "z_max": z_max_f}
    return slab


@dataclass(frozen=True)
class CanonicalSpecView:
    """Lightweight CanonicalSpec adapter for schema transforms."""

    spec: Any

    def worldbox(self) -> Optional[Sequence[Sequence[float]]]:
        domain = _get_field(self.spec, "domain")
        bbox = None
        if isinstance(domain, dict):
            bbox = _coerce_bbox(domain.get("bbox"))
        if bbox is None and isinstance(self.spec, dict):
            meta = self.spec.get("domain_meta")
            if isinstance(meta, dict):
                bbox = _coerce_bbox(meta.get("bbox"))
        return bbox

    def slab_bounds(self) -> Optional[tuple[float, float]]:
        dielectrics = _get_field(self.spec, "dielectrics", []) or []
        slab = _select_slab_layer(dielectrics)
        if not slab:
            return None
        z_min = float(slab["z_min"])
        z_max = float(slab["z_max"])
        if z_min <= z_max:
            return (z_min, z_max)
        return (z_max, z_min)

    def axis_dir(self) -> Sequence[float]:
        symbols = _get_field(self.spec, "symbols", {}) or {}
        axis = None
        if isinstance(symbols, dict):
            axis = symbols.get("axis_dir") or symbols.get("axis")
        axis = _coerce_vec3(axis)
        if axis is None:
            axis = [0.0, 0.0, 1.0]
        norm = (axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2) ** 0.5
        if norm <= 1e-12:
            return [0.0, 0.0, 1.0]
        return [axis[0] / norm, axis[1] / norm, axis[2] / norm]

    def axis_origin(self) -> Sequence[float]:
        conductors = _get_field(self.spec, "conductors", []) or []
        if conductors:
            center = _coerce_vec3(conductors[0].get("center")) if isinstance(conductors[0], dict) else None
            if center is not None:
                return center
        return [0.0, 0.0, 0.0]


@dataclass(frozen=True)
class ParamSchema:
    """Base class for parameter schema transforms."""

    name: str
    schema_id: int
    p_dim: int
    active_dims: Sequence[int]

    def transform(self, u: torch.Tensor, spec: Any, node_ctx: Optional[dict] = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def inverse(self, physical: Dict[str, torch.Tensor], spec: Any, node_ctx: Optional[dict] = None) -> torch.Tensor:
        raise NotImplementedError

    def dim_mask(self, p_dim: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(p_dim, device=device, dtype=torch.bool)
        for idx in self.active_dims:
            if idx < p_dim:
                mask[idx] = True
        return mask


class ParamSchemaRegistry:
    def __init__(self) -> None:
        self._by_id: dict[int, ParamSchema] = {}
        self._by_name: dict[str, ParamSchema] = {}

    def register(self, schema: ParamSchema) -> None:
        self._by_id[int(schema.schema_id)] = schema
        self._by_name[schema.name] = schema

    def get_by_id(self, schema_id: int) -> Optional[ParamSchema]:
        return self._by_id.get(int(schema_id))

    def get_by_name(self, name: str) -> Optional[ParamSchema]:
        return self._by_name.get(name)

    def all(self) -> Sequence[ParamSchema]:
        return list(self._by_id.values())


def _safe_logit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


def _bounds_from_worldbox(view: CanonicalSpecView, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    bbox = view.worldbox()
    if bbox is None:
        lo = torch.tensor([-1.0, -1.0, -1.0], device=device, dtype=dtype)
        hi = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
        return lo, hi
    lo = torch.tensor(bbox[0], device=device, dtype=dtype)
    hi = torch.tensor(bbox[1], device=device, dtype=dtype)
    return lo, hi


def _axis_bounds(view: CanonicalSpecView) -> tuple[float, float]:
    bbox = view.worldbox()
    if bbox is None:
        return (-1.0, 1.0)
    lo, hi = bbox
    axis = view.axis_dir()
    origin = view.axis_origin()
    corners = [
        [lo[0], lo[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [lo[0], hi[1], lo[2]],
        [lo[0], hi[1], hi[2]],
        [hi[0], lo[1], lo[2]],
        [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], lo[2]],
        [hi[0], hi[1], hi[2]],
    ]
    projections = []
    for corner in corners:
        rel = [corner[i] - origin[i] for i in range(3)]
        projections.append(rel[0] * axis[0] + rel[1] * axis[1] + rel[2] * axis[2])
    return (min(projections), max(projections))


class RealPointSchema(ParamSchema):
    def transform(self, u: torch.Tensor, spec: Any, node_ctx: Optional[dict] = None) -> Dict[str, torch.Tensor]:
        _ = node_ctx
        device = ensure_cuda(u.device)
        u = u.to(device=device)
        view = CanonicalSpecView(spec)
        lo, hi = _bounds_from_worldbox(view, device, u.dtype)
        u_pos = u[..., :3]
        pos = lo + torch.sigmoid(u_pos) * (hi - lo)
        z_imag = torch.zeros(u.shape[:-1], device=device, dtype=u.dtype)
        return {
            "position": pos,
            "z_imag": z_imag,
            "schema_id": torch.tensor(self.schema_id, device=device, dtype=torch.long),
        }

    def inverse(self, physical: Dict[str, torch.Tensor], spec: Any, node_ctx: Optional[dict] = None) -> torch.Tensor:
        _ = node_ctx
        pos = physical["position"]
        device = ensure_cuda(pos.device)
        view = CanonicalSpecView(spec)
        lo, hi = _bounds_from_worldbox(view, device, pos.dtype)
        denom = (hi - lo).clamp(min=1e-6)
        u = _safe_logit((pos - lo) / denom)
        return u


class AxisPointSchema(ParamSchema):
    def transform(self, u: torch.Tensor, spec: Any, node_ctx: Optional[dict] = None) -> Dict[str, torch.Tensor]:
        _ = node_ctx
        device = ensure_cuda(u.device)
        u = u.to(device=device)
        view = CanonicalSpecView(spec)
        axis_dir = torch.tensor(view.axis_dir(), device=device, dtype=u.dtype)
        origin = torch.tensor(view.axis_origin(), device=device, dtype=u.dtype)
        s_min, s_max = _axis_bounds(view)
        s = u[..., 0]
        s_phys = s_min + torch.sigmoid(s) * (s_max - s_min)
        pos = origin + axis_dir * s_phys.unsqueeze(-1)
        z_imag = torch.zeros(u.shape[:-1], device=device, dtype=u.dtype)
        return {
            "position": pos,
            "z_imag": z_imag,
            "schema_id": torch.tensor(self.schema_id, device=device, dtype=torch.long),
        }

    def inverse(self, physical: Dict[str, torch.Tensor], spec: Any, node_ctx: Optional[dict] = None) -> torch.Tensor:
        _ = node_ctx
        pos = physical["position"]
        device = ensure_cuda(pos.device)
        view = CanonicalSpecView(spec)
        axis_dir = torch.tensor(view.axis_dir(), device=device, dtype=pos.dtype)
        origin = torch.tensor(view.axis_origin(), device=device, dtype=pos.dtype)
        s_min, s_max = _axis_bounds(view)
        s = (pos - origin) * axis_dir
        s = s.sum(dim=-1)
        denom = max(s_max - s_min, 1e-6)
        u = _safe_logit((s - s_min) / denom)
        return u.unsqueeze(-1)


class ComplexDepthPointSchema(ParamSchema):
    def transform(self, u: torch.Tensor, spec: Any, node_ctx: Optional[dict] = None) -> Dict[str, torch.Tensor]:
        device = ensure_cuda(u.device)
        u = u.to(device=device)
        view = CanonicalSpecView(spec)
        lo, hi = _bounds_from_worldbox(view, device, u.dtype)
        slab = view.slab_bounds()
        if slab is not None:
            z_min, z_max = slab
            z_lo = torch.tensor(z_min, device=device, dtype=u.dtype)
            z_hi = torch.tensor(z_max, device=device, dtype=u.dtype)
        else:
            z_lo = lo[2]
            z_hi = hi[2]
        u_pos = u[..., :3]
        pos = lo + torch.sigmoid(u_pos) * (hi - lo)
        z_raw = u_pos[..., 2]
        pos_z = z_lo + torch.sigmoid(z_raw) * (z_hi - z_lo)
        pos = torch.stack([pos[..., 0], pos[..., 1], pos_z], dim=-1)
        z_imag = torch.nn.functional.softplus(u[..., 3]) + 1e-6
        if node_ctx and node_ctx.get("conjugate"):
            z_imag = -z_imag
        return {
            "position": pos,
            "z_imag": z_imag,
            "schema_id": torch.tensor(self.schema_id, device=device, dtype=torch.long),
        }

    def inverse(self, physical: Dict[str, torch.Tensor], spec: Any, node_ctx: Optional[dict] = None) -> torch.Tensor:
        _ = node_ctx
        pos = physical["position"]
        z_imag = physical["z_imag"]
        device = ensure_cuda(pos.device)
        view = CanonicalSpecView(spec)
        lo, hi = _bounds_from_worldbox(view, device, pos.dtype)
        slab = view.slab_bounds()
        if slab is not None:
            z_min, z_max = slab
            z_lo = torch.tensor(z_min, device=device, dtype=pos.dtype)
            z_hi = torch.tensor(z_max, device=device, dtype=pos.dtype)
        else:
            z_lo = lo[2]
            z_hi = hi[2]
        denom = (hi - lo).clamp(min=1e-6)
        xy = _safe_logit((pos[..., :2] - lo[:2]) / denom[:2])
        z = _safe_logit((pos[..., 2] - z_lo) / (z_hi - z_lo).clamp(min=1e-6))
        u_z_imag = torch.log(torch.expm1(z_imag.clamp(min=1e-6)))
        return torch.cat([xy, z.unsqueeze(-1), u_z_imag.unsqueeze(-1)], dim=-1)


class PoleSchema(ParamSchema):
    def transform(self, u: torch.Tensor, spec: Any, node_ctx: Optional[dict] = None) -> Dict[str, torch.Tensor]:
        _ = spec, node_ctx
        device = ensure_cuda(u.device)
        u = u.to(device=device)
        k_pole = u[..., 0]
        residue = u[..., 1]
        return {
            "k_pole": k_pole,
            "residue": residue,
            "schema_id": torch.tensor(self.schema_id, device=device, dtype=torch.long),
        }

    def inverse(self, physical: Dict[str, torch.Tensor], spec: Any, node_ctx: Optional[dict] = None) -> torch.Tensor:
        _ = spec, node_ctx
        k_pole = physical["k_pole"]
        device = ensure_cuda(k_pole.device)
        residue = physical["residue"].to(device=device)
        return torch.stack([k_pole, residue], dim=-1)


SCHEMA_REAL_POINT = 1
SCHEMA_AXIS_POINT = 2
SCHEMA_COMPLEX_DEPTH = 3
SCHEMA_POLE = 4

REGISTRY = ParamSchemaRegistry()
REGISTRY.register(RealPointSchema("real_point", SCHEMA_REAL_POINT, 4, (0, 1, 2)))
REGISTRY.register(AxisPointSchema("axis_point", SCHEMA_AXIS_POINT, 4, (0,)))
REGISTRY.register(ComplexDepthPointSchema("complex_depth_point", SCHEMA_COMPLEX_DEPTH, 4, (0, 1, 2, 3)))
REGISTRY.register(PoleSchema("pole", SCHEMA_POLE, 4, (0, 1)))


def get_schema_by_id(schema_id: int) -> Optional[ParamSchema]:
    return REGISTRY.get_by_id(schema_id)


def get_schema_by_name(name: str) -> Optional[ParamSchema]:
    return REGISTRY.get_by_name(name)


__all__ = [
    "CanonicalSpecView",
    "ParamSchema",
    "ParamSchemaRegistry",
    "REGISTRY",
    "SCHEMA_REAL_POINT",
    "SCHEMA_AXIS_POINT",
    "SCHEMA_COMPLEX_DEPTH",
    "SCHEMA_POLE",
    "get_schema_by_id",
    "get_schema_by_name",
]
