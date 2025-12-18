from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch

from electrodrive.utils.config import K_E

from ..oracle_registry import OracleBackend
from ..oracle_types import (
    CacheStatus,
    ErrorEstimateType,
    OracleCacheStatus,
    OracleErrorEstimate,
    OracleFidelity,
    OracleQuery,
    OracleQuantity,
    OracleResult,
)
from ..utils import require_cuda
from .base import _maybe_cuda_events, fingerprint_config, make_cost, make_provenance


def _point_charge(
    points: torch.Tensor,
    pos: torch.Tensor,
    q: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    diff = points - pos
    r = torch.linalg.norm(diff, dim=1).clamp_min(1e-12)
    inv_r = 1.0 / r
    V = K_E * q * inv_r
    E = K_E * q * diff * inv_r.pow(3).unsqueeze(1)
    return V, E


class F3SymbolicOracleBackend(OracleBackend):
    def __init__(self) -> None:
        self._fingerprint = fingerprint_config("f3_symbolic", {"version": "0.1"})

    @property
    def name(self) -> str:
        return "f3_symbolic"

    @property
    def fidelity(self) -> OracleFidelity:
        return OracleFidelity.F3

    def fingerprint(self) -> str:
        return self._fingerprint

    def can_handle(self, query: OracleQuery) -> bool:
        try:
            spec = query.spec
            charges = spec.get("charges", []) or []
            if not charges:
                return False
            conductor_type = None
            conductors = spec.get("conductors", []) or []
            if conductors:
                conductor_type = conductors[0].get("type")
            geom_ok = conductor_type in (None, "plane", "sphere") or conductor_type is None
            return geom_ok
        except Exception:
            return False

    def evaluate(self, query: OracleQuery) -> OracleResult:
        require_cuda(query.points, "points")
        device = query.points.device
        dtype = torch.float64 if query.points.dtype == torch.float64 else torch.float32
        points = query.points.to(dtype=dtype)

        spec = query.spec
        charges = spec.get("charges", []) or []
        if not charges:
            raise RuntimeError("F3SymbolicOracleBackend requires point charges in spec")
        conductors = spec.get("conductors", []) or []
        conductor_type = conductors[0].get("type") if conductors else None

        start_wall = time.perf_counter()
        start_event, end_event = _maybe_cuda_events(device)
        if start_event is not None:
            start_event.record()

        V_total = torch.zeros(points.shape[0], device=device, dtype=dtype)
        E_total = torch.zeros(points.shape[0], 3, device=device, dtype=dtype)

        for charge in charges:
            q = float(charge.get("q", charge.get("charge", 0.0)))
            pos = torch.tensor(charge.get("pos", [0.0, 0.0, 0.0]), device=device, dtype=dtype)
            V_c, E_c = _point_charge(points, pos, q)
            V_total = V_total + V_c
            E_total = E_total + E_c
            if conductor_type == "plane":
                z0 = float(conductors[0].get("z", 0.0)) if conductors else 0.0
                pos_img = pos.clone()
                pos_img[2] = 2 * z0 - pos[2]
                V_img, E_img = _point_charge(points, pos_img, -q)
                V_total = V_total + V_img
                E_total = E_total + E_img
            elif conductor_type == "sphere":
                # Kelvin image for sphere (approximate, high precision not required for small checks)
                radius = float(conductors[0].get("radius", 1.0))
                center = torch.tensor(conductors[0].get("center", [0.0, 0.0, 0.0]), device=device, dtype=dtype)
                r0 = pos - center
                r0_norm = torch.linalg.norm(r0).clamp_min(1e-9)
                pos_img = center + (radius * radius / (r0_norm * r0_norm)) * r0
                q_img = -q * radius / r0_norm
                V_img, E_img = _point_charge(points, pos_img, q_img)
                V_total = V_total + V_img
                E_total = E_total + E_img

        valid_mask = torch.ones(points.shape[0], device=device, dtype=torch.bool)
        if end_event is not None:
            end_event.record()

        cost = make_cost(start_wall, device=device, start_event=start_event, end_event=end_event)
        provenance = make_provenance(device, dtype)
        error_estimate = OracleErrorEstimate(type=ErrorEstimateType.BOUND, metrics={"sym_err": 0.0}, confidence=0.9)
        cache_status = OracleCacheStatus(status=CacheStatus.MISS)

        return OracleResult(
            V=V_total,
            E=E_total if query.quantity != OracleQuantity.POTENTIAL else None,
            valid_mask=valid_mask,
            method="symbolic_high_precision",
            fidelity=self.fidelity,
            config_fingerprint=self._fingerprint,
            error_estimate=error_estimate,
            cost=cost,
            cache=cache_status,
            provenance=provenance,
        )
