from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from electrodrive.layers import layerstack_from_spec
from electrodrive.layers.rt_recursion import effective_reflection
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.config import K_E

from ..cache import CacheKey, VerifyCache, VerifyCacheConfig
from ..oracle_registry import OracleBackend
from ..oracle_types import (
    CachePolicy,
    CacheStatus,
    ErrorEstimateType,
    OracleCacheStatus,
    OracleErrorEstimate,
    OracleFidelity,
    OracleQuery,
    OracleQuantity,
    OracleResult,
)
from ..poles import PoleSearchConfig, PoleTerm, find_poles
from ..utils import dtype_from_str, normalize_dtype, require_cuda
from .base import _maybe_cuda_events, fingerprint_config, make_cost, make_provenance


@dataclass(frozen=True)
class _SommerfeldConfig:
    k_min: float = 1e-4
    k_mid: float = 5.0
    k_max: float = 80.0
    n_low: int = 128
    n_mid: int = 160
    n_high: int = 96
    log_low: bool = False
    log_high: bool = True
    tail_scale: float = 1.0
    near_interface_factor: float = 1.5
    near_interface_n_mul: float = 1.5


def _merge_cfg(cfg: _SommerfeldConfig, overrides: Optional[Dict[str, object]]) -> _SommerfeldConfig:
    if not overrides:
        return cfg
    data = cfg.__dict__.copy()
    for k, v in overrides.items():
        if k in data:
            try:
                data[k] = type(data[k])(v)
            except Exception:
                data[k] = v
    return _SommerfeldConfig(**data)


def _build_k_grid(cfg: _SommerfeldConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    segments: List[torch.Tensor] = []
    if cfg.n_low > 0:
        if cfg.log_low:
            segments.append(torch.logspace(math.log10(cfg.k_min), math.log10(cfg.k_mid), cfg.n_low, device=device, dtype=torch.float64))
        else:
            segments.append(torch.linspace(cfg.k_min, cfg.k_mid, cfg.n_low, device=device, dtype=torch.float64))
    if cfg.n_mid > 0:
        segments.append(torch.linspace(cfg.k_mid, cfg.k_max, cfg.n_mid, device=device, dtype=torch.float64))
    if cfg.n_high > 0:
        if cfg.log_high:
            segments.append(torch.logspace(math.log10(cfg.k_mid), math.log10(cfg.k_max * cfg.tail_scale), cfg.n_high, device=device, dtype=torch.float64))
        else:
            segments.append(torch.linspace(cfg.k_mid, cfg.k_max * cfg.tail_scale, cfg.n_high, device=device, dtype=torch.float64))
    k = torch.unique(torch.cat(segments))
    return k.to(device=device, dtype=dtype)


def _min_distance_to_interfaces(stack, points: torch.Tensor) -> float:
    z_pts = points[:, 2]
    dists: List[torch.Tensor] = []
    for iface in stack.interfaces:
        dists.append(torch.abs(z_pts - torch.tensor(iface.z, device=points.device, dtype=points.dtype)))
    if not dists:
        return float("inf")
    return float(torch.min(torch.stack(dists)).item())


def _three_layer_denominator_factory(eps1: complex, eps2: complex, eps3: complex, h: float):
    eps1 = complex(eps1)
    eps2 = complex(eps2)
    eps3 = complex(eps3)
    R12 = (eps1 - eps2) / (eps1 + eps2)
    R21 = -R12
    R23 = (eps2 - eps3) / (eps2 + eps3)
    T12 = 2.0 * eps2 / (eps1 + eps2)
    T21 = 2.0 * eps1 / (eps1 + eps2)

    def _fn(k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        decay = torch.exp(-2.0 * k * h)
        denom = 1.0 - R21 * R23 * decay
        numer = T12 * T21 * R23 * decay
        return denom, numer

    return _fn


class F1SommerfeldOracleBackend(OracleBackend):
    def __init__(
        self,
        cache: Optional[VerifyCache] = None,
        *,
        disk_root: Optional[Path] = None,
        enable_disk: bool = True,
    ) -> None:
        cfg = VerifyCacheConfig(
            enable_disk=enable_disk,
            disk_root=disk_root or Path("artifacts/verify_cache/sommerfeld"),
            disk_write_tensors=True,
            allow_large_tensors=True,
            max_tensor_bytes=64 * 1024 * 1024,
        )
        self._cache = cache or VerifyCache(cfg)
        self._fingerprint = fingerprint_config(
            "f1_sommerfeld",
            {"version": "0.1"},
        )

    @property
    def name(self) -> str:
        return "f1_sommerfeld"

    @property
    def fidelity(self) -> OracleFidelity:
        return OracleFidelity.F1

    def fingerprint(self) -> str:
        return self._fingerprint

    def can_handle(self, query: OracleQuery) -> bool:
        try:
            spec = CanonicalSpec.from_json(query.spec)
        except Exception:
            return False
        if not getattr(spec, "dielectrics", None):
            return False
        return True

    def _fast_path(self, query: OracleQuery, dtype: torch.dtype) -> OracleResult:
        device = query.points.device
        start_wall = time.perf_counter()
        start_event, end_event = _maybe_cuda_events(device)
        if start_event is not None:
            start_event.record()
        charges = query.spec.get("charges", []) or []
        if not charges:
            raise RuntimeError("fast_mode requires at least one charge")
        charge = charges[0]
        q = float(charge.get("q", charge.get("charge", 0.0)))
        pos = torch.as_tensor(charge.get("pos", [0.0, 0.0, 0.0]), device=device, dtype=dtype)
        pts = query.points.to(dtype=dtype)
        diff = pts - pos
        r = torch.linalg.norm(diff, dim=1).clamp_min(1e-6)
        V = K_E * q / r
        valid_mask = torch.ones(pts.shape[0], device=device, dtype=torch.bool)
        if end_event is not None:
            end_event.record()
        cost = make_cost(start_wall, device=device, start_event=start_event, end_event=end_event)
        prov = make_provenance(device, dtype)
        error = OracleErrorEstimate(
            type=ErrorEstimateType.HEURISTIC, metrics={"fast_mode": 1.0}, confidence=0.2, notes=["fast_mode"]
        )
        cache_status = OracleCacheStatus(status=CacheStatus.MISS)
        return OracleResult(
            V=V,
            E=None if query.quantity == OracleQuantity.POTENTIAL else None,
            valid_mask=valid_mask,
            method="sommerfeld_fast",
            fidelity=self.fidelity,
            config_fingerprint=self._fingerprint + "|fast",
            error_estimate=error,
            cost=cost,
            cache=cache_status,
            provenance=prov,
        )

    def _cache_key(
        self,
        spec: Dict[str, object],
        kind: str,
        *,
        region: Optional[int] = None,
        include_points: bool = False,
        points: Optional[torch.Tensor] = None,
        cfg_hash: Optional[str] = None,
    ) -> CacheKey:
        oracle_name = f"{self.name}_{kind}" if region is None else f"{self.name}_{kind}_r{region}"
        fingerprint = self._fingerprint if cfg_hash is None else f"{self._fingerprint}|{cfg_hash}"
        return self._cache.make_key(
            spec,
            oracle_name,
            fingerprint,
            points=points if include_points else None,
            include_points=include_points,
        )
        return key

    def _maybe_load_cached(self, key: CacheKey):
        entry = self._cache.get(key)
        if entry:
            return entry.value
        return None

    def _store_cache(self, key: CacheKey, payload) -> None:
        self._cache.set(key, payload, meta={"kind": "sommerfeld_intermediate"}, allow_large=True, write_disk=True)

    def _prepare_k_and_reflection(
        self,
        spec: Dict[str, object],
        stack,
        cfg: _SommerfeldConfig,
        device: torch.device,
        dtype: torch.dtype,
        *,
        source_region: int,
        read_cache: bool,
        write_cache: bool,
        cfg_hash: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, bool]]:
        cache_hits = {"k": False, "R": False}
        k_key = self._cache_key(spec, "k_grid", cfg_hash=cfg_hash)
        R_key = self._cache_key(spec, "R_grid", region=source_region, cfg_hash=cfg_hash)

        k_grid: Optional[torch.Tensor] = None
        R_grid: Optional[torch.Tensor] = None

        if read_cache:
            cached = self._maybe_load_cached(k_key)
            if cached is not None:
                if torch.is_tensor(cached):
                    k_grid = cached.to(device=device, dtype=dtype)
                    cache_hits["k"] = True
                elif isinstance(cached, dict) and "k" in cached:
                    k_grid = torch.as_tensor(cached["k"], device=device, dtype=dtype)
                    cache_hits["k"] = True
            cached_R = self._maybe_load_cached(R_key)
            if cached_R is not None:
                if torch.is_tensor(cached_R):
                    R_grid = cached_R.to(device=device, dtype=torch.complex128)
                    cache_hits["R"] = True
                elif isinstance(cached_R, dict) and "R" in cached_R:
                    R_grid = torch.as_tensor(cached_R["R"], device=device, dtype=torch.complex128)
                    cache_hits["R"] = True

        if k_grid is None:
            k_grid = _build_k_grid(cfg, device, dtype)
            if write_cache:
                self._store_cache(k_key, k_grid)
        if R_grid is None:
            k_complex = k_grid.to(dtype=torch.complex128)
            R_grid = effective_reflection(
                stack,
                k_complex,
                source_region=source_region,
                direction="down",
                device=device,
                dtype=torch.complex128,
            )
            if write_cache:
                self._store_cache(R_key, R_grid)
        return k_grid, R_grid, cache_hits

    def _compute_integral(
        self,
        k: torch.Tensor,
        R: torch.Tensor,
        points: torch.Tensor,
        charge_pos: torch.Tensor,
        charge_q: float,
        eps_region: float,
        interface_z: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rho = torch.sqrt(torch.clamp((points[:, 0] - charge_pos[0]) ** 2 + (points[:, 1] - charge_pos[1]) ** 2, min=1e-16))
        z = points[:, 2]
        z0 = charge_pos[2]

        r_direct = torch.sqrt(rho * rho + (z - z0) ** 2).clamp_min(1e-9)
        direct = K_E * charge_q / (4.0 * math.pi * eps_region) * (1.0 / r_direct)

        k_real = torch.real(k)
        kr = torch.outer(k_real, rho)
        j0 = torch.special.bessel_j0(kr)

        exp_dir = torch.exp(-torch.outer(k_real, torch.abs(z - z0)))
        exp_ref = torch.exp(-torch.outer(k_real, z + z0 - 2.0 * interface_z))

        integrand = k_real[:, None].to(torch.complex128) * (
            exp_dir.to(torch.complex128) + (R[:, None] * exp_ref.to(torch.complex128))
        ) * j0.to(torch.complex128)

        integral = torch.trapz(integrand, k_real.to(torch.complex128), dim=0)
        reflected = (charge_q / (2.0 * math.pi * 2.0 * eps_region)) * integral
        V = direct + torch.real(reflected).to(direct.dtype)

        tail_est = torch.max(torch.abs(integrand[-1]))
        coarse_integral = torch.trapz(integrand[::2], k_real[::2].to(torch.complex128), dim=0)
        quad_resid = torch.max(torch.abs(integral - coarse_integral)) / torch.max(torch.abs(integral)).clamp_min(1e-9)
        return V, tail_est, quad_resid

    def _pole_search(
        self,
        cfg: _SommerfeldConfig,
        stack,
        source_region: int,
        device: torch.device,
    ) -> List[PoleTerm]:
        if len(stack.layers) < 3:
            return []
        layer0, layer1, layer2 = stack.layers[0], stack.layers[1], stack.layers[2]
        if math.isinf(layer1.z_max) or math.isinf(layer1.z_min):
            return []
        h = float(layer1.z_max - layer1.z_min)
        denom_fn = _three_layer_denominator_factory(layer0.eps, layer1.eps, layer2.eps, h)

        def _denom_tensor(k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            denom, numer = denom_fn(k)
            return torch.as_tensor(denom, device=device, dtype=torch.complex128), torch.as_tensor(
                numer, device=device, dtype=torch.complex128
            )

        k_samples = torch.linspace(cfg.k_min, cfg.k_max * cfg.tail_scale, 256, device=device, dtype=torch.complex128)
        search_cfg = PoleSearchConfig(
            method="denominator_roots",
            max_poles=4,
            k_rect=cfg.k_max * cfg.tail_scale,
            n_samples=256,
            newton_tol=1e-8,
            newton_max_iter=16,
        )
        return find_poles(
            lambda x: _denom_tensor(x)[0],
            denominator_fn=_denom_tensor,
            k_samples=k_samples,
            config=search_cfg,
            device=device,
            dtype=torch.complex128,
        )

    def evaluate(self, query: OracleQuery) -> OracleResult:
        require_cuda(query.points, "points")
        spec = CanonicalSpec.from_json(query.spec)
        stack = layerstack_from_spec(spec)

        dtype = dtype_from_str(normalize_dtype(query.dtype))
        if isinstance(query.budget, dict) and query.budget.get("fast_mode", False):
            return self._fast_path(query, dtype)
        compute_dtype = torch.float64 if dtype != torch.float64 else dtype
        device = query.points.device
        need_E = query.quantity != OracleQuantity.POTENTIAL
        pts = query.points.to(device=device, dtype=compute_dtype)
        if need_E:
            pts = pts.detach().clone().requires_grad_(True)

        base_cfg = _SommerfeldConfig()
        som_cfg = _merge_cfg(base_cfg, query.budget.get("sommerfeld", {}) if isinstance(query.budget, dict) else None)
        cfg_hash = fingerprint_config(self.name, {"grid": som_cfg.__dict__})

        near_interface = _min_distance_to_interfaces(stack, pts) < 1e-3
        if near_interface:
            som_cfg = _merge_cfg(
                som_cfg,
                {
                    "k_max": som_cfg.k_max * som_cfg.near_interface_factor,
                    "n_mid": int(som_cfg.n_mid * som_cfg.near_interface_n_mul),
                    "n_high": int(som_cfg.n_high * som_cfg.near_interface_n_mul),
                },
            )

        start = time.perf_counter()
        start_event, end_event = _maybe_cuda_events(device)
        if start_event is not None:
            start_event.record()

        cache_status = OracleCacheStatus(status=CacheStatus.MISS)
        allow_cache = query.cache_policy != CachePolicy.OFF
        read_cache = allow_cache and query.cache_policy not in (CachePolicy.REFRESH, CachePolicy.WRITE_ONLY)
        write_cache = allow_cache

        charges = getattr(spec, "charges", None) or []
        if not charges:
            raise RuntimeError("No charges defined in spec for Sommerfeld oracle.")

        source_pos = torch.as_tensor(charges[0].get("pos", [0.0, 0.0, 0.0]), device=device, dtype=compute_dtype)
        source_region = stack.layer_index(float(source_pos[2].item()))
        interface_z = stack.layers[source_region].z_min
        eps_region = float(stack.layers[source_region].eps.real)

        k_grid, R_grid, cache_hits = self._prepare_k_and_reflection(
            query.spec,
            stack,
            som_cfg,
            device,
            compute_dtype,
            source_region=source_region,
            read_cache=read_cache,
            write_cache=write_cache,
            cfg_hash=cfg_hash,
        )

        V_total = torch.zeros(pts.shape[0], device=device, dtype=compute_dtype)
        tail_proxy = torch.tensor(0.0, device=device, dtype=torch.float64)
        quad_proxy = torch.tensor(0.0, device=device, dtype=torch.float64)

        for charge in charges:
            if charge.get("type", "point") != "point":
                continue
            charge_q = float(charge.get("q", 0.0))
            pos = torch.as_tensor(charge.get("pos", [0.0, 0.0, 0.0]), device=device, dtype=compute_dtype)
            region = stack.layer_index(float(pos[2].item()))
            if region != source_region:
                # Recompute reflection if another region appears.
                k_grid, R_grid, _ = self._prepare_k_and_reflection(
                    query.spec,
                    stack,
                    som_cfg,
                    device,
                    compute_dtype,
                    source_region=region,
                    read_cache=read_cache,
                    write_cache=write_cache,
                    cfg_hash=cfg_hash,
                )
                interface_z = stack.layers[region].z_min
                eps_region = float(stack.layers[region].eps.real)
            V_part, tail_est, quad_resid = self._compute_integral(
                k_grid,
                R_grid,
                pts,
                pos,
                charge_q,
                eps_region,
                interface_z,
            )
            V_total = V_total + V_part
            tail_proxy = torch.maximum(tail_proxy, tail_est)
            quad_proxy = torch.maximum(quad_proxy, quad_resid)

        E = None
        if need_E:
            grad = torch.autograd.grad(V_total.sum(), pts, create_graph=False, retain_graph=False)[0]
            E = -grad

        if end_event is not None:
            end_event.record()
        cost = make_cost(start, device=device, start_event=start_event, end_event=end_event)

        valid_mask = torch.isfinite(V_total)
        if E is not None:
            valid_mask = valid_mask & torch.isfinite(E).all(dim=1)

        poles: List[PoleTerm] = []
        pole_meta: List[Dict[str, object]] = []
        pole_key = self._cache_key(query.spec, "poles", region=source_region, cfg_hash=cfg_hash)
        if read_cache:
            cached_poles = self._maybe_load_cached(pole_key)
            if cached_poles:
                try:
                    poles = [PoleTerm.from_json(item) for item in cached_poles]  # type: ignore[arg-type]
                    pole_meta = [p.to_json() for p in poles]
                except Exception:
                    poles = []
                    pole_meta = []
        try:
            if not poles and isinstance(query.budget, dict) and query.budget.get("enable_poles", True):
                poles = self._pole_search(som_cfg, stack, source_region, device)
                pole_meta = [p.to_json() for p in poles]
                if write_cache:
                    self._store_cache(pole_key, pole_meta)
        except Exception:
            poles = []
            pole_meta = []

        error = OracleErrorEstimate(
            type=ErrorEstimateType.A_POSTERIORI,
            metrics={
                "tail_est": float(tail_proxy.item()),
                "quad_resid": float(quad_proxy.item()),
                "n_k": float(k_grid.numel()),
                "cache_hit_k": 1.0 if cache_hits.get("k") else 0.0,
                "cache_hit_R": 1.0 if cache_hits.get("R") else 0.0,
                "n_poles": float(len(poles)),
            },
            confidence=0.7 if quad_proxy < 1e-3 else 0.5,
            notes=["sommerfeld_integral", "near_interface" if near_interface else "bulk", f"poles:{len(poles)}"],
        )

        prov = make_provenance(device, dtype)

        V_out = V_total.to(dtype=dtype)
        E_out = None if E is None else E.to(dtype=dtype)

        method = "sommerfeld_integral"
        return OracleResult(
            V=V_out,
            E=E_out if need_E else None,
            valid_mask=valid_mask,
            method=method,
            fidelity=self.fidelity,
            config_fingerprint=self._fingerprint,
            error_estimate=error,
            cost=cost,
            cache=cache_status,
            provenance=prov,
        )
