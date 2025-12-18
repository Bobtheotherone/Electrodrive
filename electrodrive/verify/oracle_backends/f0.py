from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Tuple

import torch

from electrodrive.core.images import AnalyticSolution
from electrodrive.core.planar_stratified_reference import (
    ThreeLayerConfig,
    potential_three_layer_region1,
)
from electrodrive.learn.collocation import (
    _NullLogger,
    _infer_geom_type_from_spec,
    _solve_analytic,
)
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.config import BEMConfig, K_E

from ..cache import VerifyCache, VerifyCacheConfig
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
from ..utils import dtype_from_str, normalize_dtype, require_cuda
from .base import _maybe_cuda_events, fingerprint_config, make_cost, make_provenance

try:
    from electrodrive.core.bem import bem_solve, BEMSolution  # type: ignore

    BEM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency for coarse BEM fallback
    BEM_AVAILABLE = False
    bem_solve = None  # type: ignore[misc,assignment]
    BEMSolution = None  # type: ignore[misc,assignment]


def _torched_constant(value: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(value, device=device, dtype=dtype)


def _eval_plane(meta: Dict[str, Any], points: torch.Tensor, need_E: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
    device, dtype = points.device, points.dtype
    q = float(meta.get("charge", 0.0))
    q_img = float(meta.get("image_charge", -q))
    r0 = torch.as_tensor(meta.get("r0", (0.0, 0.0, 1.0)), device=device, dtype=dtype)
    r_img = torch.as_tensor(meta.get("image_pos", (0.0, 0.0, -1.0)), device=device, dtype=dtype)
    diff = points - r0
    diff_img = points - r_img
    r = torch.linalg.norm(diff, dim=1).clamp_min(1e-9)
    r_img_norm = torch.linalg.norm(diff_img, dim=1).clamp_min(1e-9)
    inv_r = 1.0 / r
    inv_r_img = 1.0 / r_img_norm
    V = _torched_constant(K_E, device, dtype) * (q * inv_r + q_img * inv_r_img)
    E = None
    if need_E:
        E = _torched_constant(K_E, device, dtype) * (
            q * diff * inv_r.pow(3).unsqueeze(1) + q_img * diff_img * inv_r_img.pow(3).unsqueeze(1)
        )
    return V, E, "analytic_plane_image"


def _eval_sphere(meta: Dict[str, Any], points: torch.Tensor, need_E: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
    device, dtype = points.device, points.dtype
    q = float(meta.get("charge", 0.0))
    q_img = float(meta.get("image_charge", -q))
    r0 = torch.as_tensor(meta.get("r0", (0.0, 0.0, 1.0)), device=device, dtype=dtype)
    r_img = torch.as_tensor(meta.get("image_pos", (0.0, 0.0, -1.0)), device=device, dtype=dtype)
    diff = points - r0
    diff_img = points - r_img
    r = torch.linalg.norm(diff, dim=1).clamp_min(1e-9)
    r_img_norm = torch.linalg.norm(diff_img, dim=1).clamp_min(1e-9)
    inv_r = 1.0 / r
    inv_r_img = 1.0 / r_img_norm
    V = _torched_constant(K_E, device, dtype) * (q * inv_r + q_img * inv_r_img)
    E = None
    if need_E:
        E = _torched_constant(K_E, device, dtype) * (
            q * diff * inv_r.pow(3).unsqueeze(1) + q_img * diff_img * inv_r_img.pow(3).unsqueeze(1)
        )
    return V, E, "analytic_sphere_kelvin"


def _eval_parallel_planes(meta: Dict[str, Any], points: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
    device, dtype = points.device, points.dtype
    q = float(meta.get("q", 0.0))
    r0 = meta.get("r0", (0.0, 0.0, 0.0))
    d = float(meta.get("d", 1.0))
    n_terms = int(meta.get("N_terms", 20))
    x0, y0, z0 = map(float, r0)
    n = torch.arange(-n_terms, n_terms + 1, device=device, dtype=dtype)
    sign = torch.where((n % 2) == 0, 1.0, -1.0)
    z_img = 2.0 * n * d + sign * z0  # [M]
    q_img = sign * q
    dx = points[:, 0:1] - x0
    dy = points[:, 1:2] - y0
    dz = points[:, 2:3] - z_img.unsqueeze(0)
    r = torch.sqrt(dx * dx + dy * dy + dz * dz).clamp_min(1e-9)
    V = _torched_constant(K_E, device, dtype) * torch.sum(q_img.unsqueeze(0) / r, dim=1)
    return V, None, "analytic_parallel_planes"


def _eval_cylinder2d(meta: Dict[str, Any], points: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
    device, dtype = points.device, points.dtype
    lam = float(meta.get("lambda", 0.0))
    x0, y0 = map(float, meta.get("r0_2d", (1.0, 0.0)))
    x_img, y_img = map(float, meta.get("image_pos_2d", (0.5, 0.0)))
    lam_img = float(meta.get("image_lambda", -lam))
    dx = points[:, 0] - x0
    dy = points[:, 1] - y0
    dx_img = points[:, 0] - x_img
    dy_img = points[:, 1] - y_img
    r = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-9)
    r_img = torch.sqrt(dx_img * dx_img + dy_img * dy_img).clamp_min(1e-9)
    V = _torched_constant(K_E, device, dtype) * (lam * torch.log(1.0 / r) + lam_img * torch.log(1.0 / r_img))
    return V, None, "analytic_cylinder2d"


def _eval_three_layer(meta: Dict[str, Any], points: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
    cfg = ThreeLayerConfig(
        eps1=float(meta.get("eps1", 1.0)),
        eps2=float(meta.get("eps2", 1.0)),
        eps3=float(meta.get("eps3", 1.0)),
        h=float(meta.get("h", 1.0)),
        q=float(meta.get("q", 1.0)),
        r0=tuple(meta.get("r0", (0.0, 0.0, 1.0))),
        n_k=int(meta.get("n_k", 256) or 256),
        k_max=meta.get("k_max", None),
    )
    V = potential_three_layer_region1(points, cfg, device=points.device, dtype=points.dtype)
    return V, None, "analytic_three_layer"


def _analytic_fallback(
    solution: AnalyticSolution, points: torch.Tensor, method_name: str = "analytic_scalar_fallback_cpu"
) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
    device, dtype = points.device, points.dtype
    vals = []
    pts_cpu = points.detach().cpu().numpy()
    for p in pts_cpu:
        vals.append(solution.eval(tuple(map(float, p.tolist()))))
    V = torch.tensor(vals, device=device, dtype=dtype)
    return V, None, method_name


def _infer_three_layer_config(spec: CanonicalSpec) -> Optional[ThreeLayerConfig]:
    dielectrics = getattr(spec, "dielectrics", None) or []
    charges = getattr(spec, "charges", None) or []
    if not dielectrics or len(charges) != 1 or charges[0].get("type") != "point":
        return None

    charge_z = float(charges[0]["pos"][2])

    def _eps_for_z(z_val: float) -> Optional[float]:
        for layer in dielectrics:
            eps = layer.get("epsilon") or layer.get("eps") or layer.get("permittivity")
            if eps is None:
                continue
            z_min = layer.get("z_min", None)
            z_max = layer.get("z_max", None)
            try:
                if z_min is not None and z_val < float(z_min) - 1e-9:
                    continue
                if z_max is not None and z_val > float(z_max) + 1e-9:
                    continue
                return float(eps)
            except Exception:
                continue
        return None

    boundary_counts: Dict[float, int] = {}
    for layer in dielectrics:
        for key in ("z_min", "z_max"):
            val = layer.get(key, None)
            if val is None:
                continue
            try:
                z_val = float(val)
            except Exception:
                continue
            boundary_counts[z_val] = boundary_counts.get(z_val, 0) + 1

    shared = [z for z, cnt in boundary_counts.items() if cnt >= 2]
    bounds = sorted(shared, reverse=True)
    if len(bounds) < 2:
        all_bounds = sorted(boundary_counts.keys(), reverse=True)
        if len(all_bounds) >= 3:
            bounds = all_bounds[1:-1]
    if len(bounds) < 2:
        return None

    z_top, z_bottom = bounds[0], bounds[1]
    eps1 = _eps_for_z(max(charge_z, z_top + 1e-6))
    eps2 = _eps_for_z(0.5 * (z_top + z_bottom))
    eps3 = _eps_for_z(z_bottom - 1e-3) or _eps_for_z(z_bottom + 1e-3)
    if eps1 is None or eps2 is None or eps3 is None or charge_z < z_top:
        return None

    return ThreeLayerConfig(
        eps1=eps1,
        eps2=eps2,
        eps3=eps3,
        h=abs(z_top - z_bottom),
        q=float(charges[0]["q"]),
        r0=tuple(charges[0]["pos"]),
    )


class F0AnalyticOracleBackend(OracleBackend):
    def __init__(self, cache: Optional[VerifyCache] = None) -> None:
        cfg = VerifyCacheConfig(enable_disk=False)
        self._cache = cache or VerifyCache(cfg)
        self._fingerprint = fingerprint_config("f0_analytic", {"version": "0.1"})

    @property
    def name(self) -> str:
        return "f0_analytic"

    @property
    def fidelity(self) -> OracleFidelity:
        return OracleFidelity.F0

    def fingerprint(self) -> str:
        return self._fingerprint

    def _build_solution(self, spec_dict: Dict[str, Any]) -> Optional[AnalyticSolution]:
        spec = CanonicalSpec.from_json(spec_dict)
        return _solve_analytic(spec)

    def can_handle(self, query: OracleQuery) -> bool:
        try:
            sol = self._build_solution(query.spec)
        except Exception:
            return False
        if sol is None:
            return False
        if query.quantity == OracleQuantity.FIELD:
            meta = getattr(sol, "meta", {})
            geom = meta.get("geometry") or _infer_geom_type_from_spec(CanonicalSpec.from_json(query.spec))
            return geom in ("plane", "sphere")
        return True

    def _evaluate_solution(
        self,
        solution: AnalyticSolution,
        spec: CanonicalSpec,
        points: torch.Tensor,
        need_E: bool,
        allow_cpu_fallback: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
        meta = getattr(solution, "meta", {}) or {}
        geom = meta.get("geometry") or _infer_geom_type_from_spec(spec)
        if meta.get("kind") == "planar_three_layer":
            return _eval_three_layer(meta, points)
        if geom == "plane":
            return _eval_plane(meta, points, need_E)
        if geom == "sphere":
            return _eval_sphere(meta, points, need_E)
        if geom == "parallel_planes":
            return _eval_parallel_planes(meta, points)
        if geom == "cylinder2D":
            return _eval_cylinder2d(meta, points)
        fallback_method = "analytic_scalar_fallback_cpu"
        if not allow_cpu_fallback:
            raise RuntimeError(
                f"CPU analytic fallback '{fallback_method}' blocked for geometry '{geom or 'unknown'}'; "
                "add a CUDA analytic implementation or set budget.allow_cpu_fallback=True to permit CPU fallback."
            )
        return _analytic_fallback(solution, points, method_name=fallback_method)

    def evaluate(self, query: OracleQuery) -> OracleResult:
        require_cuda(query.points, "points")
        solution = self._build_solution(query.spec)
        if solution is None:
            raise RuntimeError("F0 analytic backend cannot handle this spec.")

        spec = CanonicalSpec.from_json(query.spec)
        dtype = dtype_from_str(normalize_dtype(query.dtype))
        device = query.points.device
        pts = query.points.to(device=device, dtype=dtype)

        start = time.perf_counter()
        start_event, end_event = _maybe_cuda_events(device)
        if start_event is not None:
            start_event.record()

        need_E = query.quantity != OracleQuantity.POTENTIAL
        cache_status = OracleCacheStatus(status=CacheStatus.MISS)
        cache_key_str: Optional[str] = None
        V: Optional[torch.Tensor] = None
        E: Optional[torch.Tensor] = None
        allow_cpu_fallback = False
        if isinstance(query.budget, dict):
            allow_cpu_fallback = bool(query.budget.get("allow_cpu_fallback", False))

        if query.cache_policy != CachePolicy.OFF:
            key = self._cache.make_key(
                query.spec,
                self.name,
                self.fingerprint(),
                points=pts,
                include_points=True,
            )
            cache_key_str = key.to_string()
            if query.cache_policy != CachePolicy.REFRESH and query.cache_policy != CachePolicy.WRITE_ONLY:
                entry = self._cache.get(key)
                if entry and isinstance(entry.value, tuple):
                    V_cached, E_cached = entry.value
                    if torch.is_tensor(V_cached):
                        V = V_cached.to(device=device, dtype=dtype)
                    if torch.is_tensor(E_cached):
                        E = E_cached.to(device=device, dtype=dtype)
                    cache_status = OracleCacheStatus(status=CacheStatus.HIT, key=cache_key_str, path=entry.meta.get("cache_path"))

        if V is None:
            V, E, method = self._evaluate_solution(solution, spec, pts, need_E, allow_cpu_fallback)
        else:
            method = "analytic_cache"

        if end_event is not None:
            end_event.record()
        cost = make_cost(start, device=device, start_event=start_event, end_event=end_event)

        valid_mask = torch.isfinite(V)
        if E is not None:
            valid_mask = valid_mask & torch.isfinite(E).all(dim=1)

        if query.cache_policy != CachePolicy.OFF and cache_status.status != CacheStatus.HIT:
            key = self._cache.make_key(
                query.spec,
                self.name,
                self.fingerprint(),
                points=pts,
                include_points=True,
            )
            cache_payload = (V.detach().contiguous(), None if E is None else E.detach().contiguous())
            self._cache.set(
                key,
                cache_payload,
                meta={"backend": self.name, "cache_path": None},
                allow_large=True,
            )
            cache_status = OracleCacheStatus(status=CacheStatus.MISS, key=cache_key_str, path=None)

        error = OracleErrorEstimate(
            type=ErrorEstimateType.NONE,
            metrics={},
            confidence=1.0,
            notes=[method],
        )

        prov = make_provenance(device, dtype)

        return OracleResult(
            V=V,
            E=E if need_E else None,
            valid_mask=valid_mask,
            method=method,
            fidelity=self.fidelity,
            config_fingerprint=self.fingerprint(),
            error_estimate=error,
            cost=cost,
            cache=cache_status,
            provenance=prov,
        )


class F0CoarseSpectralOracleBackend(OracleBackend):
    def __init__(self, n_k: int = 64, cache: Optional[VerifyCache] = None) -> None:
        cfg = VerifyCacheConfig(enable_disk=False)
        self._cache = cache or VerifyCache(cfg)
        self.n_k = int(max(8, n_k))
        self._fingerprint = fingerprint_config("f0_coarse_spectral", {"n_k": self.n_k})

    @property
    def name(self) -> str:
        return "f0_coarse_spectral"

    @property
    def fidelity(self) -> OracleFidelity:
        return OracleFidelity.F0

    def fingerprint(self) -> str:
        return self._fingerprint

    def _build_cfg(self, spec_dict: Dict[str, Any]) -> Optional[ThreeLayerConfig]:
        spec = CanonicalSpec.from_json(spec_dict)
        cfg = _infer_three_layer_config(spec)
        if cfg is None:
            return None
        cfg.n_k = self.n_k
        return cfg

    def can_handle(self, query: OracleQuery) -> bool:
        try:
            cfg = self._build_cfg(query.spec)
        except Exception:
            return False
        return cfg is not None

    def evaluate(self, query: OracleQuery) -> OracleResult:
        require_cuda(query.points, "points")
        cfg = self._build_cfg(query.spec)
        if cfg is None:
            raise RuntimeError("F0 coarse spectral backend cannot handle this spec.")
        dtype = dtype_from_str(normalize_dtype(query.dtype))
        device = query.points.device
        pts = query.points.to(device=device, dtype=dtype)

        start = time.perf_counter()
        start_event, end_event = _maybe_cuda_events(device)
        if start_event is not None:
            start_event.record()

        cache_status = OracleCacheStatus(status=CacheStatus.MISS)
        cache_key_str: Optional[str] = None
        V: Optional[torch.Tensor] = None

        if query.cache_policy != CachePolicy.OFF:
            key = self._cache.make_key(query.spec, self.name, self.fingerprint(), points=pts, include_points=True)
            cache_key_str = key.to_string()
            if query.cache_policy != CachePolicy.REFRESH and query.cache_policy != CachePolicy.WRITE_ONLY:
                entry = self._cache.get(key)
                if entry and torch.is_tensor(entry.value):
                    V = entry.value.to(device=device, dtype=dtype)
                    cache_status = OracleCacheStatus(status=CacheStatus.HIT, key=cache_key_str, path=entry.meta.get("cache_path"))

        if V is None:
            V = potential_three_layer_region1(pts, cfg, device=device, dtype=dtype)

        if end_event is not None:
            end_event.record()
        cost = make_cost(start, device=device, start_event=start_event, end_event=end_event)

        valid_mask = torch.isfinite(V)

        if query.cache_policy != CachePolicy.OFF and cache_status.status != CacheStatus.HIT:
            key = self._cache.make_key(query.spec, self.name, self.fingerprint(), points=pts, include_points=True)
            self._cache.set(
                key,
                V.detach().contiguous(),
                meta={"backend": self.name, "cache_path": None},
                allow_large=True,
            )
            cache_status = OracleCacheStatus(status=CacheStatus.MISS, key=cache_key_str, path=None)

        error = OracleErrorEstimate(
            type=ErrorEstimateType.HEURISTIC,
            metrics={"n_k": float(self.n_k)},
            confidence=0.5,
            notes=["coarse_sommerfeld"],
        )
        prov = make_provenance(device, dtype)

        return OracleResult(
            V=V,
            E=None,
            valid_mask=valid_mask,
            method="coarse_three_layer",
            fidelity=self.fidelity,
            config_fingerprint=self.fingerprint(),
            error_estimate=error,
            cost=cost,
            cache=cache_status,
            provenance=prov,
        )


class F0CoarseBEMOracleBackend(OracleBackend):
    def __init__(self, cache: Optional[VerifyCache] = None) -> None:
        cfg = VerifyCacheConfig(enable_disk=False)
        self._cache = cache or VerifyCache(cfg)
        self._fingerprint = fingerprint_config(
            "f0_coarse_bem",
            {"gmres_tol": 1e-4, "refine": 1},
        )

    @property
    def name(self) -> str:
        return "f0_coarse_bem"

    @property
    def fidelity(self) -> OracleFidelity:
        return OracleFidelity.F0

    def fingerprint(self) -> str:
        return self._fingerprint

    def _make_cfg(self) -> BEMConfig:
        cfg = BEMConfig()
        cfg.use_gpu = True
        cfg.fp64 = False
        cfg.max_refine_passes = 1
        if hasattr(cfg, "min_refine_passes"):
            try:
                setattr(cfg, "min_refine_passes", 0)
            except Exception:
                pass
        cfg.gmres_tol = 1e-4
        cfg.gmres_maxiter = 400
        cfg.gmres_restart = 64
        cfg.use_near_quadrature = False
        cfg.use_near_quadrature_matvec = False
        cfg.target_vram_fraction = 0.9
        return cfg

    def can_handle(self, query: OracleQuery) -> bool:
        if not BEM_AVAILABLE:
            return False
        try:
            CanonicalSpec.from_json(query.spec)
        except Exception:
            return False
        return True

    def evaluate(self, query: OracleQuery) -> OracleResult:
        if not BEM_AVAILABLE:
            raise RuntimeError("BEM backend unavailable for coarse F0.")
        require_cuda(query.points, "points")
        spec = CanonicalSpec.from_json(query.spec)
        dtype = dtype_from_str(normalize_dtype(query.dtype))
        device = query.points.device
        pts = query.points.to(device=device, dtype=dtype)
        cfg = self._make_cfg()

        start = time.perf_counter()
        start_event, end_event = _maybe_cuda_events(device)
        if start_event is not None:
            start_event.record()

        # coarse BEM solve; we do not cache across runs to keep this lightweight.
        out = bem_solve(spec, cfg, _NullLogger())  # type: ignore[misc]
        if isinstance(out, dict) and "error" in out:
            raise RuntimeError(f"Coarse BEM failed: {out.get('error')}")
        solution: BEMSolution = out["solution"]  # type: ignore[assignment]
        V, _ = solution.eval_V_E_batched(pts)

        if end_event is not None:
            end_event.record()
        cost = make_cost(start, device=device, start_event=start_event, end_event=end_event)

        valid_mask = torch.isfinite(V)
        error = OracleErrorEstimate(
            type=ErrorEstimateType.HEURISTIC,
            metrics={"gmres_tol": float(cfg.gmres_tol)},
            confidence=0.3,
            notes=["coarse_bem"],
        )
        prov = make_provenance(device, dtype)

        return OracleResult(
            V=V,
            E=None,
            valid_mask=valid_mask,
            method="bem_coarse",
            fidelity=self.fidelity,
            config_fingerprint=self.fingerprint(),
            error_estimate=error,
            cost=cost,
            cache=OracleCacheStatus(status=CacheStatus.MISS),
            provenance=prov,
        )
