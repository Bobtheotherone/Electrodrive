from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from electrodrive.learn.collocation import _NullLogger, _make_default_oracle_bem_config
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.config import BEMConfig

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
from ..utils import (
    dtype_from_str,
    get_git_sha,
    normalize_dtype,
    require_cuda,
    sha256_json,
    utc_now_iso,
)
from .base import _maybe_cuda_events, fingerprint_config, make_cost, make_provenance

try:
    from electrodrive.core.bem import bem_solve, BEMSolution  # type: ignore

    BEM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency handling
    BEM_AVAILABLE = False
    bem_solve = None  # type: ignore[misc,assignment]
    BEMSolution = None  # type: ignore[misc,assignment]


class F2BEMOracleBackend(OracleBackend):
    def __init__(
        self,
        *,
        disk_root: Optional[Path] = None,
    ) -> None:
        self.disk_root = disk_root or Path("artifacts/verify_cache/bem")
        self.disk_root.mkdir(parents=True, exist_ok=True)
        self._fingerprint = fingerprint_config(
            "f2_bem_ground_truth",
            {"version": "0.1"},
        )
        self._ram_solutions: Dict[str, Tuple[BEMSolution, Dict[str, Any]]] = {}

    @property
    def name(self) -> str:
        return "f2_bem"

    @property
    def fidelity(self) -> OracleFidelity:
        return OracleFidelity.F2

    def fingerprint(self) -> str:
        return self._fingerprint

    def _make_cfg(self, overrides: Optional[Dict[str, Any]] = None) -> BEMConfig:
        cfg = _make_default_oracle_bem_config(overrides or {})
        cfg.use_gpu = True
        cfg.fp64 = True
        if hasattr(cfg, "use_near_quadrature"):
            cfg.use_near_quadrature = True
        return cfg

    def _config_dict(self, cfg: BEMConfig, dtype: torch.dtype) -> Dict[str, Any]:
        cfg_dict = asdict(cfg)
        cfg_dict["dtype"] = normalize_dtype(dtype)
        cfg_dict["git"] = get_git_sha()
        return cfg_dict

    def _fingerprint_for_config(self, cfg_dict: Dict[str, Any]) -> str:
        return fingerprint_config(self.name, cfg_dict)

    def _spec_hash(self, spec_dict: Dict[str, Any]) -> str:
        return sha256_json(spec_dict)

    def _cache_dir(self, spec_hash: str, cfg_hash: str) -> Path:
        return self.disk_root / spec_hash / cfg_hash

    def _persist_solution(
        self,
        cache_dir: Path,
        solution: BEMSolution,
        *,
        cfg_dict: Dict[str, Any],
        spec_hash: str,
        cfg_hash: str,
        gmres_stats: Dict[str, Any],
        mesh_stats: Dict[str, Any],
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "centroids": solution._C.detach().cpu(),
            "areas": solution._A.detach().cpu(),
            "sigma": solution._S.detach().cpu(),
            "tile_size": int(solution._tile),
            "dtype": normalize_dtype(solution._dtype),
            "device": str(solution._device),
            "normals": None if solution._N is None else solution._N.detach().cpu(),
            "panel_vertices": None
            if getattr(solution, "_panel_vertices", None) is None
            else solution._panel_vertices.detach().cpu(),
            "meta": getattr(solution, "meta", {}),
        }
        torch.save(payload, cache_dir / "solution.pt")
        meta = {
            "spec_hash": spec_hash,
            "config_hash": cfg_hash,
            "timestamp": utc_now_iso(),
            "git_sha": get_git_sha(),
            "gmres_stats": gmres_stats,
            "mesh_stats": mesh_stats,
            "cfg": cfg_dict,
        }
        (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _load_solution(
        self,
        cache_dir: Path,
        spec: CanonicalSpec,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tuple[BEMSolution, Dict[str, Any]]]:
        meta_path = cache_dir / "meta.json"
        sol_path = cache_dir / "solution.pt"
        if not meta_path.exists() or not sol_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            payload = torch.load(sol_path, map_location=device)
        except Exception:
            return None
        C = payload["centroids"].to(device=device, dtype=dtype)
        A = payload["areas"].to(device=device, dtype=dtype)
        sigma = payload["sigma"].to(device=device, dtype=dtype)
        normals = payload.get("normals", None)
        if normals is not None:
            normals = normals.to(device=device, dtype=dtype)
        panel_vertices = payload.get("panel_vertices", None)
        if panel_vertices is not None:
            panel_vertices = panel_vertices.to(device=device, dtype=dtype)
        tile_size = int(payload.get("tile_size", 2048))
        near_quad = bool(meta.get("cfg", {}).get("use_near_quadrature", False))
        near_quad_order = int(meta.get("cfg", {}).get("near_quadrature_order", 2))
        near_quad_dist = float(meta.get("cfg", {}).get("near_quadrature_distance_factor", 1.5))
        near_quad_max_depth = meta.get("cfg", {}).get("near_quad_max_depth", None)
        solution = BEMSolution(
            spec,
            C,
            A,
            sigma,
            device,
            dtype,
            tile_size,
            normals=normals,
            panel_vertices=panel_vertices,
            near_quadrature=near_quad,
            near_quad_order=near_quad_order,
            near_quad_dist_factor=near_quad_dist,
            near_quad_max_depth=near_quad_max_depth,
        )
        solution.meta.update(payload.get("meta", {}))
        return solution, meta

    def can_handle(self, query: OracleQuery) -> bool:
        if not BEM_AVAILABLE:
            return False
        try:
            CanonicalSpec.from_json(query.spec)
        except Exception:
            return False
        return True

    def _evaluate_solution(
        self,
        solution: BEMSolution,
        points: torch.Tensor,
        need_E: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        V, E = solution.eval_V_E_batched(points)
        if not need_E:
            E = None
        return V, E

    def evaluate(self, query: OracleQuery) -> OracleResult:
        if not BEM_AVAILABLE:
            raise RuntimeError("BEM backend unavailable.")
        require_cuda(query.points, "points")
        spec = CanonicalSpec.from_json(query.spec)
        dtype = dtype_from_str(normalize_dtype(query.dtype))
        device = query.points.device
        pts = query.points.to(device=device, dtype=dtype)

        overrides = query.budget.get("bem", {}) if isinstance(query.budget, dict) else {}
        cfg = self._make_cfg(overrides if overrides else None)
        cfg_dict = self._config_dict(cfg, dtype)
        cfg_hash = self._fingerprint_for_config(cfg_dict)
        spec_hash = self._spec_hash(query.spec)
        ram_key = f"{spec_hash}:{cfg_hash}"
        cache_dir = self._cache_dir(spec_hash, cfg_hash)

        start = time.perf_counter()
        start_event, end_event = _maybe_cuda_events(device)
        if start_event is not None:
            start_event.record()

        cache_status = OracleCacheStatus(status=CacheStatus.MISS, key=cfg_hash, path=str(cache_dir))
        meta_loaded: Dict[str, Any] = {}
        solution: Optional[BEMSolution] = None

        allow_cache = query.cache_policy != CachePolicy.OFF
        read_cache = allow_cache and query.cache_policy not in (CachePolicy.REFRESH, CachePolicy.WRITE_ONLY)
        write_cache = allow_cache and query.cache_policy != CachePolicy.OFF

        if read_cache and ram_key in self._ram_solutions:
            solution, meta_loaded = self._ram_solutions[ram_key]
            cache_status = OracleCacheStatus(status=CacheStatus.HIT, key=cfg_hash, path=str(cache_dir))

        if solution is None and read_cache:
            loaded = self._load_solution(cache_dir, spec, device, dtype)
            if loaded is not None:
                solution, meta_loaded = loaded
                meta_loaded.setdefault("spec_hash", spec_hash)
                meta_loaded.setdefault("config_hash", cfg_hash)
                meta_loaded["ram_key"] = ram_key
                self._ram_solutions[ram_key] = (solution, meta_loaded)
                cache_status = OracleCacheStatus(status=CacheStatus.HIT, key=cfg_hash, path=str(cache_dir))

        gmres_stats: Dict[str, Any] = {}
        mesh_stats: Dict[str, Any] = {}

        if solution is None:
            out = bem_solve(spec, cfg, _NullLogger())  # type: ignore[misc]
            if isinstance(out, dict) and "error" in out:
                raise RuntimeError(f"BEM solve failed: {out.get('error')}")
            solution = out["solution"]  # type: ignore[assignment]
            gmres_stats = dict(out.get("gmres_stats", {}))
            mesh_stats = dict(out.get("mesh_stats", {}))
            meta_loaded = {
                "gmres_stats": gmres_stats,
                "mesh_stats": mesh_stats,
                "cfg": cfg_dict,
                "spec_hash": spec_hash,
                "config_hash": cfg_hash,
                "ram_key": ram_key,
            }
            if write_cache:
                self._persist_solution(
                    cache_dir,
                    solution,
                    cfg_dict=cfg_dict,
                    spec_hash=spec_hash,
                    cfg_hash=cfg_hash,
                    gmres_stats=gmres_stats,
                    mesh_stats=mesh_stats,
                )
                self._ram_solutions[ram_key] = (solution, meta_loaded)
            cache_status = OracleCacheStatus(status=CacheStatus.MISS, key=cfg_hash, path=str(cache_dir))
        else:
            gmres_stats = dict(meta_loaded.get("gmres_stats", {}))
            mesh_stats = dict(meta_loaded.get("mesh_stats", {}))

        V, E = self._evaluate_solution(solution, pts, query.quantity != OracleQuantity.POTENTIAL)

        if end_event is not None:
            end_event.record()
        cost = make_cost(start, device=device, start_event=start_event, end_event=end_event)

        valid_mask = torch.isfinite(V)
        if E is not None:
            valid_mask = valid_mask & torch.isfinite(E).all(dim=1)

        metrics = {}
        if gmres_stats:
            metrics["gmres_resid"] = float(gmres_stats.get("resid", gmres_stats.get("gmres_resid", 0.0)))
            metrics["gmres_iters"] = float(gmres_stats.get("iters", gmres_stats.get("gmres_iters", 0)))
        if mesh_stats:
            metrics["bc_residual_linf"] = float(mesh_stats.get("bc_residual_linf", 0.0))

        error = OracleErrorEstimate(
            type=ErrorEstimateType.A_POSTERIORI,
            metrics=metrics,
            confidence=0.8 if metrics else 0.5,
            notes=["bem_ground_truth"],
        )
        prov = make_provenance(device, dtype)

        return OracleResult(
            V=V,
            E=E,
            valid_mask=valid_mask,
            method="bem_ground_truth",
            fidelity=self.fidelity,
            config_fingerprint=cfg_hash,
            error_estimate=error,
            cost=cost,
            cache=cache_status,
            provenance=prov,
        )
