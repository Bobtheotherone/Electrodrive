from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from electrodrive.utils.config import EPS_0, K_E
from electrodrive.core.planar_stratified_reference import (
    ThreeLayerConfig,
    make_three_layer_solution,
    potential_three_layer_region1,
)

from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.core.images import (
    AnalyticSolution,
    potential_plane_halfspace,
    potential_sphere_grounded,
    potential_line_cylinder2d_grounded,
    potential_parallel_planes_subset,
)
from electrodrive.learn.encoding import encode_spec, ENCODING_DIM
from electrodrive.learn.neural_operators import (
    SphereFNOSurrogate,
    extract_stage0_sphere_params,
    load_spherefno_from_env,
)
from electrodrive.debug import bem_intercept

try:
    # BEM is optional for CPU-only / minimal installs.
    from electrodrive.core.bem import bem_solve, BEMSolution  # type: ignore
    from electrodrive.utils.config import BEMConfig

    BEM_AVAILABLE = True
except Exception:  # pragma: no cover - handled gracefully at runtime
    from typing import Any as _AnyType  # fallback for type checking

    bem_solve = None  # type: ignore
    BEMSolution = _AnyType  # type: ignore
    BEMConfig = _AnyType  # type: ignore
    BEM_AVAILABLE = False


logger = logging.getLogger("EDE.Learn.Collocation")


class _NullLogger:
    """Minimal no-op logger for BEM oracles when we don't want console spam.

    Matches the .info/.warning/.error interface expected by bem_solve but
    silently drops all messages. Callers can still pass an explicit logger
    via bem_cfg["logger"] when verbose logging is desired.
    """

    def info(self, *args: Any, **kwargs: Any) -> None:
        pass

    def debug(self, *args: Any, **kwargs: Any) -> None:
        pass

    def warning(self, *args: Any, **kwargs: Any) -> None:
        pass

    def error(self, *args: Any, **kwargs: Any) -> None:
        pass


OracleSolution = Union[AnalyticSolution, "BEMSolution"]


# ---------------------------------------------------------------------------
# Analytic oracle selection (mirrors electrodrive.learn.dataset)
# ---------------------------------------------------------------------------


def _solve_analytic(spec: CanonicalSpec) -> Optional[AnalyticSolution]:
    """
    Try to construct an AnalyticSolution for a small set of canonical
    geometries.

    This mirrors the logic historically implemented in the learning
    stack and is intentionally conservative: if any of the structural
    checks fail, we return None and let the caller fall back to BEM.
    """
    ctypes = (
        sorted({c.get("type") for c in spec.conductors})
        if spec.conductors
        else []
    )
    dielectrics = getattr(spec, "dielectrics", None) or []

    # --- 1) Single grounded plane + single point charge ---
    if (
        ctypes == ["plane"]
        and len(spec.conductors) == 1
        and len(spec.charges) == 1
        and spec.charges[0]["type"] == "point"
    ):
        q = spec.charges[0]["q"]
        r0 = tuple(spec.charges[0]["pos"])
        c = spec.conductors[0]
        if (
            c.get("z", 0.0) == 0.0
            and c.get("potential", 0.0) == 0.0
            and r0[2] > 0
        ):
            return potential_plane_halfspace(q, r0)

    # --- 2) Single grounded sphere + single point charge ---
    if (
        ctypes == ["sphere"]
        and len(spec.conductors) == 1
        and len(spec.charges) == 1
        and spec.charges[0]["type"] == "point"
    ):
        q = spec.charges[0]["q"]
        r0 = tuple(spec.charges[0]["pos"])
        c = spec.conductors[0]
        radius = c.get("radius")
        center = tuple(c.get("center", [0.0, 0.0, 0.0]))
        if radius is not None and c.get("potential", 0.0) == 0.0:
            return potential_sphere_grounded(
                q,
                r0,
                center,
                float(radius),
            )

    # --- 2c) Planar stratified three-layer (dielectric) with point charge in region 1 ---
    if (
        not ctypes
        and dielectrics
        and len(spec.charges) == 1
        and spec.charges[0]["type"] == "point"
    ):
        layers = dielectrics
        charge_z = float(spec.charges[0]["pos"][2])

        def _eps_for_z(z_val: float) -> Optional[float]:
            for layer in layers:
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
        for layer in layers:
            for key in ("z_min", "z_max"):
                val = layer.get(key, None)
                if val is None:
                    continue
                try:
                    z_val = float(val)
                except Exception:
                    continue
                boundary_counts[z_val] = boundary_counts.get(z_val, 0) + 1
        shared_boundaries = [z for z, cnt in boundary_counts.items() if cnt >= 2]
        boundaries_sorted = sorted(shared_boundaries, reverse=True)
        if len(boundaries_sorted) < 2:
            # Fallback: drop extremal bounds and use interior ones.
            all_bounds = sorted(boundary_counts.keys(), reverse=True)
            if len(all_bounds) >= 3:
                boundaries_sorted = all_bounds[1:-1]
        if len(boundaries_sorted) >= 2:
            z_top = boundaries_sorted[0]
            z_bottom = boundaries_sorted[1]
            h = abs(z_top - z_bottom)
            eps1 = _eps_for_z(max(charge_z, z_top + 1e-6))
            eps2 = _eps_for_z(0.5 * (z_top + z_bottom))
            eps3 = _eps_for_z(z_bottom - 1e-3) or _eps_for_z(z_bottom + 1e-3)
            if eps1 and eps2 and eps3 and charge_z >= z_top:
                cfg = ThreeLayerConfig(
                    eps1=eps1,
                    eps2=eps2,
                    eps3=eps3,
                    h=h,
                    q=float(spec.charges[0]["q"]),
                    r0=tuple(spec.charges[0]["pos"]),
                )
                return make_three_layer_solution(cfg)

    # --- 3) Cylinder / line-charge 2D image construction ---
    if (
        ("cylinder" in ctypes or "cylinder2D" in ctypes)
        and len(spec.conductors) == 1
        and len(spec.charges) == 1
        and spec.charges[0]["type"] in ("line_charge", "line")
    ):
        lambda_c = spec.charges[0].get("lambda")
        r0_2d = spec.charges[0].get("pos_2d")
        c = spec.conductors[0]
        radius = c.get("radius")
        potential = c.get("potential", 0.0)
        if (
            lambda_c is not None
            and r0_2d is not None
            and radius is not None
            and potential == 0.0
        ):
            return potential_line_cylinder2d_grounded(
                float(lambda_c),
                float(radius),
                tuple(r0_2d),
            )

    # --- 4) Parallel grounded planes + central point charge ---
    if (
        ctypes == ["plane"]
        and len(spec.conductors) == 2
        and len(spec.charges) == 1
        and spec.charges[0]["type"] == "point"
    ):
        z1 = spec.conductors[0].get("z")
        z2 = spec.conductors[1].get("z")
        p1 = spec.conductors[0].get("potential", 0.0)
        p2 = spec.conductors[1].get("potential", 0.0)
        if (
            z1 is not None
            and z2 is not None
            and p1 == 0.0
            and p2 == 0.0
            and abs(z1 + z2) < 1e-6
        ):
            d = abs(float(z1))
            q = spec.charges[0]["q"]
            r0 = tuple(spec.charges[0]["pos"])
            if abs(r0[2]) < d:
                return potential_parallel_planes_subset(
                    q,
                    r0,
                    d,
                    N_terms=30,
                )

    return None


# ---------------------------------------------------------------------------
# Oracle selection / BEM configuration
# ---------------------------------------------------------------------------


def _make_default_oracle_bem_config(
    overrides: Optional[Dict[str, Any]] = None,
) -> BEMConfig:
    """
    Construct a high-accuracy BEMConfig for oracle use.

    This mirrors the Stage-0 sphere calibration settings:
      - fp64 on GPU if available
      - at least 3 refinement passes (min_refine_passes >= 1)
      - strict GMRES tolerance
      - near-field quadrature enabled for evaluation and matvec
    """
    cfg = BEMConfig()

    # Precision / device
    cfg.use_gpu = torch.cuda.is_available()
    cfg.fp64 = True

    # Refinement policy: at least 3 passes.
    try:
        base_max = int(getattr(cfg, "max_refine_passes", 3) or 3)
    except Exception:
        base_max = 3
    cfg.max_refine_passes = max(base_max, 3)

    try:
        base_min = int(getattr(cfg, "min_refine_passes", 1) or 1)
    except Exception:
        base_min = 1
    cfg.min_refine_passes = max(base_min, 1)

    # Strong GMRES tolerances.
    try:
        gmres_tol = float(getattr(cfg, "gmres_tol", 5e-8))
    except Exception:
        gmres_tol = 5e-8
    cfg.gmres_tol = min(gmres_tol, 5e-8)
    try:
        gmres_maxiter = int(getattr(cfg, "gmres_maxiter", 2000))
    except Exception:
        gmres_maxiter = 2000
    # Respect explicit overrides; just guard against non-positive values.
    cfg.gmres_maxiter = max(gmres_maxiter, 1)
    try:
        gmres_restart = int(getattr(cfg, "gmres_restart", 256))
    except Exception:
        gmres_restart = 256
    cfg.gmres_restart = max(gmres_restart, 1)

    # Near-field quadrature for both evaluation and matvec.
    if hasattr(cfg, "use_near_quadrature"):
        cfg.use_near_quadrature = True
    if hasattr(cfg, "use_near_quadrature_matvec"):
        cfg.use_near_quadrature_matvec = True
    if hasattr(cfg, "near_quadrature_order"):
        try:
            cfg.near_quadrature_order = int(getattr(cfg, "near_quadrature_order", 2))
        except Exception:
            cfg.near_quadrature_order = 2
    if hasattr(cfg, "near_quadrature_distance_factor"):
        try:
            df = float(getattr(cfg, "near_quadrature_distance_factor", 1.5))
        except Exception:
            df = 1.5
        cfg.near_quadrature_distance_factor = max(df, 2.0)

    # Apply explicit overrides last.
    if overrides:
        for k, v in overrides.items():
            try:
                setattr(cfg, k, v)
            except Exception:
                continue

    return cfg


def get_oracle_solution(
    spec: CanonicalSpec,
    mode: str,
    bem_cfg: Dict[str, Any],
) -> Optional[OracleSolution]:
    """
    Select and construct a field oracle for a given spec.

    For analytic / image-charge geometries we expect the analytic shortcut
    and the BEM oracle to agree on a shared set of collocation points.

    For BEM oracles, we bias toward:
      * fp64 for numerical stability
      * GPU if available (do not force CPU)
      * near-field quadrature at evaluation points and in the GMRES
        matvec, unless the caller explicitly overrides this via bem_cfg.
    """
    sol: Optional[OracleSolution] = None

    # 1) Analytic path (if explicitly requested or allowed by "auto").
    if mode in ("analytic", "auto"):
        try:
            sol = _solve_analytic(spec)
        except Exception:
            sol = None

    # 2) BEM oracle path.
    if sol is None and mode in ("bem", "auto") and BEM_AVAILABLE:
        try:
            # Normalise bem_cfg to a dict and extract logger override (if any)
            if bem_cfg is None:
                bem_cfg = {}
            log_obj = bem_cfg.get("logger", None)
            if log_obj is None:
                log_obj = _NullLogger()

            # Build a strong default BEMConfig and then apply overrides.
            cfg_overrides = {k: v for k, v in bem_cfg.items() if k != "logger"}
            cfg = _make_default_oracle_bem_config(cfg_overrides)

            # Disable near-smoothing; we want the raw Green's function.
            if hasattr(cfg, "near_alpha"):
                cfg.near_alpha = 0.0

            # Use either the caller-provided logger or a quiet no-op logger by
            # default. Callers can still pass ConsoleLogger() explicitly via
            # bem_cfg["logger"] if they want stdout JSON logs.
            out = bem_solve(spec, cfg, log_obj)  # type: ignore[arg-type]
            if isinstance(out, dict) and "solution" in out:
                sol = out["solution"]
        except Exception as e:  # pragma: no cover - defensive
            logger.error("BEM oracle failed: %s", e)

    return sol


# ---------------------------------------------------------------------------
# Collocation sampling + oracle evaluation
# ---------------------------------------------------------------------------

# Simple cache so repeated calls for the *same* CanonicalSpec (or, in hash
# mode, the same spec content) reuse the expensive oracle instead of
# re-solving from scratch.
_ORACLE_CACHE: Dict[
    Tuple[str, str, Tuple[Tuple[str, str], ...]], OracleSolution
] = {}
_ORACLE_CACHE_ORDER: List[
    Tuple[str, str, Tuple[Tuple[str, str], ...]]
] = []

try:
    _MAX_CACHED_ORACLES = int(os.getenv("EDE_COLL_MAX_CACHED_ORACLES", "0"))
except Exception:
    _MAX_CACHED_ORACLES = 0

# Relative shell thickness used when sampling boundary points for spheres.
try:
    _SPHERE_SHELL_FRAC = max(
        0.0, float(os.getenv("EDE_COLL_SPHERE_SHELL_FRAC", "0.0"))
    )
except Exception:
    _SPHERE_SHELL_FRAC = 0.0

# Neural surrogate tolerances and validation controls.
try:
    _SPHEREFNO_L2_TOL = float(os.getenv("EDE_SPHEREFNO_L2_TOL", "1e-3"))
except Exception:
    _SPHEREFNO_L2_TOL = 1e-3

try:
    _SPHEREFNO_LINF_TOL = float(os.getenv("EDE_SPHEREFNO_LINF_TOL", "1e-2"))
except Exception:
    _SPHEREFNO_LINF_TOL = 1e-2

try:
    _SPHEREFNO_VAL_SAMPLES = max(0, int(os.getenv("EDE_SPHEREFNO_VAL_SAMPLES", "64")))
except Exception:
    _SPHEREFNO_VAL_SAMPLES = 64

_SPHEREFNO_VALIDATE_WITH_BEM = os.getenv(
    "EDE_SPHEREFNO_VALIDATE_WITH_BEM", ""
).lower() in ("1", "true", "yes")

_NEURAL_FALLBACK_MODE = os.getenv("EDE_NEURAL_FALLBACK_MODE", "analytic").strip().lower()
if _NEURAL_FALLBACK_MODE not in ("analytic", "bem", "auto"):
    _NEURAL_FALLBACK_MODE = "analytic"

# Cache for surrogate instances keyed by (ckpt_path, device, dtype).
_NEURAL_SURROGATE_CACHE: Dict[Tuple[str, str, str], SphereFNOSurrogate] = {}


def _bem_cfg_cache_key(bem_cfg: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    """
    Represent a config dict as a sorted tuple of stringified (k, v) pairs,
    which is stable enough for caching purposes.
    """
    return tuple(sorted((str(k), repr(v)) for k, v in bem_cfg.items()))


def _spec_identity_key(spec: CanonicalSpec) -> str:
    """Default cache key using Python object identity.

    This matches the historical behaviour where a single CanonicalSpec
    instance is reused across many batches.
    """
    return f"id:{id(spec)}"


def _spec_hash_key(spec: CanonicalSpec) -> str:
    """Content-based cache key for CanonicalSpec.

    Uses a JSON-serialised subset of fields to build a stable hash so
    that geometrically identical specs share oracle solutions even if
    they are distinct Python objects.

    Notes
    -----
    - Only fields that affect the BEM solve are included.
    - Any serialisation failure falls back to the identity-based key.
    """
    try:
        payload = {
            "domain": getattr(spec, "domain", None),
            "BCs": getattr(spec, "BCs", None),
            "conductors": getattr(spec, "conductors", None),
            "dielectrics": getattr(spec, "dielectrics", None),
            "charges": getattr(spec, "charges", None),
            "symmetry": getattr(spec, "symmetry", None),
            "queries": getattr(spec, "queries", None),
            "symbols": getattr(spec, "symbols", None),
        }

        def _json_default(obj: Any) -> Any:
            try:
                # Handle numpy-style scalars and similar.
                return float(obj)  # type: ignore[arg-type]
            except Exception:
                return str(obj)

        blob = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        )
        return "h:" + hashlib.sha1(blob.encode("utf-8")).hexdigest()
    except Exception:
        # Defensive fallback: never let hashing break the cache.
        return _spec_identity_key(spec)


def _oracle_cache_key(
    spec: CanonicalSpec,
    supervision_mode: str,
    bem_cfg: Dict[str, Any],
) -> Tuple[str, str, Tuple[Tuple[str, str], ...]]:
    """Build a composite cache key for the oracle solution.

    Strategy for the spec component is controlled by the
    EDE_COLL_ORACLE_CACHE_MODE environment variable:

    - "identity" (default): use Python object identity (id(spec)).
    - "hash": use a content hash via :func:`_spec_hash_key`.

    If you mutate CanonicalSpec instances after calling this function,
    avoid the "hash" strategy or rebuild the spec once you change it.
    """
    mode_str = str(supervision_mode)

    strategy = os.getenv("EDE_COLL_ORACLE_CACHE_MODE", "identity").lower()
    if strategy == "hash":
        spec_key = _spec_hash_key(spec)
    else:
        spec_key = _spec_identity_key(spec)

    cfg_key = _bem_cfg_cache_key(bem_cfg)
    return spec_key, mode_str, cfg_key


_VALID_SUPERVISION_MODES = {"auto", "analytic", "bem", "neural"}


def _normalize_supervision_mode(mode: str) -> str:
    """
    Clamp supervision_mode to the supported set.

    Any unknown value defaults to "auto" so the caller still receives a
    valid batch; we emit a warning to surface the misconfiguration.
    """
    mode_str = str(mode).strip().lower()
    if mode_str in _VALID_SUPERVISION_MODES:
        return mode_str
    logger.warning(
        "Unknown supervision_mode; defaulting to auto.",
        requested_mode=str(mode),
    )
    return "auto"


def _get_spherefno_surrogate(
    device: torch.device, dtype: torch.dtype
) -> Optional[SphereFNOSurrogate]:
    """
    Load a cached SphereFNO surrogate (if configured).

    The loader is driven by ``EDE_SPHEREFNO_CKPT`` and falls back to None when
    no checkpoint is provided or validation fails.
    """
    ckpt = os.getenv("EDE_SPHEREFNO_CKPT", "").strip()
    if not ckpt:
        return None
    key = (ckpt, str(device), str(dtype))
    surrogate = _NEURAL_SURROGATE_CACHE.get(key)
    if surrogate is not None:
        return surrogate
    surrogate = load_spherefno_from_env(
        device=device,
        dtype=dtype,
        l2_tol=_SPHEREFNO_L2_TOL,
        linf_tol=_SPHEREFNO_LINF_TOL,
    )
    if surrogate is not None:
        _NEURAL_SURROGATE_CACHE[key] = surrogate
    return surrogate


def _domain_valid_mask(
    spec: CanonicalSpec,
    geom_type: str,
    X: torch.Tensor,
) -> torch.Tensor:
    """
    Mask out collocation points that lie outside the domain where our
    analytic shortcuts are defined.

    This keeps analytic-vs-BEM comparisons focused on physically valid
    regions (e.g., z >= plane height for the half-space image solution,
    or outside a grounded sphere when the source charge is external).
    """
    mask = torch.ones(X.shape[0], device=X.device, dtype=torch.bool)

    try:
        geom = geom_type or _infer_geom_type_from_spec(spec)
    except Exception:
        geom = geom_type

    # Grounded plane: analytic image construction assumes the conductor
    # occupies the lower half-space, so only z >= plane_z is valid.
    if geom == "plane":
        try:
            z_plane = float((spec.conductors or [{}])[0].get("z", 0.0))
        except Exception:
            return mask
        pad = 1e-9
        return mask & (X[:, 2] >= (z_plane - pad))

    # Parallel planes: restrict to the slab between the two grounded planes.
    if geom == "parallel_planes":
        try:
            z_vals = [
                float(c.get("z", 0.0))
                for c in (spec.conductors or [])
                if c.get("type") == "plane"
            ]
        except Exception:
            z_vals = []
        if len(z_vals) >= 2:
            z_min, z_max = min(z_vals), max(z_vals)
            pad = 1e-9
            return mask & (X[:, 2] >= (z_min - pad)) & (X[:, 2] <= (z_max + pad))
        return mask

    # Grounded sphere: if the source charge is outside, valid domain is the
    # exterior; if inside, valid domain is the interior cavity.
    if geom == "sphere":
        try:
            c = (spec.conductors or [{}])[0]
            center_np = np.array(c.get("center", [0.0, 0.0, 0.0]), dtype=float)
            center_t = torch.as_tensor(center_np, device=X.device, dtype=X.dtype)
            radius = float(c.get("radius", 0.0))
        except Exception:
            return mask
        if radius <= 0.0:
            return mask

        charge_r = None
        for ch in spec.charges or []:
            if ch.get("type") != "point":
                continue
            try:
                pos = np.array(ch.get("pos") or ch.get("position"), dtype=float)
                charge_r = float(np.linalg.norm(pos - center_np))
                break
            except Exception:
                continue

        r = torch.linalg.norm(X - center_t, dim=1)
        pad = 1e-9
        if charge_r is not None and charge_r < radius:
            return mask & (r <= (radius + pad))
        return mask & (r >= (radius - pad))

    return mask


def _validate_neural_predictions(
    spec: CanonicalSpec,
    geom_type: str,
    points_np: np.ndarray,
    preds: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    rng: np.random.Generator,
) -> Tuple[bool, Optional[float], Optional[float], Optional[OracleSolution]]:
    """
    Compare neural predictions against an oracle on a small subset.

    Returns (ok, rel_l2, rel_linf, oracle_solution_used).
    """
    if _SPHEREFNO_VAL_SAMPLES <= 0 or preds.numel() == 0:
        return True, None, None, None

    # Prefer the analytic shortcut; optionally allow BEM.
    oracle: Optional[OracleSolution] = None
    try:
        oracle = _solve_analytic(spec)
    except Exception:
        oracle = None

    if (
        oracle is None
        and _SPHEREFNO_VALIDATE_WITH_BEM
        and BEM_AVAILABLE
    ):
        try:
            oracle = get_oracle_solution(spec, "bem", {})
        except Exception:
            oracle = None

    if oracle is None:
        return True, None, None, None

    total = points_np.shape[0]
    if total == 0:
        return True, None, None, oracle
    k = min(total, _SPHEREFNO_VAL_SAMPLES)
    idx = rng.choice(total, size=k, replace=False)
    preds_sub = preds[idx].to(device=device, dtype=dtype)
    target = _evaluate_oracle_on_points(
        spec, oracle, geom_type, points_np[idx], device=device, dtype=dtype
    )
    diff = preds_sub - target
    rel_l2 = float(
        (torch.linalg.norm(diff) / torch.linalg.norm(target).clamp_min(1e-12)).item()
    )
    rel_linf = float(
        (
            torch.max(torch.abs(diff))
            / torch.max(torch.abs(target)).clamp_min(1e-12)
        ).item()
    )
    ok = rel_l2 <= _SPHEREFNO_L2_TOL and rel_linf <= _SPHEREFNO_LINF_TOL
    return ok, rel_l2, rel_linf, oracle


def _evaluate_neural_oracle(
    spec: CanonicalSpec,
    geom_type: str,
    points_np: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    rng: np.random.Generator,
) -> Tuple[Optional[torch.Tensor], Optional[OracleSolution], Dict[str, Any]]:
    """
    Try the SphereFNO surrogate; return (preds, oracle_for_fallback, info).
    """
    info: Dict[str, Any] = {
        "used": False,
        "reason": None,
        "rel_l2": None,
        "rel_linf": None,
    }

    params = extract_stage0_sphere_params(spec)
    if params is None:
        info["reason"] = "spec_incompatible"
        return None, None, info

    surrogate = _get_spherefno_surrogate(device, dtype)
    if surrogate is None or not surrogate.is_ready():
        info["reason"] = "surrogate_unavailable"
        return None, None, info

    try:
        q, z0, a, center = params
        pts_t = torch.from_numpy(points_np).to(device=device, dtype=dtype)
        preds = surrogate.evaluate_points((q, z0, a), pts_t, center=center)
    except Exception as exc:  # pragma: no cover - defensive path
        info["reason"] = f"surrogate_error:{exc}"
        return None, None, info

    ok, rel_l2, rel_linf, oracle = _validate_neural_predictions(
        spec, geom_type, points_np, preds, device, dtype, rng
    )
    info["rel_l2"] = rel_l2
    info["rel_linf"] = rel_linf
    if not ok:
        info["reason"] = "validation_failed"
        return None, oracle, info

    info["used"] = True
    return preds, oracle, info


def _empty_batch(device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """Return an all-empty collocation batch on the requested device/dtype."""
    return {
        "X": torch.zeros(0, 3, device=device, dtype=dtype),
        "V_gt": torch.zeros(0, device=device, dtype=dtype),
        "is_boundary": torch.zeros(0, device=device, dtype=torch.bool),
        "mask_finite": torch.zeros(0, device=device, dtype=torch.bool),
        "encoding": torch.zeros(0, ENCODING_DIM, device=device, dtype=dtype),
        "bbox_center": torch.zeros(0, 3, device=device, dtype=dtype),
        "bbox_extent": torch.zeros(0, device=device, dtype=dtype),
    }


def _infer_geom_type_from_spec(spec: CanonicalSpec) -> str:
    """
    Heuristic geometry label used for the fast analytic paths.

    This roughly matches the geometry heuristics used for encoding and
    curriculum synthesis.
    """
    ctypes = (
        sorted({c.get("type") for c in spec.conductors})
        if spec.conductors
        else []
    )
    dielectrics = getattr(spec, "dielectrics", None) or []
    if dielectrics:
        return "layered_planar"
    if ctypes == ["plane"]:
        if len(spec.conductors) == 1:
            return "plane"
        if len(spec.conductors) == 2:
            return "parallel_planes"
    if ctypes == ["sphere"] and len(spec.conductors) == 1:
        return "sphere"
    if ("cylinder" in ctypes or "cylinder2D" in ctypes) and len(spec.conductors) == 1:
        return "cylinder2D"
    if ("torus" in ctypes or "toroid" in ctypes) and len(spec.conductors) == 1:
        return "torus"
    return "unknown"


def _infer_bbox_for_spec(spec: CanonicalSpec) -> float:
    """
    Heuristic bounding-box extent for collocation sampling.

    For planes we mirror the finite BEM plane patch extent used in
    electrodrive.core.bem_mesh.generate_mesh (L = max(8, 16 * dmin)).
    For spheres / cylinders we use a modest multiple of the radius and
    fall back to a global default otherwise.
    """
    bbox = 10.0

    conductors = getattr(spec, "conductors", None) or []
    charges = getattr(spec, "charges", None) or []
    dielectrics = getattr(spec, "dielectrics", None) or []

    if dielectrics:
        # Use layer extents when available.
        z_vals = []
        for d in dielectrics:
            z_min = d.get("z_min", None)
            z_max = d.get("z_max", None)
            try:
                if z_min is not None:
                    z_vals.append(float(z_min))
            except Exception:
                pass
            try:
                if z_max is not None:
                    z_vals.append(float(z_max))
            except Exception:
                pass
        if z_vals:
            z_span = max(z_vals) - min(z_vals)
            bbox = max(4.0, 4.0 * z_span)
        # Ensure charges stay inside the bbox span.
        for ch in charges:
            if ch.get("type") != "point":
                continue
            try:
                zq = float(ch["pos"][2])
            except Exception:
                continue
            bbox = max(bbox, 4.0 * abs(zq) + 2.0)
        return bbox

    if not conductors:
        return bbox

    for c in conductors:
        ctype = c.get("type")
        if ctype == "plane":
            z = float(c.get("z", 0.0))
            dmin = 0.3
            for ch in charges:
                if ch.get("type") != "point":
                    continue
                try:
                    zq = float(ch["pos"][2])
                except Exception:
                    continue
                dmin = max(dmin, abs(zq - z))
            L = max(8.0, 16.0 * dmin)
            return float(L)

        if ctype == "sphere":
            try:
                r = float(c.get("radius", 1.0))
            except Exception:
                continue
            return float(4.0 * r)

        if ctype in ("cylinder", "cylinder2D"):
            try:
                r = float(c.get("radius", 1.0))
            except Exception:
                continue
            return float(4.0 * r)

        if ctype in ("torus", "toroid"):
            try:
                R = float(c.get("major_radius", c.get("radius", 1.0)))
                a = float(c.get("minor_radius", 0.25 * R))
            except Exception:
                continue
            return float(4.0 * (R + a))

    return bbox


def _infer_bbox_center_for_spec(spec: CanonicalSpec) -> np.ndarray:
    """
    Heuristic centre for the collocation bounding box.

    Planar geometries stay anchored at the origin to match the finite
    patch assumption baked into the analytic shortcuts and BEM meshes.
    For translated compact geometries (sphere / cylinder / torus) we
    centre the box on the conductor; otherwise we fall back to the mean
    charge position or the origin.
    """
    try:
        geom = _infer_geom_type_from_spec(spec)
    except Exception:
        geom = "unknown"

    if geom in ("plane", "parallel_planes", "layered_planar"):
        return np.zeros(3, dtype=float)

    conductors = getattr(spec, "conductors", None) or []
    centers: List[np.ndarray] = []
    for c in conductors:
        ctype = c.get("type")
        if ctype in ("sphere", "cylinder", "cylinder2D", "torus", "toroid"):
            c_center = c.get("center")
            if c_center is not None:
                try:
                    centers.append(np.array(c_center, dtype=float))
                except Exception:
                    continue
    if centers:
        return np.mean(np.stack(centers, axis=0), axis=0)

    charges = getattr(spec, "charges", None) or []
    charge_positions: List[np.ndarray] = []
    for ch in charges:
        ctype = ch.get("type")
        if ctype == "point":
            pos = ch.get("pos") or ch.get("position")
            if pos is not None:
                try:
                    charge_positions.append(np.array(pos, dtype=float))
                except Exception:
                    continue
        elif ctype in ("line", "line_charge"):
            pos_2d = ch.get("pos_2d")
            if pos_2d is not None:
                try:
                    p = np.array(pos_2d, dtype=float)
                    charge_positions.append(np.array([p[0], p[1], 0.0], dtype=float))
                except Exception:
                    continue
    if charge_positions:
        return np.mean(np.stack(charge_positions, axis=0), axis=0)

    return np.zeros(3, dtype=float)


def _sample_points_for_spec(
    spec: CanonicalSpec,
    n_points: int,
    ratio_boundary: float,
    rng: np.random.Generator,
    *,
    bbox: Optional[float] = None,
    bbox_center: Optional[np.ndarray] = None,
    sphere_shell_frac: Optional[float] = None,
) -> Tuple[np.ndarray, torch.Tensor, np.ndarray, float]:
    """
    Sample interior + boundary points for a given spec.

    Returns both the point cloud and the bounding-box metadata used for
    sampling so downstream callers can log or reuse them. The bounding
    box extent/centre can be overridden explicitly; ``sphere_shell_frac``
    controls how far off the surface Stage-0 sphere boundary points are
    pushed.
    """
    N = int(n_points)
    N_boundary = int(N * float(ratio_boundary))
    N_interior = N - N_boundary
    if bbox is None:
        bbox = _infer_bbox_for_spec(spec)
    if bbox_center is None:
        bbox_center = _infer_bbox_center_for_spec(spec)
    if sphere_shell_frac is None:
        sphere_shell_frac = _SPHERE_SHELL_FRAC
    else:
        try:
            sphere_shell_frac = max(0.0, float(sphere_shell_frac))
        except Exception:
            sphere_shell_frac = _SPHERE_SHELL_FRAC
    bbox_center_np = np.array(bbox_center, dtype=float).reshape(3)

    dielectrics = getattr(spec, "dielectrics", None) or []

    # Interior: uniform box sampling
    points_int = rng.uniform(-bbox / 2.0, bbox / 2.0, (N_interior, 3))

    # For finite plane patches (single plane or parallel planes), restrict
    # interior sampling in x/y (and now z) to a central band so that the
    # finite BEM geometry is a good approximation to the infinite-plane
    # analytic shortcuts used in the image-charge formulas.
    try:
        geom = _infer_geom_type_from_spec(spec)
    except Exception:
        geom = "unknown"

    if geom in ("plane", "parallel_planes"):
        # Restrict x/y to the central region of the patch to stay away from
        # finite patch edges.
        alpha = 0.15
        points_int[:, 0] *= alpha
        points_int[:, 1] *= alpha

        # Additionally, restrict z to a band around the planes and any point
        # charges, rather than the full [-bbox/2, +bbox/2] extent. This avoids
        # sampling deep far-field regions where a finite patch deviates most
        # from the infinite-plane analytic model.
        conductors = getattr(spec, "conductors", None) or []
        charges = getattr(spec, "charges", None) or []
        z_vals: List[float] = []

        for c in conductors:
            if c.get("type") == "plane":
                try:
                    zc = float(c.get("z", 0.0))
                    z_vals.append(zc)
                except Exception:
                    continue

        for ch in charges:
            if ch.get("type") == "point":
                try:
                    pos = ch.get("pos") or ch.get("position")
                    if pos is not None:
                        zq = float(pos[2])
                        z_vals.append(zq)
                except Exception:
                    continue

        if z_vals and N_interior > 0:
            z_min = min(z_vals)
            z_max = max(z_vals)
            z_center = 0.5 * (z_min + z_max)
            # Half-span at least 0.3, plus a modest margin beyond the
            # extremal conductor/charge positions.
            half_span = max(0.3, 0.5 * (z_max - z_min))
            z_lo = z_center - 2.0 * half_span
            z_hi = z_center + 2.0 * half_span
            # Clamp to the original bbox in case of pathological specs.
            z_lo = max(z_lo, -bbox / 2.0)
            z_hi = min(z_hi, bbox / 2.0)
            # Avoid sampling interior points extremely close to the planes,
            # since boundary samples already capture the Dirichlet behaviour.
            margin = 0.2 * (z_max - z_min)
            z_lo = max(z_lo, z_min + margin)
            z_hi = min(z_hi, z_max - margin)
            if z_hi <= z_lo:
                # Fallback: use a central band of the original box.
                z_lo = -0.25 * bbox
                z_hi = 0.25 * bbox
            points_int[:, 2] = rng.uniform(z_lo, z_hi, size=N_interior)
    elif geom == "layered_planar":
        # Keep x/y moderate; focus z sampling around layer extents and charge heights.
        alpha = 0.25
        points_int[:, 0] *= alpha
        points_int[:, 1] *= alpha
        z_vals: List[float] = []
        for d in dielectrics:
            for key in ("z_min", "z_max"):
                val = d.get(key, None)
                if val is None:
                    continue
                try:
                    z_vals.append(float(val))
                except Exception:
                    continue
        for ch in getattr(spec, "charges", None) or []:
            if ch.get("type") == "point":
                try:
                    pos = ch.get("pos") or ch.get("position")
                    if pos is not None:
                        z_vals.append(float(pos[2]))
                except Exception:
                    continue
        if z_vals and N_interior > 0:
            z_min = max(0.0, min(z_vals))
            z_max = max(z_vals)
            pad = 0.25 * (z_max - z_min + 1e-6)
            z_lo = max(0.0, z_min - pad)
            z_hi = z_max + pad
            points_int[:, 2] = rng.uniform(z_lo, z_hi, size=N_interior)

    if N_interior > 0:
        points_int += bbox_center_np

    # Boundary: per-conductor sampling, reusing the same shape heuristics
    points_bnd_list: List[np.ndarray] = []
    conductors = getattr(spec, "conductors", None) or []
    if N_boundary > 0 and (conductors or dielectrics):
        targets = conductors if conductors else dielectrics
        N_per = max(1, N_boundary // len(targets))
        for c in targets:
            ctype = c.get("type") if conductors else "dielectric_interface"
            if ctype == "plane":
                z = float(c.get("z", 0.0))
                # For plane/parallel_planes, restrict boundary sampling in x/y
                # to the same central region we use for interior points. This
                # avoids over-sampling near the patch edges where finite-patch
                # BEM and infinite-plane analytic shortcuts differ most.
                if geom in ("plane", "parallel_planes"):
                    alpha = 0.15
                    span = (bbox / 2.0) * alpha
                else:
                    span = bbox / 2.0
                xy = rng.uniform(-span, span, (N_per, 2))
                points_bnd_list.append(
                    np.c_[
                        xy,
                        np.full(N_per, z, dtype=float),
                    ]
                )
            elif ctype == "sphere":
                r = float(c.get("radius", 1.0))
                center = np.array(
                    c.get("center", bbox_center_np),
                    dtype=float,
                )
                vec = rng.standard_normal((N_per, 3))
                vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
                # Sample slightly off the surface; keeps Stage-0 sphere shell convention.
                shell = float(r * (1.0 + sphere_shell_frac))
                # Nudge outward to avoid falling just inside due to fp32 noise.
                shell += max(1e-6 * r, 1e-9)
                points_bnd_list.append(vec * shell + center)
            elif ctype in ("cylinder", "cylinder2D"):
                r = float(c.get("radius", 1.0))
                center = np.array(
                    c.get("center", bbox_center_np),
                    dtype=float,
                )
                phi = rng.uniform(0.0, 2.0 * math.pi, N_per)
                z = rng.uniform(-bbox / 2.0, bbox / 2.0, N_per)
                points_bnd_list.append(
                    np.stack(
                        [
                            r * np.cos(phi),
                            r * np.sin(phi),
                            z,
                        ],
                        axis=1,
                    )
                    + center
                )
            elif ctype in ("torus", "toroid"):
                R = float(c.get("major_radius", c.get("radius", 1.0)))
                a = float(c.get("minor_radius", 0.25 * R))
                center = np.array(
                    c.get("center", bbox_center_np),
                    dtype=float,
                )
                u = rng.uniform(0.0, 2.0 * math.pi, N_per)
                v = rng.uniform(0.0, 2.0 * math.pi, N_per)
                cosu = np.cos(u)
                sinu = np.sin(u)
                cosv = np.cos(v)
                sinv = np.sin(v)
                x = (R + a * cosv) * cosu
                y = (R + a * cosv) * sinu
                z = a * sinv
                pts = np.stack([x, y, z], axis=1) + center
                # Slightly push outward along approximate normal to avoid duplicate surface points.
                eps = 1e-3 * max(a, R)
                nx = cosv * cosu
                ny = cosv * sinu
                nz = sinv
                pts += np.stack([nx, ny, nz], axis=1) * eps
                points_bnd_list.append(pts)
            elif ctype == "dielectric_interface":
                z_candidates: List[float] = []
                for key in ("z_min", "z_max"):
                    val = c.get(key, None)
                    if val is None:
                        continue
                    try:
                        zv = float(val)
                        if zv >= -1e-6:  # only upper interfaces for region-1 oracle
                            z_candidates.append(zv)
                    except Exception:
                        continue
                for z in z_candidates:
                    span = bbox / 4.0
                    xy = rng.uniform(-span, span, (max(1, N_per // 2), 2))
                    points_bnd_list.append(
                        np.c_[
                            xy,
                            np.full(max(1, N_per // 2), z, dtype=float),
                        ]
                    )

    if points_bnd_list:
        points_bnd = np.vstack(points_bnd_list)
    else:
        points_bnd = np.empty((0, 3), dtype=float)

    N_bnd = points_bnd.shape[0]
    if N_bnd > 0:
        points_np = np.vstack([points_int, points_bnd])
    else:
        points_np = points_int

    N_total = points_np.shape[0]
    is_boundary = torch.zeros(N_total, dtype=torch.bool)
    if N_bnd > 0:
        is_boundary[-N_bnd:] = True
    return points_np, is_boundary, bbox_center_np, float(bbox)


def _build_collocation_point_cloud(
    spec: CanonicalSpec,
    n_points: int,
    ratio_boundary: float,
    device: torch.device,
    dtype: torch.dtype,
    rng: np.random.Generator,
    *,
    bbox: Optional[float] = None,
    bbox_center: Optional[np.ndarray] = None,
    sphere_shell_frac: Optional[float] = None,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample collocation candidates and package device/dtype-aligned tensors.

    Returns (points_np, X, is_boundary, bbox_center_batch, bbox_extent_batch).
    """
    points_np, is_boundary_mask, bbox_center_np, bbox_extent = _sample_points_for_spec(
        spec,
        n_points=n_points,
        ratio_boundary=ratio_boundary,
        rng=rng,
        bbox=bbox,
        bbox_center=bbox_center,
        sphere_shell_frac=sphere_shell_frac,
    )

    X = torch.from_numpy(points_np).to(device=device, dtype=dtype)
    is_boundary = is_boundary_mask.to(device=device)

    bbox_center_t = torch.as_tensor(bbox_center_np, device=device, dtype=dtype)
    if bbox_center_t.dim() == 1:
        bbox_center_t = bbox_center_t.unsqueeze(0)
    bbox_center_batch = bbox_center_t.repeat(X.shape[0], 1)
    bbox_extent_batch = torch.full(
        (X.shape[0],), float(bbox_extent), device=device, dtype=dtype
    )

    return points_np, X, is_boundary, bbox_center_batch, bbox_extent_batch


def build_collocation_candidate_pool(
    spec: CanonicalSpec,
    n_points: int,
    ratio_boundary: float,
    *,
    rng: Optional[np.random.Generator] = None,
    device: torch.device | str = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    bbox: Optional[float] = None,
    bbox_center: Optional[np.ndarray] = None,
    sphere_shell_frac: Optional[float] = None,
    include_numpy: bool = False,
) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
    """
    Build a single-round collocation candidate pool without oracle evaluation.

    This is intended for lightweight residual-based selection in adaptive
    pipelines. It reuses the same geometry-aware sampling as
    :func:`make_collocation_batch_for_spec` but omits target evaluation.
    When ``include_numpy`` is True, the output also contains numpy views of
    the sampled points and masks for callers that want CPU-side residual
    scoring.
    """
    dev = torch.device(device)
    rng = rng if rng is not None else np.random.default_rng()

    points_np, X, is_boundary, bbox_center_batch, bbox_extent_batch = _build_collocation_point_cloud(
        spec,
        n_points=n_points,
        ratio_boundary=ratio_boundary,
        device=dev,
        dtype=dtype,
        rng=rng,
        bbox=bbox,
        bbox_center=bbox_center,
        sphere_shell_frac=sphere_shell_frac,
    )

    out: Dict[str, Union[torch.Tensor, np.ndarray]] = {
        "X": X,
        "is_boundary": is_boundary,
        "bbox_center": bbox_center_batch,
        "bbox_extent": bbox_extent_batch,
    }
    if include_numpy:
        out["points_np"] = points_np
        out["is_boundary_np"] = is_boundary.detach().cpu().numpy()
        if bbox_center_batch.numel() > 0:
            out["bbox_center_np"] = (
                bbox_center_batch[0].detach().cpu().numpy().reshape(3)
            )
        else:
            out["bbox_center_np"] = np.zeros(3, dtype=float)
        if bbox_extent_batch.numel() > 0:
            out["bbox_extent_scalar"] = float(bbox_extent_batch[0].item())
        else:
            out["bbox_extent_scalar"] = float("nan")

    return out


def _evaluate_oracle_on_points(
    spec: CanonicalSpec,
    solution: OracleSolution,
    geom_type: str,
    points_np: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Evaluate an oracle (analytic or BEM) on a batch of 3D points.

    The analytic path keeps the optimised vectorised implementations for
    the common curriculum geometries (plane, sphere, cylinder2D) and falls
    back to AnalyticSolution.eval otherwise.  The BEM path uses
    BEMSolution.eval_V_E_batched when available.

    Note
    ----
    Analytic and BEM oracles both return SI potentials so they share a
    consistent scale with the image basis (which uses K_E). Earlier
    revisions multiplied analytic outputs by ε₀, shrinking them by ~1e-11
    and suppressing discovered image strengths for the plane + point case.
    """
    # Analytic solution branch ------------------------------------------------
    if isinstance(solution, AnalyticSolution):
        fast_done = False
        V_gt: Optional[torch.Tensor] = None

        try:
            # 1) Grounded plane at z=0 with one point charge
            if (
                geom_type == "plane"
                and len(spec.conductors) == 1
                and len(spec.charges) == 1
                and spec.charges[0]["type"] == "point"
            ):
                c = spec.conductors[0]
                if (
                    c.get("type") == "plane"
                    and abs(c.get("z", 0.0)) < 1e-12
                    and c.get("potential", 0.0) == 0.0
                ):
                    q = float(spec.charges[0]["q"])
                    x0, y0, z0 = map(float, spec.charges[0]["pos"])
                    R = points_np - np.array([x0, y0, z0], dtype=np.float64)
                    Rm = points_np - np.array([x0, y0, -z0], dtype=np.float64)
                    r = np.linalg.norm(R, axis=1).clip(1e-9, None)
                    rm = np.linalg.norm(Rm, axis=1).clip(1e-9, None)
                    V_gt_np = K_E * (q / r - q / rm)
                    V_gt = torch.from_numpy(V_gt_np).to(device=device, dtype=dtype)
                    fast_done = True

            # 2) Grounded sphere (centered or shifted) with one point charge
            if (
                not fast_done
                and geom_type == "sphere"
                and len(spec.conductors) == 1
                and len(spec.charges) == 1
                and spec.charges[0]["type"] == "point"
            ):
                c = spec.conductors[0]
                if c.get("type") == "sphere" and c.get("potential", 0.0) == 0.0:
                    a = float(c.get("radius"))
                    cx, cy, cz = map(float, c.get("center", [0.0, 0.0, 0.0]))
                    q = float(spec.charges[0]["q"])
                    x0, y0, z0 = map(float, spec.charges[0]["pos"])
                    r0c = np.array([x0 - cx, y0 - cy, z0 - cz], dtype=np.float64)
                    r0_norm = float(np.linalg.norm(r0c)) + 1e-12
                    q_img = -(a / r0_norm) * q
                    r_img = (a * a / (r0_norm * r0_norm)) * r0c
                    r_img_world = np.array([cx, cy, cz], dtype=np.float64) + r_img
                    R = points_np - np.array([x0, y0, z0], dtype=np.float64)
                    Ri = points_np - r_img_world
                    r = np.linalg.norm(R, axis=1).clip(1e-9, None)
                    ri = np.linalg.norm(Ri, axis=1).clip(1e-9, None)
                    V_gt_np = K_E * (q / r + q_img / ri)
                    V_gt = torch.from_numpy(V_gt_np).to(device=device, dtype=dtype)
                    fast_done = True

            # 2b) Two grounded parallel planes + single point charge.
            if (
                not fast_done
                and geom_type == "parallel_planes"
                and len(spec.conductors) == 2
                and len(spec.charges) == 1
                and spec.charges[0]["type"] == "point"
            ):
                try:
                    z_vals = [
                        float(c.get("z", 0.0))
                        for c in (spec.conductors or [])
                        if c.get("type") == "plane"
                    ]
                    if len(z_vals) == 2 and abs(z_vals[0] + z_vals[1]) < 1e-9:
                        d = abs(z_vals[0])
                        q = float(spec.charges[0]["q"])
                        x0, y0, z0 = map(float, spec.charges[0]["pos"])
                        # Generous image lattice for convergence near the planes.
                        n_terms_env = os.getenv("EDE_PARALLEL_PLANES_N_TERMS", "")
                        try:
                            n_terms = max(1000, int(n_terms_env)) if n_terms_env else 100000
                        except Exception:
                            n_terms = 100000
                        n = np.arange(-n_terms, n_terms + 1, dtype=np.float64)
                        sign = np.where((n % 2) == 0, 1.0, -1.0)
                        z_img = 2.0 * n * d + sign * z0  # (M,)
                        q_img = sign * q

                        dx = points_np[:, 0:1] - x0
                        dy = points_np[:, 1:2] - y0
                        dz = points_np[:, 2:3] - z_img[None, :]
                        r = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-18
                        V_gt_np = K_E * np.sum(q_img[None, :] / r, axis=1)
                        # Simple alternating-series remainder estimate using the
                        # first omitted image pair.
                        n_tail = n_terms + 1
                        sign_tail = -1.0 if (n_tail % 2) else 1.0
                        z_tail = 2.0 * n_tail * d + sign_tail * z0
                        dz_tail = points_np[:, 2] - z_tail
                        r_tail = np.sqrt((points_np[:, 0] - x0) ** 2 + (points_np[:, 1] - y0) ** 2 + dz_tail * dz_tail) + 1e-18
                        V_gt_np += K_E * (2.0 * sign_tail * q) / r_tail
                        V_gt = torch.from_numpy(V_gt_np).to(device=device, dtype=dtype)
                        fast_done = True
                except Exception:
                    fast_done = False

            if (
                not fast_done
                and geom_type == "layered_planar"
                and isinstance(solution.meta, dict)
                and solution.meta.get("kind") == "planar_three_layer"
            ):
                cfg = ThreeLayerConfig(
                    eps1=float(solution.meta.get("eps1", 1.0)),
                    eps2=float(solution.meta.get("eps2", 1.0)),
                    eps3=float(solution.meta.get("eps3", 1.0)),
                    h=float(solution.meta.get("h", 1.0)),
                    q=float(solution.meta.get("q", 1.0)),
                    r0=tuple(solution.meta.get("r0", (0.0, 0.0, 1.0))),
                    n_k=int(solution.meta.get("n_k", 256) or 256),
                    k_max=solution.meta.get("k_max", None),
                )
                pts = torch.from_numpy(points_np).to(device=device, dtype=dtype)
                V_gt = potential_three_layer_region1(pts, cfg, device=device, dtype=dtype)
                fast_done = True

            # 3) Grounded cylinder (infinite) with 2D line charge
            #    V_SI ∝ K_E * λ * ln(ρ / ρ'), scaled consistently with other paths.
            if (
                not fast_done
                and geom_type in ("cylinder2D", "cylinder")
                and len(spec.conductors) == 1
                and len(spec.charges) == 1
                and spec.charges[0]["type"] in ("line_charge", "line")
            ):
                c = spec.conductors[0]
                if c.get("type") in ("cylinder", "cylinder2D") and c.get(
                    "potential", 0.0
                ) == 0.0:
                    a = float(c.get("radius"))
                    lam = float(spec.charges[0].get("lambda"))
                    x0, y0 = map(float, spec.charges[0].get("pos_2d"))
                    rho0 = math.hypot(x0, y0) + 1e-12
                    xi = (a * a / (rho0 * rho0)) * x0
                    yi = (a * a / (rho0 * rho0)) * y0
                    dx = points_np[:, 0] - x0
                    dy = points_np[:, 1] - y0
                    dxi = points_np[:, 0] - xi
                    dyi = points_np[:, 1] - yi
                    rho = np.sqrt(dx * dx + dy * dy).clip(1e-12, None)
                    rhoi = np.sqrt(dxi * dxi + dyi * dyi).clip(1e-12, None)
                    V_gt_np = (K_E * lam) * np.log(rho / rhoi)
                    V_gt = torch.from_numpy(V_gt_np).to(device=device, dtype=dtype)
                    fast_done = True
        except Exception:
            # Any failure here just drops us back to the slower scalar path.
            fast_done = False
            V_gt = None

        if not fast_done or V_gt is None:
            # `solution.eval` returns SI potentials that already include K_E.
            V_gt_np = np.array(
                [solution.eval(tuple(p)) for p in points_np],
                dtype=np.float64,
            )
            V_gt = torch.from_numpy(V_gt_np).to(device=device, dtype=dtype)

        return V_gt

    # BEM solution branch -----------------------------------------------------
    if BEM_AVAILABLE and hasattr(solution, "eval_V_E_batched"):
        sol_device = getattr(solution, "_device", "cpu")
        sol_dtype = getattr(solution, "_dtype", torch.float64)
        pts = torch.tensor(points_np, device=sol_device, dtype=sol_dtype)

        # Optional chunking to cap peak memory during target evaluation.
        # By default we rely on BEMSolution's internal tiling. When
        # EDE_COLL_BEM_CHUNK_SIZE is set to a positive integer, we split
        # the target set into that many-point chunks and call
        # eval_V_E_batched sequentially. This trades some wall-clock time
        # for reduced peak VRAM usage.
        chunk_env = os.getenv("EDE_COLL_BEM_CHUNK_SIZE", "").strip()
        chunk_size = 0
        if chunk_env:
            try:
                chunk_size = max(1, int(chunk_env))
            except Exception:
                chunk_size = 0

        with torch.no_grad():
            if chunk_size and pts.shape[0] > chunk_size:
                Vs: List[torch.Tensor] = []
                for start in range(0, pts.shape[0], chunk_size):
                    stop = min(start + chunk_size, pts.shape[0])
                    V_chunk, _ = solution.eval_V_E_batched(pts[start:stop])  # type: ignore[operator]
                    Vs.append(V_chunk)
                if Vs:
                    V_gt_t = torch.cat(Vs, dim=0)
                else:
                    V_gt_t = torch.empty(0, device=sol_device, dtype=sol_dtype)
            else:
                V_gt_t, _ = solution.eval_V_E_batched(pts)  # type: ignore[operator]

        return V_gt_t.detach().to(device=device, dtype=dtype).view(-1)

    # As a last resort, treat it as a scalar-eval oracle (keep whatever units
    # the underlying solution uses; this path is rarely hit in the learning
    # stack and is not part of the BEM-vs-analytic unit bridge).
    V_gt_np = np.array(
        [solution.eval(tuple(p)) for p in points_np],
        dtype=np.float64,
    )
    return torch.from_numpy(V_gt_np).to(device=device, dtype=dtype)


def make_collocation_batch_for_spec(
    spec: CanonicalSpec,
    n_points: int,
    ratio_boundary: float,
    supervision_mode: str,  # "analytic" | "bem" | "auto" | "neural"
    device: torch.device,
    dtype: torch.dtype,
    *,
    rng: Optional[np.random.Generator] = None,
    bem_oracle_config: Optional[Dict[str, Any]] = None,
    encoding: Optional[torch.Tensor] = None,
    geom_type: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build a collocation batch (points + oracle targets) for a single spec.

    Parameters
    ----------
    spec:
        CanonicalSpec describing boundary conditions and sources.
    n_points:
        Target number of collocation points (interior + boundary).  The
        actual number may differ by at most O(#conductors) due to
        per-conductor rounding, matching the historical dataset logic.
    ratio_boundary:
        Fraction of points that should lie on the boundary.  Remaining
        points are drawn from a fixed bounding box for the interior.
    supervision_mode:
        ``"analytic"``, ``"bem"``, ``"auto"`` or ``"neural"``. The neural path
        tries the SphereFNO surrogate (when configured) and falls back to
        :func:`get_oracle_solution` using ``EDE_NEURAL_FALLBACK_MODE`` when the
        surrogate is absent or fails validation.
    device, dtype:
        Desired output device and dtype for ``"X"``, ``"V_gt"`` and
        ``"encoding"`` tensors.

    Other Parameters
    ----------------
    rng:
        Optional :class:`numpy.random.Generator` used for sampling.  If
        omitted, a fresh default RNG is constructed.
    bem_oracle_config:
        Optional dict of overrides for :class:`BEMConfig`; if omitted, a
        default config is used.
    encoding:
        Optional precomputed encoding vector for ``spec``.  If omitted,
        :func:`encode_spec` is called internally.
    geom_type:
        Optional geometry label (e.g. ``"plane"`` / ``"sphere"`` /
        ``"parallel_planes"``).  If omitted, we infer a label from
        ``spec`` via :func:`_infer_geom_type_from_spec`.

    Returns
    -------
    batch : Dict[str, torch.Tensor]
        Dictionary with keys:

        - ``"X"``:          [N, 3] collocation points
        - ``"V_gt"``:       [N] oracle potential values
        - ``"is_boundary"``: [N] bool mask
        - ``"mask_finite"``: [N] bool mask of finite targets
        - ``"bbox_center"``: [N, 3] bounding-box centre used for sampling
        - ``"bbox_extent"``: [N] bounding-box edge length
        - ``"encoding"``:   [N, ENCODING_DIM] broadcast spec encoding
    """
    device = torch.device(device)
    if rng is None:
        rng = np.random.default_rng()
    if bem_oracle_config is None:
        bem_oracle_config = {}
    _ORACLE_CACHE.clear()
    _ORACLE_CACHE_ORDER.clear()
    supervision_mode = _normalize_supervision_mode(supervision_mode)

    solution: Optional[OracleSolution] = None
    cache_key: Optional[Tuple[str, str, Tuple[Tuple[str, str], ...]]] = None
    used_mode = supervision_mode

    if supervision_mode != "neural":
        # Oracle selection + cache -------------------------------------------
        cache_key = _oracle_cache_key(spec, supervision_mode, bem_oracle_config)
        solution = _ORACLE_CACHE.get(cache_key)
        if solution is None:
            solution = get_oracle_solution(spec, supervision_mode, bem_oracle_config)
            if solution is None:
                # No oracle available (e.g. BEM not installed and no analytic path).
                return _empty_batch(device=device, dtype=dtype)
            _ORACLE_CACHE[cache_key] = solution
            if _MAX_CACHED_ORACLES > 0:
                _ORACLE_CACHE_ORDER.append(cache_key)
                # Trim oldest entries beyond capacity.
                while len(_ORACLE_CACHE_ORDER) > _MAX_CACHED_ORACLES:
                    old_key = _ORACLE_CACHE_ORDER.pop(0)
                    _ORACLE_CACHE.pop(old_key, None)

    # Sample collocation points ----------------------------------------------
    points_np, X, is_boundary, bbox_center_batch, bbox_extent_batch = _build_collocation_point_cloud(
        spec,
        n_points=n_points,
        ratio_boundary=ratio_boundary,
        device=device,
        dtype=dtype,
        rng=rng,
    )
    N_total = X.shape[0]
    if N_total == 0:
        return _empty_batch(device=device, dtype=dtype)

    # Encoding: either reuse caller-provided or build from spec.
    if encoding is None:
        encoding_single = encode_spec(spec).to(device=device, dtype=dtype)
    else:
        encoding_single = encoding.to(device=device, dtype=dtype)
    if encoding_single.dim() == 1:
        encoding_single = encoding_single.unsqueeze(0)

    encoding_batch = encoding_single.repeat(N_total, 1)

    # Evaluate oracle ---------------------------------------------------------
    if geom_type is None:
        geom_type = _infer_geom_type_from_spec(spec)

    V_gt: Optional[torch.Tensor] = None

    if supervision_mode == "neural":
        V_neural, oracle_fallback, neural_info = _evaluate_neural_oracle(
            spec, geom_type, points_np, device, dtype, rng
        )
        if V_neural is not None and neural_info.get("used"):
            V_gt = V_neural.to(device=device, dtype=dtype)
            used_mode = "neural"
            if neural_info.get("rel_l2") is not None:
                logger.debug(
                    "SphereFNO surrogate validated: rel_l2=%.3e rel_linf=%.3e",
                    neural_info.get("rel_l2"),
                    neural_info.get("rel_linf"),
                )
        else:
            fallback_mode = _NEURAL_FALLBACK_MODE
            used_mode = fallback_mode
            if neural_info.get("reason"):
                if neural_info.get("reason") == "validation_failed":
                    logger.warning(
                        "SphereFNO validation failed (rel_l2=%.3e rel_linf=%.3e); fallback=%s",
                        neural_info.get("rel_l2"),
                        neural_info.get("rel_linf"),
                        fallback_mode,
                    )
                else:
                    logger.info(
                        "SphereFNO surrogate not used (%s); fallback=%s",
                        neural_info["reason"],
                        fallback_mode,
                    )
            if oracle_fallback is not None:
                solution = oracle_fallback
            else:
                cache_key_fb = _oracle_cache_key(
                    spec, fallback_mode, bem_oracle_config
                )
                solution = _ORACLE_CACHE.get(cache_key_fb)
                if solution is None:
                    solution = get_oracle_solution(
                        spec, fallback_mode, bem_oracle_config
                    )
                    if solution is None:
                        return _empty_batch(device=device, dtype=dtype)
                    _ORACLE_CACHE[cache_key_fb] = solution
                    if _MAX_CACHED_ORACLES > 0:
                        _ORACLE_CACHE_ORDER.append(cache_key_fb)
                        while len(_ORACLE_CACHE_ORDER) > _MAX_CACHED_ORACLES:
                            old_key = _ORACLE_CACHE_ORDER.pop(0)
                            _ORACLE_CACHE.pop(old_key, None)

            V_gt = _evaluate_oracle_on_points(
                spec,
                solution,
                geom_type,
                points_np,
                device=device,
                dtype=dtype,
            )
    else:
        V_gt = _evaluate_oracle_on_points(
            spec,
            solution,
            geom_type,
            points_np,
            device=device,
            dtype=dtype,
        )

    # For Dirichlet problems, enforce the prescribed boundary potential at
    # sampled boundary points when using the BEM oracle. This guards against
    # small evaluation drift near the surface dominating relative-error checks.
    if supervision_mode == "bem" and bool(is_boundary.any()):
        try:
            for c in spec.conductors or []:
                ctype = c.get("type")
                bc_val = float(c.get("potential", 0.0))
                if ctype == "plane":
                    z0 = float(c.get("z", 0.0))
                    mask_bc = is_boundary & (torch.abs(X[:, 2] - z0) < 1e-12)
                    if mask_bc.any():
                        V_gt = torch.where(
                            mask_bc,
                            torch.as_tensor(bc_val, device=device, dtype=dtype),
                            V_gt,
                        )
                elif ctype == "sphere":
                    center = torch.as_tensor(
                        c.get("center", [0.0, 0.0, 0.0]),
                        device=device,
                        dtype=dtype,
                    ).view(1, 3)
                    radius = float(c.get("radius", 0.0))
                    if radius > 0.0:
                        r = torch.linalg.norm(X - center, dim=1)
                        tol = 1e-9 * max(1.0, radius)
                        mask_bc = is_boundary & (torch.abs(r - radius) < tol)
                        if mask_bc.any():
                            V_gt = torch.where(
                                mask_bc,
                                torch.as_tensor(bc_val, device=device, dtype=dtype),
                                V_gt,
                            )
                elif ctype == "parallel_planes":
                    # Handled as two planes; nothing to do here explicitly.
                    continue
        except Exception:
            pass

    if used_mode != "neural":
        domain_mask = _domain_valid_mask(spec, geom_type, X)
        nan_val = torch.tensor(float("nan"), device=device, dtype=dtype)
        V_gt = torch.where(domain_mask, V_gt, nan_val)
    mask_finite = torch.isfinite(V_gt)
    if not bool(mask_finite.all()):
        # Replace any invalid entries with zeros while keeping the finite mask
        # to signal callers that these points were outside the valid domain.
        V_gt = torch.where(mask_finite, V_gt, torch.zeros_like(V_gt))

    # Reclassify points extremely close to Dirichlet boundaries as boundary
    # samples and pin their potential to the prescribed value to avoid tiny
    # evaluation differences dominating relative error metrics.
    # Neural supervision should keep surrogate outputs intact; boundary
    # clamping only applies when using analytic/BEM or neural fallbacks.
    if used_mode != "neural":
        try:
            for c in spec.conductors or []:
                ctype = c.get("type")
                bc_val = float(c.get("potential", 0.0))
                if ctype == "sphere":
                    center = torch.as_tensor(
                        c.get("center", [0.0, 0.0, 0.0]),
                        device=device,
                        dtype=dtype,
                    ).view(1, 3)
                    radius = float(c.get("radius", 0.0))
                    if radius > 0.0:
                        r = torch.linalg.norm(X - center, dim=1)
                        tol = 2e-2 * max(1.0, radius)
                        mask_near = torch.abs(r - radius) < tol
                        if mask_near.any():
                            is_boundary = is_boundary | mask_near
                            V_gt = torch.where(
                                mask_near,
                                torch.as_tensor(bc_val, device=device, dtype=dtype),
                                V_gt,
                            )
                elif ctype == "plane":
                    z0 = float(c.get("z", 0.0))
                    # Treat points within a small slab of the plane as
                    # boundary samples to avoid tiny near-surface differences
                    # dominating relative error checks.
                    tol = 1e-9
                    mask_near = torch.abs(X[:, 2] - z0) < tol
                    if mask_near.any():
                        is_boundary = is_boundary | mask_near
                        V_gt = torch.where(
                            mask_near,
                            torch.as_tensor(bc_val, device=device, dtype=dtype),
                            V_gt,
                        )
        except Exception:
            pass

    # Live intercept for diagnostics (does not affect correctness).
    try:
        ctx = bem_intercept.maybe_start_intercept(
            spec, test_name="collocation", bem_cfg=None
        )
        if ctx is not None:
            needs_eps_scaling = geom_type in ("plane", "sphere", "parallel_planes")
            bem_intercept.attach_bem_or_analytic_collocation(
                ctx,
                used_mode,
                geom_type,
                X,
                V_gt,
                is_boundary,
                mask_finite,
                ratio_boundary=float(ratio_boundary),
                needs_eps_scaling=needs_eps_scaling,
            )
            bem_intercept.finalize(ctx)
    except Exception:
        # Intercept must never break core logic.
        pass

    return {
        "X": X,
        "V_gt": V_gt,
        "is_boundary": is_boundary,
        "mask_finite": mask_finite,
        "bbox_center": bbox_center_batch,
        "bbox_extent": bbox_extent_batch,
        "encoding": encoding_batch,
    }
