#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import math
import os
import random
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from electrodrive.utils.logging import JsonlLogger
from electrodrive.core.bem import bem_solve  # type: ignore
from electrodrive.utils.config import BEMConfig

# ---------------------------------------------------------------------------
# Exit codes (so outer scripts can distinguish failure modes)
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_NUMERIC_FAIL = 1
EXIT_OOM_FAIL = 2
EXIT_SPEC_FAIL = 3
EXIT_EXCEPTION_FAIL = 4

# ---------------------------------------------------------------------------
# Numeric thresholds for health checks
# ---------------------------------------------------------------------------

# Very loose hard caps (to catch completely broken runs)
BC_RESID_HARD_MAX = 1e11
PDE_RESID_HARD_MAX = 1e5

# Reasonable "warn" thresholds for this benchmark; tune as needed
BC_RESID_WARN_MAX = 1e10
PDE_RESID_WARN_MAX = 1e-3


# ---------------------------------------------------------------------------
# Spec wrapper and loader
# ---------------------------------------------------------------------------


@dataclass
class DictSpecWrapper:
    """
    Minimal attribute-style wrapper around a raw dict spec.

    Provides:
      - spec.conductors
      - spec.charges
    and exposes all other top-level keys as attributes so that generate_mesh
    and bem_solve can treat it like a CanonicalSpec duck-type.
    """

    _raw: Dict[str, Any]

    def __init__(self, data: Dict[str, Any]) -> None:
        object.__setattr__(self, "_raw", data)

        # Essential fields for BEM / mesh.
        object.__setattr__(self, "conductors", data.get("conductors", []))
        object.__setattr__(self, "charges", data.get("charges", []))

        # Expose any other top-level keys as attributes too.
        for k, v in data.items():
            if not hasattr(self, k):
                object.__setattr__(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._raw)


def try_load_spec(path: Path) -> Any:
    """
    Try several likely loader entrypoints, then fall back to raw JSON.

    Mirrors the orchestration parser behaviour but starting from a file path.
    """
    candidates = [
        ("electrodrive.orchestration.parser", "load_canonical_spec"),
        ("electrodrive.orchestration.parser", "load_spec"),
        ("electrodrive.orchestration.parser", "load_problem"),
        ("electrodrive.orchestration.parser", "parse_spec"),
    ]
    last_err: Exception | None = None
    for mod_name, func_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, func_name)
            print(f"[BEM probe] Using loader: {mod_name}.{func_name}")
            return fn(str(path))
        except Exception as exc:
            last_err = exc

    print(
        "[BEM probe] Loader not found/failed "
        f"({last_err}). Falling back to raw JSON."
    )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _is_finite_scalar(x: Any) -> bool:
    """
    Robust scalar finiteness check that tolerates floats, ints, and simple strings.
    """
    if x is None:
        return False
    if isinstance(x, (int, float)):
        return math.isfinite(float(x))
    if isinstance(x, str):
        xl = x.strip().lower()
        if xl in {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity", "nan"}:
            return False
        try:
            return math.isfinite(float(x))
        except Exception:
            return False
    # Fallback for other types: not treating as finite.
    return False


def _cfg_to_dict(cfg: BEMConfig) -> Dict[str, Any]:
    """
    Best-effort config -> dict for logging/summary.
    """
    # BEMConfig is likely a dataclass or simple object with __dict__.
    return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}


def _classify_backend_error(err: str) -> str:
    """
    Map backend error string into a coarse error_type.
    """
    e = err.lower()
    if "out of memory" in e:
        return "oom"
    if "gmres" in e and ("resid=inf" in e or "non-finite" in e):
        return "gmres_numeric"
    if "fallback" in e:
        return "fallback"
    return "backend_error"


def _max_status(a: str, b: str) -> str:
    """
    Combine two statuses in {ok, warn, fail} by taking the worse.
    """
    order = {"ok": 0, "warn": 1, "fail": 2}
    return a if order[a] >= order[b] else b


@dataclass
class HealthResult:
    status: str  # "ok", "warn", "fail"
    reasons: List[str]
    mesh_stats: Dict[str, Any]
    gmres_stats: Dict[str, Any]


@dataclass
class AttemptResult:
    config_label: str
    cfg_used: Dict[str, Any]
    out_dict: Dict[str, Any] | None
    error_type: str | None  # "oom", "gmres_numeric", "fallback", "backend_error", "exception", None
    error_message: str | None
    health_status: str | None  # "ok", "warn", "fail", or None if not evaluated
    health_reasons: List[str]


# ---------------------------------------------------------------------------
# Numeric health evaluation
# ---------------------------------------------------------------------------


def evaluate_numeric_health(out_dict: Dict[str, Any]) -> HealthResult:
    mesh_stats = out_dict.get("mesh_stats") or {}
    gmres_stats = out_dict.get("gmres_stats") or {}

    status = "ok"
    reasons: List[str] = []

    # GMRES checks
    if gmres_stats:
        gmres_success = gmres_stats.get("success")
        gmres_resid = gmres_stats.get("resid")

        if gmres_success is False:
            status = "fail"
            reasons.append("gmres_success_false")

        if not _is_finite_scalar(gmres_resid):
            status = "fail"
            reasons.append("gmres_resid_non_finite")

    # BC residual checks
    bc_resid = mesh_stats.get("bc_residual_linf")
    if bc_resid is None:
        status = _max_status(status, "warn")
        reasons.append("bc_residual_missing")
    else:
        if not _is_finite_scalar(bc_resid):
            status = "fail"
            reasons.append("bc_residual_non_finite")
        else:
            try:
                bc_val = float(bc_resid)
                if abs(bc_val) > BC_RESID_HARD_MAX:
                    status = "fail"
                    reasons.append(
                        f"bc_residual_too_large({bc_val:.3e} > {BC_RESID_HARD_MAX:.1e})"
                    )
                elif abs(bc_val) > BC_RESID_WARN_MAX:
                    status = _max_status(status, "warn")
                    reasons.append(
                        f"bc_residual_above_warn({bc_val:.3e} > {BC_RESID_WARN_MAX:.1e})"
                    )
            except Exception:
                status = "fail"
                reasons.append("bc_residual_cast_failed")

    # PDE residual checks (if available)
    pde_resid = mesh_stats.get("pde_residual_linf")
    if pde_resid is not None:
        if not _is_finite_scalar(pde_resid):
            status = "fail"
            reasons.append("pde_residual_non_finite")
        else:
            try:
                pde_val = float(pde_resid)
                if abs(pde_val) > PDE_RESID_HARD_MAX:
                    status = "fail"
                    reasons.append(
                        f"pde_residual_too_large({pde_val:.3e} > {PDE_RESID_HARD_MAX:.1e})"
                    )
                elif abs(pde_val) > PDE_RESID_WARN_MAX:
                    status = _max_status(status, "warn")
                    reasons.append(
                        f"pde_residual_above_warn({pde_val:.3e} > {PDE_RESID_WARN_MAX:.1e})"
                    )
            except Exception:
                status = "fail"
                reasons.append("pde_residual_cast_failed")

    # Mesh sanity checks
    n_panels = mesh_stats.get("n_panels")
    total_area = mesh_stats.get("total_area")

    if n_panels is not None:
        try:
            if int(n_panels) <= 0:
                status = "fail"
                reasons.append(f"mesh_n_panels_nonpositive({n_panels})")
        except Exception:
            status = _max_status(status, "warn")
            reasons.append(f"mesh_n_panels_cast_failed({n_panels!r})")

    if total_area is not None:
        try:
            if float(total_area) <= 0:
                status = "fail"
                reasons.append(f"mesh_total_area_nonpositive({total_area})")
        except Exception:
            status = _max_status(status, "warn")
            reasons.append(f"mesh_total_area_cast_failed({total_area!r})")

    return HealthResult(status=status, reasons=reasons, mesh_stats=mesh_stats, gmres_stats=gmres_stats)


# ---------------------------------------------------------------------------
# Core attempt runner
# ---------------------------------------------------------------------------


def run_bem_attempt(
    spec_for_bem: Any,
    base_cfg: BEMConfig,
    logger: JsonlLogger,
    label: str,
    overrides: Dict[str, Any],
) -> AttemptResult:
    """
    Run a single BEM attempt with overrides to base_cfg.

    This is responsible for:
      - applying overrides
      - calling bem_solve
      - classifying backend error types
      - running numeric health checks
    """
    # Shallow copy config so we don't mutate the base.
    cfg = BEMConfig()
    # Copy over relevant attributes from base_cfg
    for k, v in vars(base_cfg).items():
        if not k.startswith("_"):
            setattr(cfg, k, v)

    # Apply overrides
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    cfg_dict = _cfg_to_dict(cfg)

    logger.info(
        "BEM probe attempt starting.",
        attempt_label=label,
        overrides=overrides,
        cfg=cfg_dict,
    )

    out_dict: Dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None
    health_status: str | None = None
    health_reasons: List[str] = []

    try:
        out_dict = bem_solve(spec_for_bem, cfg, logger, differentiable=False)
    except torch.cuda.OutOfMemoryError as exc:
        error_type = "oom"
        error_message = str(exc)
        logger.error(
            "BEM probe attempt OOM exception.",
            attempt_label=label,
            error=str(exc),
        )
        return AttemptResult(
            config_label=label,
            cfg_used=cfg_dict,
            out_dict=None,
            error_type=error_type,
            error_message=error_message,
            health_status=None,
            health_reasons=["oom_exception"],
        )
    except Exception as exc:
        error_type = "exception"
        error_message = str(exc)
        logger.error(
            "BEM probe attempt raised exception.",
            attempt_label=label,
            error=str(exc),
            traceback=traceback.format_exc(limit=20),
        )
        return AttemptResult(
            config_label=label,
            cfg_used=cfg_dict,
            out_dict=None,
            error_type=error_type,
            error_message=error_message,
            health_status=None,
            health_reasons=["unexpected_exception"],
        )

    # We reached here: bem_solve returned an out_dict (even if containing "error").
    backend_err = out_dict.get("error")
    if backend_err is not None:
        error_message = str(backend_err)
        error_type = _classify_backend_error(error_message)
    else:
        error_type = None

    # Decide whether to run numeric health checks.
    # - For OOM / hard backend errors: skip health -> fail attempt.
    # - For "gmres_numeric": treat as failure regardless of health (we still compute health for diagnostics).
    # - For "fallback": consider numeric health but downgrade "ok" -> "warn".
    # - For None: fully trust numeric health.
    if error_type == "oom":
        health_status = "fail"
        health_reasons = ["backend_oom_error"]
        logger.error(
            "BEM probe attempt backend OOM error.",
            attempt_label=label,
            error=error_message,
        )
    elif error_type in {"backend_error", "gmres_numeric"}:
        # Run health just for diagnostics, but treat it as fail overall.
        health = evaluate_numeric_health(out_dict)
        health_status = "fail"
        health_reasons = ["backend_error_type:" + error_type] + health.reasons
        logger.error(
            "BEM probe attempt backend error.",
            attempt_label=label,
            error_type=error_type,
            error=error_message,
            health_status=health.status,
            health_reasons=health.reasons,
        )
    elif error_type == "fallback":
        health = evaluate_numeric_health(out_dict)
        # Even if health says "ok", we treat "fallback" as at least "warn".
        if health.status == "ok":
            health_status = "warn"
        else:
            health_status = health.status
        health_reasons = ["backend_fallback"] + health.reasons
        logger.info(
            "BEM probe attempt used backend fallback.",
            attempt_label=label,
            error=error_message,
            health_status=health_status,
            health_reasons=health_reasons,
        )
    else:
        # No backend error: numeric health decides.
        health = evaluate_numeric_health(out_dict)
        health_status = health.status
        health_reasons = health.reasons
        if health_status == "ok":
            logger.info(
                "BEM probe attempt numeric health OK.",
                attempt_label=label,
                health_status=health_status,
                health_reasons=health_reasons,
            )
        elif health_status == "warn":
            logger.warning(
                "BEM probe attempt numeric health WARN.",
                attempt_label=label,
                health_status=health_status,
                health_reasons=health_reasons,
            )
        else:
            logger.error(
                "BEM probe attempt numeric health FAIL.",
                attempt_label=label,
                health_status=health_status,
                health_reasons=health_reasons,
            )

    return AttemptResult(
        config_label=label,
        cfg_used=cfg_dict,
        out_dict=out_dict,
        error_type=error_type,
        error_message=error_message,
        health_status=health_status,
        health_reasons=health_reasons,
    )


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------


def main() -> int:
    # Resolve relative to repo root (the directory containing this file)
    repo_root = Path(__file__).resolve().parent
    default_spec_rel = "specs/plane_point.json"
    spec_rel = os.environ.get("EDE_BEM_PROBE_SPEC", default_spec_rel)
    spec_path = repo_root / spec_rel
    out_dir = repo_root / "runs" / "bem_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = JsonlLogger(out_dir)

    logger.info("BEM probe starting.", spec=str(spec_path))

    # Deterministic seeds (can be overridden by env if needed)
    seed = int(os.environ.get("EDE_BEM_PROBE_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("BEM probe seeds set.", seed=seed)

    if not spec_path.exists():
        logger.error("Spec file not found for BEM probe.", spec=str(spec_path))
        print(f"[BEM probe] ERROR: spec not found at {spec_path}")
        logger.close()
        return EXIT_SPEC_FAIL

    # ---- CUDA status upfront ----
    cuda_ok = torch.cuda.is_available()
    device_type = "cuda" if cuda_ok else "cpu"
    device_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU-only"

    if cuda_ok:
        try:
            dev = torch.device("cuda")
            torch.cuda.reset_peak_memory_stats(dev)
        except Exception:
            # Best-effort; not fatal if this fails.
            pass

    logger.info(
        "CUDA status for BEM probe.",
        cuda_available=bool(cuda_ok),
        device_name=device_name,
        device_type=device_type,
    )
    print(f"[BEM probe] CUDA available = {cuda_ok}, device = {device_name}")

    # ---- Load spec ----
    try:
        spec_obj = try_load_spec(spec_path)
    except Exception as exc:
        logger.error(
            "Spec load failed for BEM probe.",
            spec=str(spec_path),
            error=str(exc),
            traceback=traceback.format_exc(limit=20),
        )
        print(f"[BEM probe] ERROR: failed to load spec: {exc}")
        logger.close()
        return EXIT_SPEC_FAIL

    logger.info(
        "Spec loaded for BEM probe.",
        spec_path=str(spec_path),
        loader_type=type(spec_obj).__name__,
    )

    # Wrap raw dicts so bem_solve / generate_mesh see attributes.
    if isinstance(spec_obj, dict):
        logger.info(
            "Wrapping raw dict spec for BEM.",
            top_level_keys=list(spec_obj.keys()),
        )
        spec_for_bem = DictSpecWrapper(spec_obj)
    else:
        # Some loaders may expose .canonical; otherwise use object as-is.
        spec_for_bem = getattr(spec_obj, "canonical", spec_obj)

    # ---- Base BEM config ----
    base_cfg = BEMConfig()
    base_cfg.run_dir = str(out_dir)
    # Let attempts override this; we default to GPU+fp64 if present.
    base_cfg.use_gpu = cuda_ok
    base_cfg.fp64 = True

    # Ensure run dir is also visible via env for other logging hooks.
    os.environ["EDE_RUN_DIR"] = str(out_dir)

    # ---- Define attempt ladder (most aggressive -> more conservative) ----
    attempts: List[Tuple[str, Dict[str, Any]]] = []

    if cuda_ok:
        attempts.extend(
            [
                (
                    "cuda_fp64_refine3",
                    {"use_gpu": True, "fp64": True, "max_refine_passes": 3},
                ),
                (
                    "cuda_fp32_refine3",
                    {"use_gpu": True, "fp64": False, "max_refine_passes": 3},
                ),
                (
                    "cuda_fp64_refine2",
                    {"use_gpu": True, "fp64": True, "max_refine_passes": 2},
                ),
                (
                    "cuda_fp32_refine2",
                    {"use_gpu": True, "fp64": False, "max_refine_passes": 2},
                ),
            ]
        )

    # Always finish with a CPU fp64 attempt as ultimate fallback.
    attempts.append(
        (
            "cpu_fp64_refine3",
            {"use_gpu": False, "fp64": True, "max_refine_passes": 3},
        )
    )

    attempt_results: List[AttemptResult] = []
    successful_ok: AttemptResult | None = None
    successful_warn: AttemptResult | None = None

    for label, overrides in attempts:
        # If CUDA is unavailable, skip configs that insist on GPU.
        if overrides.get("use_gpu") and not cuda_ok:
            continue

        result = run_bem_attempt(spec_for_bem, base_cfg, logger, label, overrides)
        attempt_results.append(result)

        if result.health_status == "ok" and result.error_type is None:
            successful_ok = result
            break
        if result.health_status == "warn" and successful_warn is None:
            successful_warn = result
            # Keep going in case we can find a clean "ok" later.

    # -----------------------------------------------------------------------
    # Decide final status and exit code
    # -----------------------------------------------------------------------
    final_attempt: AttemptResult | None = None
    final_status: str = "fail"
    exit_code: int = EXIT_NUMERIC_FAIL
    final_failure_reasons: List[str] = []

    if successful_ok is not None:
        final_attempt = successful_ok
        final_status = "ok"
        exit_code = EXIT_OK
        final_failure_reasons = successful_ok.health_reasons
    elif successful_warn is not None:
        final_attempt = successful_warn
        final_status = "warn"
        exit_code = EXIT_OK
        final_failure_reasons = successful_warn.health_reasons
    else:
        # No numerically acceptable attempt. Classify the dominant failure.
        final_status = "fail"

        # Prefer OOM if any attempt had it.
        if any(a.error_type == "oom" for a in attempt_results):
            exit_code = EXIT_OOM_FAIL
            final_failure_reasons.append("all_attempts_oom_or_oom_backend_error")
        # Otherwise, any backend_error / gmres_numeric / fallback / exception?
        elif any(
            a.error_type in {"backend_error", "gmres_numeric", "fallback"}
            for a in attempt_results
        ):
            exit_code = EXIT_NUMERIC_FAIL
            final_failure_reasons.append("all_attempts_backend_numeric_failure")
        elif any(a.error_type == "exception" for a in attempt_results):
            exit_code = EXIT_EXCEPTION_FAIL
            final_failure_reasons.append("all_attempts_raised_exceptions")
        else:
            # Fallback catch-all
            exit_code = EXIT_NUMERIC_FAIL
            final_failure_reasons.append("all_attempts_failed_unknown_reason")

        # Also accumulate per-attempt reasons for context.
        for a in attempt_results:
            tag = f"attempt:{a.config_label}"
            if a.health_reasons:
                final_failure_reasons.append(f"{tag}:{';'.join(a.health_reasons)}")
            elif a.error_type:
                final_failure_reasons.append(f"{tag}:error_type={a.error_type}")

    # -----------------------------------------------------------------------
    # Console summary and summary manifest
    # -----------------------------------------------------------------------
    if final_attempt is not None and final_attempt.out_dict is not None:
        mesh_stats = final_attempt.out_dict.get("mesh_stats") or {}
        gmres_stats = final_attempt.out_dict.get("gmres_stats") or {}
    else:
        mesh_stats = {}
        gmres_stats = {}

    n_panels = mesh_stats.get("n_panels")
    total_area = mesh_stats.get("total_area")
    h_final = mesh_stats.get("h_final")
    tile_final = mesh_stats.get("tile_size_final")
    bc_resid = mesh_stats.get("bc_residual_linf")
    pde_resid = mesh_stats.get("pde_residual_linf")

    print(
        f"[BEM probe] mesh: n_panels = {n_panels}, "
        f"total_area = {total_area}, "
        f"h_final = {h_final}, "
        f"tile_size_final = {tile_final}"
    )
    print(f"[BEM probe] bc_residual_linf = {bc_resid}")
    if pde_resid is not None:
        print(f"[BEM probe] pde_residual_linf = {pde_resid}")

    # GPU memory summary: try GMRES stats, then mesh stats, then None.
    gpu_peak_dict = None
    for src in (gmres_stats, mesh_stats):
        if isinstance(src, dict):
            cand = src.get("gpu_mem_peak_detail") or src.get("gpu_mem_peak_mb")
            if isinstance(cand, dict):
                gpu_peak_dict = cand
                break

    if gpu_peak_dict is not None:
        print(
            "[BEM probe] GPU peak MB "
            f"(allocated={gpu_peak_dict.get('allocated')}, "
            f"reserved={gpu_peak_dict.get('reserved')})"
        )
    else:
        print("[BEM probe] GPU peak MB (no detailed GPU stats in mesh/gmres stats)")

    if gmres_stats:
        print(
            f"[BEM probe] GMRES iters = {gmres_stats.get('iters')}, "
            f"resid = {gmres_stats.get('resid')}, "
            f"success = {gmres_stats.get('success')}, "
            f"code = {gmres_stats.get('code')}"
        )
    else:
        print("[BEM probe] GMRES stats not present in solver output.")

    # Build attempts summary for JSON
    attempts_summary: List[Dict[str, Any]] = []
    for a in attempt_results:
        attempts_summary.append(
            {
                "config_label": a.config_label,
                "cfg_used": a.cfg_used,
                "error_type": a.error_type,
                "error_message": a.error_message,
                "health_status": a.health_status,
                "health_reasons": a.health_reasons,
            }
        )

    summary = {
        "spec_path": str(spec_path),
        "cuda_available": cuda_ok,
        "device_name": device_name,
        "final_status": final_status,
        "exit_code": exit_code,
        "failure_reasons": final_failure_reasons,
        "attempts": attempts_summary,
    }

    summary_path = out_dir / "bem_probe_summary.json"
    try:
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(
            "BEM probe summary written.",
            path=str(summary_path),
            final_status=final_status,
            exit_code=exit_code,
        )
    except Exception as exc:
        logger.error(
            "Failed to write BEM probe summary.",
            path=str(summary_path),
            error=str(exc),
        )

    # Final log + exit
    if final_status in {"ok", "warn"}:
        msg = "BEM probe SUCCEEDED." if final_status == "ok" else "BEM probe SUCCEEDED with warnings."
        print(f"[BEM probe] {msg}")
    else:
        print(f"[BEM probe] PROBE FAILED. Reasons: {', '.join(final_failure_reasons)}")

    logger.info(
        "BEM probe final status.",
        final_status=final_status,
        exit_code=exit_code,
        failure_reasons=final_failure_reasons,
    )
    logger.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
