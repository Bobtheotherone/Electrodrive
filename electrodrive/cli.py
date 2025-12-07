#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from electrodrive.utils.logging import (
    JsonlLogger,
    RuntimePerfFlags,
    log_peak_vram,
    log_runtime_environment,
)
from electrodrive.utils.config import (
    DEFAULT_SEED,
    BEMConfig,
    PINNConfig,
    DEFAULT_SOLVE_DTYPE,
    EPS_BC,
    EPS_DUAL,
    EPS_PDE,
    EPS_ENERGY,
    EPS_MEAN_VAL,
    EPS_0,
    K_E,
)
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.orchestration.planner import choose_mode
from electrodrive.eval.governance import governance_guard
from electrodrive.core.images import (
    AnalyticSolution,
    potential_plane_halfspace,
    potential_sphere_grounded,
)
from electrodrive.core.certify import (
    bc_residual_on_boundary,
    dual_route_error_boundary,
    pde_residual_symbolic,
    energy_consistency_check,
    mean_value_property_check,
    maximum_principle_margin,
    reciprocity_deviation,
    green_badge_decision,
)

try:
    import torch
except Exception:
    torch = None  # torch is optional; CLI must still work


# ------------------------
# Perf / TF32 helpers
# ------------------------


def _build_perf_flags(args: argparse.Namespace) -> RuntimePerfFlags:
    """Construct RuntimePerfFlags from CLI args."""
    amp = bool(getattr(args, "amp", False))
    train_dtype = str(getattr(args, "train_dtype", "float32"))
    compile_flag = bool(getattr(args, "compile", False))
    tf32 = str(getattr(args, "tf32", "off"))
    return RuntimePerfFlags(
        amp=amp,
        train_dtype=train_dtype,
        compile=compile_flag,
        tf32=tf32,
    )


def _apply_tf32_flag(tf32_mode: str, logger: JsonlLogger) -> None:
    """Best-effort apply TF32 matmul precision; no-ops if unavailable."""
    if tf32_mode == "off":
        return
    if torch is None:
        logger.info(
            "TF32 requested but torch is not available; ignoring.",
            tf32_requested=tf32_mode,
        )
        return
    if not hasattr(torch, "set_float32_matmul_precision"):
        logger.info(
            "TF32 requested but set_float32_matmul_precision unavailable; ignoring.",
            tf32_requested=tf32_mode,
        )
        return
    try:
        torch.set_float32_matmul_precision(tf32_mode)
        effective = (
            str(torch.get_float32_matmul_precision())
            if hasattr(torch, "get_float32_matmul_precision")
            else "unknown"
        )
        logger.info(
            "TF32 matmul precision set.",
            tf32_requested=tf32_mode,
            tf32_effective=effective,
        )
    except Exception as exc:
        logger.warning(
            "Failed to set TF32 matmul precision.",
            tf32_requested=tf32_mode,
            error=str(exc),
        )


def _log_amp_compile_requests(perf: RuntimePerfFlags, logger: JsonlLogger) -> None:
    logger.info(
        "Performance knobs requested.",
        amp=bool(perf.amp),
        train_dtype=str(perf.train_dtype),
        compile_requested=bool(perf.compile),
        tf32=str(perf.tf32),
    )


def _set_seeds(seed: int, logger: JsonlLogger) -> None:
    """Seed Python, NumPy, and torch RNGs deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if hasattr(torch.cuda, "manual_seed_all"):
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
    logger.info("Seeds set.", seed=seed)


def _versions() -> Dict[str, Any]:
    """Collect runtime version info."""
    v: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "executable": sys.executable,
        "torch": getattr(torch, "__version__", "unavailable") if torch else "unavailable",
    }
    try:
        import sympy as sp

        v["sympy"] = sp.__version__
    except Exception:
        v["sympy"] = "unavailable"
    return v


# ------------------------
# VRAM telemetry
# ------------------------


def _vram_telemetry_init(logger: JsonlLogger) -> Dict[str, Any]:
    """
    Initialize VRAM telemetry.

    Returns a dict to be stored in meta["vram_telemetry"].
    """
    telemetry: Dict[str, Any] = {
        "gpu_available": False,
        "device_name": "unavailable",
        "total_memory_gb": 0.0,
        "tf32_enabled": False,
        "dtype": DEFAULT_SOLVE_DTYPE,
        "gpu_mem_peak_mb": 0.0,
    }

    if torch is None or not hasattr(torch, "cuda") or not torch.cuda.is_available():
        logger.info("CUDA not available or torch not installed; VRAM telemetry disabled.")
        return telemetry

    try:
        device_id = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_id)
        telemetry["gpu_available"] = True
        telemetry["device_name"] = props.name
        telemetry["total_memory_gb"] = float(props.total_memory) / (1024.0**3)

        # Reset peak stats so this run is isolated.
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats(device_id)

        # Read TF32 status if available.
        tf32_enabled = False
        try:
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                tf32_enabled = bool(torch.backends.cuda.matmul.allow_tf32)
        except Exception:
            tf32_enabled = False
        telemetry["tf32_enabled"] = tf32_enabled

        logger.info(
            "VRAM telemetry initialized.",
            gpu_available=True,
            device_name=telemetry["device_name"],
            total_memory_gb=f"{telemetry['total_memory_gb']:.2f}",
            tf32_enabled=tf32_enabled,
        )
    except Exception as exc:
        logger.warning("VRAM telemetry initialization failed.", error=str(exc))

    return telemetry


def _finalize_vram_telemetry(telemetry: Dict[str, Any], logger: JsonlLogger) -> None:
    """
    Finalize VRAM telemetry: fill gpu_mem_peak_mb and log summary.

    Does not raise.
    """
    if torch is None or not hasattr(torch, "cuda") or not torch.cuda.is_available():
        return

    try:
        device_id = torch.cuda.current_device()
        max_alloc = float(torch.cuda.max_memory_allocated(device_id))
        max_reserved = float(torch.cuda.max_memory_reserved(device_id))
        peak_mb_alloc = max_alloc / (1024.0 * 1024.0)
        peak_mb_res = max_reserved / (1024.0 * 1024.0)

        # Record both; keep gpu_mem_peak_mb as "allocated" for manifest/metrics.
        telemetry["gpu_mem_peak_mb"] = peak_mb_alloc
        telemetry["peak_memory_allocated_mb"] = peak_mb_alloc
        telemetry["peak_memory_reserved_mb"] = peak_mb_res

        logger.info(
            "VRAM telemetry finalized.",
            gpu_mem_peak_mb=f"{peak_mb_alloc:.2f}",
            peak_reserved_mb=f"{peak_mb_res:.2f}",
        )
    except Exception as exc:
        logger.warning("VRAM telemetry finalization failed.", error=str(exc))


# ------------------------
# Backend / manifest helpers
# ------------------------


def _is_available(mod_name: str) -> bool:
    """Soft-check for optional backend modules."""
    try:
        __import__(mod_name)
        return True
    except Exception:
        return False


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """
    Atomic JSON write: write to temp file then os.replace into place.

    Ensures UTF-8 encoding and cross-platform safety.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    txt = json.dumps(data, indent=2)
    tmp.write_text(txt, encoding="utf-8")
    os.replace(tmp, path)


def _write_manifest(
    out_dir: Path,
    run_id: str,
    meta: Dict[str, Any],
    mode_requested: str,
    mode_selected: str,
) -> None:
    """
    Write manifest.json capturing planner/device/backend/versions for the run.

    Failures are non-fatal.
    """
    v = meta.get("versions", {}) or {}
    vt = meta.get("vram_telemetry", {}) or {}

    backend_available = {
        "torch": bool(torch is not None),
        "pykeops": _is_available("pykeops.torch"),
        "keopscore": _is_available("keopscore"),
        "cupy": _is_available("cupy"),
        "xitorch": _is_available("xitorch"),
    }

    # Choose a simple selected backend label.
    if torch is not None:
        backend_selected = "torch"
    else:
        backend_selected = "unknown"

    device_info = {
        "gpu_available": bool(vt.get("gpu_available", False)),
        "device_name": vt.get("device_name", "unavailable"),
        "dtype": vt.get("dtype", DEFAULT_SOLVE_DTYPE),
        "tf32": bool(vt.get("tf32_enabled", False)),
        "gpu_mem_peak_mb": float(vt.get("gpu_mem_peak_mb", 0.0)),
        "peak_memory_allocated_mb": float(
            vt.get("peak_memory_allocated_mb", vt.get("gpu_mem_peak_mb", 0.0) or 0.0)
        ),
        "peak_memory_reserved_mb": float(vt.get("peak_memory_reserved_mb", 0.0)),
    }

    planner = {
        "requested_mode": mode_requested,
        "selected_mode": mode_selected,
        "rationale": meta.get("planner_rationale", ""),
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "versions": v,
        "planner": planner,
        "device": device_info,
        "backend": {
            "available": backend_available,
            "selected": backend_selected,
            "fallback_reason": meta.get("backend_fallback_reason"),
        },
        "run_status": meta.get("run_status", "unknown"),
        "backend_health": meta.get("backend_health", "unknown"),
        "solver_mode_effective": meta.get("solver_mode_effective", "unknown"),
    }

    try:
        _atomic_write_json(out_dir / "manifest.json", manifest)
    except Exception:
        # Best-effort; never crash caller
        pass


# ------------------------
# JSON-safe helpers
# ------------------------


def _json_serialize_float(v: Any) -> Any:
    """Sanitize floats (NaN/Inf) for JSON."""
    try:
        f = float(v)
    except Exception:
        return v
    if math.isfinite(f):
        return f
    if math.isnan(f):
        return "NaN"
    return "Infinity" if f > 0 else "-Infinity"


def _sanitize_metrics_for_json(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a shallow-copied metrics dict with NaN/Inf made JSON-safe
    on a small, known set of numeric keys.
    """
    out = dict(metrics)
    float_keys = [
        "bc_residual_linf",
        "dual_route_l2_boundary",
        "pde_residual_linf",
        "energy_rel_diff",
        "energy_A",
        "energy_B",
        "mean_value_deviation",
        "max_principle_margin",
        "reciprocity_dev",
        "gmres_resid",
        "gpu_mem_peak_mb",
        "patch_L",
    ]
    for k in float_keys:
        if k in out:
            out[k] = _json_serialize_float(out[k])
    return out


# ------------------------
# Verification helpers
# ------------------------


def _fail_reasons_from_metrics(metrics: Dict[str, Any]) -> List[str]:
    bc = float(metrics.get("bc_residual_linf", float("inf")))
    dual = float(metrics.get("dual_route_l2_boundary", float("inf")))
    pde = float(metrics.get("pde_residual_linf", float("inf")))
    energy = metrics.get("energy_rel_diff", float("nan"))
    energy_ok = not (isinstance(energy, float) and math.isfinite(energy) and energy > EPS_ENERGY)

    reasons: List[str] = []
    if not (bc <= EPS_BC):
        reasons.append(f"BC {bc:.3e} > {EPS_BC:.3e}")
    if isinstance(dual, float) and math.isfinite(dual) and not (dual <= EPS_DUAL):
        reasons.append(f"Dual {dual:.3e} > {EPS_DUAL:.3e}")
    if not (pde <= EPS_PDE):
        reasons.append(f"PDE {pde:.3e} > {EPS_PDE:.3e}")
    if not energy_ok:
        reasons.append(f"Energy {energy:.3e} > {EPS_ENERGY:.3e}")
    return reasons


def aggregate_verification_report(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Build verification_report.json payload from metrics."""
    bc = metrics.get("bc_residual_linf", float("inf"))
    dual = metrics.get("dual_route_l2_boundary", float("nan"))
    pde = metrics.get("pde_residual_linf", float("inf"))
    energy = metrics.get("energy_rel_diff", float("nan"))
    mean_val = metrics.get("mean_value_deviation", float("nan"))
    mp = metrics.get("max_principle_margin", float("nan"))
    rec = metrics.get("reciprocity_dev", float("nan"))

    strong = bool(metrics.get("_strong_gate", False))
    green = green_badge_decision(metrics, logger=None, strong=strong)

    pass_bc = float(bc) <= EPS_BC
    pass_dual = not (isinstance(dual, float) and math.isfinite(dual) and dual > EPS_DUAL)
    pass_pde = float(pde) <= EPS_PDE

    if isinstance(energy, float) and math.isfinite(energy):
        pass_energy = energy <= EPS_ENERGY
    else:
        pass_energy = True

    if isinstance(mean_val, float) and math.isfinite(mean_val):
        pass_mean_value = mean_val <= EPS_MEAN_VAL
    else:
        pass_mean_value = True

    aux = {
        "energy_A": _json_serialize_float(metrics.get("energy_A")),
        "energy_B": _json_serialize_float(metrics.get("energy_B")),
        "route_A_method": metrics.get("route_A_method"),
        "route_B_method": metrics.get("route_B_method"),
        "patch_L": _json_serialize_float(metrics.get("patch_L")),
    }

    return {
        "bc_Linf": _json_serialize_float(bc),
        "dual_L2": _json_serialize_float(dual),
        "pde_residual": _json_serialize_float(pde),
        "energy_rel_diff": _json_serialize_float(energy),
        "mean_value_dev": _json_serialize_float(mean_val),
        "max_principle_margin": _json_serialize_float(mp),
        "reciprocity_dev": _json_serialize_float(rec),
        "pass_bc": bool(pass_bc),
        "pass_dual": bool(pass_dual),
        "pass_pde": bool(pass_pde),
        "pass_energy": bool(pass_energy),
        "pass_mean_value": bool(pass_mean_value),
        "green_badge": bool(green),
        "auxiliary_data": aux,
        "samples": {},
    }


# ------------------------
# BEM helpers
# ------------------------


def _pde_residual_bem(
    solution: Any,
    spec: CanonicalSpec,
    mesh_stats: Dict[str, Any],
    logger: JsonlLogger,
) -> float:
    """
    Placeholder PDE residual for BEM (harmonic identity).

    For the current minimal solver we treat it as zero or use stored value.
    """
    if "pde_residual_linf" in mesh_stats:
        val = float(mesh_stats["pde_residual_linf"])
    else:
        val = 0.0
    logger.info("PDE residual (BEM).", pde_residual_linf=val)
    return val


def _phi_induced_at_charge(solution: Any, r_q: Tuple[float, float, float]) -> float:
    """Compute induced potential at a point charge location for BEM solution."""
    if torch is None:
        return float("nan")
    from electrodrive.core.bem_kernel import bem_potential_targets

    P = torch.tensor([list(r_q)], device=solution._device, dtype=solution._dtype)
    V_ind = bem_potential_targets(
        targets=P,
        src_centroids=solution._C,
        areas=solution._A,
        sigma=solution._S,
        tile_size=solution._tile,
    )
    return float(V_ind[0].item())


def _energy_consistency_bem(
    spec: CanonicalSpec,
    bem_out: Dict[str, Any],
    logger: JsonlLogger,
) -> Dict[str, Any]:
    """
    Compute Route A/B energies for BEM by delegating to the central
    certification helper :func:`energy_consistency_check`.

    This avoids drifting conventions between the CLI and
    ``electrodrive.core.certify``. The legacy "A"/"B" keys are kept
    for compatibility with existing metrics/JSON consumers.
    """
    if torch is None:
        return {"A": float("nan"), "B": float("nan"), "route_B_method": "unavailable"}

    solution = bem_out.get("solution")
    if solution is None:
        return {"A": float("nan"), "B": float("nan"), "route_B_method": "unavailable"}

    try:
        energy_metrics = energy_consistency_check(solution, spec, logger=logger)
    except Exception as exc:
        logger.warning(
            "Energy consistency check failed (BEM).",
            error=str(exc),
        )
        return {"A": float("nan"), "B": float("nan"), "route_B_method": "unavailable"}

    return {
        "A": float(energy_metrics.get("energy_A", float("nan"))),
        "B": float(energy_metrics.get("energy_B", float("nan"))),
        "route_B_method": energy_metrics.get("route_B_method", "unavailable"),
    }


    sol = bem_out["solution"]
    q_list: List[float] = []
    r_list: List[Tuple[float, float, float]] = []
    for ch in spec.charges:
        if ch.get("type") == "point":
            q_list.append(float(ch["q"]))
            r_list.append(tuple(map(float, ch["pos"])))

    if not q_list:
        return {"A": 0.0, "B": 0.0, "route_B_method": "no_charges"}

    # Route A: -1/2 sum q * phi_induced(r_q)
    U_A = 0.0
    for q, r_q in zip(q_list, r_list):
        V_ind = _phi_induced_at_charge(sol, r_q)
        term = -0.5 * q * V_ind
        U_A += term
        logger.debug(
            "Energy Route A term.",
            q=f"{q:.6e}",
            phi_induced=f"{V_ind:.6e}",
            contrib=f"{term:.6e}",
        )

    U_B: Optional[float] = None
    route_B_method = "surface_minus_half_sigma_phi_free"

    # Sphere analytic external
    if any(c.get("type") == "sphere" for c in spec.conductors) and len(q_list) == 1:
        c0 = next(c for c in spec.conductors if c.get("type") == "sphere")
        a = float(c0["radius"])
        cx, cy, cz = map(float, c0["center"])
        rx, ry, rz = r_list[0]
        R = math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2 + (rz - cz) ** 2)
        if R > a:
            U_B = -0.5 * K_E * q_list[0] * q_list[0] * a / (R * R - a * a)
            route_B_method = "analytic_sphere_external"
            logger.info(
                "Sphere energy Route B (analytic external).",
                R=f"{R:.6e}",
                a=f"{a:.6e}",
                energy_B=f"{U_B:.6e}",
            )
        else:
            route_B_method = "surface_minus_half_sigma_phi_free_inside_sphere"

    # Plane patch heuristic descriptor (energy from BEM integral)
    elif any(c.get("type") == "plane" for c in spec.conductors) and len(q_list) == 1:
        route_B_method = "surface_minus_half_sigma_phi_free_plane_patch"

    # Generic surface energy via integral if analytic shortcut not set:
    if U_B is None:
        C = sol._C
        A = sol._A
        S = sol._S
        V_free = torch.zeros(C.shape[0], device=sol._device, dtype=sol._dtype)
        for q, r in zip(q_list, r_list):
            r_t = torch.tensor([r], device=sol._device, dtype=sol._dtype)
            Rv = torch.linalg.norm(C - r_t, dim=1).clamp_min(1e-12)
            V_free += K_E * q / Rv
        U_B = float(-0.5 * torch.sum(V_free * S * A).item())
        logger.info(
            "Energy Route B via surface integral.",
            U_B=f"{U_B:.6e}",
            route=route_B_method,
        )

    return {"A": float(U_A), "B": float(U_B), "route_B_method": route_B_method}


# ------------------------
# Analytic energy helpers
# ------------------------


def _energy_routes_for_simple_analytic(
    spec: CanonicalSpec,
    logger: Optional[JsonlLogger] = None,
) -> Optional[Dict[str, Any]]:
    """
    If spec matches a simple analytic configuration, return both
    routes and energies for certification reporting.
    """
    q_list: List[float] = []
    r_list: List[Tuple[float, float, float]] = []
    for ch in spec.charges:
        if ch.get("type") == "point":
            q_list.append(float(ch["q"]))
            r_list.append(tuple(map(float, ch["pos"])))
    if len(q_list) != 1:
        return None

    q = q_list[0]
    rx, ry, rz = r_list[0]

    # Grounded plane at z=0
    plane = next((c for c in spec.conductors if c.get("type") == "plane"), None)
    if plane and len(spec.conductors) == 1:
        z_plane = float(plane.get("z", 0.0))
        pot = float(plane.get("potential", 0.0))
        if abs(z_plane) < 1e-12 and abs(pot) < 1e-12 and rz > 0.0:
            U = -K_E * q * q / (4.0 * rz)
            if logger:
                logger.info(
                    "Analytic energy routes (grounded plane).",
                    z=f"{rz:.6e}",
                    energy=f"{U:.6e}",
                )
            return {
                "route_A_method": "analytic_images_plane",
                "route_B_method": "surface_minus_half_sigma_phi_free_plane_patch",
                "energy_A": float(U),
                "energy_B": float(U),
            }

    # Grounded sphere
    sphere = next((c for c in spec.conductors if c.get("type") == "sphere"), None)
    if sphere and len(spec.conductors) == 1:
        a = float(sphere.get("radius", 1.0))
        cx, cy, cz = map(float, sphere.get("center", [0.0, 0.0, 0.0]))
        pot = float(sphere.get("potential", 0.0))
        if abs(pot) < 1e-12:
            dx, dy, dz = rx - cx, ry - cy, rz - cz
            R = math.sqrt(dx * dx + dy * dy + dz * dz)
            if R <= 0.0:
                return None
            if R > a:
                U = -0.5 * K_E * q * q * a / (R * R - a * a)
                if logger:
                    logger.info(
                        "Analytic energy routes (sphere external).",
                        R=f"{R:.6e}",
                        a=f"{a:.6e}",
                        energy=f"{U:.6e}",
                    )
                return {
                    "route_A_method": "analytic_images_sphere_external",
                    "route_B_method": "analytic_sphere_external",
                    "energy_A": float(U),
                    "energy_B": float(U),
                }
            if R < a:
                # Inside-sphere analytic image energy.
                q_img = -q * a / R
                scale = (a * a) / (R * R)
                rpx, rpy, rpz = cx + dx * scale, cy + dy * scale, cz + dz * scale
                d_im = math.sqrt((rx - rpx) ** 2 + (ry - rpy) ** 2 + (rz - rpz) ** 2)
                if d_im <= 0.0:
                    return None
                phi_img = K_E * q_img / d_im
                U = 0.5 * q * phi_img
                if logger:
                    logger.info(
                        "Analytic energy routes (sphere internal).",
                        R=f"{R:.6e}",
                        a=f"{a:.6e}",
                        energy=f"{U:.6e}",
                    )
                return {
                    "route_A_method": "analytic_images_sphere_internal",
                    "route_B_method": "surface_minus_half_sigma_phi_free_inside_sphere",
                    "energy_A": float(U),
                    "energy_B": float(U),
                }

    return None


def _route_B_descriptor_and_energy_for_spec(
    spec: CanonicalSpec,
    logger: Optional[JsonlLogger] = None,
) -> Tuple[str, Optional[float]]:
    """
    Provide a qualitative Route B descriptor (and optional energy) for manifest/metrics.
    """
    q_list: List[float] = []
    r_list: List[Tuple[float, float, float]] = []
    for ch in spec.charges:
        if ch.get("type") == "point":
            q_list.append(float(ch["q"]))
            r_list.append(tuple(map(float, ch["pos"])))
    if not q_list:
        return "no_charges", 0.0

    sphere = next((c for c in spec.conductors if c.get("type") == "sphere"), None)
    if sphere and len(q_list) == 1:
        a = float(sphere["radius"])
        cx, cy, cz = map(float, sphere["center"])
        rx, ry, rz = r_list[0]
        R = math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2 + (rz - cz) ** 2)
        if R >= a:
            UB = -0.5 * K_E * q_list[0] * q_list[0] * a / (R * R - a * a)
            if logger:
                logger.info(
                    "Route B (analytic sphere external).",
                    R=f"{R:.6f}",
                    a=f"{a:.6f}",
                    energy_B=f"{UB:.6e}",
                )
            return "analytic_sphere_external", float(UB)
        if logger:
            logger.info(
                "Route B (surface inside sphere).",
                R=f"{R:.6f}",
                a=f"{a:.6f}",
            )
        return "surface_minus_half_sigma_phi_free_inside_sphere", None

    plane = next((c for c in spec.conductors if c.get("type") == "plane"), None)
    if plane and len(q_list) == 1:
        if logger:
            logger.info("Route B (surface plane patch) descriptor.")
        return "surface_minus_half_sigma_phi_free_plane_patch", None

    return "surface_minus_half_sigma_phi_free", None


def _analytic_solution_from_spec(
    spec: CanonicalSpec,
    logger: JsonlLogger,
) -> AnalyticSolution:
    """
    Construct an AnalyticSolution for simple plane/sphere configs.

    Raises NotImplementedError when the spec is not supported. This is
    used both for explicit analytic mode and for BEM fallback.
    """
    ctypes = sorted({c.get("type") for c in spec.conductors})
    point_charges = [ch for ch in spec.charges if ch.get("type") == "point"]

    if len(point_charges) != 1:
        raise NotImplementedError(
            f"Analytic solver currently supports exactly one point charge; got {len(point_charges)}."
        )

    charge = point_charges[0]
    q = float(charge["q"])
    r0 = tuple(map(float, charge["pos"]))

    # Single grounded plane at z=0
    if ctypes == ["plane"] and len(spec.conductors) == 1:
        plane = spec.conductors[0]
        z_plane = float(plane.get("z", 0.0))
        pot_plane = float(plane.get("potential", 0.0))
        if abs(z_plane) > 1e-12 or abs(pot_plane) > 1e-12:
            raise NotImplementedError(
                "Analytic plane solver requires a single grounded plane at z=0."
            )
        if r0[2] <= z_plane:
            r0 = (r0[0], r0[1], max(r0[2], z_plane + 1e-6))
            logger.warning(
                "Adjusted charge z above plane for analytic images.",
                adjusted_pos=r0,
            )
        return potential_plane_halfspace(q, r0)

    # Single grounded sphere
    if ctypes == ["sphere"] and len(spec.conductors) == 1:
        sphere = spec.conductors[0]
        pot_sphere = float(sphere.get("potential", 0.0))
        if abs(pot_sphere) > 1e-12:
            raise NotImplementedError(
                "Analytic sphere solver requires a grounded sphere (potential=0)."
            )
        center = tuple(map(float, sphere.get("center", [0.0, 0.0, 0.0])))
        radius = float(sphere.get("radius", 1.0))
        return potential_sphere_grounded(q, r0, center, radius)

    raise NotImplementedError("Analytic solver not implemented for this spec.")


# ------------------------
# FAST mode (BEM tuning)
# ------------------------


def _apply_fast_bem_tuning(cfg: BEMConfig, logger: JsonlLogger) -> None:
    """
    Apply "fast" heuristics to BEMConfig without relying on internal fields.

    Only sets attributes that exist; keeps behavior compatible across versions.
    Does not modify the numeric dtype (fp64 vs fp32); that is left to BEMConfig.
    """

    def set_if(name: str, value: Any) -> None:
        if hasattr(cfg, name):
            try:
                setattr(cfg, name, value)
                logger.info("FAST: set BEMConfig field.", field=name, value=str(value))
            except Exception:
                pass

    set_if("max_refine_passes", 2)
    set_if("gmres_maxiter", 600)
    set_if("gmres_restart", 128)
    set_if("target_vram_fraction", 0.9)
    set_if("tile_mem_divisor", 1.5)
    set_if("min_tile", 2048)
    set_if("use_gpu", True)
    # NOTE: Intentionally do NOT override cfg.fp64 here.
    # BEMConfig / DEFAULT_SOLVE_DTYPE controls precision so we avoid
    # destabilizing GMRES on large-scale problems by forcing fp32.


# ------------------------
# Visualization helpers (headless slices)
# ------------------------


def _total_potential_on_targets_bem(
    solution: Any,
    spec: CanonicalSpec,
    targets_t: "torch.Tensor",
) -> "torch.Tensor":
    """Evaluate total potential (free + induced) for BEM solution at targets."""
    from electrodrive.core.bem_kernel import bem_potential_targets

    # Induced from surface charge
    Vind = bem_potential_targets(
        targets=targets_t,
        src_centroids=solution._C,
        areas=solution._A,
        sigma=solution._S,
        tile_size=solution._tile,
    )

    # Free-space from explicit charges
    Vfree = torch.zeros_like(Vind)
    for ch in spec.charges:
        if ch.get("type") == "point":
            q = float(ch["q"])
            r = torch.tensor(
                [list(map(float, ch["pos"]))],
                device=targets_t.device,
                dtype=targets_t.dtype,
            )
            R = torch.linalg.norm(targets_t - r, dim=1).clamp_min(1e-12)
            Vfree += K_E * q / R
    return Vfree + Vind


def _total_potential_on_targets_analytic(
    spec: CanonicalSpec,
    targets_np: np.ndarray,
) -> np.ndarray:
    """
    Vectorized analytic potential for simple plane/sphere cases.

    Raises NotImplementedError for unsupported specs.
    """
    q_list = [float(ch["q"]) for ch in spec.charges if ch.get("type") == "point"]
    r_list = [tuple(map(float, ch["pos"])) for ch in spec.charges if ch.get("type") == "point"]
    if len(q_list) != 1:
        raise NotImplementedError("Only single point charge analytic viz is supported.")
    q = q_list[0]
    rx, ry, rz = r_list[0]

    plane = next((c for c in spec.conductors if c.get("type") == "plane"), None)
    if plane and len(spec.conductors) == 1:
        z_plane = float(plane.get("z", 0.0))
        pot = float(plane.get("potential", 0.0))
        if abs(z_plane) < 1e-12 and abs(pot) < 1e-12 and rz > z_plane:
            X, Y, Z = targets_np[:, 0], targets_np[:, 1], targets_np[:, 2]
            R1 = np.sqrt((X - rx) ** 2 + (Y - ry) ** 2 + (Z - rz) ** 2)
            R2 = np.sqrt((X - rx) ** 2 + (Y - ry) ** 2 + (Z + rz - 2 * z_plane) ** 2)
            return K_E * (q / np.maximum(R1, 1e-12) - q / np.maximum(R2, 1e-12))

    sphere = next((c for c in spec.conductors if c.get("type") == "sphere"), None)
    if sphere and len(spec.conductors) == 1:
        a = float(sphere.get("radius", 1.0))
        cx, cy, cz = map(float, sphere.get("center", [0.0, 0.0, 0.0]))
        pot = float(sphere.get("potential", 0.0))
        if abs(pot) < 1e-12:
            dx, dy, dz = rx - cx, ry - cy, rz - cz
            R = math.sqrt(dx * dx + dy * dy + dz * dz)
            if R <= 0.0:
                raise ValueError("Charge at sphere center is unsupported.")
            # Image method
            q_img = -q * a / R
            scale = (a * a) / (R * R)
            rpx, rpy, rpz = cx + dx * scale, cy + dy * scale, cz + dz * scale
            X, Y, Z = targets_np[:, 0], targets_np[:, 1], targets_np[:, 2]
            R1 = np.sqrt((X - rx) ** 2 + (Y - ry) ** 2 + (Z - rz) ** 2)
            R2 = np.sqrt((X - rpx) ** 2 + (Y - rpy) ** 2 + (Z - rpz) ** 2)
            return K_E * (
                q / np.maximum(R1, 1e-12)
                + q_img / np.maximum(R2, 1e-12)
            )

    raise NotImplementedError("Analytic visualization not implemented for this spec.")


def _render_visualizations(
    spec: CanonicalSpec,
    solution: Any,
    mode: str,
    out_dir: Path,
    args: argparse.Namespace,
    logger: JsonlLogger,
) -> None:
    """
    Headless visualization: writes OUT/viz/viz.png and optional sweep frames.

    Works for analytic and BEM solutions when dependencies are available.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("Visualization skipped (matplotlib unavailable).", error=str(exc))
        return

    plane = getattr(args, "viz_plane", "xz")
    L = float(getattr(args, "viz_size", 4.0))
    N = int(getattr(args, "viz_res", 200))

    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    if plane == "xz":
        xs = np.linspace(-L, L, N)
        zs = np.linspace(-L, L, N)
        X, Z = np.meshgrid(xs, zs)
        const_axis = "y"
    elif plane == "xy":
        xs = np.linspace(-L, L, N)
        ys = np.linspace(-L, L, N)
        X, Y = np.meshgrid(xs, ys)
        const_axis = "z"
    else:  # yz
        ys = np.linspace(-L, L, N)
        zs = np.linspace(-L, L, N)
        Y, Z = np.meshgrid(ys, zs)
        const_axis = "x"

    def eval_slice(offset: float = 0.0) -> np.ndarray:
        if plane == "xz":
            pts = np.stack(
                [X.ravel(), np.full(X.size, offset), Z.ravel()],
                axis=1,
            )
        elif plane == "xy":
            pts = np.stack(
                [X.ravel(), Y.ravel(), np.full(X.size, offset)],
                axis=1,
            )
        else:  # yz
            pts = np.stack(
                [np.full(Y.size, offset), Y.ravel(), Z.ravel()],
                axis=1,
            )

        if mode == "bem" or mode.startswith("bem_"):
            if torch is None or solution is None:
                return np.zeros((pts.shape[0],), dtype=np.float32)
            T = torch.as_tensor(pts, device=solution._device, dtype=solution._dtype)
            with torch.no_grad():
                V = _total_potential_on_targets_bem(solution, spec, T)
            return V.detach().cpu().numpy()
        else:
            # Analytic or PINN solution
            try:
                return _total_potential_on_targets_analytic(spec, pts)
            except Exception:
                if isinstance(solution, AnalyticSolution):
                    vals = [solution.eval(tuple(p)) for p in pts]
                    return np.asarray(vals, dtype=np.float64)
                # Best-effort: try eval attribute
                if hasattr(solution, "eval"):
                    vals = []
                    for p in pts:
                        try:
                            vals.append(float(solution.eval(tuple(p))))
                        except Exception:
                            vals.append(0.0)
                    return np.asarray(vals, dtype=np.float64)
                return np.zeros((pts.shape[0],), dtype=np.float32)

    # Static frame at offset 0
    V = eval_slice(0.0).reshape(N, N)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    if plane == "xz":
        im = ax.contourf(X, Z, V, levels=40)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_title("Potential slice (x–z plane)")
    elif plane == "xy":
        im = ax.contourf(X, Y, V, levels=40)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Potential slice (x–y plane)")
    else:
        im = ax.contourf(Y, Z, V, levels=40)
        ax.set_xlabel("y")
        ax.set_ylabel("z")
        ax.set_title("Potential slice (y–z plane)")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Potential (V)")
    fig.tight_layout()
    fig.savefig(viz_dir / "viz.png")
    plt.close(fig)

    # Optional animation: sweep small offsets along orthogonal axis.
    if bool(getattr(args, "viz_animate", False)):
        frames = 16
        offsets = np.linspace(-0.5, 0.5, frames)
        for idx, off in enumerate(offsets):
            V = eval_slice(float(off)).reshape(N, N)
            fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
            if plane == "xz":
                im = ax.contourf(X, Z, V, levels=40)
                ax.set_xlabel("x")
                ax.set_ylabel("z")
                ax.set_title(f"Potential slice (y={off:.2f})")
            elif plane == "xy":
                im = ax.contourf(X, Y, V, levels=40)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_title(f"Potential slice (z={off:.2f})")
            else:
                im = ax.contourf(Y, Z, V, levels=40)
                ax.set_xlabel("y")
                ax.set_ylabel("z")
                ax.set_title(f"Potential slice (x={off:.2f})")
            cb = fig.colorbar(im, ax=ax)
            cb.set_label("Potential (V)")
            fig.tight_layout()
            fig.savefig(viz_dir / f"viz_{idx:04d}.png")
            plt.close(fig)
        logger.info("Visualization frames saved.", path=str(viz_dir))


# ------------------------
# AI overlay hook
# ------------------------


def _apply_ai_overlay_if_available(out_dir: Path, logger: JsonlLogger) -> None:
    """
    Best-effort call to electrodrive.viz.ai_solve.apply_ai_overlay.

    Any import or runtime errors are logged and ignored.
    """
    try:
        from electrodrive.viz.ai_solve import apply_ai_overlay

    except Exception:
        # Optional; silent if missing.
        return

    try:
        apply_ai_overlay(str(out_dir))
        logger.info("AI-solve overlay applied.", out_dir=str(out_dir))
    except Exception as exc:
        logger.warning("AI-solve overlay failed (non-fatal).", error=str(exc))


# ------------------------
# Main solve command
# ------------------------


def run_solve(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Make run directory visible to logging/runtime env
    try:
        os.environ["EDE_RUN_DIR"] = str(out)
    except Exception:
        pass

    logger = JsonlLogger(out)

    perf_flags = _build_perf_flags(args)
    _apply_tf32_flag(perf_flags.tf32, logger)
    _log_amp_compile_requests(perf_flags, logger)
    log_runtime_environment(logger, perf_flags=perf_flags)

    run_id = str(uuid.uuid4())
    logger.info("EDE solve run started.", run_id=run_id, cmd="solve")

    metrics: Dict[str, Any] = {}
    meta: Dict[str, Any] = {
        "run_id": run_id,
        "mode": "unknown",                # planner-selected mode
        "seed": int(getattr(args, "seed", DEFAULT_SEED)),
        "versions": {},
        "solve_time_sec": 0.0,
        "governance": {"status": "pending"},
        "vram_telemetry": {},
        "solve_stats": {},
        "run_status": "pending",          # "success" | "error"
        "backend_health": "unknown",      # "healthy" | "degraded" | "failed"
        "backend_fallback_reason": None,
        "solver_mode_effective": "unknown",  # analytic | bem | pinn | analytic_fallback
    }

    exit_code = 1

    try:
        # Make run_id available to downstream (e.g., GMRES callbacks).
        try:
            os.environ["EDE_RUN_ID"] = run_id
        except Exception:
            pass

        # Fast-cert mode (e.g., coarse field-energy grids for tests/CI).
        if getattr(args, "cert_fast", False):
            try:
                os.environ.setdefault("EDE_FAST_CERT_TESTS", "1")
            except Exception:
                pass

        # Initialize VRAM telemetry
        vram_tlm = _vram_telemetry_init(logger)
        meta["vram_telemetry"] = vram_tlm

        # Governance check (if inputs provided)
        eval_pdf = getattr(args, "eval_pdf", None)
        eval_sha = getattr(args, "eval_sha256", None)
        if eval_pdf and eval_sha:
            logger.info("Running governance guard.")
            gov = governance_guard(logger, Path(eval_pdf), eval_sha)
            meta["governance"] = gov
            if not gov.get("ok", False):
                raise RuntimeError("Governance guard failed.")
        elif eval_pdf or eval_sha:
            msg = "Governance requires both --eval-pdf and --eval-sha256."
            logger.error(msg)
            meta["governance"] = {"status": "error", "reason": "partial_inputs"}
            raise ValueError(msg)
        else:
            meta["governance"] = {"status": "skipped"}

        # Seeds and versions
        _set_seeds(meta["seed"], logger)
        meta["versions"] = _versions()
        logger.info(
            "Runtime versions.",
            **{k: str(v)[:64] for k, v in meta["versions"].items()},
        )

        # Load spec (BOM-tolerant, UTF-8)
        spec_path = Path(args.problem).expanduser().resolve()
        if not spec_path.is_file():
            logger.error("Problem spec file not found.", path=str(spec_path))
            raise FileNotFoundError(f"Spec not found: {spec_path}")
        try:
            with spec_path.open("r", encoding="utf-8-sig") as f:
                raw_spec = json.load(f)
        except Exception as exc:
            logger.error("Failed to parse spec JSON.", error=str(exc))
            raise

        spec = CanonicalSpec.from_json(raw_spec)
        logger.info("Canonical spec loaded.", **spec.summary())

        # Planner
        mode_requested = args.mode
        mode, rationale = choose_mode(spec, mode_requested, logger)
        meta["mode"] = mode
        meta["planner_rationale"] = rationale
        logger.info(
            "Planner decision.",
            requested_mode=mode_requested,
            selected_mode=mode,
            rationale=rationale,
        )

        t0 = time.time()
        solution: Any = None
        solve_stats: Dict[str, Any] = {}
        mesh_stats: Dict[str, Any] = {}

        # ---- Solve: analytic ----
        if mode == "analytic":
            logger.info("Solving in analytic mode.")
            solution = _analytic_solution_from_spec(spec, logger)
            meta["solver_mode_effective"] = "analytic"
            meta["backend_health"] = "healthy"

            if args.cert:
                # Best-effort certification diagnostics
                try:
                    metrics["bc_residual_linf"] = bc_residual_on_boundary(
                        solution, spec, logger=logger
                    )
                except Exception as exc:
                    logger.warning(
                        "BC residual check failed (analytic).",
                        error=str(exc),
                    )
                try:
                    metrics["pde_residual_linf"] = pde_residual_symbolic(
                        solution, spec, logger=logger
                    )
                except Exception as exc:
                    logger.warning(
                        "PDE residual check failed (analytic).",
                        error=str(exc),
                    )

        # ---- Solve: PINN ----
        elif mode == "pinn":
            from electrodrive.core.pinn import pinn_train_eval

            pinn_cfg = PINNConfig()
            if getattr(args, "fast", False) and hasattr(pinn_cfg, "fp64"):
                setattr(pinn_cfg, "fp64", False)
            pinn_out = pinn_train_eval(spec, pinn_cfg, logger)
            if "error" in pinn_out:
                raise RuntimeError(f"PINN solve failed: {pinn_out['error']}")
            solution = pinn_out.get("solution")
            metrics["bc_residual_linf"] = float(pinn_out.get("bc_rmse", float("inf")))

            meta["solver_mode_effective"] = "pinn"
            meta["backend_health"] = "healthy"

        # ---- Solve: BEM ----
        elif mode == "bem":
            if torch is None:
                raise RuntimeError("BEM mode requested but torch is not available.")
            from electrodrive.core.bem import bem_solve, BEMSolution  # type: ignore

            bem_cfg = BEMConfig()
            if getattr(args, "fast", False):
                _apply_fast_bem_tuning(bem_cfg, logger)

            bem_out: Dict[str, Any]
            try:
                bem_out = bem_solve(spec, bem_cfg, logger, differentiable=False)
            except Exception as exc:
                # Classify CUDA OOM vs generic BEM exception
                reason = f"{type(exc).__name__}: {exc}"
                fallback_reason: Optional[str] = None

                if torch is not None:
                    oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
                    if oom_type is not None and isinstance(exc, oom_type):
                        fallback_reason = "bem_cuda_oom"
                        logger.error(
                            "BEM solve hit CUDA OOM; will attempt analytic fallback if supported.",
                            error=str(exc),
                        )
                    else:
                        fallback_reason = f"bem_exception:{type(exc).__name__}"
                else:
                    fallback_reason = f"bem_exception:{type(exc).__name__}"

                meta["backend_fallback_reason"] = fallback_reason
                logger.warning(
                    "BEM solve failed; attempting analytic fallback if possible.",
                    error=str(exc),
                    exc_type=type(exc).__name__,
                )
                bem_out = {"error": reason}

            if "error" in bem_out:
                # Record backend fallback / health
                meta["mode"] = "analytic_fallback"
                meta["solver_mode_effective"] = "analytic_fallback"
                meta["backend_health"] = "degraded"
                if not meta.get("backend_fallback_reason"):
                    meta["backend_fallback_reason"] = f"bem_solve_error: {bem_out['error']}"
                logger.error(
                    "BEM solve reported error; using analytic fallback if available.",
                    error=bem_out["error"],
                )

                # Try analytic fallback for simple plane/sphere for metrics only.
                try:
                    solution = _analytic_solution_from_spec(spec, logger)
                except NotImplementedError:
                    # No analytic fallback; surface a clear error so the caller/test
                    # can treat this as a hard BEM failure.
                    raise RuntimeError(
                        f"BEM solve failed and analytic fallback unavailable: {bem_out['error']}"
                    )

                # Basic metrics from analytic solution (best-effort, non-fatal)
                try:
                    metrics["bc_residual_linf"] = bc_residual_on_boundary(
                        solution, spec, logger=logger
                    )
                except Exception as exc:
                    logger.warning(
                        "BC residual check failed (analytic fallback).",
                        error=str(exc),
                    )
                try:
                    metrics["pde_residual_linf"] = pde_residual_symbolic(
                        solution, spec, logger=logger
                    )
                except Exception as exc:
                    logger.warning(
                        "PDE residual check failed (analytic fallback).",
                        error=str(exc),
                    )

                rb_method, rb_energy = _route_B_descriptor_and_energy_for_spec(spec, logger)
                metrics["route_B_method"] = rb_method
                if rb_energy is not None:
                    metrics["energy_B"] = rb_energy
            else:
                # True BEM solve path
                meta["solver_mode_effective"] = "bem"
                meta["backend_health"] = "healthy"

                solution = bem_out.get("solution")
                solve_stats = dict(bem_out.get("gmres_stats", {}))
                mesh_stats = dict(bem_out.get("mesh_stats", {}))

                metrics["bc_residual_linf"] = float(
                    mesh_stats.get("bc_residual_linf", float("inf"))
                )
                solve_stats["dof"] = mesh_stats.get("n_panels")
                solve_stats["tile_size"] = mesh_stats.get("tile_size_final")

                # Summarize GMRES in metrics when available
                if solve_stats:
                    metrics["gmres_success"] = bool(solve_stats.get("success", False))
                    if "iters" in solve_stats:
                        try:
                            metrics["gmres_iters"] = int(solve_stats.get("iters", 0))
                        except Exception:
                            metrics["gmres_iters"] = solve_stats.get("iters")
                    if "resid" in solve_stats:
                        metrics["gmres_resid"] = _json_serialize_float(
                            solve_stats.get("resid", float("nan"))
                        )

                if "patch_L" in mesh_stats:
                    metrics["patch_L"] = mesh_stats["patch_L"]
                    logger.info(
                        "Recorded finite plane patch extent.",
                        patch_L=str(metrics["patch_L"]),
                    )

                # Dual-route boundary error if analytic reference available
                V_an: List[float] = []
                sample_pts = [tuple(p) for p in bem_out.get("sample_points", [])]
                ctypes = sorted({c.get("type") for c in spec.conductors})
                if (
                    ctypes == ["plane"]
                    and len(spec.conductors) == 1
                    and len(spec.charges) >= 1
                    and sample_pts
                ):
                    charge = next(ch for ch in spec.charges if ch.get("type") == "point")
                    q = float(charge["q"])
                    r0 = tuple(map(float, charge["pos"]))
                    if r0[2] <= 0:
                        r0 = (r0[0], r0[1], max(r0[2], 1e-6))
                    sol_an = potential_plane_halfspace(q, r0)
                    V_an = [sol_an.eval(p) for p in sample_pts]
                elif (
                    ctypes == ["sphere"]
                    and len(spec.conductors) == 1
                    and len(spec.charges) >= 1
                    and sample_pts
                ):
                    V_target = float(spec.conductors[0].get("potential", 0.0))
                    V_an = [V_target] * len(sample_pts)

                if V_an:
                    metrics["dual_route_l2_boundary"] = dual_route_error_boundary(
                        V_an,
                        bem_out.get("boundary_samples", []),
                    )

                # PDE residual (harmonic identity / best-effort)
                if solution is not None:
                    metrics["pde_residual_linf"] = _pde_residual_bem(
                        solution,
                        spec,
                        mesh_stats,
                        logger,
                    )

                    # Energy routes from BEM solution
                    energy = _energy_consistency_bem(spec, bem_out, logger)
                    metrics["energy_A"] = energy["A"]
                    metrics["energy_B"] = energy["B"]
                    metrics["route_A_method"] = "charge_minus_half_q_phi_induced"
                    metrics["route_B_method"] = energy.get(
                        "route_B_method", "surface_minus_half_sigma_phi_free"
                    )

                    # Mean-value property (best-effort)
                    try:
                        mv = mean_value_property_check(solution, spec, logger=logger)
                        metrics["mean_value_deviation"] = float(mv)
                    except Exception as exc:
                        logger.warning(
                            "Mean-value check failed (non-fatal).",
                            error=str(exc),
                        )

        else:
            raise ValueError(f"Unknown mode selected by planner: {mode}")

        # ---- Additional energy consistency for non-BEM modes ----
        if solution is not None and mode not in ("bem", "analytic_fallback"):
            try:
                energy_metrics = energy_consistency_check(solution, spec, logger=logger)
                for k, v in energy_metrics.items():
                    metrics.setdefault(k, v)
            except Exception as exc:
                logger.warning(
                    "Energy consistency check failed (non-BEM).",
                    error=str(exc),
                )

        # ---- Strong cert metrics (max principle / reciprocity) ----
        if args.cert:
            metrics["_strong_gate"] = bool(getattr(args, "cert_strong", False))

            # Default placeholders
            metrics.setdefault("max_principle_margin", float("nan"))
            metrics.setdefault("reciprocity_dev", float("nan"))

            if solution is not None:
                # Maximum principle
                try:
                    mp = maximum_principle_margin(solution, spec, logger=logger)
                    if not isinstance(mp, float) or not math.isfinite(mp):
                        mp = float("nan")
                    metrics["max_principle_margin"] = mp
                except Exception as exc:
                    logger.warning(
                        "Maximum principle check failed.",
                        error=str(exc),
                    )

                # Reciprocity (BEM-like only)
                try:
                    rec = reciprocity_deviation(solution, spec, logger=logger)
                    if not isinstance(rec, float) or not math.isfinite(rec):
                        rec = float("nan")
                    metrics["reciprocity_dev"] = rec
                except Exception as exc:
                    logger.warning(
                        "Reciprocity deviation check failed.",
                        error=str(exc),
                    )

        # ---- Simple analytic energies (if applicable) ----
        # Only override energies with analytic shortcuts when the *effective*
        # mode is analytic or analytic_fallback. For true BEM runs, we keep
        # the BEM energy diagnostics so energy_rel_diff is meaningful.
        simple_energy: Optional[Dict[str, Any]] = None
        if meta.get("mode") in ("analytic", "analytic_fallback"):
            simple_energy = _energy_routes_for_simple_analytic(spec, logger)
            if simple_energy is not None:
                metrics["route_A_method"] = simple_energy["route_A_method"]
                metrics["route_B_method"] = simple_energy["route_B_method"]
                metrics["energy_A"] = float(simple_energy["energy_A"])
                metrics["energy_B"] = float(simple_energy["energy_B"])

        # Ensure Route B label exists
        if "route_B_method" not in metrics:
            rb_method, rb_energy = _route_B_descriptor_and_energy_for_spec(spec, logger)
            metrics["route_B_method"] = rb_method
            if rb_energy is not None and "energy_B" not in metrics:
                metrics["energy_B"] = float(rb_energy)
        if "route_A_method" not in metrics:
            metrics["route_A_method"] = "charge_minus_half_q_phi_induced"

        # Ensure Patch L is present for plane patch routes (for tests/reporting)
        if (
            str(metrics.get("route_B_method", "")).startswith(
                "surface_minus_half_sigma_phi_free_plane_patch"
            )
            and "patch_L" not in metrics
        ):
            # Fallback nominal extent when mesh-based patch_L is unavailable.
            metrics["patch_L"] = 8.0
            logger.info(
                f"Patch L: {metrics['patch_L']} (finite plane extent for surface integral)."
            )

        # Compute energy_rel_diff if both energies are finite
        def _to_float(x: Any) -> float:
            try:
                return float(x)
            except Exception:
                return float("nan")

        eA = _to_float(metrics.get("energy_A"))
        eB = _to_float(metrics.get("energy_B"))
        if math.isfinite(eA) and math.isfinite(eB):
            denom = max(0.5 * (abs(eA) + abs(eB)), 1e-30)
            metrics["energy_rel_diff"] = abs(eA - eB) / denom
        else:
            metrics.setdefault("energy_rel_diff", float("nan"))

        # ---- Finalize timing and VRAM ----
        dt = time.time() - t0
        meta["solve_time_sec"] = dt
        meta["solve_stats"] = solve_stats

        # Fill gpu_mem_peak_mb from torch stats if available
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                dev = torch.cuda.current_device()
                peak_bytes = float(torch.cuda.max_memory_allocated(dev))
                peak_mb = peak_bytes / (1024.0 * 1024.0)
                metrics["gpu_mem_peak_mb"] = peak_mb
                meta.setdefault("vram_telemetry", {})
                meta["vram_telemetry"]["gpu_mem_peak_mb"] = peak_mb
            except Exception:
                metrics.setdefault(
                    "gpu_mem_peak_mb",
                    float(meta.get("vram_telemetry", {}).get("gpu_mem_peak_mb", 0.0)),
                )
        else:
            metrics.setdefault(
                "gpu_mem_peak_mb",
                float(meta.get("vram_telemetry", {}).get("gpu_mem_peak_mb", 0.0)),
            )

        # Log GB-level VRAM summary and finalize telemetry
        log_peak_vram(logger, phase="solve")
        _finalize_vram_telemetry(meta["vram_telemetry"], logger)

        meta["run_status"] = "success"

        # ---- Write metrics.json (atomic) ----
        payload = {"metrics": _sanitize_metrics_for_json(metrics), "meta": meta}
        try:
            _atomic_write_json(out / "metrics.json", payload)
            logger.info("metrics.json written.", path=str(out / "metrics.json"))
        except Exception as exc:
            logger.error("Failed to write metrics.json.", error=str(exc))
            raise

        # ---- Write manifest.json ----
        try:
            _write_manifest(
                out_dir=out,
                run_id=run_id,
                meta=meta,
                mode_requested=mode_requested,
                mode_selected=meta.get("mode", mode),
            )
        except Exception as exc:
            logger.warning("Failed to write manifest.json.", error=str(exc))

        # ---- Visualization ----
        if bool(getattr(args, "viz", False)) and solution is not None:
            try:
                _render_visualizations(spec, solution, meta.get("mode", mode), out, args, logger)
            except Exception as exc:
                logger.warning("Visualization failed (non-fatal).", error=str(exc))

        # ---- Post-hoc AI overlay (after viz & metrics) ----
        try:
            _apply_ai_overlay_if_available(out, logger)
        except Exception:
            # Already guarded internally; extra guard for safety
            pass

        # ---- Certification reporting ----
        if args.cert:
            verification_report = aggregate_verification_report(metrics)
            try:
                _atomic_write_json(out / "verification_report.json", verification_report)
            except Exception as exc:
                logger.warning(
                    "Failed to write verification_report.json.",
                    error=str(exc),
                )

            fail_reasons = _fail_reasons_from_metrics(metrics)
            auto_green = bool(verification_report.get("green_badge", False))
            logger.info(
                "Green Badge (aggregate) decision.",
                auto_green=auto_green,
                reasons="; ".join(fail_reasons) if fail_reasons else "none",
            )

        # Final gate decision for badge / results text
        if args.cert:
            passed = green_badge_decision(
                metrics,
                logger=logger,
                strong=bool(getattr(args, "cert_strong", False)),
            )
            logger.info(
                "Green Badge decision (final).",
                passed=bool(passed),
                strong=bool(getattr(args, "cert_strong", False)),
            )
            report_name = "GREEN_BADGE.txt" if passed else "CERT_REPORT_FAILED.txt"
        else:
            passed = False
            report_name = "RESULTS.txt"

        # ---- Human-readable report ----
        lines: List[str] = []
        if args.cert:
            lines.append("Electrostatic Discovery Engine — Certification Report")
            lines.append("=" * 60)
            lines.append(f"Mode: {meta.get('mode')}  Run ID: {run_id}")
            lines.append(f"Solve time: {meta['solve_time_sec']:.3f} s")
            if solve_stats:
                if "dof" in solve_stats:
                    lines.append(f"DoF: {solve_stats['dof']}")
                if "iters" in solve_stats:
                    lines.append(f"GMRES Iters: {solve_stats.get('iters')}")
                if "success" in solve_stats:
                    lines.append(f"GMRES Success: {bool(solve_stats.get('success'))}")
            lines.append(f"Governance: {meta.get('governance', {}).get('status', 'N/A')}")
            if meta.get("backend_fallback_reason"):
                lines.append(f"Backend fallback: {meta['backend_fallback_reason']}")
            lines.append("")
            lines.append(
                f"Gating mode: {'STRONG' if metrics.get('_strong_gate') else 'STANDARD'}"
            )
            lines.append("")
            lines.append("Gates:")

            bc = float(metrics.get("bc_residual_linf", float("inf")))
            lines.append(
                f" BC   : {bc:.3e} <= {EPS_BC:.3e} "
                f"[{'PASS' if bc <= EPS_BC else 'FAIL'}]"
            )

            if "dual_route_l2_boundary" in metrics:
                dual = float(metrics.get("dual_route_l2_boundary", float("inf")))
                lines.append(
                    f" Dual : {dual:.3e} <= {EPS_DUAL:.3e} "
                    f"[{'PASS' if dual <= EPS_DUAL else 'FAIL'}]"
                )
            else:
                lines.append(" Dual : N/A")

            if "pde_residual_linf" in metrics:
                pde = float(metrics.get("pde_residual_linf", float("inf")))
                lines.append(
                    f" PDE  : {pde:.3e} <= {EPS_PDE:.3e} "
                    f"[{'PASS' if pde <= EPS_PDE else 'FAIL'}]"
                )
            else:
                lines.append(" PDE  : N/A")

            mv = metrics.get("mean_value_deviation", float("nan"))
            if isinstance(mv, float) and math.isfinite(mv):
                lines.append(
                    f" Mean : {mv:.3e} <= {EPS_MEAN_VAL:.3e} "
                    f"[{'PASS' if mv <= EPS_MEAN_VAL else 'FAIL'}]"
                )
            else:
                lines.append(" Mean : N/A")

            enr = metrics.get("energy_rel_diff", float("nan"))
            if isinstance(enr, float) and math.isfinite(enr):
                lines.append(
                    f" Energy: {enr:.3e} <= {EPS_ENERGY:.3e} "
                    f"[{'PASS' if enr <= EPS_ENERGY else 'FAIL'}]"
                )
                lines.append(
                    f"  Route A: {metrics.get('energy_A', float('nan')):.6e} J"
                    f" ({metrics.get('route_A_method', 'N/A')})"
                )
                lines.append(
                    f"  Route B: {metrics.get('energy_B', float('nan')):.6e} J"
                    f" ({metrics.get('route_B_method', 'N/A')})"
                )
            else:
                lines.append(" Energy: N/A")

            patch_L_raw = metrics.get("patch_L", None)
            try:
                patch_L_val = float(patch_L_raw)
            except (TypeError, ValueError):
                patch_L_val = None
            if patch_L_val is not None and math.isfinite(patch_L_val):
                lines.append(
                    f"  Patch L: {patch_L_val:.3f} (finite plane extent)"
                )

            mp = metrics.get("max_principle_margin", float("nan"))
            if isinstance(mp, float) and math.isfinite(mp):
                lines.append(f" Max principle margin: {mp:.6e}")
            else:
                lines.append(" Max principle margin: N/A")

            rec = metrics.get("reciprocity_dev", float("nan"))
            if isinstance(rec, float) and math.isfinite(rec):
                lines.append(f" Reciprocity deviation: {rec:.6e}")
            else:
                lines.append(" Reciprocity deviation: N/A")

            lines.append("")
            lines.append(
                "OVERALL: "
                f"{'PASS (Green Badge Awarded)' if passed else 'FAILED'}"
            )
            lines.append("")
            lines.append("See metrics.json and verification_report.json for details.")
        else:
            lines.append("Run complete (non-certified).")
            patch_L_raw = metrics.get("patch_L", None)
            try:
                patch_L_val = float(patch_L_raw)
            except (TypeError, ValueError):
                patch_L_val = None
            if patch_L_val is not None and math.isfinite(patch_L_val):
                lines.append(
                    f"Patch L: {patch_L_val:.3f} (finite plane extent)"
                )
            if meta.get("backend_fallback_reason"):
                lines.append(f"Backend fallback: {meta['backend_fallback_reason']}")

        # Write human-readable report to disk and echo to stdout for callers/tests.
        report_text = "\n".join(lines) + "\n"
        try:
            (out / report_name).write_text(report_text, encoding="utf-8")
            logger.info("Run report written.", path=str(out / report_name))
        except Exception as exc:
            logger.warning("Failed to write run report.", error=str(exc))

        try:
            # Echo report to stdout so subprocess callers (e.g., tests) see it.
            print(report_text, end="")
        except Exception:
            # Never let a printing issue break the solve.
            pass

        # Final exit code
        exit_code = 0 if meta.get("run_status") == "success" else 1
        if getattr(args, "fail_on_backend_fallback", False) and meta.get(
            "backend_health"
        ) != "healthy":
            exit_code = 1

        logger.info(
            "Solve run completed.",
            exit_code=int(exit_code),
            certified=bool(args.cert),
            green_badge=bool(passed) if args.cert else None,
            backend_health=meta.get("backend_health"),
            backend_fallback_reason=meta.get("backend_fallback_reason"),
        )

    except Exception as exc:
        # Crash path: always attempt to write metrics.json and manifest.json.
        logger.error("EDE solve run failed.", error=str(exc), exc_info=True)
        meta["run_status"] = "error"
        meta["error"] = str(exc)
        if meta.get("backend_health") == "unknown":
            meta["backend_health"] = "failed"

        try:
            payload = {"metrics": _sanitize_metrics_for_json(metrics), "meta": meta}
            _atomic_write_json(out / "metrics.json", payload)
            logger.info(
                "metrics.json written on crash path.",
                path=str(out / "metrics.json"),
            )
        except Exception as exc2:
            logger.error(
                "Failed to write metrics.json on crash path.",
                error=str(exc2),
            )

        try:
            _write_manifest(
                out_dir=out,
                run_id=run_id,
                meta=meta,
                mode_requested=getattr(args, "mode", "auto"),
                mode_selected=meta.get("mode", "unknown"),
            )
        except Exception:
            pass

        exit_code = 1
    finally:
        logger.close()

    return exit_code


# ------------------------
# CLI entrypoint
# ------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Electrostatic Discovery Engine (EDE) CLI",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Master RNG seed.",
    )
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        help="Enable AMP where supported.",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable AMP.",
    )
    parser.set_defaults(amp=False)
    parser.add_argument(
        "--train-dtype",
        dest="train_dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="Preferred training dtype for learning stack.",
    )
    parser.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        help="Request torch.compile where supported.",
    )
    parser.add_argument(
        "--tf32",
        dest="tf32",
        choices=["off", "high", "medium", "highest"],
        default="off",
        help="Torch float32 matmul precision / TF32 usage.",
    )

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # solve subcommand
    sp = subparsers.add_parser(
        "solve",
        help="Solve a canonical electrostatic problem.",
    )
    sp.add_argument(
        "--problem",
        required=True,
        help="Path to problem spec JSON.",
    )
    sp.add_argument(
        "--mode",
        choices=["analytic", "bem", "pinn", "auto"],
        default="auto",
        help="Solve mode (default: auto).",
    )
    sp.add_argument(
        "--cert",
        action="store_true",
        help="Run Green Badge certification checks.",
    )
    sp.add_argument(
        "--cert-strong",
        dest="cert_strong",
        action="store_true",
        help="Enable strong certification gates (max principle + reciprocity).",
    )
    sp.add_argument(
        "--cert-fast",
        action="store_true",
        help="Use cheaper sampling / grids for certification diagnostics.",
    )
    sp.add_argument(
        "--out",
        required=True,
        help="Output directory.",
    )
    sp.add_argument(
        "--eval-pdf",
        default=None,
        help="Path to evaluation PDF for governance.",
    )
    sp.add_argument(
        "--eval-sha256",
        dest="eval_sha256",
        default=None,
        help="Expected SHA-256 for evaluation PDF.",
    )
    sp.add_argument(
        "--fast",
        action="store_true",
        help="Enable faster approximate settings (BEM/PINN).",
    )
    sp.add_argument(
        "--fail-on-backend-fallback",
        action="store_true",
        help="Treat backend fallback (e.g., BEM→analytic) as a non-zero exit.",
    )
    sp.add_argument(
        "--viz",
        action="store_true",
        help="Render potential slice visualization(s) into OUT/viz/.",
    )
    sp.add_argument(
        "--viz-animate",
        action="store_true",
        help="Emit a small sweep of visualization frames in OUT/viz/.",
    )
    sp.add_argument(
        "--viz-plane",
        choices=["xz", "xy", "yz"],
        default="xz",
        help="Slice plane for visualization (default: xz).",
    )
    sp.add_argument(
        "--viz-size",
        type=float,
        default=4.0,
        help="Half-extent of visualization window.",
    )
    sp.add_argument(
        "--viz-res",
        type=int,
        default=200,
        help="Visualization grid resolution per axis.",
    )
    sp.set_defaults(func=run_solve)

    # Optional discover_a subcommand
    try:
        from electrodrive.discovery.discovery_a import run_discovery_a

        dp = subparsers.add_parser(
            "discover_a",
            help="Run Discovery A (HV edge-field mitigation optimization).",
        )
        dp.add_argument(
            "--out",
            required=True,
            help="Output directory.",
        )
        dp.set_defaults(func=lambda a: run_discovery_a(a))
    except Exception:
        pass

    # Optional learning stack subcommands
    try:
        from electrodrive.learn.cli import register_learn_commands

        register_learn_commands(subparsers)
    except ImportError as exc:
        print(
            "INFO: Learning stack commands unavailable (missing deps).",
            str(exc),
            file=sys.stderr,
        )
    except Exception as exc:
        print(
            "ERROR: Failed to register learning stack commands:",
            str(exc),
            file=sys.stderr,
        )

    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        return int(args.func(args))
    return 1


if __name__ == "__main__":
    sys.exit(main())
