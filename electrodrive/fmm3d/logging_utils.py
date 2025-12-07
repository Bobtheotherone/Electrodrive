from __future__ import annotations

"""
Logging helpers specific to the FMM stack.

Responsibilities
----------------
- Provide lightweight wrappers around the project's JsonlLogger.
- Centralize FMM-specific log keys (N, p, backend, MAC, etc.).
- Make it easy to correlate BEM-level logs with FMM-level events.
- Optionally emit structured JSONL records for offline analysis.
- Provide robust, crash-safe debugging tools for mathematical analysis.
- Ensure verbose console logging works additively with structured loggers.

Environment variables
---------------------
EDE_FMM_ENABLE_JSONL
    If set to a non-zero / truthy value ("1", "true", "yes", "on"),
    FMM test results are emitted as one-JSON-object-per-line.
    (JSONL is also implicitly enabled if EDE_FMM_JSONL_PATH is set.)

EDE_FMM_JSONL_PATH
    Optional filesystem path. If set, log_test_result_jsonl appends one
    JSON object per line to this file (UTF-8, no BOM) and JSONL emission
    is considered enabled even if EDE_FMM_ENABLE_JSONL is unset.

EDE_FMM_JSONL_NO_STDOUT
    If truthy and EDE_FMM_JSONL_PATH is set, suppress printing JSON records
    to stdout. This is useful when you want clean human-readable stdout and
    a separate machine-readable JSONL file.

EDE_FMM_LOG_LEVEL
    Optional log level hint for log_fmm_event. One of:
    {"debug", "info", "warning", "error", "critical"} (case-insensitive).

EDE_FMM_JSONL_DEBUG
    If set to a truthy value, emit verbose debug information about the
    JSONL logging pipeline to stderr. Intended for diagnosing wiring issues.

ELECTRODRIVE_FMM_DEBUG_VERBOSE
    Legacy flag. If set to "1", enables verbose console logging.
"""

import json
import math
import os
import sys
from typing import Any, Mapping, MutableMapping, Optional, Union, List

import torch
from torch import Tensor

try:
    # Project-wide structured logger type. We do not *require* this to exist
    # at runtime; all helpers degrade gracefully if it cannot be imported.
    from electrodrive.utils.logging import JsonlLogger  # type: ignore
except Exception:  # pragma: no cover
    JsonlLogger = None  # type: ignore


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

_TRUE_VALUES = {"1", "true", "yes", "on", "y"}
_VERBOSE_ENV = "ELECTRODRIVE_FMM_DEBUG_VERBOSE"


def _normalize_bool_env(name: str, default: bool = False) -> bool:
    """
    Interpret an environment variable as a boolean.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default

    val = raw.strip().lower()
    if val in _TRUE_VALUES:
        return True
    if val in {"0", "false", "no", "off", "n"}:
        return False
    return default


def want_verbose_debug(default: bool = False) -> bool:
    """
    Single source of truth for 'turn on noisy FMM debug logs'.

    True if either:
      - ELECTRODRIVE_FMM_DEBUG_VERBOSE is truthy, or
      - EDE_FMM_LOG_LEVEL == 'debug'
    """
    # Legacy knob
    raw = os.environ.get(_VERBOSE_ENV, "")
    if raw.strip().lower() in _TRUE_VALUES:
        return True

    # New knob via log level
    lvl = os.environ.get("EDE_FMM_LOG_LEVEL", "").strip().lower()
    if lvl == "debug":
        return True

    return default


def _debug_jsonl(msg: str) -> None:
    """
    Best-effort debug output for JSONL wiring issues.
    """
    if not _normalize_bool_env("EDE_FMM_JSONL_DEBUG", default=False):
        return
    try:
        sys.stderr.write(f"[EDE_FMM_JSONL_DEBUG] {msg}\n")
        sys.stderr.flush()
    except Exception:
        return


def want_jsonl() -> bool:
    """
    Return True if JSONL emission is enabled.
    """
    enabled_flag = _normalize_bool_env("EDE_FMM_ENABLE_JSONL", default=False)
    has_path = bool(os.environ.get("EDE_FMM_JSONL_PATH", "").strip())
    enabled = enabled_flag or has_path
    _debug_jsonl(
        "want_jsonl: "
        f"EDE_FMM_ENABLE_JSONL={os.environ.get('EDE_FMM_ENABLE_JSONL')!r}, "
        f"EDE_FMM_JSONL_PATH={os.environ.get('EDE_FMM_JSONL_PATH')!r} "
        f"-> {enabled}"
    )
    return enabled


def get_log_level() -> str:
    """
    Return a normalized log level for FMM events based on EDE_FMM_LOG_LEVEL.
    """
    raw = os.environ.get("EDE_FMM_LOG_LEVEL", "info")
    lvl = raw.strip().lower()
    if lvl not in {"debug", "info", "warning", "error", "critical"}:
        return "info"
    return lvl


# ---------------------------------------------------------------------------
# JSON-safe conversion
# ---------------------------------------------------------------------------


def _to_jsonable(obj: Any) -> Any:
    """
    Best-effort conversion of arbitrary Python objects into something
    JSON-serializable.
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            if math.isnan(obj):
                return "NaN"
            return "Infinity" if obj > 0.0 else "-Infinity"
        return float(obj)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return repr(obj)


# ---------------------------------------------------------------------------
# Loggers: Console & Combined
# ---------------------------------------------------------------------------

class ConsoleLogger:
    """
    Simple stdout logger for FMM debug / spectral traces.
    Safe to use anywhere; no external logging config needed.
    """
    def info(self, msg: str, **_: Any) -> None:
        print(f"[FMM-SPECTRAL] {msg}", flush=True)

    def warning(self, msg: str, **_: Any) -> None:
        print(f"[FMM-SPECTRAL-WARN] {msg}", flush=True)

    def error(self, msg: str, **_: Any) -> None:
        print(f"[FMM-SPECTRAL-ERR] {msg}", flush=True)

    def debug(self, msg: str, **_: Any) -> None:
        print(f"[FMM-DEBUG] {msg}", flush=True)


class CombinedLogger:
    """
    Fans out log calls to multiple loggers.
    Used to ensure verbose console logs occur even if a structured logger is present.
    """
    def __init__(self, *loggers: Any) -> None:
        self._loggers = [l for l in loggers if l is not None]

    def _broadcast(self, method_name: str, msg: str, **kwargs: Any) -> None:
        for lg in self._loggers:
            # Try specific method (e.g. lg.debug), fallback to info, or skip if missing
            fn = getattr(lg, method_name, None)
            if fn is None and method_name != "info":
                fn = getattr(lg, "info", None)
            
            if callable(fn):
                try:
                    fn(msg, **kwargs)
                except Exception:
                    pass

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._broadcast("debug", msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._broadcast("info", msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._broadcast("warning", msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._broadcast("error", msg, **kwargs)


def get_logger(logger: Optional[Any] = None) -> Any:
    """
    Returns the appropriate logger instance.
    
    Logic:
    1. If verbose debug is OFF:
       - Return `logger` if provided.
       - Return None if `logger` is None.
       
    2. If verbose debug is ON:
       - Return `ConsoleLogger` if `logger` is None.
       - Return `CombinedLogger(logger, ConsoleLogger)` if `logger` is provided.
         (This ensures console output is not suppressed by the presence of a structured logger).
    """
    verbose = want_verbose_debug()
    console = ConsoleLogger() if verbose else None

    if logger is None:
        return console

    if console is None:
        return logger

    # Check for idempotency to avoid nested wrappers (optional safety)
    if isinstance(logger, CombinedLogger):
        return logger

    # Verbose mode + structured logger -> send to both
    return CombinedLogger(logger, console)


# ---------------------------------------------------------------------------
# Robust Tensor Debugging
# ---------------------------------------------------------------------------

def _safe_tensor(x: Any) -> Optional[Tensor]:
    """
    Best-effort conversion of arbitrary array-like objects to a torch.Tensor.
    Returns None if conversion fails.
    """
    if isinstance(x, torch.Tensor):
        return x
    try:
        return torch.as_tensor(x)
    except Exception:
        return None


def debug_tensor_stats(name: str, x: Any, logger: Optional[Any] = None) -> None:
    """
    Robust debug print that never crashes if x is list/None/etc.
    If logger is None, it respects the global verbosity setting (creating a ConsoleLogger if needed).
    """
    # If no logger provided, check if we want verbose debug globally
    if logger is None:
        if not want_verbose_debug():
            return
        logger = ConsoleLogger()

    # 1. Handle None
    if x is None:
        logger.debug(f"{name}: <None>")
        return

    # 2. Robust conversion
    t = _safe_tensor(x)
    
    # 3. If conversion failed (e.g. list of mixed types), log type info safely
    if t is None:
        safe_type = type(x).__name__
        try:
            length = len(x) # type: ignore
            logger.debug(f"{name}: <{safe_type} len={length}> (not a tensor)")
        except Exception:
            logger.debug(f"{name}: <{safe_type}> (not a tensor)")
        return

    if t.numel() == 0:
        logger.debug(f"{name}: empty tensor")
        return

    shape = tuple(t.shape)
    try:
        if t.is_complex():
            mag = t.abs()
            logger.debug(
                f"{name}: shape={shape} (complex), "
                f"MaxMag={mag.max().item():.3e}, "
                f"MeanMag={mag.mean().item():.3e}"
            )
        else:
            # FIX: ensure floats for mean calculation to avoid LongTensor issues
            t_float = t.float()
            logger.debug(
                f"{name}: shape={shape}, "
                f"min={t_float.min().item():.3e}, "
                f"max={t_float.max().item():.3e}, "
                f"mean={t_float.mean().item():.3e}"
            )
    except Exception as exc:
        # If item() fails or complex types cause issues
        logger.debug(f"{name}: shape={shape} (stats failed: {exc})")


# ---------------------------------------------------------------------------
# Low-level JSONL file writer
# ---------------------------------------------------------------------------


def _write_jsonl_to_path(path: str, line: str) -> None:
    """
    Append a single JSONL line to the given path (UTF-8, no BOM).
    """
    if not path:
        return
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")
    except Exception as exc:
        _debug_jsonl(f"write_jsonl_to_path: ERROR {exc!r}")
        return


# ---------------------------------------------------------------------------
# Thin wrapper around the project JsonlLogger
# ---------------------------------------------------------------------------


def log_fmm_event(
    logger: Optional[Any],
    event: str,
    **fields: Any,
) -> None:
    """
    Emit a structured FMM log event if a logger is available.
    Respects verbosity settings: if logger is None but verbose is on,
    it will log to ConsoleLogger.
    """
    # 1. Exit early if no logger AND not verbose
    if logger is None and not want_verbose_debug():
        return

    # 2. Resolve logger (handles fan-out if verbose is on)
    resolved_logger = get_logger(logger)
    if resolved_logger is None:
        return

    level = get_log_level()
    try:
        # Try the configured level (debug, info, etc.), fallback to info
        log_fn = getattr(resolved_logger, level, None)
        if log_fn is None or not callable(log_fn):
            log_fn = getattr(resolved_logger, "info", None)
        
        if callable(log_fn):
            log_fn(event, **fields)
    except Exception:
        return


# ---------------------------------------------------------------------------
# Structured JSONL output for FMM / BEM tests
# ---------------------------------------------------------------------------


def log_test_result_jsonl(
    suite: str,
    result: Any,
) -> None:
    """
    Emit a one-line JSON object summarizing a test result (if enabled).
    """
    enabled = want_jsonl()
    if not enabled:
        return

    try:
        name = getattr(result, "name", "<unknown>")
        ok = bool(getattr(result, "ok", False))
        max_abs_err = float(getattr(result, "max_abs_err", float("nan")))
        rel_l2_err = float(getattr(result, "rel_l2_err", float("nan")))
        extra = getattr(result, "extra", None)

        record: MutableMapping[str, Any] = {
            "event": "fmm_test_result",
            "suite": suite,
            "name": str(name),
            "ok": ok,
            "max_abs_err": max_abs_err,
            "rel_l2_err": rel_l2_err,
        }

        if isinstance(extra, Mapping):
            for k, v in extra.items():
                key = str(k)
                if key in record:
                    key = f"extra_{key}"
                record[key] = _to_jsonable(v)

        safe_record = _to_jsonable(record)
        line = json.dumps(
            safe_record,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        jsonl_path = os.environ.get("EDE_FMM_JSONL_PATH", "").strip()
        if jsonl_path:
            _write_jsonl_to_path(jsonl_path, line)
            if _normalize_bool_env("EDE_FMM_JSONL_NO_STDOUT", default=False):
                return

        # Fallback / default: also emit to stdout.
        try:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        except Exception:
            return

    except Exception as exc:
        _debug_jsonl(f"log_test_result_jsonl: ERROR building record: {exc!r}")
        return


# ---------------------------------------------------------------------------
# Deep Mathematical Debugging Helpers
# ---------------------------------------------------------------------------


def log_spectral_stats(
    logger: Optional[Any],
    stage_name: str,
    tensor: Union[Tensor, List[Any], Any],
    l_max: int,
    threshold: float = 1e5
) -> None:
    """
    Analyzes the spectral content of a Multipole/Local expansion tensor.
    Robust against List/Tensor mismatches.
    
    Falls back to ConsoleLogger if logger is None but global verbosity is enabled.
    """
    # If no logger provided, check global verbosity
    if logger is None:
        if not want_verbose_debug():
            return
        logger = ConsoleLogger()

    # Robust conversion
    tensor = _safe_tensor(tensor)
    
    # If conversion failed or not a tensor, exit gracefully
    if not isinstance(tensor, torch.Tensor):
        return

    try:
        if tensor.numel() == 0:
            return

        # FIX: Handle 1D tensors (like L2P potential) by unsqueezing
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(1)

        max_val = tensor.abs().max().item()
        msg = f"[{stage_name}] MaxMag={max_val:.2e}"

        if not torch.isfinite(tensor).all():
            logger.error(f"[{stage_name}] FATAL: Tensor contains NaNs or Infs!")
            return

        # Spectral Breakdown (Double precision for stats)
        avg_coeffs = tensor.abs().to(torch.float64).mean(dim=0)

        spectral_power = []
        current_idx = 0
        for l in range(l_max + 1):
            n_m = 2 * l + 1
            if current_idx + n_m > avg_coeffs.shape[0]:
                break
            coeffs_l = avg_coeffs[current_idx : current_idx + n_m]
            norm_l = torch.norm(coeffs_l, p=2).item()
            spectral_power.append(norm_l)
            current_idx += n_m

        spectrum_str = ", ".join([f"l{l}={val:.1e}" for l, val in enumerate(spectral_power)])
        logger.info(f"{msg} | Spectrum: [{spectrum_str}]")

        if len(spectral_power) > 1 and l_max > 1:
            if spectral_power[-1] > (spectral_power[0] * 1e3 + 1e-9):
                logger.error(
                    f"[{stage_name}] DIVERGENCE DETECTED: High-order terms are dominating! "
                    f"l=0: {spectral_power[0]:.2e}, l={l_max}: {spectral_power[-1]:.2e}"
                )

        if max_val > threshold:
            logger.warning(f"[{stage_name}] Tensor values exceeding threshold {threshold:.1e}!")

    except Exception as exc:
        logger.warning(f"[{stage_name}] Spectral logging failed: {exc!r}")
        return


__all__ = [
    "log_fmm_event",
    "log_test_result_jsonl",
    "log_spectral_stats",
    "want_jsonl",
    "get_log_level",
    "want_verbose_debug",
    "ConsoleLogger",
    "CombinedLogger",
    "debug_tensor_stats",
    "get_logger",
]