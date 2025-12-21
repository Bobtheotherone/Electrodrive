"""CUDA enforcement helpers for flow modules."""

from __future__ import annotations

import os
from typing import Optional

import torch


def resolve_device(device: Optional[torch.device | str] = None) -> torch.device:
    """Resolve a device from input or EDE_DEVICE, defaulting to CUDA if available."""
    if device is None:
        env = os.getenv("EDE_DEVICE", "").strip()
        if env:
            try:
                device = torch.device(env)
            except Exception:
                device = None
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: Optional[torch.dtype | str] = None) -> torch.dtype:
    """Resolve a dtype from input or EDE_DTYPE, defaulting to float32."""
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str) and dtype:
        key = dtype.strip().lower()
    else:
        key = os.getenv("EDE_DTYPE", "").strip().lower()
    if key in {"float64", "double"}:
        return torch.float64
    if key in {"float16", "fp16"}:
        return torch.float16
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float32


def ensure_cuda(device: Optional[torch.device | str] = None) -> torch.device:
    """Return a CUDA device or raise if CUDA is unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for electrodrive.flows.")
    resolved = resolve_device(device)
    if resolved.type != "cuda":
        raise ValueError(f"CUDA device required for electrodrive.flows, got {resolved}.")
    return resolved


def flow_compile_enabled() -> bool:
    """Return True when torch.compile should be enabled for flow models."""
    raw = os.getenv("EDE_FLOW_COMPILE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}
