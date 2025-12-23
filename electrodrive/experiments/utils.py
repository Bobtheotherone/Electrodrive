"""Utility helpers for experiment runners (GPU-first, minimal dependencies)."""

from __future__ import annotations

import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch


def assert_cuda_or_die() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. GPU-first rule: aborting.")


def assert_cuda_tensor(tensor: torch.Tensor, name: str = "tensor") -> None:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise RuntimeError(f"{name} is not CUDA (found {tensor.device})")


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        if obj != obj:
            return "NaN"
        if obj == float("inf"):
            return "Infinity"
        if obj == float("-inf"):
            return "-Infinity"
        return obj
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, torch.Tensor):
        t = obj.detach()
        if t.numel() == 1:
            return t.item()
        return t.cpu().tolist()
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    return str(obj)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    try:
        import yaml
    except Exception:
        write_json(path, payload)
        return
    path.write_text(yaml.safe_dump(_sanitize(payload), sort_keys=False), encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    rec = json.dumps(_sanitize(payload), separators=(",", ":"))
    with path.open("a", encoding="utf-8") as f:
        f.write(rec + "\n")


def git_sha(cwd: str | None = None) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
        )
    except Exception:
        return "unknown"


def git_status(cwd: str | None = None) -> str:
    try:
        return (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=cwd, text=True).strip()
        )
    except Exception:
        return "unknown"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_env(key: str, value: str) -> None:
    if value:
        os.environ[key] = value
