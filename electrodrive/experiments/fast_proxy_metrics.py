from __future__ import annotations

import math
from typing import Callable

import torch


def _eval_tensor(fn: Callable[[torch.Tensor], torch.Tensor], pts: torch.Tensor) -> torch.Tensor:
    out = fn(pts)
    if isinstance(out, tuple):
        out = out[0]
    if not torch.is_tensor(out):
        raise ValueError("fast proxy eval must return a torch.Tensor")
    return out.reshape(-1)


def far_field_ratio(
    eval_fn: Callable[[torch.Tensor], torch.Tensor],
    near_pts: torch.Tensor,
    far_pts: torch.Tensor,
) -> float:
    if not near_pts.is_cuda or not far_pts.is_cuda:
        raise ValueError("fast proxy points must be CUDA tensors")
    if near_pts.numel() == 0 or far_pts.numel() == 0:
        return float("nan")
    near_vals = _eval_tensor(eval_fn, near_pts)
    far_vals = _eval_tensor(eval_fn, far_pts)
    near_mag = torch.mean(torch.abs(near_vals))
    far_mag = torch.mean(torch.abs(far_vals))
    ratio = far_mag / near_mag.clamp_min(1e-12)
    if not torch.isfinite(ratio):
        return float("nan")
    return float(ratio.item())


def interface_jump(
    eval_fn: Callable[[torch.Tensor], torch.Tensor],
    pts_up: torch.Tensor,
    pts_dn: torch.Tensor,
) -> float:
    if not pts_up.is_cuda or not pts_dn.is_cuda:
        raise ValueError("fast proxy points must be CUDA tensors")
    if pts_up.numel() == 0 or pts_dn.numel() == 0:
        return 0.0
    vals_up = _eval_tensor(eval_fn, pts_up)
    vals_dn = _eval_tensor(eval_fn, pts_dn)
    if vals_up.numel() == 0 or vals_dn.numel() == 0:
        return 0.0
    diff = torch.abs(vals_up - vals_dn)
    if diff.numel() == 0:
        return 0.0
    max_jump = torch.max(diff)
    if not torch.isfinite(max_jump):
        return float("nan")
    return float(max_jump.item())


def condition_ratio(A: torch.Tensor) -> float:
    if not A.is_cuda:
        raise ValueError("condition proxy expects CUDA matrix")
    if A.numel() == 0 or A.shape[1] == 0:
        return float("nan")
    col_norms = torch.linalg.norm(A, dim=0)
    max_val = torch.max(col_norms)
    min_val = torch.min(col_norms).clamp_min(1e-12)
    ratio = max_val / min_val
    if not torch.isfinite(ratio):
        return float("nan")
    return float(ratio.item())


def log10_bucket(value: float, *, min_exp: int = -8, max_exp: int = 4) -> str:
    if not math.isfinite(value) or value <= 0.0:
        return "nonfinite"
    exp = int(math.floor(math.log10(value)))
    exp = max(min(exp, max_exp), min_exp)
    return f"1e{exp}"


__all__ = [
    "condition_ratio",
    "far_field_ratio",
    "interface_jump",
    "log10_bucket",
]
