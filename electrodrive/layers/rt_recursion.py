from __future__ import annotations

import math
from typing import Literal, Sequence

import torch

from electrodrive.layers.stack import LayerStack

Direction = Literal["down", "up"]


def _ensure_cuda_tensor(k: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k_t = torch.as_tensor(k, device=device, dtype=dtype)
    if k_t.device.type != "cuda":
        raise ValueError("effective_reflection must run on CUDA (GPU-first rule).")
    return k_t


def _safe_exp_neg2kh(k: torch.Tensor, h: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if math.isinf(h):
        return torch.zeros_like(k, device=device, dtype=dtype)
    if abs(h) < 1e-15:
        return torch.ones_like(k, device=device, dtype=dtype)
    exponent = -2.0 * k * h
    real_part = torch.clamp(exponent.real, min=-80.0, max=80.0)
    imag_part = exponent.imag
    return torch.exp(real_part) * (torch.cos(imag_part) + 1j * torch.sin(imag_part))


def _reflection_recursion(
    eps_seq: Sequence[torch.Tensor],
    thickness_seq: Sequence[float],
    k: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    R_next = torch.zeros_like(k, device=device, dtype=dtype)
    for idx in reversed(range(len(thickness_seq))):
        eps_i = eps_seq[idx]
        eps_j = eps_seq[idx + 1]
        denom = eps_i + eps_j
        denom = denom + (denom == 0).to(dtype=dtype) * torch.tensor(1e-12, device=device, dtype=dtype)
        R_ij = (eps_i - eps_j) / denom
        R_ji = -R_ij
        T_ij = 2.0 * eps_j / denom
        T_ji = 2.0 * eps_i / denom
        decay = _safe_exp_neg2kh(k, thickness_seq[idx], device, dtype)
        R_next = R_ij + (T_ij * T_ji * R_next * decay) / (1.0 - R_ji * R_next * decay)
    return R_next


def effective_reflection(
    stack: LayerStack,
    k: torch.Tensor,
    source_region: int,
    direction: Direction,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """
    GPU-based reflection recursion for planar stratified media (electrostatic limit).

    Returns effective reflection seen from `source_region` into the specified direction.
    """
    device = torch.device(device)
    k_t = _ensure_cuda_tensor(k, device, dtype)
    if not torch.is_complex(k_t):
        k_t = k_t.to(dtype=dtype)

    layers = stack.layers
    if len(layers) < 2:
        raise ValueError("LayerStack must contain at least two layers for reflection recursion.")

    if direction == "down":
        if source_region >= len(layers) - 1:
            raise ValueError("No lower interface exists for the requested source_region.")
        idxs = list(range(source_region, len(layers)))
    elif direction == "up":
        if source_region <= 0:
            raise ValueError("No upper interface exists for the requested source_region.")
        idxs = list(range(source_region, -1, -1))
    else:
        raise ValueError(f"Unsupported direction '{direction}' (expected 'up' or 'down').")

    eps_seq = [
        torch.as_tensor(layers[i].eps, device=device, dtype=dtype)
        for i in idxs
    ]
    thickness_seq = [stack.thickness(i) for i in idxs[1:]]

    return _reflection_recursion(eps_seq, thickness_seq, k_t, device, dtype)
