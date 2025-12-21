"""GPU-batched ODE solvers for flow integration."""

from __future__ import annotations

from typing import Callable, Optional

import torch


def _expand_mask(mask: Optional[torch.Tensor], u: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.dtype != torch.bool:
        mask = mask.to(dtype=torch.bool)
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    if mask.shape[-1] == 1 and u.shape[-1] != 1:
        mask = mask.expand(u.shape)
    return mask


def _apply_mask(v: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return v
    mask_f = mask.to(dtype=v.dtype)
    return v * mask_f


def euler_step(
    u: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mask = _expand_mask(mask, u)
    v = velocity_fn(u, t)
    v = _apply_mask(v, mask)
    return u + dt * v


def heun_step(
    u: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mask = _expand_mask(mask, u)
    v1 = _apply_mask(velocity_fn(u, t), mask)
    u_pred = u + dt * v1
    v2 = _apply_mask(velocity_fn(u_pred, t + dt), mask)
    return u + 0.5 * dt * (v1 + v2)


def rk4_step(
    u: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mask = _expand_mask(mask, u)
    k1 = _apply_mask(velocity_fn(u, t), mask)
    k2 = _apply_mask(velocity_fn(u + 0.5 * dt * k1, t + 0.5 * dt), mask)
    k3 = _apply_mask(velocity_fn(u + 0.5 * dt * k2, t + 0.5 * dt), mask)
    k4 = _apply_mask(velocity_fn(u + dt * k3, t + dt), mask)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


__all__ = ["euler_step", "heun_step", "rk4_step"]
