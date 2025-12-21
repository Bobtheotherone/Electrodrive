"""Conditional flow matching trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from electrodrive.flows.device_guard import ensure_cuda, resolve_dtype
from electrodrive.flows.models import ConditionBatch, ParamFlowNet


@dataclass(frozen=True)
class FlowMatchingConfig:
    noise_std: float = 1.0


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return torch.mean((pred - target) ** 2)
    mask_f = mask.to(dtype=pred.dtype)
    diff = (pred - target) ** 2 * mask_f
    denom = mask_f.sum().clamp_min(1.0)
    return diff.sum() / denom


class FlowMatchingTrainer:
    def __init__(
        self,
        model: ParamFlowNet,
        optimizer: torch.optim.Optimizer,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype | str] = None,
        config: Optional[FlowMatchingConfig] = None,
    ) -> None:
        self.device = ensure_cuda(device)
        self.dtype = resolve_dtype(dtype)
        self.model = model.to(device=self.device, dtype=self.dtype)
        self.optimizer = optimizer
        self.config = config or FlowMatchingConfig()

    def train_step(
        self,
        u0: torch.Tensor,
        cond: ConditionBatch,
        *,
        node_mask: Optional[torch.Tensor] = None,
        dim_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> dict:
        device = ensure_cuda(self.device)
        u0 = u0.to(device=device, dtype=self.dtype)
        cond = cond.to(device=device, dtype=self.dtype)

        if generator is None:
            generator = torch.Generator(device=device)

        batch = u0.shape[0]
        t = torch.rand((batch,), device=device, dtype=self.dtype, generator=generator)
        noise = torch.randn(u0.shape, device=device, dtype=self.dtype, generator=generator)
        u1 = u0 + self.config.noise_std * noise
        ut = (1.0 - t).view(-1, 1, 1) * u0 + t.view(-1, 1, 1) * u1
        v_star = u1 - u0

        pred = self.model(ut, t, cond)
        mask = dim_mask
        if mask is None and node_mask is not None:
            mask = node_mask.unsqueeze(-1)
        loss = _masked_mse(pred, v_star, mask)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.detach(),
            "t_mean": float(t.mean().item()),
        }


__all__ = ["FlowMatchingConfig", "FlowMatchingTrainer"]
