"""Minimal CUDA-first training entrypoints for flow models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from electrodrive.flows.device_guard import ensure_cuda, resolve_dtype
from electrodrive.flows.flow_matching import FlowMatchingTrainer
from electrodrive.flows.models import ConditionBatch, ParamFlowNet
from electrodrive.flows.types import FlowConfig


@dataclass
class RandomFlowDataset:
    batch_size: int
    tokens: int
    latent_dim: int

    def sample(self, device: torch.device, dtype: torch.dtype, generator: torch.Generator) -> tuple:
        u0 = torch.randn(
            (self.batch_size, self.tokens, self.latent_dim), device=device, dtype=dtype, generator=generator
        )
        node_mask = torch.ones((self.batch_size, self.tokens), device=device, dtype=torch.bool)
        schema_ids = torch.zeros((self.batch_size, self.tokens), device=device, dtype=torch.long)
        spec_embed = torch.zeros((self.batch_size, 1), device=device, dtype=dtype)
        cond = ConditionBatch(
            spec_embed=spec_embed,
            ast_tokens=None,
            ast_mask=None,
            node_feats=None,
            node_mask=node_mask,
            schema_ids=schema_ids,
        )
        dim_mask = torch.ones((self.batch_size, self.tokens, self.latent_dim), device=device, dtype=torch.bool)
        return u0, cond, node_mask, dim_mask


def train_flow_matching(
    *,
    model: ParamFlowNet,
    steps: int,
    config: Optional[FlowConfig] = None,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype | str] = None,
    seed: int = 0,
    lr: float = 1e-3,
    out_dir: Optional[str | Path] = None,
) -> dict:
    device = ensure_cuda(device)
    dtype = resolve_dtype(dtype)
    cfg = config or FlowConfig()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = FlowMatchingTrainer(model, optimizer, device=device, dtype=dtype)

    dataset = RandomFlowDataset(batch_size=4, tokens=max(1, cfg.max_tokens or 1), latent_dim=cfg.latent_dim)
    generator = torch.Generator(device=device).manual_seed(seed)

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    metrics = {}
    for step in range(int(steps)):
        u0, cond, node_mask, dim_mask = dataset.sample(device, dtype, generator)
        metrics = trainer.train_step(u0, cond, node_mask=node_mask, dim_mask=dim_mask, generator=generator)
        if out_path is not None and step % 50 == 0:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "metrics": {k: float(v) if torch.is_tensor(v) else v for k, v in metrics.items()},
            }
            torch.save(ckpt, out_path / f"flow_step_{step:06d}.pt")

    return metrics


__all__ = ["RandomFlowDataset", "train_flow_matching"]
