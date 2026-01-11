"""Interfaces for flow-based parameter sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch


@dataclass(frozen=True)
class ParamPayload:
    """Batched parameter payload for program-conditioned sampling."""

    u_latent: torch.Tensor
    node_mask: torch.Tensor
    dim_mask: torch.Tensor | None
    schema_ids: torch.Tensor
    node_to_token: Sequence[Sequence[int]] | torch.Tensor
    seed: int | None
    config_hash: str | None
    device: torch.device
    dtype: torch.dtype

    def for_program(self, index: int) -> "ParamPayload":
        """Select the payload slice for a single program."""
        if self.node_mask.dim() == 2:
            node_mask = self.node_mask[index]
        else:
            node_mask = self.node_mask

        if self.u_latent.dim() >= 3:
            u_latent = self.u_latent[index]
        else:
            u_latent = self.u_latent

        dim_mask = None
        if self.dim_mask is not None:
            dim_mask = self.dim_mask[index] if self.dim_mask.dim() >= 3 else self.dim_mask

        schema_ids = self.schema_ids[index] if self.schema_ids.dim() == 2 else self.schema_ids

        node_to_token = self.node_to_token
        if isinstance(node_to_token, torch.Tensor):
            if node_to_token.dim() >= 2:
                node_to_token = node_to_token[index]
        elif node_to_token and isinstance(node_to_token[0], (list, tuple)):
            if len(node_to_token) > index:
                node_to_token = node_to_token[index]
        return ParamPayload(
            u_latent=u_latent,
            node_mask=node_mask,
            dim_mask=dim_mask,
            schema_ids=schema_ids,
            node_to_token=node_to_token,
            seed=self.seed,
            config_hash=self.config_hash,
            device=self.device,
            dtype=self.dtype,
        )


@dataclass(frozen=True)
class ProgramBatch:
    """Program batch with token and parameter-node layout."""

    programs: Sequence[Any]
    ast_tokens: torch.Tensor
    ast_mask: torch.Tensor
    node_table: torch.Tensor
    node_mask: torch.Tensor
    schema_ids: torch.Tensor


@dataclass(frozen=True)
class FlowConfig:
    """Config for flow-based param sampling."""

    latent_dim: int = 4
    model_dim: int = 128
    max_tokens: int | None = None
    max_ast_len: int | None = None
    n_steps: int = 4
    solver: str = "euler"
    temperature: float = 1.0
    latent_clip: float | None = None
    dtype: str = "fp32"
    seed: int | None = None


class ParamSampler(Protocol):
    """Protocol for CUDA-first parameter samplers."""

    def sample(
        self,
        programs: Sequence[Any] | ProgramBatch,
        spec: Any,
        spec_embedding: torch.Tensor,
        *,
        seed: int | Sequence[int | None] | None,
        device: torch.device,
        dtype: torch.dtype,
        **cfg: Any,
    ) -> ParamPayload:
        """Return a ParamPayload allocated on CUDA."""
        ...
