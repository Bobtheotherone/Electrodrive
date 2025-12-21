"""Transformer-style flow model for parameter tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from electrodrive.flows.device_guard import ensure_cuda


@dataclass(frozen=True)
class ConditionBatch:
    spec_embed: Optional[torch.Tensor]
    ast_tokens: Optional[torch.Tensor]
    ast_mask: Optional[torch.Tensor]
    node_feats: Optional[torch.Tensor]
    node_mask: torch.Tensor
    schema_ids: torch.Tensor

    def to(self, device: torch.device, dtype: torch.dtype) -> "ConditionBatch":
        spec_embed = self.spec_embed.to(device=device, dtype=dtype) if self.spec_embed is not None else None
        ast_tokens = self.ast_tokens.to(device=device) if self.ast_tokens is not None else None
        ast_mask = self.ast_mask.to(device=device) if self.ast_mask is not None else None
        node_feats = self.node_feats.to(device=device, dtype=dtype) if self.node_feats is not None else None
        node_mask = self.node_mask.to(device=device)
        schema_ids = self.schema_ids.to(device=device)
        return ConditionBatch(
            spec_embed=spec_embed,
            ast_tokens=ast_tokens,
            ast_mask=ast_mask,
            node_feats=node_feats,
            node_mask=node_mask,
            schema_ids=schema_ids,
        )


class FlowBlock(nn.Module):
    def __init__(self, model_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(model_dim, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.SiLU(),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(model_dim, eps=1e-5)
        self.norm3 = nn.LayerNorm(model_dim, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor],
        ast_tokens: Optional[torch.Tensor],
        ast_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        q = self.norm1(x)
        attn_out, _ = self.self_attn(q, q, q, key_padding_mask=key_padding_mask)
        x = x + attn_out
        if ast_tokens is not None:
            q = self.norm2(x)
            ast_key_padding = None
            if ast_mask is not None:
                ast_key_padding = ~ast_mask.to(dtype=torch.bool)
            cross_out, _ = self.cross_attn(q, ast_tokens, ast_tokens, key_padding_mask=ast_key_padding)
            x = x + cross_out
        x = x + self.ff(self.norm3(x))
        return x


class ParamFlowNet(nn.Module):
    """Parameter flow network over token sequences."""

    def __init__(
        self,
        *,
        latent_dim: int,
        model_dim: int = 128,
        num_schemas: int = 16,
        ast_vocab_size: int = 8,
        spec_embed_dim: int = 0,
        node_feat_dim: int = 0,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.model_dim = int(model_dim)
        self.token_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.schema_embed = nn.Embedding(num_schemas, self.model_dim)
        self.ast_embed = nn.Embedding(ast_vocab_size, self.model_dim)
        self.spec_proj = nn.Linear(spec_embed_dim, self.model_dim) if spec_embed_dim > 0 else None
        self.node_proj = nn.Linear(node_feat_dim, self.model_dim) if node_feat_dim > 0 else None
        self.blocks = nn.ModuleList([FlowBlock(self.model_dim, n_heads, dropout) for _ in range(n_layers)])
        self.out_proj = nn.Linear(self.model_dim, self.latent_dim)

    def forward(self, ut: torch.Tensor, t: torch.Tensor, cond: ConditionBatch) -> torch.Tensor:
        device = ensure_cuda(ut.device)
        ut = ut.to(device=device)
        cond = cond.to(device=device, dtype=ut.dtype)

        if t.dim() == 0:
            t = t.view(1)
        if t.dim() == 1:
            t = t.view(-1, 1)
        t = t.to(device=device, dtype=ut.dtype)

        node_mask = cond.node_mask.to(device=device, dtype=torch.bool)
        schema_ids = cond.schema_ids.to(device=device)
        schema_ids = schema_ids.clamp(0, self.schema_embed.num_embeddings - 1)

        x = self.token_proj(ut)
        x = x + self.schema_embed(schema_ids)
        time_emb = self.time_mlp(t)
        x = x + time_emb.unsqueeze(1)

        if cond.spec_embed is not None and self.spec_proj is not None:
            spec_emb = self.spec_proj(cond.spec_embed)
            x = x + spec_emb.unsqueeze(1)
        if cond.node_feats is not None and self.node_proj is not None:
            x = x + self.node_proj(cond.node_feats)

        ast_tokens = None
        if cond.ast_tokens is not None:
            ast_tokens = self.ast_embed(cond.ast_tokens)

        key_padding_mask = ~node_mask
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask, ast_tokens=ast_tokens, ast_mask=cond.ast_mask)

        out = self.out_proj(x)
        out = out * node_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out


__all__ = ["ConditionBatch", "ParamFlowNet"]
