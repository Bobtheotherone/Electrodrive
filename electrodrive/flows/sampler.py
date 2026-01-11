"""Deterministic CUDA-first sampler for parameter flow models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from electrodrive.flows.device_guard import ensure_cuda, resolve_dtype
from electrodrive.flows.models import ConditionBatch, ParamFlowNet
from electrodrive.flows.ode_solvers import euler_step, heun_step, rk4_step
from electrodrive.flows.schemas import REGISTRY, ParamSchemaRegistry, SCHEMA_REAL_POINT
from electrodrive.flows.types import FlowConfig, ParamPayload, ProgramBatch

try:
    from electrodrive.gfn.dsl.nodes import (
        AddBranchCutBlock,
        AddMotifBlock,
        AddPoleBlock,
        AddPrimitiveBlock,
    )
except Exception:  # pragma: no cover - optional dependency
    AddPrimitiveBlock = AddPoleBlock = AddBranchCutBlock = AddMotifBlock = object  # type: ignore

try:
    from electrodrive.gfn.dsl.tokenize import TOKEN_MAP, tokenize_program
except Exception:  # pragma: no cover
    TOKEN_MAP = {"pad": 0}

    def tokenize_program(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise RuntimeError("tokenize_program unavailable")


_PARAM_NODE_TYPES = (AddPrimitiveBlock, AddPoleBlock, AddBranchCutBlock, AddMotifBlock)


def _is_param_node(node: Any) -> bool:
    return isinstance(node, _PARAM_NODE_TYPES)


@dataclass
class ParamFlowSampler:
    model: Optional[ParamFlowNet] = None
    config: FlowConfig = FlowConfig()
    registry: ParamSchemaRegistry = REGISTRY
    _dim_mask_cache: Dict[Tuple[int, str], torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def _schema_id_from_node(self, node: Any) -> int:
        schema_id = getattr(node, "schema_id", None)
        if schema_id is not None:
            try:
                return int(schema_id)
            except Exception:
                schema_id = None
        schema_name = getattr(node, "schema_name", None)
        if schema_name:
            schema = self.registry.get_by_name(str(schema_name))
            if schema is not None:
                return int(schema.schema_id)
        return 0

    def _build_token_layout(self, programs: Sequence[Any]) -> tuple[List[List[int]], List[List[int]], List[int]]:
        node_to_token: List[List[int]] = []
        schema_ids_list: List[List[int]] = []
        token_counts: List[int] = []
        for program in programs:
            nodes = getattr(program, "nodes", []) or []
            mapping = [-1 for _ in nodes]
            schema_row: List[int] = []
            token_idx = 0
            for node_idx, node in enumerate(nodes):
                if _is_param_node(node):
                    mapping[node_idx] = token_idx
                    schema_row.append(self._schema_id_from_node(node))
                    token_idx += 1
            # node_to_token is small CPU metadata; keep it off CUDA to avoid GPU sync.
            node_to_token.append(mapping)
            schema_ids_list.append(schema_row)
            token_counts.append(token_idx)
        return node_to_token, schema_ids_list, token_counts

    def build_program_batch(
        self,
        programs: Sequence[Any],
        *,
        device: torch.device,
        max_ast_len: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> ProgramBatch:
        """Build a deterministic ProgramBatch for downstream flow sampling."""
        device = ensure_cuda(device)
        program_list = list(programs)
        batch = len(program_list)

        if batch == 0:
            empty = torch.empty((0, 0), device=device, dtype=torch.long)
            return ProgramBatch(
                programs=program_list,
                ast_tokens=empty,
                ast_mask=empty.to(dtype=torch.bool),
                node_table=empty,
                node_mask=empty.to(dtype=torch.bool),
                schema_ids=empty,
            )

        if max_ast_len is None:
            max_ast_len = max(len(getattr(p, "nodes", []) or []) for p in program_list)
            max_ast_len = max(1, int(max_ast_len))

        ast_tokens = torch.stack(
            [tokenize_program(p, max_len=max_ast_len, device=device) for p in program_list], dim=0
        )
        ast_mask = ast_tokens != TOKEN_MAP.get("pad", 0)

        node_to_token, schema_ids_list, token_counts = self._build_token_layout(program_list)
        max_tokens_local = max(token_counts) if token_counts else 0
        if max_tokens is not None:
            max_tokens_local = max(max_tokens_local, int(max_tokens))

        node_table = torch.full(
            (batch, max_tokens_local), -1, device=device, dtype=torch.long
        )
        node_mask = torch.zeros((batch, max_tokens_local), device=device, dtype=torch.bool)
        schema_ids = torch.zeros((batch, max_tokens_local), device=device, dtype=torch.long)

        for idx, mapping in enumerate(node_to_token):
            for node_idx, token_idx in enumerate(mapping):
                if token_idx < 0 or token_idx >= max_tokens_local:
                    continue
                node_table[idx, token_idx] = node_idx
            count = token_counts[idx]
            if count > 0:
                node_mask[idx, :count] = True
            row = schema_ids_list[idx]
            if row:
                schema_ids[idx, : len(row)] = torch.as_tensor(row, device=device, dtype=torch.long)

        return ProgramBatch(
            programs=program_list,
            ast_tokens=ast_tokens,
            ast_mask=ast_mask,
            node_table=node_table,
            node_mask=node_mask,
            schema_ids=schema_ids,
        )

    def _dim_mask_table(self, latent_dim: int, device: torch.device) -> torch.Tensor:
        key = (int(latent_dim), str(device))
        cached = self._dim_mask_cache.get(key)
        if cached is not None and cached.device == device:
            return cached

        schemas = self.registry.all()
        max_id = max((int(schema.schema_id) for schema in schemas), default=0)
        table = torch.zeros((max_id + 1, latent_dim), device=device, dtype=torch.bool)

        default_schema = self.registry.get_by_id(SCHEMA_REAL_POINT)
        if default_schema is not None:
            table[:] = default_schema.dim_mask(latent_dim, device=device)
        for schema in schemas:
            schema_id = int(schema.schema_id)
            if schema_id < 0:
                continue
            table[schema_id] = schema.dim_mask(latent_dim, device=device)

        self._dim_mask_cache[key] = table
        return table

    def _build_dim_mask(
        self, schema_ids: torch.Tensor, latent_dim: int, device: torch.device
    ) -> torch.Tensor:
        table = self._dim_mask_table(latent_dim, device)
        if table.numel() == 0:
            return torch.zeros((*schema_ids.shape, latent_dim), device=device, dtype=torch.bool)
        max_id = table.shape[0] - 1
        schema_ids = schema_ids.to(device=device, dtype=torch.long)
        invalid = (schema_ids < 0) | (schema_ids > max_id)
        schema_ids = schema_ids.clamp(0, max_id)
        schema_ids = torch.where(invalid, torch.zeros_like(schema_ids), schema_ids)
        flat = schema_ids.reshape(-1)
        dim_mask = table.index_select(0, flat).view(*schema_ids.shape, latent_dim)
        return dim_mask

    def _build_condition_batch(
        self,
        programs: Sequence[Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
        node_mask: torch.Tensor,
        schema_ids: torch.Tensor,
        spec_embedding: torch.Tensor,
        ast_tokens: Optional[torch.Tensor] = None,
        ast_mask: Optional[torch.Tensor] = None,
        max_ast_len: Optional[int],
    ) -> ConditionBatch:
        spec_embed = spec_embedding.to(device=device, dtype=dtype)
        if spec_embed.dim() == 1:
            spec_embed = spec_embed.unsqueeze(0)
        if spec_embed.shape[0] == 1 and len(programs) > 1:
            spec_embed = spec_embed.expand(len(programs), -1)

        if ast_tokens is not None:
            ast_tokens = ast_tokens.to(device=device)
            if ast_mask is None:
                ast_mask = ast_tokens != TOKEN_MAP.get("pad", 0)
            else:
                ast_mask = ast_mask.to(device=device)
        elif programs:
            if max_ast_len is None:
                max_ast_len = max(len(getattr(p, "nodes", []) or []) for p in programs)
                max_ast_len = max(1, int(max_ast_len))
            try:
                ast_tokens = torch.stack(
                    [tokenize_program(p, max_len=max_ast_len, device=device) for p in programs], dim=0
                )
                ast_mask = ast_tokens != TOKEN_MAP.get("pad", 0)
            except Exception:
                ast_tokens = None
                ast_mask = None

        return ConditionBatch(
            spec_embed=spec_embed,
            ast_tokens=ast_tokens,
            ast_mask=ast_mask,
            node_feats=None,
            node_mask=node_mask,
            schema_ids=schema_ids,
        )

    def sample_latents(
        self,
        batch: int,
        tokens: int,
        latent_dim: int,
        schema_ids: torch.Tensor,
        node_mask: torch.Tensor,
        *,
        seed: Optional[int],
        device: torch.device,
        dtype: torch.dtype,
        temperature: float = 1.0,
    ) -> ParamPayload:
        device = ensure_cuda(device)
        dtype = resolve_dtype(dtype)
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(int(seed))
        else:
            generator.manual_seed(torch.seed())

        schema_ids = schema_ids.to(device=device)
        node_mask = node_mask.to(device=device)
        dim_mask = self._build_dim_mask(schema_ids, latent_dim, device)

        u_latent = torch.randn(
            (batch, tokens, latent_dim), device=device, dtype=dtype, generator=generator
        )
        u_latent = u_latent * float(temperature)
        node_to_token = [list(range(tokens)) for _ in range(batch)]
        return ParamPayload(
            u_latent=u_latent,
            node_mask=node_mask,
            dim_mask=dim_mask,
            schema_ids=schema_ids,
            node_to_token=node_to_token,
            seed=seed,
            config_hash=None,
            device=device,
            dtype=dtype,
        )

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
        device = ensure_cuda(device)
        dtype = resolve_dtype(dtype)

        latent_dim = int(cfg.get("latent_dim", self.config.latent_dim))
        n_steps = int(cfg.get("n_steps", self.config.n_steps))
        solver = str(cfg.get("solver", self.config.solver))
        temperature = float(cfg.get("temperature", 1.0))
        latent_clip = cfg.get("latent_clip", self.config.latent_clip)
        max_tokens = cfg.get("max_tokens", self.config.max_tokens)
        max_ast_len = cfg.get("max_ast_len", self.config.max_ast_len)

        if n_steps < 1 or n_steps > 8:
            raise ValueError("n_steps must be in [1, 8].")

        program_batch = programs if isinstance(programs, ProgramBatch) else None
        program_list = list(program_batch.programs if program_batch is not None else programs)
        batch = len(program_list)
        if batch == 0:
            empty = torch.empty((0, 0, latent_dim), device=device, dtype=dtype)
            return ParamPayload(
                u_latent=empty,
                node_mask=torch.empty((0, 0), device=device, dtype=torch.bool),
                dim_mask=torch.empty((0, 0, latent_dim), device=device, dtype=torch.bool),
                schema_ids=torch.empty((0, 0), device=device, dtype=torch.long),
                node_to_token=[],
                seed=seed,
                config_hash=None,
                device=device,
                dtype=dtype,
            )

        node_to_token, schema_ids_list, token_counts = self._build_token_layout(program_list)
        max_tokens_local = max(token_counts) if token_counts else 0
        if max_tokens is not None:
            max_tokens_local = max(max_tokens_local, int(max_tokens))
        if program_batch is not None:
            node_mask = program_batch.node_mask.to(device=device)
            schema_ids = program_batch.schema_ids.to(device=device)
            if node_mask.ndim != 2:
                node_mask = node_mask.view(batch, -1)
            if schema_ids.ndim != 2:
                schema_ids = schema_ids.view(batch, -1)
            if max_tokens_local > node_mask.shape[1]:
                pad = max_tokens_local - node_mask.shape[1]
                node_mask = torch.cat(
                    [node_mask, torch.zeros((batch, pad), device=device, dtype=torch.bool)],
                    dim=1,
                )
                schema_ids = torch.cat(
                    [schema_ids, torch.zeros((batch, pad), device=device, dtype=torch.long)],
                    dim=1,
                )
        else:
            node_mask = torch.zeros((batch, max_tokens_local), device=device, dtype=torch.bool)
            schema_ids = torch.zeros((batch, max_tokens_local), device=device, dtype=torch.long)
            for idx, count in enumerate(token_counts):
                if count <= 0:
                    continue
                node_mask[idx, :count] = True
                row = schema_ids_list[idx]
                if row:
                    schema_ids[idx, : len(row)] = torch.as_tensor(row, device=device, dtype=torch.long)

        dim_mask = self._build_dim_mask(schema_ids, latent_dim, device)

        seed_list: Optional[List[Optional[int]]] = None
        if isinstance(seed, torch.Tensor):
            seed_list = [int(v) for v in seed.flatten().tolist()]
        elif isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
            seed_list = [int(v) if v is not None else None for v in list(seed)]

        if seed_list is not None:
            if len(seed_list) == 1 and batch != 1:
                seed_list = seed_list * batch
            if len(seed_list) != batch:
                raise ValueError("seed list length must match batch size.")
            base_seed = int(torch.seed())
            u_latent = torch.empty((batch, max_tokens_local, latent_dim), device=device, dtype=dtype)
            # Per-program seeding preserves determinism when seeds differ.
            for idx, seed_val in enumerate(seed_list):
                gen = torch.Generator(device=device)
                if seed_val is None:
                    seed_val = base_seed + idx
                gen.manual_seed(int(seed_val))
                u_latent[idx] = torch.randn(
                    (max_tokens_local, latent_dim), device=device, dtype=dtype, generator=gen
                )
        else:
            generator = torch.Generator(device=device)
            if seed is not None:
                generator.manual_seed(int(seed))
            else:
                generator.manual_seed(torch.seed())
            u_latent = torch.randn(
                (batch, max_tokens_local, latent_dim),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        u_latent = u_latent * temperature

        if self.model is not None:
            ast_tokens = program_batch.ast_tokens.to(device=device) if program_batch is not None else None
            ast_mask = program_batch.ast_mask.to(device=device) if program_batch is not None else None
            cond = self._build_condition_batch(
                program_list,
                device=device,
                dtype=dtype,
                node_mask=node_mask,
                schema_ids=schema_ids,
                spec_embedding=spec_embedding,
                ast_tokens=ast_tokens,
                ast_mask=ast_mask,
                max_ast_len=max_ast_len,
            )
            self.model.to(device=device, dtype=dtype)
            u_latent = self._integrate(u_latent, cond, n_steps=n_steps, solver=solver, mask=dim_mask)

        if latent_clip is not None:
            clip = float(latent_clip)
            if clip > 0:
                u_latent = u_latent.clamp(min=-clip, max=clip)

        return ParamPayload(
            u_latent=u_latent,
            node_mask=node_mask,
            dim_mask=dim_mask,
            schema_ids=schema_ids,
            node_to_token=node_to_token,
            seed=seed if isinstance(seed, int) or seed is None else None,
            config_hash=None,
            device=device,
            dtype=dtype,
        )

    def _integrate(
        self,
        u_init: torch.Tensor,
        cond: ConditionBatch,
        *,
        n_steps: int,
        solver: str,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        device = ensure_cuda(u_init.device)
        u = u_init
        dt = 1.0 / float(n_steps)

        if solver == "euler":
            step_fn = euler_step
        elif solver == "heun":
            step_fn = heun_step
        elif solver == "rk4":
            step_fn = rk4_step
        else:
            raise ValueError(f"Unknown solver '{solver}'.")

        batch = u.shape[0]
        for i in range(n_steps):
            t = torch.full((batch,), i * dt, device=device, dtype=u.dtype)
            if self.model is None:
                break
            u = step_fn(u, t, dt, lambda uu, tt: self.model(uu, tt, cond), mask=mask)
        return u


__all__ = ["ParamFlowSampler"]
