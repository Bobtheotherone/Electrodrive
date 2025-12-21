"""Hybrid GFlowNet + flow-based generator for parametric basis elements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
import warnings

import os
import torch

from electrodrive.flows import FlowConfig, ParamFlowNet, ParamFlowSampler
from electrodrive.flows.device_guard import ensure_cuda, flow_compile_enabled, resolve_dtype
from electrodrive.gfn.integration.compile import compile_program_to_basis
from electrodrive.gfn.integration.gfn_basis_generator import GFlowNetProgramGenerator, _spec_metadata_from_spec
from electrodrive.gfn.rollout import SpecBatchItem
from electrodrive.images.basis import ImageBasisElement
from electrodrive.utils.device import get_default_device


@dataclass(frozen=True)
class FlowCheckpoint:
    model_state: Mapping[str, torch.Tensor]
    model_config: Mapping[str, Any]
    sampler_config: Mapping[str, Any]


class HybridGFlowFlowGenerator(GFlowNetProgramGenerator):
    """Generate parametric basis elements with GFlowNet programs + flow sampling."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        flow_checkpoint_path: Optional[str],
        flow_config: Optional[FlowConfig] = None,
        allow_random_flow: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        max_steps: Optional[int] = None,
        temperature_schedule: Optional[Any] = None,
        debug_keep_states: bool = False,
    ) -> None:
        device = ensure_cuda(device or get_default_device())
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            dtype=dtype,
            max_steps=max_steps,
            temperature_schedule=temperature_schedule,
            debug_keep_states=debug_keep_states,
        )
        self.flow_checkpoint_path = flow_checkpoint_path
        self.flow_checkpoint_id = _flow_checkpoint_identity(flow_checkpoint_path)

        flow_cfg = flow_config or FlowConfig()
        use_sampler_defaults = flow_config is None
        checkpoint = None
        if flow_checkpoint_path:
            checkpoint = _load_flow_checkpoint(flow_checkpoint_path, device)
        elif not allow_random_flow:
            raise ValueError("Flow checkpoint required for gfn_flow; random weights are not allowed.")

        model_cfg = _resolve_flow_model_config(checkpoint, flow_cfg, spec_dim=self.policy.config.spec_dim)
        self.flow_model = ParamFlowNet(**model_cfg).to(device=device, dtype=resolve_dtype(flow_cfg.dtype))
        if checkpoint is not None:
            self.flow_model.load_state_dict(checkpoint.model_state, strict=True)
        elif allow_random_flow:
            self.flow_checkpoint_id = "random"
        self.flow_model.eval()
        if flow_compile_enabled():
            if hasattr(torch, "compile"):
                try:
                    self.flow_model = torch.compile(self.flow_model)
                except Exception as exc:  # pragma: no cover - optional acceleration
                    warnings.warn(
                        f"EDE_FLOW_COMPILE=1 but torch.compile failed ({exc}); continuing without compilation.",
                        RuntimeWarning,
                    )
            else:  # pragma: no cover - torch<2.0
                warnings.warn(
                    "EDE_FLOW_COMPILE=1 ignored because torch.compile is unavailable.",
                    RuntimeWarning,
                )

        self.flow_config = _merge_flow_config(flow_cfg, checkpoint, model_cfg, use_sampler_defaults=use_sampler_defaults)
        self.flow_dtype = resolve_dtype(self.flow_config.dtype)
        self.param_sampler = ParamFlowSampler(model=self.flow_model, config=self.flow_config)
        self._flow_spec_dim = int(model_cfg.get("spec_embed_dim", 0))

    def generate(
        self,
        *,
        spec: Any,
        spec_embedding: torch.Tensor,
        n_candidates: int,
        seed: Optional[int] = None,
    ) -> List[ImageBasisElement]:
        if n_candidates <= 0:
            return []
        spec_embedding = spec_embedding.to(device=self.device, dtype=self.dtype).view(-1)
        if spec_embedding.numel() != self.policy.config.spec_dim:
            raise ValueError(
                "Spec embedding dimension mismatch: "
                f"expected {self.policy.config.spec_dim}, got {spec_embedding.numel()}."
            )
        if self._flow_spec_dim > 0 and spec_embedding.numel() != self._flow_spec_dim:
            raise ValueError(
                "Flow model spec embedding dimension mismatch: "
                f"expected {self._flow_spec_dim}, got {spec_embedding.numel()}."
            )

        spec_meta = _spec_metadata_from_spec(spec)
        spec_batch = [
            SpecBatchItem(
                spec=spec,
                spec_meta=spec_meta,
                spec_embedding=spec_embedding,
                seed=None,
            )
            for _ in range(n_candidates)
        ]

        candidates: List[ImageBasisElement] = []
        debug_states: List[Any] = []
        max_attempts = max(1, n_candidates * 4)
        base_seed = seed if seed is not None else self.flow_config.seed
        for attempt in range(max_attempts):
            rollout = self._rollout(spec_batch, seed=None if seed is None else seed + attempt)
            final_states = rollout.final_states or ()
            programs = [state.program for state in final_states]
            if not programs:
                continue
            program_batch = self.param_sampler.build_program_batch(
                programs,
                device=self.device,
                max_ast_len=self.flow_config.max_ast_len,
                max_tokens=self.flow_config.max_tokens,
            )
            param_seed = None if base_seed is None else int(base_seed) + attempt
            payload = self.param_sampler.sample(
                program_batch,
                spec,
                spec_embedding,
                seed=param_seed,
                device=self.device,
                dtype=self.flow_dtype,
                n_steps=self.flow_config.n_steps,
                solver=self.flow_config.solver,
                temperature=self.flow_config.temperature,
                max_tokens=self.flow_config.max_tokens,
                max_ast_len=self.flow_config.max_ast_len,
            )
            for idx, program in enumerate(programs):
                per_payload = payload.for_program(idx)
                elems, _, _ = compile_program_to_basis(
                    program,
                    spec,
                    self.device,
                    param_payload=per_payload,
                    strict=True,
                )
                candidates.extend(elems)
                if len(candidates) >= n_candidates:
                    break
            if self.debug_keep_states:
                debug_states.extend(final_states)
            if len(candidates) >= n_candidates:
                break

        if self.debug_keep_states:
            self.debug_states = [state for state in debug_states]
        return candidates[:n_candidates]


def _flow_checkpoint_identity(path: Optional[str]) -> str:
    if not path:
        return "none"
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return f"{path}:missing"
    return f"{path}:{int(stat.st_mtime)}"


def _load_flow_checkpoint(path: str, device: torch.device) -> FlowCheckpoint:
    payload = torch.load(Path(path), map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("Invalid flow checkpoint payload (expected dict).")
    state = payload.get("model") or payload.get("model_state") or payload.get("state_dict") or None
    if state is None:
        raise ValueError("Flow checkpoint missing model state.")
    if not isinstance(state, dict):
        raise ValueError("Flow checkpoint model state must be a dict.")
    model_cfg = payload.get("model_config", {}) or payload.get("config", {})
    sampler_cfg = payload.get("sampler_config", {}) or payload.get("flow_config", {})
    return FlowCheckpoint(model_state=state, model_config=model_cfg, sampler_config=sampler_cfg)


def _resolve_flow_model_config(
    checkpoint: Optional[FlowCheckpoint],
    flow_cfg: FlowConfig,
    *,
    spec_dim: int,
) -> Dict[str, Any]:
    if checkpoint is None:
        return {
            "latent_dim": int(flow_cfg.latent_dim),
            "model_dim": int(flow_cfg.model_dim),
            "num_schemas": 16,
            "ast_vocab_size": 8,
            "spec_embed_dim": int(spec_dim),
            "node_feat_dim": 0,
            "n_heads": 4,
            "n_layers": 2,
            "dropout": 0.0,
        }
    return _infer_flow_model_config(checkpoint.model_state, checkpoint.model_config)


def _infer_flow_model_config(
    state: Mapping[str, torch.Tensor],
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    def _int(val: Any, default: int) -> int:
        try:
            return int(val)
        except Exception:
            return int(default)

    latent_dim = _int(cfg.get("latent_dim"), state.get("token_proj.0.weight", torch.empty(0)).shape[1] or 4)
    model_dim = _int(cfg.get("model_dim"), state.get("token_proj.0.weight", torch.empty(0)).shape[0] or 128)
    num_schemas = _int(cfg.get("num_schemas"), state.get("schema_embed.weight", torch.empty(0)).shape[0] or 16)
    ast_vocab_size = _int(cfg.get("ast_vocab_size"), state.get("ast_embed.weight", torch.empty(0)).shape[0] or 8)
    spec_embed_dim = _int(cfg.get("spec_embed_dim"), 0)
    if "spec_proj.weight" in state:
        spec_embed_dim = _int(spec_embed_dim, state["spec_proj.weight"].shape[1])
    node_feat_dim = _int(cfg.get("node_feat_dim"), 0)
    if "node_proj.weight" in state:
        node_feat_dim = _int(node_feat_dim, state["node_proj.weight"].shape[1])
    block_indices = {
        int(key.split(".")[1])
        for key in state
        if key.startswith("blocks.") and key.split(".")[1].isdigit()
    }
    inferred_layers = max(block_indices) + 1 if block_indices else 2
    n_layers = _int(cfg.get("n_layers"), inferred_layers)
    n_heads = _int(cfg.get("n_heads"), 4)
    dropout = float(cfg.get("dropout", 0.0))

    return {
        "latent_dim": latent_dim,
        "model_dim": model_dim,
        "num_schemas": num_schemas,
        "ast_vocab_size": ast_vocab_size,
        "spec_embed_dim": spec_embed_dim,
        "node_feat_dim": node_feat_dim,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "dropout": dropout,
    }


def _merge_flow_config(
    base: FlowConfig,
    checkpoint: Optional[FlowCheckpoint],
    model_cfg: Mapping[str, Any],
    *,
    use_sampler_defaults: bool,
) -> FlowConfig:
    sampler_cfg = checkpoint.sampler_config if checkpoint is not None else {}
    if use_sampler_defaults:
        n_steps = int(sampler_cfg.get("n_steps", base.n_steps))
        solver = str(sampler_cfg.get("solver", base.solver))
        temperature = float(sampler_cfg.get("temperature", base.temperature))
        dtype = str(sampler_cfg.get("dtype", base.dtype))
        seed = sampler_cfg.get("seed", base.seed)
    else:
        n_steps = int(base.n_steps)
        solver = str(base.solver)
        temperature = float(base.temperature)
        dtype = str(base.dtype)
        seed = base.seed
    return FlowConfig(
        latent_dim=int(model_cfg.get("latent_dim", base.latent_dim)),
        model_dim=int(model_cfg.get("model_dim", base.model_dim)),
        max_tokens=base.max_tokens if base.max_tokens is not None else sampler_cfg.get("max_tokens"),
        max_ast_len=base.max_ast_len if base.max_ast_len is not None else sampler_cfg.get("max_ast_len"),
        n_steps=n_steps,
        solver=solver,
        temperature=temperature,
        dtype=dtype,
        seed=seed,
    )


__all__ = ["HybridGFlowFlowGenerator"]
