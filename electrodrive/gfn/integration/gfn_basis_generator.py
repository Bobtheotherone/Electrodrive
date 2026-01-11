"""GFlowNet-backed BasisGenerator for program-driven basis proposals."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Any, List, Mapping, Optional, Sequence

import torch

from electrodrive.gfn.dsl import Grammar
from electrodrive.gfn.dsl.action import Action, ActionToken
from electrodrive.gfn.env import ElectrodriveProgramEnv, PartialProgramState, SpecMetadata
from electrodrive.gfn.integration.compile import compile_program_to_basis
from electrodrive.gfn.policy import LogZNet, PolicyNet, PolicyNetConfig, action_factor_sizes_from_table, build_action_factor_table
from electrodrive.gfn.replay import sanitize_state_for_replay
from electrodrive.gfn.rollout import SpecBatchItem, TemperatureSchedule, rollout_on_policy
from electrodrive.images.basis import BasisGenerator, ImageBasisElement
from electrodrive.learn.collocation import _infer_geom_type_from_spec
from electrodrive.utils.device import get_default_device


@dataclass(frozen=True)
class GFlowNetCheckpoint:
    """Container for checkpoint payloads used by the generator."""

    policy_state: Mapping[str, torch.Tensor]
    logz_state: Mapping[str, torch.Tensor]
    policy_config: PolicyNetConfig
    grammar: Grammar
    max_length: int
    min_length_for_stop: int
    action_vocab: Optional[Mapping[ActionToken, int]] = None


class GFlowNetProgramGenerator(BasisGenerator):
    """Generate basis candidates by sampling GFlowNet programs."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        max_steps: Optional[int] = None,
        temperature_schedule: Optional[TemperatureSchedule] = None,
        debug_keep_states: bool = False,
    ) -> None:
        super().__init__()
        if not checkpoint_path:
            raise ValueError("GFlowNet generator requires a checkpoint; random weights are not allowed.")
        self.device = device or get_default_device()
        self.dtype = dtype
        self.temperature_schedule = temperature_schedule
        self.max_steps = max_steps
        self.debug_keep_states = debug_keep_states
        self.debug_states: Optional[List[PartialProgramState]] = None
        self._current_spec: Optional[Any] = None

        payload = self._load_checkpoint(checkpoint_path)
        self.grammar = payload.grammar
        max_length = int(payload.max_length)
        if max_steps is not None:
            max_length = max(max_length, int(max_steps))
        policy_config = payload.policy_config
        if int(policy_config.max_seq_len) != max_length:
            policy_config = replace(policy_config, max_seq_len=max_length)
        self.env = ElectrodriveProgramEnv(
            grammar=self.grammar,
            max_length=max_length,
            min_length_for_stop=payload.min_length_for_stop,
            device=self.device,
        )
        self.factor_table = build_action_factor_table(self.env, device=self.device)
        factor_sizes = action_factor_sizes_from_table(self.factor_table)
        if payload.action_vocab is not None:
            _validate_action_vocab(payload.action_vocab, self.grammar)

        self.policy = PolicyNet(
            policy_config,
            factor_sizes,
            device=self.device,
            token_vocab_size=self.env.token_vocab_size,
        )
        self.policy.load_state_dict(payload.policy_state)
        self.policy.eval()

        self.logz = LogZNet(policy_config.spec_dim, device=self.device)
        self.logz.load_state_dict(payload.logz_state)
        self.logz.eval()

    def set_spec(self, spec: Any) -> None:
        """Set the spec used by :meth:`forward`."""
        self._current_spec = spec

    def forward(
        self,
        z_global: torch.Tensor,
        charge_nodes: Sequence[Any],
        conductor_nodes: Sequence[Any],
        n_candidates: int,
    ) -> List[ImageBasisElement]:
        """BasisGenerator compatibility wrapper (spec must be set)."""
        _ = charge_nodes, conductor_nodes
        if self._current_spec is None:
            raise ValueError("GFlowNetProgramGenerator requires set_spec() before forward().")
        return self.generate(
            spec=self._current_spec,
            spec_embedding=z_global,
            n_candidates=n_candidates,
        )

    def generate(
        self,
        *,
        spec: Any,
        spec_embedding: torch.Tensor,
        n_candidates: int,
        seed: Optional[int] = None,
    ) -> List[ImageBasisElement]:
        """Generate candidate basis elements from a spec embedding."""
        if n_candidates <= 0:
            return []
        spec_embedding = spec_embedding.to(device=self.device, dtype=self.dtype).view(-1)
        if spec_embedding.numel() != self.policy.config.spec_dim:
            raise ValueError(
                "Spec embedding dimension mismatch: "
                f"expected {self.policy.config.spec_dim}, got {spec_embedding.numel()}."
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
        debug_states: List[PartialProgramState] = []
        max_attempts = max(1, n_candidates * 4)
        for attempt in range(max_attempts):
            rollout = self._rollout(spec_batch, seed=seed if seed is None else seed + attempt)
            final_states = rollout.final_states or ()
            programs = [state.program for state in final_states]
            for program in programs:
                elems, _, _ = compile_program_to_basis(program, spec, self.device)
                candidates.extend(elems)
                if len(candidates) >= n_candidates:
                    break
            if self.debug_keep_states:
                debug_states.extend(final_states)
            if len(candidates) >= n_candidates:
                break
        if self.debug_keep_states:
            self.debug_states = [sanitize_state_for_replay(state) for state in debug_states]
        return candidates[:n_candidates]

    def _rollout(
        self,
        spec_batch: Sequence[SpecBatchItem],
        *,
        seed: Optional[int],
    ) -> Any:
        max_steps = self.max_steps if self.max_steps is not None else self.env.max_length
        if seed is None:
            return rollout_on_policy(
                self.env,
                self.policy,
                spec_batch,
                max_steps=max_steps,
                temperature_schedule=self.temperature_schedule,
            )
        devices: List[int] = []
        if self.device.type == "cuda":
            if self.device.index is not None:
                devices = [self.device.index]
            else:
                devices = [torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(int(seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
            return rollout_on_policy(
                self.env,
                self.policy,
                spec_batch,
                max_steps=max_steps,
                temperature_schedule=self.temperature_schedule,
            )

    def _load_checkpoint(self, checkpoint_path: str) -> GFlowNetCheckpoint:
        payload = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("Invalid GFlowNet checkpoint payload (expected dict)")

        if "policy_state" not in payload or "logz_state" not in payload:
            raise ValueError("GFlowNet checkpoint missing policy_state or logz_state")
        policy_cfg_raw = payload.get("policy_config")
        if not isinstance(policy_cfg_raw, dict):
            raise ValueError("GFlowNet checkpoint missing policy_config")
        policy_config = PolicyNetConfig(**policy_cfg_raw)

        grammar_cfg = payload.get("grammar", {})
        grammar = Grammar(
            families=tuple(grammar_cfg.get("families", Grammar().families)),
            motifs=tuple(grammar_cfg.get("motifs", Grammar().motifs)),
            approx_types=tuple(grammar_cfg.get("approx_types", Grammar().approx_types)),
            schema_ids=tuple(grammar_cfg.get("schema_ids", Grammar().schema_ids)),
            base_pole_budget=int(grammar_cfg.get("base_pole_budget", Grammar().base_pole_budget)),
            branch_cut_budget=int(grammar_cfg.get("branch_cut_budget", Grammar().branch_cut_budget)),
            interface_id_choices=tuple(grammar_cfg.get("interface_id_choices", (0,))),
            pole_count_choices=tuple(grammar_cfg.get("pole_count_choices", ())),
            branch_budget_choices=tuple(grammar_cfg.get("branch_budget_choices", ())),
            primitive_schema_ids=_coerce_optional_tuple(grammar_cfg.get("primitive_schema_ids", None)),
            dcim_schema_ids=_coerce_optional_tuple(grammar_cfg.get("dcim_schema_ids", None)),
            conjugate_ref_choices=tuple(grammar_cfg.get("conjugate_ref_choices", (0,))),
        )
        max_length = int(grammar_cfg.get("max_length", policy_config.max_seq_len))
        min_length_for_stop = int(grammar_cfg.get("min_length_for_stop", 1))
        if policy_config.max_seq_len != max_length:
            raise ValueError(
                "Checkpoint policy_config.max_seq_len does not match grammar max_length "
                f"({policy_config.max_seq_len} != {max_length})."
            )
        if getattr(policy_config, "token_vocab_size", 0) <= 0:
            policy_config = replace(policy_config, token_vocab_size=grammar.token_vocab_size())

        action_vocab = _parse_action_vocab(payload.get("action_vocab"))
        if action_vocab is None:
            _logger.warning("Checkpoint missing action_vocab; reconstructing from grammar.")
        else:
            object.__setattr__(grammar, "_cached_action_vocab", action_vocab)
        return GFlowNetCheckpoint(
            policy_state=payload["policy_state"],
            logz_state=payload["logz_state"],
            policy_config=policy_config,
            grammar=grammar,
            max_length=max_length,
            min_length_for_stop=min_length_for_stop,
            action_vocab=action_vocab,
        )


def _spec_metadata_from_spec(
    spec: Any,
    *,
    extra_overrides: Optional[Mapping[str, Any]] = None,
) -> SpecMetadata:
    geom_type = _infer_geom_type_from_spec(spec)
    dielectrics = getattr(spec, "dielectrics", None) or []
    bc_type = getattr(spec, "BCs", "") or ""
    extra = {
        "allow_branch_cut": bool(dielectrics),
        "allow_pole": True,
        "allow_real_primitives": True,
        "layered": bool(dielectrics),
    }
    if extra_overrides:
        extra.update(dict(extra_overrides))
    return SpecMetadata(
        geom_type=geom_type,
        n_dielectrics=len(dielectrics),
        bc_type=str(bc_type),
        extra=extra,
    )


def _parse_action_vocab(raw: object) -> Optional[Mapping[ActionToken, int]]:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        if not raw:
            return None
        sample_key = next(iter(raw.keys()))
        if isinstance(sample_key, tuple):
            coerced = {_coerce_action_token(key): int(val) for key, val in raw.items()}
            return coerced
        return None
    if isinstance(raw, (list, tuple)):
        tokens: List[ActionToken] = []
        for entry in raw:
            if isinstance(entry, Mapping):
                action = Action(**entry)
                tokens.append(action.to_token())
            else:
                tokens.append(_coerce_action_token(entry))
        return {token: idx + 1 for idx, token in enumerate(tokens)}
    return None


def _coerce_action_token(raw: object) -> ActionToken:
    if isinstance(raw, tuple):
        return tuple(_coerce_action_token(x) if isinstance(x, (list, tuple)) else x for x in raw)  # type: ignore[return-value]
    if isinstance(raw, list):
        return tuple(_coerce_action_token(x) if isinstance(x, (list, tuple)) else x for x in raw)  # type: ignore[return-value]
    raise TypeError("Action token must be a tuple or list")


def _coerce_optional_tuple(value: object) -> Optional[tuple[int, ...]]:
    if value is None:
        return None
    return tuple(int(v) for v in value)


def _validate_action_vocab(action_vocab: Mapping[ActionToken, int], grammar: Grammar) -> None:
    expected = {action.to_token(): idx + 1 for idx, action in enumerate(grammar.enumerate_actions())}
    if expected != dict(action_vocab):
        raise ValueError("Checkpoint action_vocab does not match the current grammar ordering.")


_logger = logging.getLogger(__name__)


__all__ = ["GFlowNetProgramGenerator"]
