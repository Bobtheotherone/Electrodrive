"""GFlowNet environment for constructing electrostatic programs."""

from __future__ import annotations

import random
from typing import List, Mapping, Optional, Sequence, Tuple

import torch

from electrodrive.gfn.dsl import (
    Action,
    AddBranchCutBlock,
    AddMotifBlock,
    AddPoleBlock,
    AddPrimitiveBlock,
    ConjugatePair,
    Grammar,
    Program,
    StopProgram,
)
from electrodrive.gfn.dsl.tokenize import tokenize_program
from electrodrive.gfn.env.state import PartialProgramState, SpecMetadata
from electrodrive.utils.device import get_default_device


class ElectrodriveProgramEnv:
    """Append-only program construction environment with deterministic reversal."""

    ACTION_ENCODING_SIZE = 8
    _ACTION_TYPE_VOCAB: Tuple[str, ...] = (
        "add_primitive",
        "add_motif",
        "add_pole",
        "add_branch_cut",
        "conjugate_pair",
        "stop",
    )

    def __init__(
        self,
        grammar: Optional[Grammar] = None,
        *,
        max_length: int = 64,
        min_length_for_stop: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        self.grammar = grammar or Grammar()
        self.device = device or get_default_device()
        self.max_length = max_length
        self.min_length_for_stop = min_length_for_stop
        self._rng = random.Random()
        self.torch_gen = (
            torch.Generator(device=self.device) if self.device.type == "cuda" else torch.Generator()
        )

        self.actions: Tuple[Action, ...] = self.grammar.enumerate_actions()
        self.action_to_index: dict[tuple, int] = {}
        for idx, action in enumerate(self.actions):
            key = self._action_key(action)
            if key in self.action_to_index:
                raise ValueError(f"Duplicate action key for {action}")
            self.action_to_index[key] = idx
        self.action_type_to_id = {name: idx for idx, name in enumerate(self._ACTION_TYPE_VOCAB)}
        self.id_to_action_type = {idx: name for name, idx in self.action_type_to_id.items()}
        self.family_to_id = {name: idx for idx, name in enumerate(self.grammar.families)}
        self.id_to_family = {idx: name for name, idx in self.family_to_id.items()}
        self.motif_to_id = {name: idx for idx, name in enumerate(self.grammar.motifs)}
        self.id_to_motif = {idx: name for name, idx in self.motif_to_id.items()}
        self.approx_to_id = {name: idx for idx, name in enumerate(self.grammar.approx_types)}
        self.id_to_approx = {idx: name for name, idx in self.approx_to_id.items()}
        stop_indices = [i for i, a in enumerate(self.actions) if a.action_type == "stop"]
        self.stop_index = stop_indices[0] if stop_indices else None
        self._encoded_actions = torch.stack([self.encode_action(action) for action in self.actions]).to(self.device)

    def reset(self, spec: object, spec_meta: SpecMetadata, seed: Optional[int] = None) -> PartialProgramState:
        """Reset the environment to an empty program."""
        if seed is not None:
            self._rng.seed(seed)
            self.torch_gen.manual_seed(seed)
        spec_hash = self._infer_spec_hash(spec)
        program = Program()
        return self._make_state(spec_hash, spec_meta, program)

    def step(
        self,
        state: PartialProgramState,
        action: Action,
    ) -> Tuple[PartialProgramState, bool, Mapping[str, object]]:
        """Apply an action to the state and return the next state."""
        if self.is_terminal(state):
            return state, True, {"reason": "terminal"}

        key = self._action_key(action)
        idx = self.action_to_index.get(key)
        if idx is None:
            raise ValueError("Unknown action: must be enumerated by Grammar.enumerate_actions()")
        mask_cpu = state.action_mask
        if mask_cpu is None or len(mask_cpu) != len(self.actions):
            base_mask = self.grammar.get_action_mask(state, state.spec_meta)
            mask_cpu = self._compute_mask_cpu(state, base_mask)
        if not mask_cpu[idx]:
            raise ValueError("Illegal action under current state mask")

        node = self._action_to_node(action)
        next_program = state.program.with_node(node)
        next_state = self._make_state(state.spec_hash, state.spec_meta, next_program)
        done = self.is_terminal(next_state)
        info = {"state_hash": next_state.state_hash}
        return next_state, done, info

    def step_batch(
        self,
        states: Sequence[PartialProgramState],
        actions: Sequence[Action],
    ) -> Tuple[List[PartialProgramState], torch.Tensor]:
        """Apply a batch of actions to states."""
        if len(states) != len(actions):
            raise ValueError("States and actions must align")
        next_states: List[PartialProgramState] = []
        done_flags = torch.zeros(len(states), dtype=torch.bool, device=self.device)
        for idx, (state, action) in enumerate(zip(states, actions)):
            next_state, done, _ = self.step(state, action)
            next_states.append(next_state)
            done_flags[idx] = done
        return next_states, done_flags

    def reverse_step(self, state: PartialProgramState, action: Optional[Action] = None) -> PartialProgramState:
        """Deterministic reversal by popping the last node."""
        program_nodes = list(state.program.nodes)
        if not program_nodes:
            return state
        popped = program_nodes.pop()
        if action is not None and not self._action_matches_node(action, popped):
            raise ValueError(f"Reverse action mismatch: popped {popped} does not match {action}")
        prev_program = Program(nodes=tuple(program_nodes))
        return self._make_state(state.spec_hash, state.spec_meta, prev_program)

    def is_terminal(self, state: PartialProgramState) -> bool:
        """Check whether a state is terminal."""
        nodes = state.program.nodes
        return bool(nodes) and isinstance(nodes[-1], StopProgram)

    def legal_actions(
        self,
        state: PartialProgramState,
    ) -> Tuple[Tuple[Action, ...], torch.Tensor, torch.Tensor]:
        """Return actions, encoded tensors, and mask on the current device."""
        mask_batch, _ = self.get_action_mask_batch([state])
        return self.actions, self._encoded_actions, mask_batch[0]

    def _mask_cache_key(self, state: PartialProgramState) -> Tuple[object, ...]:
        extra_items = tuple(sorted((str(key), repr(value)) for key, value in state.spec_meta.extra.items()))
        return (
            state.state_hash,
            state.spec_meta.geom_type,
            state.spec_meta.n_dielectrics,
            state.spec_meta.bc_type,
            extra_items,
            self.max_length,
            self.min_length_for_stop,
            str(self.device),
        )

    def get_action_mask(self, state: PartialProgramState) -> torch.Tensor:
        """Return a CUDA-first boolean mask for legal actions."""
        cache_key = self._mask_cache_key(state)
        cached = state.mask_cuda
        if cached is not None and state.mask_cache_key == cache_key and self._device_matches(cached.device):
            return cached
        mask_cpu = state.action_mask
        if mask_cpu is None or len(mask_cpu) != len(self.actions):
            base_mask = self.grammar.get_action_mask(state, state.spec_meta)
            mask_cpu = self._compute_mask_cpu(state, base_mask)
            state = state.with_action_mask(mask_cpu)
        cuda_mask = torch.as_tensor(mask_cpu, dtype=torch.bool, device=self.device)
        state = state.with_cached_mask(cuda_mask, cache_key)
        return cuda_mask

    def get_action_mask_batch(
        self, states: Sequence[PartialProgramState]
    ) -> Tuple[torch.Tensor, List[PartialProgramState]]:
        """Return a batched CUDA-first boolean mask for legal actions."""
        if not states:
            return torch.empty((0, len(self.actions)), dtype=torch.bool, device=self.device), []
        if len(states) == 1:
            mask = self.get_action_mask(states[0])
            return mask.unsqueeze(0), [states[0]]

        masks: List[torch.Tensor] = []
        updated_states: List[PartialProgramState] = []
        for state in states:
            cache_key = self._mask_cache_key(state)
            cached = state.mask_cuda
            if cached is not None and state.mask_cache_key == cache_key and self._device_matches(cached.device):
                masks.append(cached)
                updated_states.append(state)
                continue
            mask_cpu = state.action_mask
            if mask_cpu is None or len(mask_cpu) != len(self.actions):
                base_mask = self.grammar.get_action_mask(state, state.spec_meta)
                mask_cpu = self._compute_mask_cpu(state, base_mask)
                state = state.with_action_mask(mask_cpu)
            cuda_mask = torch.as_tensor(mask_cpu, dtype=torch.bool, device=self.device)
            state = state.with_cached_mask(cuda_mask, cache_key)
            masks.append(cuda_mask)
            updated_states.append(state)
        return torch.stack(masks, dim=0), updated_states

    def get_logpb(self, state: PartialProgramState, action: Action, next_state: PartialProgramState) -> torch.Tensor:
        """Deterministic backward policy baseline (append-only ⇒ logPB=0)."""
        _ = state, action, next_state
        return torch.zeros((), device=self.device)

    def encode_action(self, action: Action) -> torch.Tensor:
        """Pack an action into a fixed-width int32 tensor on the default device."""
        action_type_id = self.action_type_to_id.get(action.action_type, -1)
        subtype_id = -1
        conductor_id = int(action.discrete_args.get("conductor_id", -1))
        family_id = -1
        motif_id = -1
        interface_id = int(action.discrete_args.get("interface_id", -1))
        arg0 = -1
        arg1 = -1
        schema_id = action.discrete_args.get("schema_id")
        if schema_id is not None:
            try:
                arg1 = int(schema_id)
            except Exception:
                arg1 = -1

        if action.action_type == "add_primitive":
            family = action.discrete_args.get("family_name") or action.action_subtype
            if family is not None:
                family_id = self.family_to_id.get(str(family), -1)
                subtype_id = family_id
            motif_id = int(action.discrete_args.get("motif_id", -1))
        elif action.action_type == "add_motif":
            motif = action.discrete_args.get("motif_type") or action.action_subtype
            if motif is not None:
                motif_id = self.motif_to_id.get(str(motif), -1)
                subtype_id = motif_id
        elif action.action_type == "add_pole":
            arg0 = int(action.discrete_args.get("n_poles", -1))
        elif action.action_type == "add_branch_cut":
            approx = action.discrete_args.get("approx_type") or action.action_subtype
            if approx is not None:
                subtype_id = self.approx_to_id.get(str(approx), -1)
            arg0 = int(action.discrete_args.get("budget", -1))
        elif action.action_type == "conjugate_pair":
            arg0 = int(action.discrete_args.get("block_ref", -1))

        encoded = torch.tensor(
            [
                action_type_id,
                subtype_id,
                conductor_id,
                family_id,
                motif_id,
                interface_id,
                arg0,
                arg1,
            ],
            dtype=torch.int32,
            device=self.device,
        )
        return encoded

    def decode_action(self, tensor: torch.Tensor) -> Action:
        """Convert a packed tensor back into an Action."""
        if tensor.numel() < self.ACTION_ENCODING_SIZE:
            raise ValueError(f"Encoded action must have {self.ACTION_ENCODING_SIZE} ints, got {tensor.numel()}")
        fields = tensor.to("cpu").to(torch.int32).tolist()
        (
            action_type_id,
            subtype_id,
            conductor_id,
            family_id,
            motif_id,
            interface_id,
            arg0,
            arg1,
        ) = fields[: self.ACTION_ENCODING_SIZE]
        action_type = self.id_to_action_type.get(action_type_id, "stop")
        discrete_args = {}
        action_subtype = None
        if action_type == "add_primitive":
            if family_id >= 0:
                family_name = self.id_to_family.get(family_id, "unknown")
                discrete_args["family_name"] = family_name
                action_subtype = family_name
            if conductor_id >= 0:
                discrete_args["conductor_id"] = conductor_id
            if motif_id >= 0:
                discrete_args["motif_id"] = motif_id
            if arg1 >= 0:
                discrete_args["schema_id"] = arg1
        elif action_type == "add_motif":
            if motif_id >= 0:
                motif_name = self.id_to_motif.get(motif_id, "unknown")
                discrete_args["motif_type"] = motif_name
                action_subtype = motif_name
        elif action_type == "add_pole":
            if interface_id >= 0:
                discrete_args["interface_id"] = interface_id
            if arg0 >= 0:
                discrete_args["n_poles"] = arg0
            if arg1 >= 0:
                discrete_args["schema_id"] = arg1
        elif action_type == "add_branch_cut":
            if interface_id >= 0:
                discrete_args["interface_id"] = interface_id
            if subtype_id >= 0:
                discrete_args["approx_type"] = self.id_to_approx.get(subtype_id, "unknown")
            if arg0 >= 0:
                discrete_args["budget"] = arg0
            if arg1 >= 0:
                discrete_args["schema_id"] = arg1
        elif action_type == "conjugate_pair":
            if arg0 >= 0:
                discrete_args["block_ref"] = arg0

        return Action(action_type=action_type, action_subtype=action_subtype, discrete_args=discrete_args)

    def _make_state(self, spec_hash: str, spec_meta: SpecMetadata, program: Program) -> PartialProgramState:
        provisional = PartialProgramState(spec_hash=spec_hash, spec_meta=spec_meta, ast_partial=program)
        base_mask = self.grammar.get_action_mask(provisional, spec_meta)
        mask_cpu = self._compute_mask_cpu(provisional, base_mask)
        token_ids = self._tokenize_program(program)
        return PartialProgramState(
            spec_hash=spec_hash,
            spec_meta=spec_meta,
            ast_partial=program,
            action_mask=mask_cpu,
            mask_cuda=None,
            mask_cache_key=None,
            ast_token_ids=token_ids,
            cached_embeddings=None,
        )

    def _compute_mask_cpu(self, state: PartialProgramState, base_mask: Sequence[bool]) -> tuple[bool, ...]:
        """Apply additional structural constraints without creating tensors."""
        mask = list(base_mask)
        length = len(state.program)
        if self.is_terminal(state):
            return tuple(False for _ in mask)

        if length >= self.max_length:
            mask = [False for _ in mask]
            if self.stop_index is not None:
                mask[self.stop_index] = True
            return tuple(mask)

        if self.stop_index is not None and length < self.min_length_for_stop:
            mask[self.stop_index] = False

        for idx, action in enumerate(self.actions):
            if not mask[idx]:
                continue
            mask[idx] = self._action_allowed_by_geometry(action, state)
            if not mask[idx]:
                continue
            if action.action_type == "conjugate_pair":
                ref = int(action.discrete_args.get("block_ref", 0))
                if length == 0 or ref >= length:
                    mask[idx] = False
        return tuple(bool(x) for x in mask)

    def _action_allowed_by_geometry(self, action: Action, state: PartialProgramState) -> bool:
        extra = state.spec_meta.extra
        if action.action_type == "add_branch_cut" and not extra.get("allow_branch_cut", True):
            return False
        if action.action_type == "add_pole" and not extra.get("allow_pole", True):
            return False
        if action.action_type == "add_motif" and action.action_subtype == "layered" and not extra.get("layered", False):
            return False
        return True

    def _tokenize_program(self, program: Program) -> torch.Tensor:
        return tokenize_program(program, max_len=self.max_length, device=self.device)

    @staticmethod
    def _infer_spec_hash(spec: object) -> str:
        if isinstance(spec, str):
            return spec
        for attr in ("spec_hash", "hash", "id"):
            if hasattr(spec, attr):
                return str(getattr(spec, attr))
        return str(spec)

    def _device_matches(self, other: torch.device) -> bool:
        if other.type != self.device.type:
            return False
        if self.device.index is None or other.index is None:
            return True
        return self.device.index == other.index

    def _action_to_node(self, action: Action) -> object:
        """Convert an action into the corresponding AST node."""
        schema_id = action.discrete_args.get("schema_id")
        if schema_id is not None:
            try:
                schema_id = int(schema_id)
            except Exception:
                schema_id = None
        if action.action_type == "add_primitive":
            family = action.discrete_args.get("family_name") or action.action_subtype or "baseline"
            conductor = int(action.discrete_args.get("conductor_id", 0))
            motif_id = int(action.discrete_args.get("motif_id", 0))
            return AddPrimitiveBlock(
                family_name=str(family),
                conductor_id=conductor,
                motif_id=motif_id,
                schema_id=schema_id,
            )
        if action.action_type == "add_motif":
            motif = action.discrete_args.get("motif_type") or action.action_subtype or "connector"
            args = action.discrete_args.copy()
            args.pop("motif_type", None)
            return AddMotifBlock(motif_type=str(motif), args=args)
        if action.action_type == "add_pole":
            interface_id = action.discrete_args.get("interface_id", 0)
            n_poles = int(action.discrete_args.get("n_poles", 1))
            return AddPoleBlock(interface_id=interface_id, n_poles=n_poles, schema_id=schema_id)
        if action.action_type == "add_branch_cut":
            interface_id = action.discrete_args.get("interface_id", 0)
            approx_type = action.discrete_args.get("approx_type") or action.action_subtype or "pade"
            budget = int(action.discrete_args.get("budget", 1))
            return AddBranchCutBlock(
                interface_id=interface_id,
                approx_type=str(approx_type),
                budget=budget,
                schema_id=schema_id,
            )
        if action.action_type == "conjugate_pair":
            block_ref = int(action.discrete_args.get("block_ref", 0))
            return ConjugatePair(block_ref=block_ref)
        if action.action_type == "stop":
            return StopProgram()
        raise ValueError(f"Unknown action type {action.action_type}")

    def _action_key(self, action: Action) -> tuple:
        items = []
        for key, value in action.discrete_args.items():
            normalized = self._normalize_action_value(value)
            if key == "schema_id" and normalized in ("", 0, -1):
                continue
            items.append((key, normalized))
        return (action.action_type, action.action_subtype or "", tuple(sorted(items)))

    @staticmethod
    def _normalize_action_value(value: object) -> object:
        if value is None:
            return ""
        if isinstance(value, (bool, int, str)):
            return value
        return str(value)

    def _action_matches_node(self, action: Action, node: object) -> bool:
        if getattr(node, "type_name", None) != action.action_type:
            return False
        if isinstance(node, AddPrimitiveBlock):
            family = action.discrete_args.get("family_name") or action.action_subtype
            if family is not None and str(family) != node.family_name:
                return False
            conductor = action.discrete_args.get("conductor_id")
            if conductor is not None and int(conductor) != node.conductor_id:
                return False
            motif_id = action.discrete_args.get("motif_id")
            if motif_id is not None and int(motif_id) != node.motif_id:
                return False
            schema_id = action.discrete_args.get("schema_id")
            if schema_id is not None and int(schema_id) != int(node.schema_id or 0):
                return False
            return True
        if isinstance(node, AddMotifBlock):
            motif = action.discrete_args.get("motif_type") or action.action_subtype
            if motif is not None and str(motif) != node.motif_type:
                return False
            for key, value in action.discrete_args.items():
                if key == "motif_type":
                    continue
                if key not in node.args or node.args[key] != value:
                    return False
            return True
        if isinstance(node, AddPoleBlock):
            interface_id = action.discrete_args.get("interface_id")
            if interface_id is not None and int(interface_id) != int(node.interface_id):
                return False
            n_poles = action.discrete_args.get("n_poles")
            if n_poles is not None and int(n_poles) != node.n_poles:
                return False
            schema_id = action.discrete_args.get("schema_id")
            if schema_id is not None and int(schema_id) != int(node.schema_id or 0):
                return False
            return True
        if isinstance(node, AddBranchCutBlock):
            interface_id = action.discrete_args.get("interface_id")
            if interface_id is not None and int(interface_id) != int(node.interface_id):
                return False
            approx_type = action.discrete_args.get("approx_type") or action.action_subtype
            if approx_type is not None and str(approx_type) != node.approx_type:
                return False
            budget = action.discrete_args.get("budget")
            if budget is not None and int(budget) != node.budget:
                return False
            schema_id = action.discrete_args.get("schema_id")
            if schema_id is not None and int(schema_id) != int(node.schema_id or 0):
                return False
            return True
        if isinstance(node, ConjugatePair):
            block_ref = action.discrete_args.get("block_ref")
            if block_ref is not None and int(block_ref) != int(node.block_ref):
                return False
            return True
        if isinstance(node, StopProgram):
            return True
        return False
