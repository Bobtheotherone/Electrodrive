"""Minimal grammar constraints for the program DSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

from electrodrive.gfn.dsl.action import Action, ActionToken
from electrodrive.gfn.dsl.nodes import (
    AddBranchCutBlock,
    AddMotifBlock,
    AddPoleBlock,
    AddPrimitiveBlock,
    ConjugatePair,
    Node,
    StopProgram,
)
from electrodrive.gfn.dsl.program import Program

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from electrodrive.gfn.env.state import PartialProgramState, SpecMetadata


@dataclass(frozen=True)
class Grammar:
    """Pure grammar rules for legal actions.

    This object stays free of solver calls and only encodes structural
    constraints and enumerations of valid blocks/motifs.
    """

    families: Tuple[str, ...] = ("baseline",)
    motifs: Tuple[str, ...] = ("connector",)
    approx_types: Tuple[str, ...] = ("pade",)
    schema_ids: Tuple[int, ...] = ()
    base_pole_budget: int = 1
    branch_cut_budget: int = 1
    interface_id_choices: Tuple[int, ...] = (0,)
    pole_count_choices: Tuple[int, ...] = ()
    branch_budget_choices: Tuple[int, ...] = ()
    primitive_schema_ids: Optional[Tuple[int, ...]] = None
    dcim_schema_ids: Optional[Tuple[int, ...]] = None
    conjugate_ref_choices: Tuple[int, ...] = (0,)
    _cached_actions: Tuple[Action, ...] = field(default_factory=tuple, init=False, repr=False, compare=False)
    _cached_action_vocab: Mapping[ActionToken, int] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def enumerate_actions(self) -> Tuple[Action, ...]:
        """Enumerate the static action templates used for masking."""
        if self._cached_actions:
            return self._cached_actions

        actions: list[Action] = []
        primitive_schema_ids = _resolve_schema_ids(self.primitive_schema_ids, self.schema_ids)
        dcim_schema_ids = _resolve_schema_ids(self.dcim_schema_ids, self.schema_ids)
        pole_counts = self.pole_count_choices or (self.base_pole_budget,)
        branch_budgets = self.branch_budget_choices or (self.branch_cut_budget,)
        motif_ids = tuple(range(len(self.motifs))) or (0,)
        include_motif_id = len(self.motifs) > 1

        for family in self.families:
            for motif_id in motif_ids:
                if primitive_schema_ids:
                    for schema_id in primitive_schema_ids:
                        discrete_args = {"family_name": family}
                        if include_motif_id:
                            discrete_args["motif_id"] = motif_id
                        discrete_args["schema_id"] = schema_id
                        actions.append(
                            Action(
                                action_type="add_primitive",
                                action_subtype=family,
                                discrete_args=discrete_args,
                            )
                        )
                else:
                    discrete_args = {"family_name": family}
                    if include_motif_id:
                        discrete_args["motif_id"] = motif_id
                    actions.append(
                        Action(
                            action_type="add_primitive",
                            action_subtype=family,
                            discrete_args=discrete_args,
                        )
                    )

        for motif in self.motifs:
            actions.append(
                Action(
                    action_type="add_motif",
                    action_subtype=motif,
                    discrete_args={"motif_type": motif},
                )
            )

        for interface_id in self.interface_id_choices:
            for n_poles in pole_counts:
                if dcim_schema_ids:
                    for schema_id in dcim_schema_ids:
                        actions.append(
                            Action(
                                action_type="add_pole",
                                discrete_args={
                                    "interface_id": interface_id,
                                    "n_poles": n_poles,
                                    "schema_id": schema_id,
                                },
                            )
                        )
                else:
                    actions.append(
                        Action(
                            action_type="add_pole",
                            discrete_args={"interface_id": interface_id, "n_poles": n_poles},
                        )
                    )

            for approx_type in self.approx_types:
                for budget in branch_budgets:
                    if dcim_schema_ids:
                        for schema_id in dcim_schema_ids:
                            actions.append(
                                Action(
                                    action_type="add_branch_cut",
                                    discrete_args={
                                        "interface_id": interface_id,
                                        "approx_type": approx_type,
                                        "budget": budget,
                                        "schema_id": schema_id,
                                    },
                                )
                            )
                    else:
                        actions.append(
                            Action(
                                action_type="add_branch_cut",
                                discrete_args={
                                    "interface_id": interface_id,
                                    "approx_type": approx_type,
                                    "budget": budget,
                                },
                            )
                        )

        for block_ref in self.conjugate_ref_choices:
            actions.append(Action(action_type="conjugate_pair", discrete_args={"block_ref": block_ref}))
        actions.append(Action(action_type="stop"))

        object.__setattr__(self, "_cached_actions", tuple(actions))
        return self._cached_actions

    def action_vocab(self) -> Mapping[ActionToken, int]:
        """Return a deterministic action-token vocabulary (pad id is 0)."""
        if self._cached_action_vocab:
            return self._cached_action_vocab
        actions = self.enumerate_actions()
        vocab = {action.to_token(): idx + 1 for idx, action in enumerate(actions)}
        object.__setattr__(self, "_cached_action_vocab", vocab)
        return self._cached_action_vocab

    def token_vocab_size(self) -> int:
        """Return the size of the action token vocabulary including padding."""
        if self._cached_action_vocab:
            max_id = max(self._cached_action_vocab.values(), default=0)
            return int(max_id) + 1
        return len(self.enumerate_actions()) + 1

    def action_to_token(self, node: Node) -> int:
        """Return the token id for a node, or 0 if it is unknown."""
        action = _action_from_node(node, include_motif_id=len(self.motifs) > 1)
        if action is None:
            return 0
        return int(self.action_vocab().get(action.to_token(), 0))

    def token_to_action(self, token: int) -> Optional[Action]:
        """Return the Action for a token id, or None for padding/unknown."""
        idx = int(token) - 1
        actions = self.enumerate_actions()
        if idx < 0 or idx >= len(actions):
            return None
        return actions[idx]

    def get_action_mask(
        self,
        state: Optional["PartialProgramState"],
        spec_meta: Optional["SpecMetadata"],
    ) -> Tuple[bool, ...]:
        """Return a boolean mask aligned with :meth:`enumerate_actions`."""
        actions = self.enumerate_actions()
        nodes = _extract_nodes(state.ast_partial) if state is not None else ()
        has_nodes = len(nodes) > 0
        terminated = _ends_with_stop(nodes)

        if terminated:
            return tuple(False for _ in actions)

        n_interfaces = None
        if spec_meta is not None:
            try:
                n_interfaces = max(0, int(spec_meta.n_dielectrics) - 1)
            except Exception:
                n_interfaces = None

        mask: list[bool] = []
        for action in actions:
            allowed = True
            if action.action_type == "conjugate_pair" and not has_nodes:
                allowed = False
            if allowed and n_interfaces is not None and action.action_type in ("add_pole", "add_branch_cut"):
                interface_id = action.discrete_args.get("interface_id")
                try:
                    interface_id = int(interface_id)
                except Exception:
                    interface_id = -1
                if n_interfaces <= 0 or interface_id < 0 or interface_id >= n_interfaces:
                    allowed = False
            mask.append(allowed)
        return tuple(mask)

    def legal_actions(
        self,
        state: Optional["PartialProgramState"],
        spec_meta: Optional["SpecMetadata"],
    ) -> Tuple[Action, ...]:
        """Filter enumerated actions using the current mask."""
        actions = self.enumerate_actions()
        mask = self.get_action_mask(state, spec_meta)
        return tuple(action for action, allowed in zip(actions, mask) if allowed)


def _extract_nodes(ast_partial: Optional[object]) -> Tuple[Node, ...]:
    if ast_partial is None:
        return ()
    if isinstance(ast_partial, Program):
        return ast_partial.nodes
    if isinstance(ast_partial, (list, tuple)):
        typed_nodes = tuple(node for node in ast_partial if isinstance(node, Node))
        return typed_nodes
    return ()


def _ends_with_stop(nodes: Sequence[Node]) -> bool:
    return bool(nodes) and isinstance(nodes[-1], StopProgram)


def _action_from_node(node: Node, *, include_motif_id: bool) -> Optional[Action]:
    if isinstance(node, AddPrimitiveBlock):
        discrete_args = {
            "family_name": str(node.family_name),
        }
        if include_motif_id:
            discrete_args["motif_id"] = int(node.motif_id)
        if node.schema_id is not None:
            discrete_args["schema_id"] = int(node.schema_id)
        return Action(
            action_type="add_primitive",
            action_subtype=str(node.family_name),
            discrete_args=discrete_args,
        )
    if isinstance(node, AddMotifBlock):
        return Action(
            action_type="add_motif",
            action_subtype=str(node.motif_type),
            discrete_args={"motif_type": str(node.motif_type)},
        )
    if isinstance(node, AddPoleBlock):
        discrete_args = {"interface_id": int(node.interface_id), "n_poles": int(node.n_poles)}
        if node.schema_id is not None:
            discrete_args["schema_id"] = int(node.schema_id)
        return Action(action_type="add_pole", discrete_args=discrete_args)
    if isinstance(node, AddBranchCutBlock):
        discrete_args = {
            "interface_id": int(node.interface_id),
            "approx_type": str(node.approx_type),
            "budget": int(node.budget),
        }
        if node.schema_id is not None:
            discrete_args["schema_id"] = int(node.schema_id)
        return Action(action_type="add_branch_cut", discrete_args=discrete_args)
    if isinstance(node, ConjugatePair):
        return Action(action_type="conjugate_pair", discrete_args={"block_ref": int(node.block_ref)})
    if isinstance(node, StopProgram):
        return Action(action_type="stop")
    return None


def _resolve_schema_ids(
    preferred: Optional[Tuple[int, ...]],
    fallback: Tuple[int, ...],
) -> Optional[Tuple[int, ...]]:
    if preferred is not None:
        return preferred
    if fallback:
        return fallback
    return None


__all__ = ["Grammar"]
