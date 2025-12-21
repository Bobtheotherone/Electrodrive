"""Minimal grammar constraints for the program DSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple, TYPE_CHECKING

from electrodrive.gfn.dsl.action import Action
from electrodrive.gfn.dsl.nodes import Node, StopProgram
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
    _cached_actions: Tuple[Action, ...] = field(default_factory=tuple, init=False, repr=False, compare=False)

    def enumerate_actions(self) -> Tuple[Action, ...]:
        """Enumerate the static action templates used for masking."""
        if self._cached_actions:
            return self._cached_actions

        actions: list[Action] = []
        if self.schema_ids:
            for family in self.families:
                for schema_id in self.schema_ids:
                    actions.append(
                        Action(
                            action_type="add_primitive",
                            action_subtype=family,
                            discrete_args={"family_name": family, "schema_id": schema_id},
                        )
                    )
        else:
            for family in self.families:
                actions.append(
                    Action(
                        action_type="add_primitive",
                        action_subtype=family,
                        discrete_args={"family_name": family},
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
        if self.schema_ids:
            for schema_id in self.schema_ids:
                actions.append(
                    Action(
                        action_type="add_pole",
                        discrete_args={
                            "interface_id": 0,
                            "n_poles": self.base_pole_budget,
                            "schema_id": schema_id,
                        },
                    )
                )
                actions.append(
                    Action(
                        action_type="add_branch_cut",
                        discrete_args={
                            "interface_id": 0,
                            "approx_type": self.approx_types[0],
                            "budget": self.branch_cut_budget,
                            "schema_id": schema_id,
                        },
                    )
                )
        else:
            actions.append(
                Action(
                    action_type="add_pole",
                    discrete_args={"interface_id": 0, "n_poles": self.base_pole_budget},
                )
            )
            actions.append(
                Action(
                    action_type="add_branch_cut",
                    discrete_args={
                        "interface_id": 0,
                        "approx_type": self.approx_types[0],
                        "budget": self.branch_cut_budget,
                    },
                )
            )
        actions.append(Action(action_type="conjugate_pair", discrete_args={"block_ref": 0}))
        actions.append(Action(action_type="stop"))

        object.__setattr__(self, "_cached_actions", tuple(actions))
        return self._cached_actions

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

        mask: list[bool] = []
        for action in actions:
            allowed = True
            if action.action_type == "conjugate_pair" and not has_nodes:
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


__all__ = ["Grammar"]
