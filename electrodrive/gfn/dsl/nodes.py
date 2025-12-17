"""AST node definitions for the GFlowNet program DSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Tuple, Union


@dataclass(frozen=True)
class Node:
    """Base AST node representing a construction step."""

    type_name: ClassVar[str] = "node"
    commutative_fields: ClassVar[Tuple[str, ...]] = ()

    def to_dict(self) -> Mapping[str, Any]:
        """Return a serializable representation."""
        return {"type": self.type_name}


@dataclass(frozen=True)
class AddPrimitiveBlock(Node):
    """Insert a primitive basis element tied to a family and conductor."""

    family_name: str
    conductor_id: int
    motif_id: int
    type_name: ClassVar[str] = "add_primitive"

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "type": self.type_name,
            "family_name": self.family_name,
            "conductor_id": self.conductor_id,
            "motif_id": self.motif_id,
        }


@dataclass(frozen=True)
class AddMotifBlock(Node):
    """Attach a higher-level motif with typed arguments."""

    motif_type: str
    args: Mapping[str, Any] = field(default_factory=dict)
    type_name: ClassVar[str] = "add_motif"
    # Args ordering is preserved by default; commutativity must be requested by callers.
    commutative_fields: ClassVar[Tuple[str, ...]] = ()

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "type": self.type_name,
            "motif_type": self.motif_type,
            "args": dict(self.args),
        }


@dataclass(frozen=True)
class AddPoleBlock(Node):
    """Insert a pole-based approximation block at an interface."""

    interface_id: Union[int, str]
    n_poles: int
    type_name: ClassVar[str] = "add_pole"

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "type": self.type_name,
            "interface_id": self.interface_id,
            "n_poles": self.n_poles,
        }


@dataclass(frozen=True)
class AddBranchCutBlock(Node):
    """Add a branch-cut approximation block."""

    interface_id: Union[int, str]
    approx_type: str
    budget: int
    type_name: ClassVar[str] = "add_branch_cut"

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "type": self.type_name,
            "interface_id": self.interface_id,
            "approx_type": self.approx_type,
            "budget": self.budget,
        }


@dataclass(frozen=True)
class ConjugatePair(Node):
    """Insert a conjugate pair corresponding to a previous block."""

    block_ref: Union[int, str]
    type_name: ClassVar[str] = "conjugate_pair"

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "type": self.type_name,
            "block_ref": self.block_ref,
        }


@dataclass(frozen=True)
class StopProgram(Node):
    """Terminate the construction."""

    type_name: ClassVar[str] = "stop"

    def to_dict(self) -> Mapping[str, Any]:
        return {"type": self.type_name}


__all__ = [
    "AddBranchCutBlock",
    "AddMotifBlock",
    "AddPoleBlock",
    "AddPrimitiveBlock",
    "ConjugatePair",
    "Node",
    "StopProgram",
]
