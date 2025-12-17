"""Program container and hashing utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Tuple

from electrodrive.gfn.dsl.canonicalization import hash_program, program_to_canonical_bytes
from electrodrive.gfn.dsl.nodes import Node


@dataclass(frozen=True)
class Program:
    """A partially or fully constructed program."""

    nodes: Tuple[Node, ...] = field(default_factory=tuple)

    def with_node(self, node: Node) -> "Program":
        """Return a new program with an additional node appended."""
        return Program(nodes=self.nodes + (node,))

    def extend(self, nodes: Iterable[Node]) -> "Program":
        """Return a new program with a batch of nodes appended."""
        return Program(nodes=self.nodes + tuple(nodes))

    @property
    def canonical_bytes(self) -> bytes:
        """Canonical byte representation suitable for hashing or caching."""
        return program_to_canonical_bytes(self.nodes)

    def hash(self, spec_hash: str) -> str:
        """Compute a stable hash for this program given a spec hash."""
        return hash_program(spec_hash, self.canonical_bytes)

    def to_list(self) -> list[Node]:
        """Return a mutable list copy of the nodes."""
        return list(self.nodes)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.nodes)


__all__ = ["Program"]
