"""Canonicalization helpers for deterministic program serialization."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from electrodrive.gfn.dsl.nodes import Node


def canonicalize_value(value: Any, *, sort_sequences: bool = False) -> Any:
    """Convert values into a deterministic, JSON-friendly form.

    - Mappings are sorted by key.
    - Sets and flagged sequences are sorted to enforce commutativity.
    - Tuples/lists are converted to tuples to ensure hash stability before the
      final JSON dump, which renders them as lists.
    """
    if isinstance(value, Mapping):
        return {k: canonicalize_value(value[k], sort_sequences=sort_sequences) for k in sorted(value)}
    if isinstance(value, set):
        return tuple(sorted((canonicalize_value(v, sort_sequences=True) for v in value), key=_sort_key))
    if isinstance(value, (list, tuple)):
        items = tuple(canonicalize_value(v, sort_sequences=sort_sequences) for v in value)
        if sort_sequences:
            return tuple(sorted(items, key=_sort_key))
        return items
    return value


def node_to_canonical_dict(node: Node) -> Mapping[str, Any]:
    """Return a deterministic mapping representing a node."""
    raw = node.to_dict()
    canonical = {}
    for key in sorted(raw):
        canonical[key] = canonicalize_value(
            raw[key],
            sort_sequences=key in getattr(node, "commutative_fields", ()),
        )
    return canonical


def program_to_canonical_bytes(nodes: Sequence[Node]) -> bytes:
    """Serialize a program into canonical bytes."""
    canonical_nodes = [node_to_canonical_dict(node) for node in nodes]
    encoded = json.dumps(canonical_nodes, sort_keys=True, separators=(",", ":"))
    return encoded.encode("utf-8")


def hash_program(spec_hash: str, canonical_ast: bytes) -> str:
    """Hash a program given a spec hash and canonical AST bytes."""
    digest = hashlib.sha256()
    digest.update(spec_hash.encode("utf-8"))
    digest.update(canonical_ast)
    return digest.hexdigest()


def _sort_key(value: Any) -> str:
    """Stable ordering helper for mixed-type sequences."""
    return repr(value)


__all__ = [
    "canonicalize_value",
    "hash_program",
    "node_to_canonical_dict",
    "program_to_canonical_bytes",
]
