from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from electrodrive.gfn.dsl import AddMotifBlock, Program
from electrodrive.gfn.dsl.canonicalization import program_to_canonical_bytes


def test_canonicalization_is_deterministic() -> None:
    node_a = AddMotifBlock(motif_type="connector", args={"b": 2, "a": 1})
    node_b = AddMotifBlock(motif_type="connector", args={"a": 1, "b": 2})
    program_a = Program(nodes=(node_a,))
    program_b = Program(nodes=(node_b,))

    assert program_a.canonical_bytes == program_b.canonical_bytes
    assert program_to_canonical_bytes(program_a.nodes) == program_a.canonical_bytes


def test_program_hash_stability() -> None:
    spec_hash = "spec-xyz"
    program = Program(nodes=(AddMotifBlock(motif_type="connector", args={"a": 1}),))
    digest_1 = program.hash(spec_hash)
    digest_2 = program.hash(spec_hash)
    assert digest_1 == digest_2


def test_program_hashable_with_motif_args() -> None:
    program = Program(nodes=(AddMotifBlock(motif_type="connector", args={"a": 1}),))
    mapping = {program: "ok"}
    assert mapping[program] == "ok"


def test_ordered_sequence_in_mapping_is_preserved_by_default() -> None:
    node_ordered = AddMotifBlock(motif_type="connector", args={"path": [3, 1, 2]})
    node_sorted = AddMotifBlock(motif_type="connector", args={"path": [1, 2, 3]})
    canonical_ordered = Program(nodes=(node_ordered,)).canonical_bytes
    canonical_sorted = Program(nodes=(node_sorted,)).canonical_bytes

    assert canonical_ordered != canonical_sorted
    assert b"[3,1,2]" in canonical_ordered
    assert b"[1,2,3]" in canonical_sorted


@dataclass(frozen=True)
class CommutativeMotif(AddMotifBlock):
    commutative_fields: ClassVar[tuple[str, ...]] = ("args",)
    type_name: ClassVar[str] = "add_motif_commutative"


def test_commutative_fields_still_sort_when_requested() -> None:
    base = CommutativeMotif(motif_type="connector", args={"path": [3, 1, 2]})
    normalized = CommutativeMotif(motif_type="connector", args={"path": [1, 2, 3]})
    canonical_base = Program(nodes=(base,)).canonical_bytes
    canonical_normalized = Program(nodes=(normalized,)).canonical_bytes

    assert canonical_base == canonical_normalized
    assert b"[1,2,3]" in canonical_base
