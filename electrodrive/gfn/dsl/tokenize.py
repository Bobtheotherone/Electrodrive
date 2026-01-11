"""Tokenization utilities for program ASTs."""

from __future__ import annotations

from typing import Mapping, Optional, TYPE_CHECKING

import torch

from electrodrive.gfn.dsl.program import Program


if TYPE_CHECKING:  # pragma: no cover - typing only
    from electrodrive.gfn.dsl.grammar import Grammar


PAD_TOKEN_ID = 0
TOKEN_MAP: Mapping[str, int] = {
    "pad": PAD_TOKEN_ID,
    "add_primitive": 1,
    "add_motif": 2,
    "add_pole": 3,
    "add_branch_cut": 4,
    "conjugate_pair": 5,
    "stop": 6,
}


def tokenize_program(
    program: Program,
    max_len: int,
    device: torch.device,
    *,
    grammar: Optional["Grammar"] = None,
) -> torch.Tensor:
    """Tokenize a program into a fixed-length sequence on the target device."""
    if grammar is None:
        tokens = [TOKEN_MAP.get(node.type_name, PAD_TOKEN_ID) for node in program.nodes]
    else:
        tokens = [grammar.action_to_token(node) for node in program.nodes]
    tokens = tokens[:max_len]
    if len(tokens) < max_len:
        tokens.extend([PAD_TOKEN_ID] * (max_len - len(tokens)))
    return torch.tensor(tokens, dtype=torch.int64, device=device)


__all__ = ["PAD_TOKEN_ID", "TOKEN_MAP", "tokenize_program"]
