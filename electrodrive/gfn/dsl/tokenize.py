"""Tokenization utilities for program ASTs."""

from __future__ import annotations

from typing import Mapping

import torch

from electrodrive.gfn.dsl.program import Program


TOKEN_MAP: Mapping[str, int] = {
    "pad": 0,
    "add_primitive": 1,
    "add_motif": 2,
    "add_pole": 3,
    "add_branch_cut": 4,
    "conjugate_pair": 5,
    "stop": 6,
}


def tokenize_program(program: Program, max_len: int, device: torch.device) -> torch.Tensor:
    """Tokenize a program into a fixed-length sequence on the target device."""
    tokens = [TOKEN_MAP.get(node.type_name, TOKEN_MAP["pad"]) for node in program.nodes]
    tokens = tokens[:max_len]
    if len(tokens) < max_len:
        tokens.extend([TOKEN_MAP["pad"]] * (max_len - len(tokens)))
    return torch.tensor(tokens, dtype=torch.int64, device=device)


__all__ = ["TOKEN_MAP", "tokenize_program"]
