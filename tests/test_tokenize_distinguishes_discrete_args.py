from __future__ import annotations

import torch

from electrodrive.flows.schemas import SCHEMA_COMPLEX_DEPTH, SCHEMA_REAL_POINT
from electrodrive.gfn.dsl import AddPoleBlock, AddPrimitiveBlock, Grammar, Program
from electrodrive.gfn.dsl.tokenize import tokenize_program


def test_tokenize_distinguishes_interface_ids() -> None:
    grammar = Grammar(interface_id_choices=(0, 1), pole_count_choices=(1,))
    prog0 = Program(nodes=(AddPoleBlock(interface_id=0, n_poles=1),))
    prog1 = Program(nodes=(AddPoleBlock(interface_id=1, n_poles=1),))
    tok0 = tokenize_program(prog0, max_len=2, device=torch.device("cpu"), grammar=grammar)
    tok1 = tokenize_program(prog1, max_len=2, device=torch.device("cpu"), grammar=grammar)
    assert int(tok0[0].item()) != int(tok1[0].item())


def test_tokenize_distinguishes_schema_ids() -> None:
    grammar = Grammar(primitive_schema_ids=(SCHEMA_REAL_POINT, SCHEMA_COMPLEX_DEPTH))
    prog_real = Program(
        nodes=(AddPrimitiveBlock(family_name="baseline", conductor_id=0, motif_id=0, schema_id=SCHEMA_REAL_POINT),)
    )
    prog_complex = Program(
        nodes=(AddPrimitiveBlock(family_name="baseline", conductor_id=0, motif_id=0, schema_id=SCHEMA_COMPLEX_DEPTH),)
    )
    tok_real = tokenize_program(prog_real, max_len=2, device=torch.device("cpu"), grammar=grammar)
    tok_complex = tokenize_program(prog_complex, max_len=2, device=torch.device("cpu"), grammar=grammar)
    assert int(tok_real[0].item()) != int(tok_complex[0].item())
