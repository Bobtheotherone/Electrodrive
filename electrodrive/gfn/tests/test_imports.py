from __future__ import annotations

import electrodrive.gfn as gfn
from electrodrive.gfn.dsl import Action, Grammar, Program
from electrodrive.gfn.dsl.nodes import AddPrimitiveBlock
from electrodrive.gfn.env import PartialProgramState, SpecMetadata


def test_gfn_imports_are_safe() -> None:
    grammar = Grammar()
    spec_meta = SpecMetadata(geom_type="plate", n_dielectrics=2, bc_type="dirichlet")
    state = PartialProgramState(spec_hash="spec123", spec_meta=spec_meta, ast_partial=Program())

    assert hasattr(gfn, "Program")
    assert callable(gfn.compile_program_to_basis)
    mask = grammar.get_action_mask(state, spec_meta)
    assert isinstance(mask, tuple)
    assert len(mask) == len(grammar.enumerate_actions())
    legal = grammar.legal_actions(state, spec_meta)
    assert all(isinstance(action, Action) for action in legal)


def test_program_append_and_hash() -> None:
    spec_meta = SpecMetadata(geom_type="plate", n_dielectrics=1, bc_type="neumann")
    program = Program().with_node(AddPrimitiveBlock(family_name="baseline", conductor_id=0, motif_id=0))
    state = PartialProgramState(spec_hash="abc", spec_meta=spec_meta, ast_partial=program)
    assert state.program is program
    assert isinstance(state.state_hash, str)
