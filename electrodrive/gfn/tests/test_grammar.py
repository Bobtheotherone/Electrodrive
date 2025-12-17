from __future__ import annotations

from electrodrive.gfn.dsl import Grammar, Program, StopProgram
from electrodrive.gfn.env import PartialProgramState, SpecMetadata


def test_terminated_program_masks_all_actions() -> None:
    grammar = Grammar()
    spec_meta = SpecMetadata(geom_type="plate", n_dielectrics=1, bc_type="dirichlet")
    program = Program(nodes=(StopProgram(),))
    state = PartialProgramState(spec_hash="spec", spec_meta=spec_meta, ast_partial=program)

    mask = grammar.get_action_mask(state, spec_meta)
    assert all(allowed is False for allowed in mask)
    assert grammar.legal_actions(state, spec_meta) == ()
