from __future__ import annotations

from electrodrive.gfn.dsl import Grammar, Program
from electrodrive.gfn.env import PartialProgramState, SpecMetadata


def test_rich_action_space_expands() -> None:
    base_count = len(Grammar().enumerate_actions())
    grammar = Grammar(
        families=("baseline", "layered_complex"),
        motifs=("connector", "alt"),
        approx_types=("pade", "branch"),
        interface_id_choices=(0, 1),
        pole_count_choices=(1, 2),
        branch_budget_choices=(1, 3),
        primitive_schema_ids=(1, 3),
        dcim_schema_ids=(4,),
        conjugate_ref_choices=(0, 2),
    )
    actions = grammar.enumerate_actions()
    assert len(actions) > base_count
    assert any(
        action.action_type == "add_pole"
        and int(action.discrete_args.get("interface_id", -1)) == 1
        and int(action.discrete_args.get("n_poles", -1)) == 2
        for action in actions
    )
    assert any(
        action.action_type == "add_branch_cut"
        and action.discrete_args.get("approx_type") == "branch"
        and int(action.discrete_args.get("budget", -1)) == 3
        for action in actions
    )
    assert any(
        action.action_type == "add_primitive"
        and int(action.discrete_args.get("schema_id", -1)) == 3
        for action in actions
    )
    assert any(
        action.action_type == "conjugate_pair"
        and int(action.discrete_args.get("block_ref", -1)) == 2
        for action in actions
    )


def test_action_mask_blocks_invalid_interface_ids() -> None:
    grammar = Grammar(interface_id_choices=(0, 1, 2))
    program = Program()
    spec_meta = SpecMetadata(geom_type="plate", n_dielectrics=2, bc_type="dirichlet")
    state = PartialProgramState(spec_hash="spec", spec_meta=spec_meta, ast_partial=program)
    actions = grammar.enumerate_actions()
    mask = grammar.get_action_mask(state, spec_meta)
    for action, allowed in zip(actions, mask):
        if action.action_type in ("add_pole", "add_branch_cut"):
            interface_id = int(action.discrete_args.get("interface_id", -1))
            if interface_id >= 1:
                assert allowed is False


def test_action_mask_blocks_poles_for_zero_interfaces() -> None:
    grammar = Grammar(interface_id_choices=(0, 1))
    program = Program()
    spec_meta = SpecMetadata(geom_type="plate", n_dielectrics=1, bc_type="dirichlet")
    state = PartialProgramState(spec_hash="spec", spec_meta=spec_meta, ast_partial=program)
    actions = grammar.enumerate_actions()
    mask = grammar.get_action_mask(state, spec_meta)
    assert any(action.action_type == "add_pole" for action in actions)
    assert any(action.action_type == "add_branch_cut" for action in actions)
    for action, allowed in zip(actions, mask):
        if action.action_type in ("add_pole", "add_branch_cut"):
            assert allowed is False
