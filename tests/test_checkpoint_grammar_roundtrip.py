from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from electrodrive.flows.schemas import SCHEMA_COMPLEX_DEPTH, SCHEMA_REAL_POINT
from electrodrive.gfn.dsl import Grammar
from electrodrive.gfn.env import ElectrodriveProgramEnv
from electrodrive.gfn.integration import GFlowNetProgramGenerator
from electrodrive.gfn.policy import (
    LogZNet,
    PolicyNet,
    PolicyNetConfig,
    action_factor_sizes_from_table,
    build_action_factor_table,
)


def _write_checkpoint(path: Path, *, grammar: Grammar, spec_dim: int, include_vocab: bool) -> None:
    device = torch.device("cpu")
    env = ElectrodriveProgramEnv(grammar=grammar, max_length=4, min_length_for_stop=1, device=device)
    table = build_action_factor_table(env, device=device)
    sizes = action_factor_sizes_from_table(table)
    policy_cfg = PolicyNetConfig(
        spec_dim=spec_dim, max_seq_len=env.max_length, token_vocab_size=env.token_vocab_size
    )
    policy = PolicyNet(policy_cfg, sizes, device=device, token_vocab_size=env.token_vocab_size)
    logz = LogZNet(policy_cfg.spec_dim, device=device)
    payload = {
        "policy_state": policy.state_dict(),
        "logz_state": logz.state_dict(),
        "policy_config": asdict(policy_cfg),
        "grammar": {
            "families": grammar.families,
            "motifs": grammar.motifs,
            "approx_types": grammar.approx_types,
            "schema_ids": grammar.schema_ids,
            "base_pole_budget": grammar.base_pole_budget,
            "branch_cut_budget": grammar.branch_cut_budget,
            "interface_id_choices": grammar.interface_id_choices,
            "pole_count_choices": grammar.pole_count_choices,
            "branch_budget_choices": grammar.branch_budget_choices,
            "primitive_schema_ids": grammar.primitive_schema_ids,
            "dcim_schema_ids": grammar.dcim_schema_ids,
            "conjugate_ref_choices": grammar.conjugate_ref_choices,
            "max_length": env.max_length,
            "min_length_for_stop": env.min_length_for_stop,
        },
    }
    if include_vocab:
        payload["action_vocab"] = grammar.action_vocab()
    torch.save(payload, path)


def test_checkpoint_roundtrip_expanded_grammar_and_vocab(tmp_path: Path) -> None:
    grammar = Grammar(
        families=("baseline", "layered_complex"),
        motifs=("connector", "alt"),
        approx_types=("pade", "cheb"),
        interface_id_choices=(0, 1),
        pole_count_choices=(1, 2),
        branch_budget_choices=(1, 3),
        primitive_schema_ids=(SCHEMA_REAL_POINT, SCHEMA_COMPLEX_DEPTH),
        dcim_schema_ids=(SCHEMA_COMPLEX_DEPTH,),
        conjugate_ref_choices=(0, 2),
    )
    ckpt_path = tmp_path / "gfn_new.pt"
    _write_checkpoint(ckpt_path, grammar=grammar, spec_dim=4, include_vocab=True)

    generator = GFlowNetProgramGenerator(checkpoint_path=str(ckpt_path), device=torch.device("cpu"))
    loaded = generator.grammar
    assert loaded.interface_id_choices == grammar.interface_id_choices
    assert loaded.pole_count_choices == grammar.pole_count_choices
    assert loaded.branch_budget_choices == grammar.branch_budget_choices
    assert loaded.primitive_schema_ids == grammar.primitive_schema_ids
    assert loaded.dcim_schema_ids == grammar.dcim_schema_ids
    assert loaded.conjugate_ref_choices == grammar.conjugate_ref_choices
    assert loaded.action_vocab() == grammar.action_vocab()


def test_checkpoint_roundtrip_legacy_defaults(tmp_path: Path) -> None:
    grammar = Grammar()
    ckpt_path = tmp_path / "gfn_old.pt"
    _write_checkpoint(ckpt_path, grammar=grammar, spec_dim=4, include_vocab=False)

    generator = GFlowNetProgramGenerator(checkpoint_path=str(ckpt_path), device=torch.device("cpu"))
    loaded = generator.grammar
    assert loaded.interface_id_choices == (0,)
    assert loaded.pole_count_choices == ()
    assert loaded.branch_budget_choices == ()
    assert loaded.primitive_schema_ids is None
    assert loaded.dcim_schema_ids is None
    assert loaded.conjugate_ref_choices == (0,)
