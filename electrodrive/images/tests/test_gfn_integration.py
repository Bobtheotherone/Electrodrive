"""Integration tests for GFlowNet program compilation and discovery."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from electrodrive.gfn.dsl import AddPrimitiveBlock, Program, StopProgram
from electrodrive.gfn.env import ElectrodriveProgramEnv
from electrodrive.gfn.integration import GFlowNetProgramGenerator, compile_program_to_basis
from electrodrive.gfn.policy import PolicyNet, PolicyNetConfig, LogZNet, action_factor_sizes_from_table, build_action_factor_table
from electrodrive.gfn.dsl import Grammar
from electrodrive.images.basis import BASIS_FAMILY_ENUM
from electrodrive.images.learned_generator import SimpleGeoEncoder
from electrodrive.images.search import discover_images
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.device import get_default_device
from electrodrive.utils.logging import JsonlLogger


def _plane_point_spec() -> CanonicalSpec:
    return CanonicalSpec.from_json(
        {
            "domain": "R3",
            "BCs": "dirichlet",
            "conductors": [{"type": "plane", "z": 0.0}],
            "charges": [{"type": "point", "pos": [0.0, 0.0, 0.5], "q": 1.0}],
        }
    )


def _write_gfn_checkpoint(path: Path, *, spec_dim: int, device: torch.device) -> None:
    grammar = Grammar()
    env = ElectrodriveProgramEnv(grammar=grammar, max_length=2, min_length_for_stop=1, device=device)
    table = build_action_factor_table(env, device=device)
    sizes = action_factor_sizes_from_table(table)
    policy_cfg = PolicyNetConfig(
        spec_dim=spec_dim, max_seq_len=env.max_length, token_vocab_size=env.token_vocab_size
    )
    policy = PolicyNet(policy_cfg, sizes, device=device, token_vocab_size=env.token_vocab_size)
    logz = LogZNet(spec_dim=spec_dim, device=device)
    ckpt = {
        "policy_state": policy.state_dict(),
        "logz_state": logz.state_dict(),
        "policy_config": asdict(policy_cfg),
        "grammar": {
            "families": list(grammar.families),
            "motifs": list(grammar.motifs),
            "approx_types": list(grammar.approx_types),
            "schema_ids": list(grammar.schema_ids),
            "base_pole_budget": grammar.base_pole_budget,
            "branch_cut_budget": grammar.branch_cut_budget,
            "interface_id_choices": list(grammar.interface_id_choices),
            "pole_count_choices": list(grammar.pole_count_choices),
            "branch_budget_choices": list(grammar.branch_budget_choices),
            "primitive_schema_ids": list(grammar.primitive_schema_ids)
            if grammar.primitive_schema_ids is not None
            else None,
            "dcim_schema_ids": list(grammar.dcim_schema_ids) if grammar.dcim_schema_ids is not None else None,
            "conjugate_ref_choices": list(grammar.conjugate_ref_choices),
            "max_length": env.max_length,
            "min_length_for_stop": env.min_length_for_stop,
        },
        "action_vocab": grammar.action_vocab(),
    }
    torch.save(ckpt, path)


def test_compile_program_to_basis_basic() -> None:
    device = get_default_device()
    spec = _plane_point_spec()
    program = Program(
        nodes=(
            AddPrimitiveBlock(family_name="point", conductor_id=0, motif_id=2),
            StopProgram(),
        )
    )
    elems, group_ids, meta = compile_program_to_basis(program, spec, device)
    assert elems, meta
    info = getattr(elems[0], "_group_info", {})
    assert info.get("conductor_id") == 0
    assert info.get("motif_index") == 2
    assert "family_name" in info
    family_name = info["family_name"]
    family_code = BASIS_FAMILY_ENUM[family_name]
    expected_gid = info["conductor_id"] * 1000 + family_code * 10 + info["motif_index"]
    assert int(group_ids[0].item()) == expected_gid
    if torch.cuda.is_available():
        assert elems[0].params["position"].device.type == "cuda"


def test_gfn_discover_images_smoke(tmp_path: Path) -> None:
    device = get_default_device()
    spec = _plane_point_spec()
    ckpt_path = tmp_path / "gfn_checkpoint.pt"
    _write_gfn_checkpoint(ckpt_path, spec_dim=32, device=device)

    logger = JsonlLogger(tmp_path)
    system = discover_images(
        spec=spec,
        basis_types=["point"],
        n_max=1,
        reg_l1=1e-3,
        restarts=0,
        logger=logger,
        operator_mode=True,
        n_points_override=32,
        ratio_boundary_override=0.5,
        basis_generator_mode="gfn",
        geo_encoder=SimpleGeoEncoder(),
        gfn_checkpoint=str(ckpt_path),
        gfn_seed=123,
    )
    assert system.elements
    info = getattr(system.elements[0], "_group_info", {})
    assert "conductor_id" in info and "family_name" in info and "motif_index" in info
    if torch.cuda.is_available():
        assert system.elements[0].params["position"].device.type == "cuda"


def test_gfn_generator_debug_states_sanitized(tmp_path: Path) -> None:
    device = get_default_device()
    spec = _plane_point_spec()
    ckpt_path = tmp_path / "gfn_checkpoint.pt"
    _write_gfn_checkpoint(ckpt_path, spec_dim=8, device=device)

    generator = GFlowNetProgramGenerator(
        checkpoint_path=str(ckpt_path),
        device=device,
        debug_keep_states=True,
    )
    embedding = torch.zeros(8, device=device)
    candidates = generator.generate(
        spec=spec,
        spec_embedding=embedding,
        n_candidates=1,
        seed=7,
    )
    assert candidates
    assert generator.debug_states is not None
    for state in generator.debug_states:
        assert state.mask_cuda is None
        assert state.mask_cache_key is None
