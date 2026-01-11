from __future__ import annotations

import torch

from electrodrive.flows.schemas import SCHEMA_COMPLEX_DEPTH
from electrodrive.flows.types import ParamPayload
from electrodrive.gfn.dsl import AddBranchCutBlock, AddPoleBlock, AddPrimitiveBlock
from electrodrive.gfn.integration import GFlowNetProgramGenerator, compile_program_to_basis
from electrodrive.gfn.integration.gfn_basis_generator import _spec_metadata_from_spec
from electrodrive.gfn.rollout import SpecBatchItem, rollout_on_policy
from electrodrive.gfn.train.run_train import run_train_from_config
from electrodrive.images.basis import DCIMBranchCutImageBasis, DCIMPoleImageBasis
from electrodrive.images.learned_generator import SimpleGeoEncoder
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.device import ensure_cuda_available_or_skip


def _layered_spec() -> CanonicalSpec:
    return CanonicalSpec.from_json(
        {
            "domain": {"bbox": [[-1, -1, -2], [1, 1, 2]]},
            "dielectrics": [
                {"name": "region1", "epsilon": 1.0, "z_min": 0.5, "z_max": 2.0},
                {"name": "slab", "epsilon": 4.0, "z_min": 0.0, "z_max": 0.5},
                {"name": "region3", "epsilon": 1.0, "z_min": -2.0, "z_max": 0.0},
            ],
            "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.2]}],
            "BCs": "dielectric_interfaces",
        }
    )


def _payload_for_program(program, device: torch.device) -> ParamPayload:
    mapping = []
    schema_ids = []
    token_idx = 0
    for node in getattr(program, "nodes", []) or []:
        if isinstance(node, (AddPrimitiveBlock, AddPoleBlock, AddBranchCutBlock)):
            mapping.append(token_idx)
            schema_ids.append(int(node.schema_id or SCHEMA_COMPLEX_DEPTH))
            token_idx += 1
        else:
            mapping.append(-1)
    if token_idx == 0:
        token_idx = 1
        schema_ids.append(SCHEMA_COMPLEX_DEPTH)
        mapping = [0]
    u_latent = torch.zeros((token_idx, 4), device=device, dtype=torch.float32)
    return ParamPayload(
        u_latent=u_latent,
        node_mask=torch.ones((token_idx,), device=device, dtype=torch.bool),
        dim_mask=None,
        schema_ids=torch.tensor(schema_ids, device=device, dtype=torch.long),
        node_to_token=mapping,
        seed=123,
        config_hash="smoke_cfg",
        device=device,
        dtype=torch.float32,
    )


def test_train_and_sample_rich_programs_smoke(tmp_path) -> None:
    ensure_cuda_available_or_skip("rich GFN smoke requires CUDA")
    device = torch.device("cuda")
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ckpt_path = tmp_path / "gfn_rich_ckpt.pt"
    config = {
        "seed": seed,
        "output_path": str(ckpt_path),
        "spec_dim": 8,
        "steps": 2,
        "batch_size": 6,
        "max_length": 6,
        "min_length_for_stop": 3,
        "grammar": {
            "families": ["layered_complex"],
            "motifs": [],
            "approx_types": ["pade"],
            "interface_id_choices": [0, 1],
            "pole_count_choices": [1, 2],
            "branch_budget_choices": [1, 2],
            "primitive_schema_ids": [SCHEMA_COMPLEX_DEPTH],
            "dcim_schema_ids": [SCHEMA_COMPLEX_DEPTH],
            "conjugate_ref_choices": [],
        },
        "spec_extra": {"allow_real_primitives": False},
    }
    run_train_from_config(config)
    assert ckpt_path.exists()

    generator = GFlowNetProgramGenerator(
        checkpoint_path=str(ckpt_path),
        device=device,
        debug_keep_states=True,
    )
    spec = _layered_spec()
    encoder = SimpleGeoEncoder(latent_dim=8, hidden_dim=16)
    spec_embedding, _, _ = encoder.encode(spec, device=device, dtype=torch.float32)
    spec_meta = _spec_metadata_from_spec(spec, extra_overrides={"allow_real_primitives": False})
    has_dcim = False
    has_complex = False
    compiled_elems = []

    for trial_seed in (seed, seed + 1, seed + 2):
        spec_batch = [
            SpecBatchItem(
                spec=spec,
                spec_meta=spec_meta,
                spec_embedding=spec_embedding,
                seed=trial_seed + idx,
            )
            for idx in range(16)
        ]
        gen = torch.Generator(device=device).manual_seed(trial_seed)
        rollout = rollout_on_policy(
            generator.env,
            generator.policy,
            spec_batch,
            max_steps=generator.env.max_length,
            generator=gen,
        )
        programs = [state.program for state in rollout.final_states or ()]
        assert programs

        for program in programs:
            if any(isinstance(node, (AddPoleBlock, AddBranchCutBlock)) for node in program.nodes):
                has_dcim = True
            if any(
                isinstance(node, AddPrimitiveBlock) and int(node.schema_id or 0) == SCHEMA_COMPLEX_DEPTH
                for node in program.nodes
            ):
                has_complex = True
            payload = _payload_for_program(program, device)
            elems, _, _ = compile_program_to_basis(program, spec, device, param_payload=payload)
            compiled_elems.extend(elems)

        if has_dcim and has_complex:
            break

    assert has_dcim
    assert has_complex
    assert any(isinstance(elem, (DCIMPoleImageBasis, DCIMBranchCutImageBasis)) for elem in compiled_elems)
    assert any(
        float(torch.as_tensor(elem.params.get("z_imag", 0.0), device=device).abs().max().item()) > 0.0
        for elem in compiled_elems
    )
