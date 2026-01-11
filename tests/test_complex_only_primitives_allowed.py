from __future__ import annotations

import torch

from electrodrive.flows.schemas import SCHEMA_COMPLEX_DEPTH
from electrodrive.flows.types import ParamPayload
from electrodrive.gfn.dsl import AddPrimitiveBlock, Grammar, Program
from electrodrive.gfn.env import ElectrodriveProgramEnv, SpecMetadata
from electrodrive.gfn.integration import compile_program_to_basis
from electrodrive.utils.device import ensure_cuda_available_or_skip


def test_env_rejects_real_primitives_when_disallowed() -> None:
    grammar = Grammar()
    env = ElectrodriveProgramEnv(grammar=grammar, max_length=2, min_length_for_stop=1, device=torch.device("cpu"))
    spec_meta = SpecMetadata(
        geom_type="plane",
        n_dielectrics=1,
        bc_type="dirichlet",
        extra={"allow_real_primitives": False},
    )
    state = env.reset(spec="spec", spec_meta=spec_meta, seed=0)
    mask = env.get_action_mask(state).to("cpu").tolist()
    for action, allowed in zip(env.actions, mask):
        if action.action_type == "add_primitive":
            assert allowed is False


def test_env_accepts_complex_schema_primitives_when_real_disallowed() -> None:
    grammar = Grammar(primitive_schema_ids=(SCHEMA_COMPLEX_DEPTH,))
    env = ElectrodriveProgramEnv(grammar=grammar, max_length=2, min_length_for_stop=1, device=torch.device("cpu"))
    spec_meta = SpecMetadata(
        geom_type="plane",
        n_dielectrics=1,
        bc_type="dirichlet",
        extra={"allow_real_primitives": False},
    )
    state = env.reset(spec="spec", spec_meta=spec_meta, seed=0)
    mask = env.get_action_mask(state).to("cpu").tolist()
    allowed = [
        action
        for action, ok in zip(env.actions, mask)
        if ok and action.action_type == "add_primitive" and action.discrete_args.get("schema_id") == SCHEMA_COMPLEX_DEPTH
    ]
    assert allowed
    env.step(state, allowed[0])


def test_compile_complex_schema_produces_imag_depth() -> None:
    ensure_cuda_available_or_skip("complex schema compile needs CUDA")
    device = torch.device("cuda")
    program = Program(
        nodes=(
            AddPrimitiveBlock(
                family_name="baseline",
                conductor_id=0,
                motif_id=0,
                schema_id=SCHEMA_COMPLEX_DEPTH,
            ),
        )
    )
    u_latent = torch.zeros((1, 4), device=device, dtype=torch.float32)
    payload = ParamPayload(
        u_latent=u_latent,
        node_mask=torch.tensor([True], device=device),
        dim_mask=None,
        schema_ids=torch.tensor([SCHEMA_COMPLEX_DEPTH], device=device, dtype=torch.long),
        node_to_token=[0],
        seed=123,
        config_hash="test_cfg",
        device=device,
        dtype=torch.float32,
    )
    elems, _, _ = compile_program_to_basis(program, {}, device, param_payload=payload)
    assert elems
    z_imag = elems[0].params.get("z_imag")
    assert z_imag is not None
    assert float(torch.as_tensor(z_imag).min().item()) > 0.0
