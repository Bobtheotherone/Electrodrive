import json

import pytest
import torch

from electrodrive.flows.schemas import SCHEMA_REAL_POINT
from electrodrive.flows.types import ParamPayload
from electrodrive.gfn.dsl import AddPrimitiveBlock, Program
from electrodrive.gfn.dsl.tokenize import TOKEN_MAP, tokenize_program
from electrodrive.gfn.integration import compile_program_to_basis
from electrodrive.utils.device import ensure_cuda_available_or_skip


def test_schema_id_persists_in_canonical_bytes() -> None:
    program = Program(
        nodes=(
            AddPrimitiveBlock(
                family_name="baseline",
                conductor_id=0,
                motif_id=0,
                schema_id=SCHEMA_REAL_POINT,
            ),
        )
    )
    tokens = tokenize_program(program, max_len=4, device=torch.device("cpu"))
    assert int(tokens[0].item()) == TOKEN_MAP["add_primitive"]

    payload = json.loads(program.canonical_bytes.decode("utf-8"))
    assert payload[0]["schema_id"] == SCHEMA_REAL_POINT


def _make_payload(device: torch.device) -> ParamPayload:
    u_latent = torch.tensor([[[0.1, -0.2, 0.3, 0.4]]], device=device, dtype=torch.float32)
    node_mask = torch.tensor([[True]], device=device)
    schema_ids = torch.tensor([[SCHEMA_REAL_POINT]], device=device, dtype=torch.long)
    return ParamPayload(
        u_latent=u_latent,
        node_mask=node_mask,
        dim_mask=None,
        schema_ids=schema_ids,
        node_to_token=[[0]],
        seed=123,
        config_hash="test_cfg",
        device=device,
        dtype=torch.float32,
    )


def test_param_payload_compile_is_deterministic() -> None:
    ensure_cuda_available_or_skip("step10 param payload compile determinism")
    device = torch.device("cuda")
    program = Program(
        nodes=(
            AddPrimitiveBlock(
                family_name="baseline",
                conductor_id=0,
                motif_id=0,
                schema_id=SCHEMA_REAL_POINT,
            ),
        )
    )
    payload = _make_payload(device)

    elems_a, _, _ = compile_program_to_basis(program, {}, device, param_payload=payload)
    elems_b, _, _ = compile_program_to_basis(program, {}, device, param_payload=payload)

    assert elems_a and elems_b
    params_a = elems_a[0].params
    params_b = elems_b[0].params
    assert torch.equal(params_a["position"], params_b["position"])
    assert torch.equal(params_a["z_imag"], params_b["z_imag"])


def test_strict_mode_requires_param_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDE_STRICT_PARAM_PAYLOAD", "1")
    program = Program(
        nodes=(
            AddPrimitiveBlock(family_name="baseline", conductor_id=0, motif_id=0),
        )
    )
    with pytest.raises(RuntimeError):
        compile_program_to_basis(program, {}, torch.device("cpu"))
