import pytest
import torch

from electrodrive.gfdsl.ast import Param, RealImageChargeNode
from electrodrive.gfdsl.compile import validate_program
from electrodrive.gfdsl.io import deserialize_program, serialize_program_json


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for GPU-first param tests"
)


def test_param_preserves_cuda_tensor():
    raw = torch.tensor([1.0, 2.0], device="cuda")
    p = Param(raw=raw, trainable=False)
    assert p.raw.device.type == "cuda"
    assert p.raw.requires_grad is False


def test_value_can_materialize_on_cuda():
    p = Param(1.0, trainable=False)
    v = p.value(device="cuda")
    assert v.device.type == "cuda"


def test_serialization_roundtrip_works_from_cuda():
    program = RealImageChargeNode(
        params={
            "position": Param(torch.tensor([0.1, -0.2, 0.3], device="cuda")),
        }
    )
    validate_program(program)

    json_payload = serialize_program_json(program)
    roundtrip = deserialize_program(json_payload)

    assert program.canonical_dict(include_raw=True) == roundtrip.canonical_dict(include_raw=True)
    assert program.full_hash() == roundtrip.full_hash()
