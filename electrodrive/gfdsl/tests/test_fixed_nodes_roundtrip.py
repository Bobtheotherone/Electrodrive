import torch

from electrodrive.gfdsl.ast import FixedChargeNode, FixedDipoleNode, Param, SumNode
from electrodrive.gfdsl.compile import CompileContext
from electrodrive.gfdsl.io import deserialize_program, serialize_program_json


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_fixed_nodes_roundtrip_and_lower():
    device = _device()
    program = SumNode(
        children=(
            FixedChargeNode(
                params={
                    "position": Param(torch.tensor([0.1, -0.2, 0.3], device=device)),
                    "charge": Param(torch.tensor(2.0, device=device)),
                }
            ),
            FixedDipoleNode(
                params={
                    "position": Param(torch.tensor([0.0, 0.0, -0.1], device=device)),
                    "moment": Param(torch.tensor([0.5, -0.3, 0.2], device=device)),
                }
            ),
        )
    )

    payload = serialize_program_json(program)
    roundtrip = deserialize_program(payload)
    assert isinstance(roundtrip.children[0], FixedChargeNode)
    assert isinstance(roundtrip.children[1], FixedDipoleNode)

    ctx = CompileContext(device=device, dtype=torch.float32, eval_backend="dense")
    contrib = roundtrip.lower(ctx)
    assert contrib.evaluator.K == 0
    assert contrib.fixed_term is not None

    X = torch.randn(4, 3, device=device, dtype=ctx.dtype)
    y_fix = contrib.fixed_term(X)
    assert y_fix.shape == (X.shape[0],)
    assert y_fix.device.type == ctx.device.type
