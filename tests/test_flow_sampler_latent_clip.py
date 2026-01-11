import torch
import pytest

from electrodrive.flows.sampler import ParamFlowSampler
from electrodrive.flows.types import FlowConfig


class DummyProgram:
    nodes = []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for flow sampler")
def test_flow_sampler_latent_clip_applied():
    sampler = ParamFlowSampler(model=None, config=FlowConfig(latent_dim=4))
    device = torch.device("cuda")

    payload = sampler.sample(
        [DummyProgram()],
        spec={},
        spec_embedding=torch.zeros(1, device=device),
        seed=123,
        device=device,
        dtype=torch.float32,
        temperature=5.0,
        latent_clip=0.25,
        max_tokens=4,
        max_ast_len=4,
    )

    max_abs = float(payload.u_latent.abs().max().item())
    assert max_abs <= 0.250001
