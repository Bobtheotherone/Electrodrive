import pytest
import torch

from electrodrive.flows.models import ConditionBatch, ParamFlowNet
from electrodrive.flows.sampler import ParamFlowSampler
from electrodrive.flows.schemas import CanonicalSpecView, get_schema_by_name
from electrodrive.gfn.dsl.nodes import AddPrimitiveBlock
from electrodrive.gfn.dsl.program import Program
from electrodrive.utils.device import ensure_cuda_available_or_skip


class _ZeroVelocityModel(torch.nn.Module):
    def forward(self, ut: torch.Tensor, t: torch.Tensor, cond: ConditionBatch) -> torch.Tensor:
        _ = t, cond
        return torch.zeros_like(ut)


def test_param_flownet_shapes_cuda() -> None:
    ensure_cuda_available_or_skip("step10 flows core")
    device = torch.device("cuda")
    B, K, P = 2, 3, 4
    model = ParamFlowNet(
        latent_dim=P,
        model_dim=32,
        num_schemas=8,
        ast_vocab_size=16,
        spec_embed_dim=8,
        node_feat_dim=4,
        n_heads=4,
        n_layers=1,
    ).to(device)

    ut = torch.randn(B, K, P, device=device)
    t = torch.rand(B, device=device)
    cond = ConditionBatch(
        spec_embed=torch.randn(B, 8, device=device),
        ast_tokens=torch.randint(0, 16, (B, 5), device=device),
        ast_mask=torch.ones(B, 5, device=device, dtype=torch.bool),
        node_feats=torch.randn(B, K, 4, device=device),
        node_mask=torch.tensor([[True, True, False], [True, True, True]], device=device),
        schema_ids=torch.randint(0, 8, (B, K), device=device),
    )

    out = model(ut, t, cond)
    assert out.shape == (B, K, P)
    assert out.is_cuda


def test_schema_transforms_bounds() -> None:
    ensure_cuda_available_or_skip("step10 flows core")
    device = torch.device("cuda")
    spec = {
        "domain": {"bbox": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]},
        "dielectrics": [{"name": "slab", "z_min": -0.5, "z_max": 0.25}],
        "conductors": [],
        "charges": [],
    }
    view = CanonicalSpecView(spec)
    slab = view.slab_bounds()
    assert slab is not None
    z_bottom, z_top = slab

    schema = get_schema_by_name("complex_depth_point")
    assert schema is not None
    u = torch.randn(2, 1, 4, device=device)
    params = schema.transform(u, spec, node_ctx=None)
    z_imag = params["z_imag"]
    pos = params["position"]
    assert torch.all(z_imag >= 0)
    assert torch.all(pos[..., 2] >= z_bottom)
    assert torch.all(pos[..., 2] <= z_top)


def test_sampler_determinism() -> None:
    ensure_cuda_available_or_skip("step10 flows core")
    device = torch.device("cuda")
    program = Program(nodes=(AddPrimitiveBlock(family_name="baseline", conductor_id=0, motif_id=0),))
    sampler = ParamFlowSampler(model=_ZeroVelocityModel())

    payload1 = sampler.sample(
        [program],
        spec=None,
        spec_embedding=torch.zeros(1, device=device),
        seed=123,
        device=device,
        dtype=torch.float32,
        n_steps=2,
    )
    payload2 = sampler.sample(
        [program],
        spec=None,
        spec_embedding=torch.zeros(1, device=device),
        seed=123,
        device=device,
        dtype=torch.float32,
        n_steps=2,
    )
    payload3 = sampler.sample(
        [program],
        spec=None,
        spec_embedding=torch.zeros(1, device=device),
        seed=124,
        device=device,
        dtype=torch.float32,
        n_steps=2,
    )

    assert torch.equal(payload1.u_latent, payload2.u_latent)
    assert not torch.equal(payload1.u_latent, payload3.u_latent)


def test_sampler_step_limits() -> None:
    ensure_cuda_available_or_skip("step10 flows core")
    device = torch.device("cuda")
    program = Program(nodes=(AddPrimitiveBlock(family_name="baseline", conductor_id=0, motif_id=0),))
    sampler = ParamFlowSampler(model=_ZeroVelocityModel())

    payload = sampler.sample(
        [program],
        spec=None,
        spec_embedding=torch.zeros(1, device=device),
        seed=42,
        device=device,
        dtype=torch.float32,
        n_steps=1,
    )
    assert payload.u_latent.shape[1] == 1

    with pytest.raises(ValueError):
        sampler.sample(
            [program],
            spec=None,
            spec_embedding=torch.zeros(1, device=device),
            seed=0,
            device=device,
            dtype=torch.float32,
            n_steps=9,
        )
