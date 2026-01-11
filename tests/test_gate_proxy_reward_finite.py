from __future__ import annotations

import torch

from electrodrive.flows.schemas import SCHEMA_COMPLEX_DEPTH
from electrodrive.gfn.dsl import AddPoleBlock, AddPrimitiveBlock, Program, StopProgram
from electrodrive.gfn.reward.gate_proxy_reward import GateProxyRewardComputer, GateProxyRewardConfig
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


def test_gate_proxy_reward_finite() -> None:
    ensure_cuda_available_or_skip("gate proxy reward requires CUDA")
    device = torch.device("cuda")
    spec = _layered_spec()
    program_complex = Program(
        nodes=(
            AddPrimitiveBlock(
                family_name="layered_complex",
                conductor_id=0,
                motif_id=0,
                schema_id=SCHEMA_COMPLEX_DEPTH,
            ),
            StopProgram(),
        )
    )
    program_dcim = Program(
        nodes=(
            AddPoleBlock(interface_id=0, n_poles=2, schema_id=SCHEMA_COMPLEX_DEPTH),
            StopProgram(),
        )
    )
    config = GateProxyRewardConfig(
        n_points=64,
        ratio_boundary=0.5,
        gateA_n_interior=32,
        gateB_n_xy=16,
        gateC_n_dir=16,
        gateD_n_points=32,
        gateA_prefer_autograd=False,
        param_fallback=True,
    )
    reward = GateProxyRewardComputer(device=device, config=config)

    for program in (program_complex, program_dcim):
        terms = reward.compute(program, spec, device=device, seed=123)
        for value in terms.as_dict().values():
            assert torch.isfinite(value).item()
