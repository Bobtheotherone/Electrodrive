"""Tests for GFlowNet reward computation."""

from __future__ import annotations

import torch

from electrodrive.gfn.dsl import AddPrimitiveBlock, Program, StopProgram
from electrodrive.gfn.reward import RewardComputer, RewardConfig, RewardWeights
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.device import ensure_cuda_available_or_skip, get_default_device


def _plane_point_spec() -> CanonicalSpec:
    return CanonicalSpec.from_json(
        {
            "domain": "R3",
            "BCs": "dirichlet",
            "conductors": [{"type": "plane", "z": 0.0}],
            "charges": [{"type": "point", "pos": [0.0, 0.0, 0.5], "q": 1.0}],
        }
    )


def test_reward_compute_cuda_deterministic() -> None:
    ensure_cuda_available_or_skip("reward compute CUDA test")
    device = get_default_device()
    spec = _plane_point_spec()
    program = Program(
        nodes=(
            AddPrimitiveBlock(family_name="point", conductor_id=0, motif_id=0),
            StopProgram(),
        )
    )
    config = RewardConfig(
        n_points=16,
        ratio_boundary=0.5,
        max_iter=20,
        latency_warmup=0,
        weights=RewardWeights(beta=0.0),
    )
    reward_computer = RewardComputer(device=device, config=config)

    terms_a = reward_computer.compute(program, spec, device=device, seed=123)
    terms_b = reward_computer.compute(program, spec, device=device, seed=123)

    assert terms_a.relerr.device.type == "cuda"
    assert torch.allclose(terms_a.logR, terms_b.logR, rtol=1e-4, atol=1e-4)
    assert torch.isfinite(terms_a.logR).item()
    assert terms_a.latency_ms.item() >= 0.0


def test_reward_empty_compilation_penalized() -> None:
    device = get_default_device()
    spec = _plane_point_spec()
    program = Program(nodes=(StopProgram(),))
    config = RewardConfig(logR_clip=(-11.0, 5.0), latency_warmup=0)
    reward_computer = RewardComputer(device=device, config=config)

    terms = reward_computer.compute(program, spec, device=device, seed=7)
    assert torch.isclose(terms.logR, torch.tensor(config.logR_clip[0], device=device)).item()
    assert terms.latency_ms.item() >= 0.0
