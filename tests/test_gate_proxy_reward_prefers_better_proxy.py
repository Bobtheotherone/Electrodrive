from __future__ import annotations

from typing import Any, Dict

import torch

from electrodrive.gfn.dsl import AddPrimitiveBlock, Program, StopProgram
from electrodrive.gfn.reward.gate_proxy_reward import (
    GateProxyRewardComputer,
    GateProxyRewardConfig,
    GateProxyRewardWeights,
)
from electrodrive.orchestration.parser import CanonicalSpec


def _plane_point_spec() -> CanonicalSpec:
    return CanonicalSpec.from_json(
        {
            "domain": "R3",
            "BCs": "dirichlet",
            "conductors": [{"type": "plane", "z": 0.0}],
            "charges": [{"type": "point", "pos": [0.0, 0.0, 0.5], "q": 1.0}],
        }
    )


def _proxy_metrics(
    *,
    worst_ratio: float,
    b_jump: float,
    far_slope: float,
    near_slope: float,
    spurious: float,
    rel_change: float,
) -> Dict[str, Any]:
    return {
        "proxy_gateA_worst_ratio": worst_ratio,
        "proxy_gateB_max_v_jump": b_jump,
        "proxy_gateB_max_d_jump": b_jump,
        "proxy_gateC_far_slope": far_slope,
        "proxy_gateC_near_slope": near_slope,
        "proxy_gateC_spurious_fraction": spurious,
        "proxy_gateD_rel_change": rel_change,
    }


def test_gate_proxy_reward_prefers_better_proxy(monkeypatch) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = _plane_point_spec()
    prog_good = Program(
        nodes=(
            AddPrimitiveBlock(family_name="baseline", conductor_id=0, motif_id=0),
            StopProgram(),
        )
    )
    prog_bad = Program(
        nodes=(
            AddPrimitiveBlock(family_name="baseline", conductor_id=0, motif_id=0),
            AddPrimitiveBlock(family_name="baseline", conductor_id=0, motif_id=0),
            StopProgram(),
        )
    )
    weights = GateProxyRewardWeights(
        gateA=1.0,
        gateB=1.0,
        gateC=1.0,
        gateD=1.0,
        speed=0.0,
        complexity=0.1,
        dcim_bonus=0.0,
        complex_bonus=0.0,
    )
    config = GateProxyRewardConfig(
        n_points=16,
        gateA_n_interior=8,
        gateB_n_xy=4,
        gateC_n_dir=4,
        gateD_n_points=8,
        use_reference_potential=False,
        param_fallback=False,
        weights=weights,
    )
    reward = GateProxyRewardComputer(device=device, config=config)

    good_metrics = _proxy_metrics(
        worst_ratio=0.5,
        b_jump=1e-4,
        far_slope=-1.0,
        near_slope=-1.0,
        spurious=0.0,
        rel_change=1e-4,
    )
    bad_metrics = _proxy_metrics(
        worst_ratio=10.0,
        b_jump=0.1,
        far_slope=-0.2,
        near_slope=-0.4,
        spurious=0.2,
        rel_change=0.2,
    )

    def fake_metrics(*args, **kwargs) -> Dict[str, Any]:
        program = kwargs.get("program")
        return good_metrics if program is prog_good else bad_metrics

    monkeypatch.setattr(reward, "_compute_proxy_metrics", fake_metrics)

    terms_good = reward.compute(prog_good, spec, device=device, seed=1)
    terms_bad = reward.compute(prog_bad, spec, device=device, seed=1)
    assert terms_good.logR.item() > terms_bad.logR.item()

    def same_metrics(*args, **kwargs) -> Dict[str, Any]:
        return good_metrics

    monkeypatch.setattr(reward, "_compute_proxy_metrics", same_metrics)
    terms_short = reward.compute(prog_good, spec, device=device, seed=2)
    terms_long = reward.compute(prog_bad, spec, device=device, seed=2)
    assert terms_short.logR.item() > terms_long.logR.item()
