from pathlib import Path

import pytest
import torch

from electrodrive.verify.oracle_backends import F1SommerfeldOracleBackend
from electrodrive.verify.oracle_types import CachePolicy, OracleFidelity, OracleQuery, OracleQuantity


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for F1 cache tests")


def test_f1_cache_roundtrip(tmp_path: Path) -> None:
    _skip_if_no_cuda()
    device = torch.device("cuda")
    dtype = torch.float64
    h = 0.5
    spec = {
        "domain": "layered",
        "BCs": "dielectric_interfaces",
        "dielectrics": [
            {"name": "layer0", "epsilon": 2.0, "z_min": 0.0, "z_max": float("inf")},
            {"name": "layer1", "epsilon": 5.0, "z_min": -h, "z_max": 0.0},
            {"name": "layer2", "epsilon": 1.0, "z_min": -float("inf"), "z_max": -h},
        ],
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.6]}],
    }
    points = torch.tensor([[0.0, 0.0, 0.1], [0.1, -0.05, 0.2]], device=device, dtype=dtype)
    budget = {"sommerfeld": {"n_low": 64, "n_mid": 96, "n_high": 64, "k_max": 40.0}}
    backend = F1SommerfeldOracleBackend(disk_root=tmp_path / "som_cache")

    query = OracleQuery(
        spec=spec,
        points=points,
        quantity=OracleQuantity.POTENTIAL,
        requested_fidelity=OracleFidelity.F1,
        cache_policy=CachePolicy.USE_CACHE,
        budget=budget,
    )

    res1 = backend.evaluate(query)
    assert res1.V is not None
    assert torch.isfinite(res1.V).all()
    hit_k1 = res1.error_estimate.metrics.get("cache_hit_k", 0.0)
    hit_r1 = res1.error_estimate.metrics.get("cache_hit_R", 0.0)
    assert hit_k1 == 0.0 or hit_r1 == 0.0

    res2 = backend.evaluate(query)
    assert res2.V is not None
    assert torch.isfinite(res2.V).all()
    hit_k2 = res2.error_estimate.metrics.get("cache_hit_k", 0.0)
    hit_r2 = res2.error_estimate.metrics.get("cache_hit_R", 0.0)
    assert (hit_k2 == 1.0) or (hit_r2 == 1.0)
