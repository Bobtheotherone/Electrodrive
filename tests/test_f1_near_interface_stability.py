import pytest
import torch

from electrodrive.verify.oracle_backends import F1SommerfeldOracleBackend
from electrodrive.verify.oracle_types import CachePolicy, OracleFidelity, OracleQuery, OracleQuantity


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for F1 near-interface tests")


def test_f1_near_interface_stable() -> None:
    _skip_if_no_cuda()
    h = 0.5
    spec = {
        "domain": "layered",
        "BCs": "dielectric_interfaces",
        "dielectrics": [
            {"name": "layer0", "epsilon": 2.0, "z_min": 0.0, "z_max": float("inf")},
            {"name": "layer1", "epsilon": 5.0, "z_min": -h, "z_max": 0.0},
            {"name": "layer2", "epsilon": 1.0, "z_min": -float("inf"), "z_max": -h},
        ],
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.1]}],
    }
    points = torch.tensor(
        [[0.0, 0.0, 1e-4], [0.2, -0.1, 5e-3], [0.0, 0.0, 0.02]],
        device="cuda",
        dtype=torch.float64,
    )
    backend = F1SommerfeldOracleBackend()
    query = OracleQuery(
        spec=spec,
        points=points,
        quantity=OracleQuantity.POTENTIAL,
        requested_fidelity=OracleFidelity.F1,
        cache_policy=CachePolicy.OFF,
        budget={"sommerfeld": {"n_low": 96, "n_mid": 160, "n_high": 120, "k_max": 60.0}},
    )
    result = backend.evaluate(query)
    assert result.V is not None
    assert torch.isfinite(result.V).all()
    assert result.valid_mask.all()
    assert result.error_estimate.metrics.get("tail_est", 0.0) >= 0.0
    assert result.error_estimate.metrics.get("quad_resid", 1.0) < 0.5
