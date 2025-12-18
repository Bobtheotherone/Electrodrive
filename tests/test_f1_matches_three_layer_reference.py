import pytest
import torch

from electrodrive.core.planar_stratified_reference import ThreeLayerConfig, potential_three_layer_region1
from electrodrive.verify.oracle_backends import F1SommerfeldOracleBackend
from electrodrive.verify.oracle_types import CachePolicy, OracleFidelity, OracleQuery, OracleQuantity


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for F1 oracle tests")


def test_f1_matches_three_layer_reference() -> None:
    _skip_if_no_cuda()
    eps1, eps2, eps3 = 2.0, 4.0, 1.5
    h = 0.6
    spec = {
        "domain": "layered",
        "BCs": "dielectric_interfaces",
        "dielectrics": [
            {"name": "layer0", "epsilon": eps1, "z_min": 0.0, "z_max": float("inf")},
            {"name": "layer1", "epsilon": eps2, "z_min": -h, "z_max": 0.0},
            {"name": "layer2", "epsilon": eps3, "z_min": -float("inf"), "z_max": -h},
        ],
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.8]}],
    }
    points = torch.tensor(
        [[0.0, 0.0, 0.2], [0.1, -0.1, 0.5], [0.2, 0.2, 1.2]],
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
        budget={"sommerfeld": {"n_low": 80, "n_mid": 120, "n_high": 80, "k_max": 50.0}},
    )
    result = backend.evaluate(query)
    cfg = ThreeLayerConfig(eps1=eps1, eps2=eps2, eps3=eps3, h=h, q=1.0, r0=(0.0, 0.0, 0.8), n_k=256, k_max=50.0)
    ref = potential_three_layer_region1(points, cfg, device=points.device, dtype=points.dtype)
    assert result.V is not None
    assert torch.allclose(result.V, ref, rtol=1e-2, atol=1e-3)
