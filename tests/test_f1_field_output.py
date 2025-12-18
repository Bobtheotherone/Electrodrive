import pytest
import torch

from electrodrive.verify.oracle_backends import F1SommerfeldOracleBackend
from electrodrive.verify.oracle_types import CachePolicy, OracleFidelity, OracleQuery, OracleQuantity


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for F1 E-field test")


def _eval_potential(backend: F1SommerfeldOracleBackend, spec, point: torch.Tensor, budget: dict) -> torch.Tensor:
    query = OracleQuery(
        spec=spec,
        points=point.unsqueeze(0),
        quantity=OracleQuantity.POTENTIAL,
        requested_fidelity=OracleFidelity.F1,
        cache_policy=CachePolicy.OFF,
        budget=budget,
    )
    res = backend.evaluate(query)
    assert res.V is not None
    return res.V[0]


def test_f1_field_matches_fd_sanity() -> None:
    _skip_if_no_cuda()
    device = torch.device("cuda")
    dtype = torch.float64
    h = 0.4
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
    points = torch.tensor(
        [[0.0, 0.0, 0.1], [0.05, -0.02, 0.2]],
        device=device,
        dtype=dtype,
    )
    budget = {"sommerfeld": {"n_low": 80, "n_mid": 120, "n_high": 80, "k_max": 50.0}}
    backend = F1SommerfeldOracleBackend(enable_disk=False)

    field_query = OracleQuery(
        spec=spec,
        points=points,
        quantity=OracleQuantity.FIELD,
        requested_fidelity=OracleFidelity.F1,
        cache_policy=CachePolicy.OFF,
        budget=budget,
    )
    res = backend.evaluate(field_query)
    assert res.E is not None
    assert res.E.shape == (points.shape[0], 3)
    assert torch.isfinite(res.E).all()
    assert res.valid_mask.all()

    # Finite-difference sanity at first point
    eps = 1e-4
    p0 = points[0].clone()
    fd_grad = []
    for i in range(3):
        shift = torch.zeros(3, device=device, dtype=dtype)
        shift[i] = eps
        vp = _eval_potential(backend, spec, p0 + shift, budget)
        vm = _eval_potential(backend, spec, p0 - shift, budget)
        fd_grad.append((vp - vm) / (2 * eps))
    fd_grad_t = torch.stack(fd_grad)
    e_pred = -fd_grad_t
    e_true = res.E[0]
    abs_err = torch.abs(e_pred - e_true)
    rel_err = abs_err / torch.clamp(torch.abs(e_true), min=1e-6)
    assert torch.all((abs_err < 1e-1) | (rel_err < 0.2))
