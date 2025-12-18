import pytest
import torch

from electrodrive.verify.oracle_backends import F0AnalyticOracleBackend
from electrodrive.verify.oracle_types import OracleFidelity, OracleQuery, OracleQuantity


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for F0 oracle tests")


def test_plane_dirichlet_boundary_zero() -> None:
    _skip_if_no_cuda()
    spec = {
        "domain": "halfspace",
        "BCs": "Dirichlet",
        "conductors": [{"type": "plane", "z": 0.0, "potential": 0.0}],
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 1.0]}],
        "dielectrics": [],
    }
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.2, 0.0], [0.0, 0.0, 0.5]],
        device="cuda",
        dtype=torch.float32,
    )
    backend = F0AnalyticOracleBackend()
    query = OracleQuery(
        spec=spec,
        points=points,
        quantity=OracleQuantity.POTENTIAL,
        requested_fidelity=OracleFidelity.F0,
    )
    assert backend.can_handle(query)
    result = backend.evaluate(query)
    boundary_mask = torch.isclose(points[:, 2], torch.tensor(0.0, device=points.device, dtype=points.dtype))
    assert boundary_mask.any()
    assert result.V is not None
    assert torch.max(torch.abs(result.V[boundary_mask])).item() < 1e-4
