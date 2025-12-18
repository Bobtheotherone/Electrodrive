import pytest
import torch

from electrodrive.verify.oracle_backends import F0AnalyticOracleBackend
from electrodrive.verify.oracle_manager import OracleManager
from electrodrive.verify.oracle_registry import OracleBackend, OracleRegistry
from electrodrive.verify.oracle_types import OracleFidelity, OracleQuery, OracleQuantity


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for oracle manager tests")


class _FailingBackend(OracleBackend):
    @property
    def name(self) -> str:
        return "failing_backend"

    @property
    def fidelity(self) -> OracleFidelity:
        return OracleFidelity.F0

    def can_handle(self, _query: OracleQuery) -> bool:
        return True

    def evaluate(self, _query: OracleQuery):
        raise RuntimeError("intentional failure")

    def fingerprint(self) -> str:
        return "failing_backend_fingerprint"


def test_manager_fallback_on_exception() -> None:
    _skip_if_no_cuda()
    registry = OracleRegistry()
    registry.register(_FailingBackend())
    registry.register(F0AnalyticOracleBackend())
    manager = OracleManager(registry)

    spec = {
        "domain": "halfspace",
        "BCs": "Dirichlet",
        "conductors": [{"type": "plane", "z": 0.0, "potential": 0.0}],
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 1.0]}],
        "dielectrics": [],
    }
    points = torch.tensor([[0.0, 0.0, 0.5], [0.1, 0.0, 0.7]], device="cuda", dtype=torch.float32)
    query = OracleQuery(
        spec=spec,
        points=points,
        quantity=OracleQuantity.POTENTIAL,
        requested_fidelity=OracleFidelity.F0,
    )

    result, backend = manager.evaluate_with_backend(query)

    assert backend.name == "f0_analytic"
    assert result.V is not None
