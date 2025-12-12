import torch

from electrodrive.images.search import ImageSystem, _numeric_diagnostics
from electrodrive.images.basis import PointChargeBasis
from tools.three_layer_capacity import lstsq_metrics


def test_numeric_diagnostics_ok_and_catastrophic():
    pts = torch.zeros(4, 3)
    g = torch.ones(4)
    elem = PointChargeBasis({"position": torch.tensor([0.0, 0.0, 1.0])})
    system = ImageSystem([elem], torch.tensor([0.0]))
    diag = _numeric_diagnostics(system, pts, g)
    assert diag["numeric_status"] == "ok"

    system_bad = ImageSystem([elem], torch.tensor([float("inf")]))
    diag_bad = _numeric_diagnostics(system_bad, pts, g)
    assert diag_bad["numeric_status"] == "nonfinite"


def test_condition_status_threshold():
    A = torch.eye(4)
    V = torch.ones(4)
    is_b = torch.zeros(4, dtype=torch.bool)
    stats = lstsq_metrics(A, V, is_b)
    assert stats["cond_est"] < 1e3
