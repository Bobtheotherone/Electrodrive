import torch

from electrodrive.experiments.run_discovery import _validate_weights, _weights_serializable
from electrodrive.experiments.utils import write_json
from electrodrive.images.basis import PointChargeBasis
from electrodrive.images.search import ImageSystem


def test_rejects_nonfinite_weights_before_verification() -> None:
    elem = PointChargeBasis({"position": torch.tensor([0.0, 0.0, 1.0])})
    system = ImageSystem([elem], torch.tensor([float("nan")]))
    ok, reason = _validate_weights(system.weights)
    assert not ok
    assert reason == "weights_nonfinite"


def test_weights_serializable_blocks_nonfinite_summary(tmp_path) -> None:
    bad_weights = torch.tensor([float("nan")])
    serial, reason = _weights_serializable(bad_weights)
    assert serial is None
    assert reason == "weights_nonfinite"

    good_weights = torch.tensor([1.0, -2.0])
    serial, reason = _weights_serializable(good_weights)
    assert reason == ""
    assert all(isinstance(v, float) for v in serial)

    path = tmp_path / "summary.json"
    write_json(path, {"weights": serial})
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
