import json
from pathlib import Path

import numpy as np
import torch

from electrodrive.learn import collocation
from electrodrive.learn.neural_operators import SphereFNO
from electrodrive.orchestration.parser import CanonicalSpec


def _load_stage0_spec() -> CanonicalSpec:
    spec_path = Path("specs") / "sphere_axis_point_external.json"
    return CanonicalSpec.from_json(json.loads(spec_path.read_text()))


def test_neural_surrogate_used(monkeypatch, tmp_path):
    # Build a tiny checkpoint with validated metrics.
    ckpt_path = tmp_path / "spherefno_ckpt.pth"
    model = SphereFNO()
    torch.save(
        {"model_state_dict": model.state_dict(), "metrics": {"val_rel_l2": 1e-4, "val_rel_linf": 1e-4}},
        ckpt_path,
    )
    monkeypatch.setenv("EDE_SPHEREFNO_CKPT", str(ckpt_path))
    monkeypatch.setenv("EDE_DEVICE", "cpu")

    # Skip validation sampling to keep runtime tiny and force surrogate usage.
    monkeypatch.setattr(collocation, "_SPHEREFNO_VAL_SAMPLES", 0)
    monkeypatch.setattr(collocation, "_NEURAL_FALLBACK_MODE", "analytic")

    spec = _load_stage0_spec()
    rng = np.random.default_rng(0)
    points = np.zeros((4, 3))

    preds, oracle_fb, info = collocation._evaluate_neural_oracle(
        spec, "sphere", points, torch.device("cpu"), torch.float32, rng
    )
    assert info["used"]
    assert preds is not None
    assert oracle_fb is None

    batch = collocation.make_collocation_batch_for_spec(
        spec=spec,
        n_points=16,
        ratio_boundary=0.5,
        supervision_mode="neural",
        device=torch.device("cpu"),
        dtype=torch.float32,
        rng=rng,
    )
    assert batch["V_gt"].shape[0] == batch["X"].shape[0]
    assert batch["V_gt"].numel() > 0


def test_neural_fallback_to_analytic(monkeypatch):
    # Remove surrogate; ensure we fall back cleanly.
    monkeypatch.delenv("EDE_SPHEREFNO_CKPT", raising=False)
    monkeypatch.setenv("EDE_DEVICE", "cpu")
    monkeypatch.setattr(collocation, "_SPHEREFNO_VAL_SAMPLES", 0)
    monkeypatch.setattr(collocation, "_NEURAL_FALLBACK_MODE", "analytic")

    spec = _load_stage0_spec()
    rng = np.random.default_rng(1)
    batch = collocation.make_collocation_batch_for_spec(
        spec=spec,
        n_points=12,
        ratio_boundary=0.5,
        supervision_mode="neural",
        device=torch.device("cpu"),
        dtype=torch.float32,
        rng=rng,
    )
    assert batch["V_gt"].numel() == batch["X"].shape[0]
    # Fallback should have produced finite values.
    assert bool(torch.isfinite(batch["V_gt"]).all())


def test_neural_validation_failure_triggers_fallback(monkeypatch, tmp_path):
    class _BadSurrogate:
        def is_ready(self) -> bool:
            return True

        def evaluate_points(self, params, pts, center=None):
            # Deliberately poor predictions to force validation failure.
            return torch.zeros((pts.shape[0],), device=pts.device, dtype=pts.dtype)

    # Force loading of the stub surrogate.
    monkeypatch.setenv("EDE_SPHEREFNO_CKPT", str(tmp_path / "dummy.pth"))
    monkeypatch.setenv("EDE_DEVICE", "cpu")
    monkeypatch.setattr(collocation, "_NEURAL_SURROGATE_CACHE", {})
    monkeypatch.setattr(collocation, "load_spherefno_from_env", lambda **kwargs: _BadSurrogate())
    monkeypatch.setattr(collocation, "_SPHEREFNO_VAL_SAMPLES", 4)
    monkeypatch.setattr(collocation, "_SPHEREFNO_L2_TOL", 1e-8)
    monkeypatch.setattr(collocation, "_SPHEREFNO_LINF_TOL", 1e-8)
    monkeypatch.setattr(collocation, "_NEURAL_FALLBACK_MODE", "analytic")

    spec = _load_stage0_spec()
    rng = np.random.default_rng(7)
    points = np.zeros((6, 3))

    preds, oracle_fb, info = collocation._evaluate_neural_oracle(
        spec, "sphere", points, torch.device("cpu"), torch.float32, rng
    )
    assert preds is None
    assert info["reason"] == "validation_failed"
    # Validation failure should provide an oracle for fallback when possible.
    assert oracle_fb is not None or info["rel_l2"] is None

    batch = collocation.make_collocation_batch_for_spec(
        spec=spec,
        n_points=10,
        ratio_boundary=0.4,
        supervision_mode="neural",
        device=torch.device("cpu"),
        dtype=torch.float32,
        rng=rng,
    )
    assert batch["V_gt"].numel() == batch["X"].shape[0]
    assert bool(torch.isfinite(batch["V_gt"]).all())
