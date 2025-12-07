import numpy as np
import torch

from electrodrive.learn import collocation
from electrodrive.learn.neural_operators import SphereFNOSurrogate
from electrodrive.orchestration.parser import CanonicalSpec


class DummySphereModel(torch.nn.Module):
    def __init__(self, value: float = 1.0, n_theta: int = 4, n_phi: int = 4) -> None:
        super().__init__()
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.pos_enc = torch.zeros(1)
        self.value = float(value)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        if params.dim() == 1:
            params = params.unsqueeze(0)
        return torch.full(
            (params.shape[0], self.n_theta, self.n_phi),
            self.value,
            device=params.device,
            dtype=params.dtype,
        )


def _stage0_spec() -> CanonicalSpec:
    return CanonicalSpec.from_json(
        {
            "domain": "R3",
            "BCs": "dirichlet",
            "conductors": [
                {"type": "sphere", "radius": 1.0, "potential": 0.0, "center": [0, 0, 0]}
            ],
            "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 1.5]}],
        }
    )


def _plane_spec() -> CanonicalSpec:
    return CanonicalSpec.from_json(
        {
            "domain": "R3",
            "BCs": "dirichlet",
            "conductors": [{"type": "plane", "z": 0.0, "potential": 0.0}],
            "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 1.0]}],
        }
    )


def test_neural_collocation_uses_surrogate(monkeypatch):
    surrogate = SphereFNOSurrogate(
        model=DummySphereModel(value=2.5, n_theta=8, n_phi=8),
        validated=True,
        radial_extension="clamp_zero",
    )
    monkeypatch.setattr(collocation, "_NEURAL_SURROGATE_CACHE", {})
    monkeypatch.setattr(collocation, "_get_spherefno_surrogate", lambda device, dtype: surrogate)
    monkeypatch.setattr(
        collocation,
        "_validate_neural_predictions",
        lambda spec, geom_type, pts, preds, device, dtype, rng: (True, 0.0, 0.0, None),
    )

    batch = collocation.make_collocation_batch_for_spec(
        spec=_stage0_spec(),
        n_points=10,
        ratio_boundary=0.5,
        supervision_mode="neural",
        device=torch.device("cpu"),
        dtype=torch.float32,
        rng=np.random.default_rng(0),
    )
    assert batch["V_gt"].numel() == batch["X"].shape[0]
    assert torch.allclose(batch["V_gt"], torch.full_like(batch["V_gt"], 2.5))


def test_neural_collocation_geometry_mismatch_fallback(monkeypatch):
    surrogate = SphereFNOSurrogate(
        model=DummySphereModel(value=3.0, n_theta=8, n_phi=8),
        validated=True,
        radial_extension="clamp_zero",
    )
    monkeypatch.setattr(collocation, "_NEURAL_SURROGATE_CACHE", {})
    monkeypatch.setattr(collocation, "_get_spherefno_surrogate", lambda device, dtype: surrogate)
    monkeypatch.setattr(
        collocation,
        "_validate_neural_predictions",
        lambda spec, geom_type, pts, preds, device, dtype, rng: (True, 0.0, 0.0, None),
    )
    batch = collocation.make_collocation_batch_for_spec(
        spec=_plane_spec(),
        n_points=6,
        ratio_boundary=0.5,
        supervision_mode="neural",
        device=torch.device("cpu"),
        dtype=torch.float32,
        rng=np.random.default_rng(1),
    )
    assert batch["V_gt"].numel() == batch["X"].shape[0]
    assert not torch.allclose(batch["V_gt"], torch.full_like(batch["V_gt"], 3.0))


def test_neural_collocation_validation_failure_fallback(monkeypatch):
    surrogate = SphereFNOSurrogate(
        model=DummySphereModel(value=4.0, n_theta=8, n_phi=8),
        validated=True,
        radial_extension="clamp_zero",
    )
    monkeypatch.setattr(collocation, "_NEURAL_SURROGATE_CACHE", {})
    monkeypatch.setattr(collocation, "_get_spherefno_surrogate", lambda device, dtype: surrogate)
    monkeypatch.setattr(
        collocation,
        "_validate_neural_predictions",
        lambda spec, geom_type, pts, preds, device, dtype, rng: (False, 1.0, 1.0, None),
    )
    batch = collocation.make_collocation_batch_for_spec(
        spec=_stage0_spec(),
        n_points=10,
        ratio_boundary=0.5,
        supervision_mode="neural",
        device=torch.device("cpu"),
        dtype=torch.float32,
        rng=np.random.default_rng(2),
    )
    assert batch["V_gt"].numel() == batch["X"].shape[0]
    assert not torch.allclose(batch["V_gt"], torch.full_like(batch["V_gt"], 4.0))
