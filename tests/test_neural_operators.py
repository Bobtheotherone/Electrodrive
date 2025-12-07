import torch

from electrodrive.learn.neural_operators import (
    SphereFNOSurrogate,
    extract_stage0_sphere_params,
)
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


def test_extract_stage0_sphere_params_on_axis():
    spec = CanonicalSpec.from_json(
        {
            "domain": "R3",
            "BCs": "dirichlet",
            "conductors": [
                {"type": "sphere", "radius": 1.0, "potential": 0.0, "center": [0, 0, 0]}
            ],
            "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 1.5]}],
        }
    )
    out = extract_stage0_sphere_params(spec)
    assert out is not None
    q, z0, a, center = out
    assert q == 1.0
    assert a == 1.0
    assert center == (0.0, 0.0, 0.0)
    assert torch.isclose(torch.tensor(z0), torch.tensor(1.5))


def test_extract_stage0_sphere_params_off_axis():
    spec = CanonicalSpec.from_json(
        {
            "domain": "R3",
            "BCs": "dirichlet",
            "conductors": [
                {"type": "sphere", "radius": 1.0, "potential": 0.0, "center": [0, 0, 0]}
            ],
            "charges": [{"type": "point", "q": 1.0, "pos": [0.5, 0.0, 1.5]}],
        }
    )
    assert extract_stage0_sphere_params(spec) is None


def test_surrogate_evaluate_points_inv_r_and_clamp():
    model = DummySphereModel(value=2.0, n_theta=4, n_phi=4)
    surrogate = SphereFNOSurrogate(model=model, validated=True, radial_extension="inv_r")
    pts = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 0.5]], dtype=torch.float32)
    vals = surrogate.evaluate_points((1.0, 1.0, 1.0), pts, center=(0.0, 0.0, 0.0))
    assert torch.allclose(vals[0], torch.tensor(1.0))  # 2.0 * (1/2)
    assert torch.allclose(vals[1], torch.tensor(0.0))  # inside sphere -> 0

    surrogate_zero = SphereFNOSurrogate(model=model, validated=True, radial_extension="clamp_zero")
    vals_zero = surrogate_zero.evaluate_points((1.0, 1.0, 1.0), pts, center=(0.0, 0.0, 0.0))
    assert torch.allclose(vals_zero[0], torch.tensor(2.0))
    assert torch.allclose(vals_zero[1], torch.tensor(0.0))
