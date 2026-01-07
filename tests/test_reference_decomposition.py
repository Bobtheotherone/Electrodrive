import torch

from electrodrive.learn.collocation import compute_layered_reference_potential
from electrodrive.orchestration.parser import CanonicalSpec


def _layered_spec() -> CanonicalSpec:
    spec = {
        "domain": {"bbox": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]},
        "conductors": [],
        "dielectrics": [
            {"name": "region1", "epsilon": 2.0, "z_min": 0.0, "z_max": 5.0},
            {"name": "slab", "epsilon": 4.0, "z_min": -0.5, "z_max": 0.0},
            {"name": "region3", "epsilon": 1.0, "z_min": -5.0, "z_max": -0.5},
        ],
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.4]}],
        "BCs": "dielectric_interfaces",
        "symmetry": ["rot_z"],
        "queries": [],
    }
    return CanonicalSpec.from_json(spec)


def _run_decomposition(device: torch.device) -> None:
    dtype = torch.float64
    spec = _layered_spec()
    pts = torch.tensor(
        [
            [0.2, -0.1, 0.3],
            [-0.5, 0.4, -0.2],
            [0.6, -0.3, 0.1],
            [-0.4, 0.2, -0.4],
        ],
        device=device,
        dtype=dtype,
    )
    V_ref = compute_layered_reference_potential(spec, pts, device=device, dtype=dtype)
    V_corr_true = 1e5 * pts[:, 2]
    V_oracle = V_ref + V_corr_true

    V_corr_target = V_oracle - V_ref
    assert torch.allclose(V_corr_target, V_corr_true, atol=1e-6, rtol=1e-5)

    V_full = V_ref + V_corr_target
    assert torch.allclose(V_full, V_oracle, atol=1e-6, rtol=1e-5)


def test_reference_decomposition_cpu() -> None:
    _run_decomposition(torch.device("cpu"))


def test_reference_decomposition_cuda() -> None:
    if not torch.cuda.is_available():
        return
    _run_decomposition(torch.device("cuda"))
