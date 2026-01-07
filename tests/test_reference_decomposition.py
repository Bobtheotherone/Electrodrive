import torch

from electrodrive.experiments.reference_math import add_reference, stable_subtract_reference
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
        "charges": [{"type": "point", "q": 1e-6, "pos": [0.0, 0.0, 0.4]}],
        "BCs": "dielectric_interfaces",
        "symmetry": ["rot_z"],
        "queries": [],
    }
    return CanonicalSpec.from_json(spec)


def _run_decomposition(device: torch.device) -> None:
    dtype = torch.float32
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
    V_ref64 = compute_layered_reference_potential(
        spec,
        pts.to(dtype=torch.float64),
        device=device,
        dtype=torch.float64,
    )
    V_ref = V_ref64.to(dtype=dtype)
    V_corr_true = 1.0 * pts[:, 2]
    V_oracle = (V_ref + V_corr_true).to(dtype=dtype)

    V_corr_target = stable_subtract_reference(V_oracle, V_ref64, out_dtype=dtype)
    assert V_corr_target.dtype == dtype
    assert torch.allclose(V_corr_target, V_corr_true, atol=5e-4, rtol=5e-4)

    V_full = add_reference(V_corr_target, V_ref)
    assert torch.allclose(V_full, V_oracle, atol=5e-4, rtol=5e-4)


def test_reference_decomposition_cpu() -> None:
    _run_decomposition(torch.device("cpu"))


def test_reference_decomposition_cuda() -> None:
    if not torch.cuda.is_available():
        return
    _run_decomposition(torch.device("cuda"))
