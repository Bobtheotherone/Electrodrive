import torch

from electrodrive.verify.gate_proxies import proxy_gateB, proxy_gateC, proxy_gateD
from electrodrive.orchestration.parser import CanonicalSpec


def _layered_spec() -> CanonicalSpec:
    spec = {
        "domain": {"bbox": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]},
        "conductors": [],
        "dielectrics": [
            {"name": "region1", "epsilon": 1.0, "z_min": 0.0, "z_max": 5.0},
            {"name": "slab", "epsilon": 4.0, "z_min": -0.3, "z_max": 0.0},
            {"name": "region3", "epsilon": 1.0, "z_min": -5.0, "z_max": -0.3},
        ],
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.2]}],
        "BCs": "dielectric_interfaces",
        "symmetry": ["rot_z"],
        "queries": [],
    }
    return CanonicalSpec.from_json(spec)


def _run_proxy_tests(device: torch.device) -> None:
    dtype = torch.float32
    spec = _layered_spec()
    pts = torch.randn(64, 3, device=device, dtype=dtype)

    def constant(p: torch.Tensor) -> torch.Tensor:
        return p[:, 0] * 0.0

    def linear(p: torch.Tensor) -> torch.Tensor:
        return p[:, 2]

    def monopole(p: torch.Tensor) -> torch.Tensor:
        r = torch.linalg.norm(p - torch.tensor([0.0, 0.0, 0.2], device=p.device, dtype=p.dtype), dim=1).clamp_min(1e-3)
        return 1.0 / r

    prox_b_const = proxy_gateB(spec, constant, n_xy=16, delta=0.01, device=device, dtype=dtype)
    prox_c_const = proxy_gateC(constant, near_radii=(0.125, 0.5), far_radii=(10.0, 20.0), n_dir=64, device=device, dtype=dtype)
    prox_d_const = proxy_gateD(constant, pts, delta=0.05)

    assert torch.isfinite(torch.tensor(list(prox_b_const.values()))).all()
    assert torch.isfinite(torch.tensor(list(prox_c_const.values()))).all()
    assert torch.isfinite(torch.tensor(list(prox_d_const.values()))).all()
    assert prox_b_const["proxy_gateB_max_v_jump"] <= 1e-6
    assert prox_b_const["proxy_gateB_max_d_jump"] <= 1e-6
    assert abs(prox_c_const["proxy_gateC_far_slope"]) < 0.2
    assert prox_d_const["proxy_gateD_rel_change"] <= 1e-6

    prox_b_linear = proxy_gateB(spec, linear, n_xy=16, delta=0.01, device=device, dtype=dtype)
    prox_c_linear = proxy_gateC(linear, near_radii=(0.125, 0.5), far_radii=(10.0, 20.0), n_dir=64, device=device, dtype=dtype)
    assert prox_b_linear["proxy_gateB_max_d_jump"] > 0.0
    assert prox_c_linear["proxy_gateC_far_slope"] > 0.5
    assert prox_c_linear["proxy_gateC_near_slope"] > 0.5

    prox_c_monopole = proxy_gateC(monopole, near_radii=(0.125, 0.5), far_radii=(10.0, 20.0), n_dir=64, device=device, dtype=dtype)
    assert abs(prox_c_monopole["proxy_gateC_far_slope"] + 1.0) < 0.3
    assert abs(prox_c_monopole["proxy_gateC_far_slope"] + 1.0) < abs(prox_c_const["proxy_gateC_far_slope"] + 1.0)
    assert abs(prox_c_monopole["proxy_gateC_far_slope"] + 1.0) < abs(prox_c_linear["proxy_gateC_far_slope"] + 1.0)


def test_gate_proxies_cpu() -> None:
    _run_proxy_tests(torch.device("cpu"))


def test_gate_proxies_cuda() -> None:
    if not torch.cuda.is_available():
        return
    _run_proxy_tests(torch.device("cuda"))
