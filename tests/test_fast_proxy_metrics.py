import torch

from electrodrive.experiments.fast_proxy_metrics import far_field_ratio, interface_jump
from electrodrive.experiments.layered_sampling import sample_layered_interface_pairs
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.device import ensure_cuda_available_or_skip


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


def test_fast_proxy_metrics_monotonic():
    ensure_cuda_available_or_skip("fast proxy metrics")
    device = torch.device("cuda")
    dtype = torch.float32
    near = torch.randn(32, 3, device=device, dtype=dtype)
    near = near / torch.linalg.norm(near, dim=1, keepdim=True).clamp_min(1e-6) * 0.5
    far = torch.randn(32, 3, device=device, dtype=dtype)
    far = far / torch.linalg.norm(far, dim=1, keepdim=True).clamp_min(1e-6) * 10.0

    def eval_good(pts: torch.Tensor) -> torch.Tensor:
        r = torch.linalg.norm(pts, dim=1).clamp_min(1e-6)
        return 1.0 / r

    def eval_bad(pts: torch.Tensor) -> torch.Tensor:
        r = torch.linalg.norm(pts, dim=1)
        return r

    ratio_good = far_field_ratio(eval_good, near, far)
    ratio_bad = far_field_ratio(eval_bad, near, far)
    assert ratio_bad > ratio_good

    spec = _layered_spec()
    pts_up, pts_dn = sample_layered_interface_pairs(
        spec,
        n_xy=16,
        device=device,
        dtype=dtype,
        seed=7,
        delta=1e-2,
        domain_scale=1.0,
    )

    def eval_continuous(pts: torch.Tensor) -> torch.Tensor:
        return pts[:, 2]

    def eval_jump(pts: torch.Tensor) -> torch.Tensor:
        return torch.where(pts[:, 2] >= 0.0, torch.ones_like(pts[:, 2]), -torch.ones_like(pts[:, 2]))

    jump_good = interface_jump(eval_continuous, pts_up, pts_dn)
    jump_bad = interface_jump(eval_jump, pts_up, pts_dn)
    assert jump_bad > jump_good
