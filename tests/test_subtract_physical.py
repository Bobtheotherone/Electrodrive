import torch
import numpy as np

from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.config import K_E
from electrodrive.images.search import get_collocation_data
from electrodrive.learn.collocation import compute_layered_reference_potential
from electrodrive.core.planar_stratified_reference import (
    ThreeLayerConfig,
    potential_three_layer_region1,
)
from electrodrive.utils.logging import JsonlLogger


class _StubLogger(JsonlLogger):
    def __init__(self):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _plane_spec():
    return CanonicalSpec.from_json(
        {
            "domain": {"bbox": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]},
            "conductors": [{"type": "plane", "z": 0.0, "potential": 0.0}],
            "dielectrics": [],
            "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.5]}],
            "BCs": "dirichlet",
        }
    )


def test_subtract_physical_plane_induced_only():
    spec = _plane_spec()
    n_points = 128
    ratio_boundary = 0.5
    rng = np.random.default_rng(0)

    X_sub, V_sub, _ = get_collocation_data(
        spec,
        logger=_StubLogger(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        return_is_boundary=True,
        rng=rng,
        n_points_override=n_points,
        ratio_override=ratio_boundary,
        subtract_physical_potential=True,
    )

    X = X_sub
    V_induced = V_sub

    # Free-space reference for the same charge.
    pos = torch.tensor([0.0, 0.0, 0.5], dtype=torch.float32)
    diff = X - pos
    R = torch.linalg.norm(diff, dim=1).clamp_min(1e-9)
    V_free = K_E * (1.0 / R)

    # Grounded plane analytic: V_plane = K_E * (q/r - q/r_mirror)
    Rm = torch.linalg.norm(X - torch.tensor([0.0, 0.0, -0.5], dtype=torch.float32), dim=1).clamp_min(1e-9)
    V_plane = K_E * (1.0 / R - 1.0 / Rm)
    V_recon = V_induced + V_free

    assert torch.isfinite(V_induced).all()
    mae = torch.mean(torch.abs(V_recon - V_plane))
    rel = mae / (torch.mean(torch.abs(V_plane)) + 1e-9)
    assert rel < 1e-6


def test_subtract_physical_layered_smoke():
    spec = CanonicalSpec.from_json(
        {
            "domain": {"bbox": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]},
            "conductors": [],
            "dielectrics": [
                {"name": "region1", "epsilon": 1.0, "z_min": 0.0, "z_max": 5.0},
                {"name": "slab", "epsilon": 4.0, "z_min": -0.3, "z_max": 0.0},
                {"name": "region3", "epsilon": 1.0, "z_min": -5.0, "z_max": -0.3},
            ],
            "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.2]}],
            "BCs": "dielectric_interfaces",
        }
    )
    rng = np.random.default_rng(1)
    X_full, V_full, _ = get_collocation_data(
        spec,
        logger=_StubLogger(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        return_is_boundary=True,
        rng=rng,
        n_points_override=64,
        ratio_override=0.5,
        subtract_physical_potential=False,
    )
    rng = np.random.default_rng(1)
    X_sub, V_sub, _ = get_collocation_data(
        spec,
        logger=_StubLogger(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        return_is_boundary=True,
        rng=rng,
        n_points_override=64,
        ratio_override=0.5,
        subtract_physical_potential=True,
    )
    assert torch.isfinite(V_sub).all()
    # Induced targets should be finite and not blow up relative to full.
    assert torch.mean(torch.abs(V_sub)) < 1e10


def test_subtract_physical_layered_homogeneous_eps1():
    spec = CanonicalSpec.from_json(
        {
            "domain": {"bbox": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]},
            "conductors": [],
            "dielectrics": [
                {"name": "region1", "epsilon": 1.0, "z_min": 0.0, "z_max": 5.0},
                {"name": "slab", "epsilon": 4.0, "z_min": -0.3, "z_max": 0.0},
                {"name": "region3", "epsilon": 1.0, "z_min": -5.0, "z_max": -0.3},
            ],
            "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.2]}],
            "BCs": "dielectric_interfaces",
        }
    )

    n_points = 128
    ratio_boundary = 0.5
    rng = np.random.default_rng(1)
    logger = _StubLogger()

    X_sub, V_induced, _ = get_collocation_data(
        spec,
        logger=logger,
        device=torch.device("cpu"),
        dtype=torch.float32,
        return_is_boundary=True,
        rng=rng,
        n_points_override=n_points,
        ratio_override=ratio_boundary,
        subtract_physical_potential=True,
    )

    eps1 = 1.0
    eps2 = 4.0
    eps3 = 1.0
    h = 0.3
    cfg = ThreeLayerConfig(
        eps1=eps1,
        eps2=eps2,
        eps3=eps3,
        h=h,
        q=1.0,
        r0=(0.0, 0.0, 0.2),
    )
    V_full = potential_three_layer_region1(
        X_sub, cfg, device=torch.device("cpu"), dtype=torch.float32
    )
    V_ref = compute_layered_reference_potential(
        spec, X_sub, device=torch.device("cpu"), dtype=torch.float32
    )
    V_recon = V_induced + V_ref

    num = torch.mean(torch.abs(V_recon - V_full))
    den = torch.mean(torch.abs(V_full)).clamp_min(1e-9)
    rel = (num / den).item()
    assert rel < 1e-2  # reconstruction within 1% on average

    assert torch.mean(torch.abs(V_induced)) < torch.mean(torch.abs(V_full))
