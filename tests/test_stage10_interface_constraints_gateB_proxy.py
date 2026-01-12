import pytest
import torch

from electrodrive.experiments import run_discovery as rd
from electrodrive.experiments.layered_sampling import sample_layered_interior
from electrodrive.images.basis import PointChargeBasis
from electrodrive.images.search import assemble_basis_matrix, ImageSystem
from electrodrive.learn.collocation import compute_layered_reference_potential
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.verify.gate_proxies import proxy_gateB


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for interface-constraint Gate B proxy test"
)


def test_interface_constraints_reduce_gateB_proxy() -> None:
    device = torch.device("cuda")
    dtype = torch.float32
    spec = CanonicalSpec.from_json(
        {
            "domain": {"bbox": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]},
            "conductors": [{"type": "plane", "z": 0.0, "potential": 0.0}],
            "dielectrics": [
                {"z_min": -1.0, "z_max": 0.0, "epsilon": 2.0},
                {"z_min": 0.0, "z_max": 1.0, "epsilon": 1.0},
            ],
            "charges": [{"type": "point", "q": 1.0, "charge": 1.0, "pos": [0.0, 0.0, 0.5]}],
            "BCs": "dielectric_interfaces",
        }
    )

    elements = [
        PointChargeBasis({"position": torch.tensor([0.0, 0.0, 0.5], device=device, dtype=dtype)}),
        PointChargeBasis({"position": torch.tensor([0.25, 0.0, 0.35], device=device, dtype=dtype)}),
        PointChargeBasis({"position": torch.tensor([-0.2, 0.1, -0.3], device=device, dtype=dtype)}),
    ]

    interface_delta = 5e-2
    seed = 123
    constraint_data = rd._build_interface_constraint_data(
        spec,
        device=device,
        dtype=dtype,
        seed=seed,
        domain_scale=1.0,
        interface_delta=interface_delta,
        points_per_interface=8,
        weight=10.0,
        use_ref=False,
    )
    assert constraint_data is not None

    pts_up = constraint_data["pts_up"]
    pts_dn = constraint_data["pts_dn"]
    interior = sample_layered_interior(
        spec,
        16,
        device=device,
        dtype=dtype,
        seed=seed + 7,
        exclusion_radius=0.0,
        interface_band=0.0,
        domain_scale=1.0,
    )
    X_train = torch.cat([pts_up, pts_dn, interior], dim=0)
    V_train = compute_layered_reference_potential(spec, X_train, device=device, dtype=dtype)

    A_train = assemble_basis_matrix(elements, X_train)
    weights_base = rd._fast_weights(
        A_train,
        V_train,
        reg=1e-6,
        normalize=True,
        max_abs_b=float(torch.max(torch.abs(V_train)).item()),
    )
    assert weights_base.numel() == len(elements)
    system_base = ImageSystem(elements, weights_base)

    A_train_aug, V_train_aug, _ = rd._apply_interface_constraints(
        A_train=A_train,
        V_train=V_train,
        is_boundary=None,
        elements=elements,
        constraint_data=constraint_data,
    )
    weights_cons = rd._fast_weights(
        A_train_aug,
        V_train_aug,
        reg=1e-6,
        normalize=True,
        max_abs_b=float(torch.max(torch.abs(V_train)).item()),
    )
    assert weights_cons.numel() == len(elements)
    system_cons = ImageSystem(elements, weights_cons)

    proxy_base = proxy_gateB(
        spec,
        system_base.potential,
        n_xy=8,
        delta=interface_delta,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    proxy_cons = proxy_gateB(
        spec,
        system_cons.potential,
        n_xy=8,
        delta=interface_delta,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    assert proxy_cons["proxy_gateB_max_d_jump"] <= proxy_base["proxy_gateB_max_d_jump"]
