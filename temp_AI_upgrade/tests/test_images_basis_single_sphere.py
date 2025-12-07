from __future__ import annotations

import torch

from electrodrive.images.basis import SphereKelvinImageBasis, generate_candidate_basis
from electrodrive.orchestration.spec_registry import load_stage0_sphere_external


def test_single_sphere_kelvin_candidates_external_axis():
    spec = load_stage0_sphere_external()
    candidates = generate_candidate_basis(
        spec,
        basis_types=["sphere_kelvin_ladder"],
        n_candidates=4,
        device="cpu",
        dtype=torch.float64,
    )

    assert len(candidates) == 4
    center = torch.tensor(spec.conductors[0]["center"], dtype=torch.float64)
    radius = float(spec.conductors[0]["radius"])
    charge_pos = torch.tensor(spec.charges[0]["pos"], dtype=torch.float64)
    r = torch.linalg.norm(charge_pos - center).item()
    expected = center + (radius * radius / (r * r)) * (charge_pos - center)
    expected_z = float(expected[2].item())

    first = candidates[0]
    assert isinstance(first, SphereKelvinImageBasis)
    assert abs(float(first.params["position"][2]) - expected_z) < 0.15 * radius

    for elem in candidates:
        assert isinstance(elem, SphereKelvinImageBasis)
        pos = elem.params["position"]
        dist = torch.linalg.norm(pos - center).item()
        assert dist > 0.0 and dist < radius + 1e-6
        # Axis-aligned charge is above the center, so images should sit on the same side.
        assert float(pos[2].item()) > float(center[2].item()) - 1e-6


def test_axis_point_candidates_charge_facing():
    spec = load_stage0_sphere_external()
    candidates = generate_candidate_basis(
        spec,
        basis_types=["axis_point"],
        n_candidates=4,
        device="cpu",
        dtype=torch.float64,
    )
    assert len(candidates) == 4
    center = torch.tensor(spec.conductors[0]["center"], dtype=torch.float64)
    radius = float(spec.conductors[0]["radius"])
    charge_pos = torch.tensor(spec.charges[0]["pos"], dtype=torch.float64)
    axis_dir = (charge_pos - center) / torch.linalg.norm(charge_pos - center)

    distances = []
    for elem in candidates:
        pos = elem.params["position"]
        assert float(torch.dot((pos - center), axis_dir)) > 0.0
        distances.append(torch.linalg.norm(pos - center).item())
    assert any(d > radius for d in distances)


def test_single_sphere_ring_charge_aware_orientation():
    spec = load_stage0_sphere_external()
    candidates = generate_candidate_basis(
        spec,
        basis_types=["sphere_equatorial_ring"],
        n_candidates=2,
        device="cpu",
        dtype=torch.float64,
    )
    assert len(candidates) == 1
    ring = candidates[0]
    center = torch.tensor(spec.conductors[0]["center"], dtype=torch.float64)
    charge_pos = torch.tensor(spec.charges[0]["pos"], dtype=torch.float64)
    radius = float(spec.conductors[0]["radius"])

    normal = ring.params.get("normal", None)
    assert normal is not None
    normal = normal / torch.linalg.norm(normal)
    axis_dir = (charge_pos - center) / torch.linalg.norm(charge_pos - center)
    cos_ang = float(torch.dot(normal, axis_dir).item())
    assert cos_ang > 0.9

    ring_center = ring.params["center"]
    proj = float(torch.dot(ring_center - center, axis_dir).item())
    assert proj > 0.0

    ring_radius = float(ring.params["radius"].item())
    assert ring_radius < radius
