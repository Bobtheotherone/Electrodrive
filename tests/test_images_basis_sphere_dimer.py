from __future__ import annotations

import math

import torch
from electrodrive.images.basis import (
    BASIS_FAMILY_ENUM,
    SphereKelvinImageBasis,
    SphereEquatorialRingBasis,
    RingLadderBasis,
    build_dictionary,
    generate_candidate_basis,
)
from electrodrive.images.operator import BasisOperator
from electrodrive.orchestration.spec_registry import (
    load_stage0_sphere_external,
    load_stage1_sphere_dimer_inside,
)


def test_single_sphere_kelvin_candidates():
    spec = load_stage0_sphere_external()
    candidates = generate_candidate_basis(
        spec=spec,
        basis_types=["sphere_kelvin_ladder"],
        n_candidates=4,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert len(candidates) >= 1
    for elem in candidates:
        assert isinstance(elem, SphereKelvinImageBasis)
        pos = elem.params["position"]
        r = torch.linalg.norm(pos - torch.zeros_like(pos))
        assert float(r) < 1.0 + 1e-6  # inside sphere of radius 1


def test_kelvin_ladder_candidates_inside_spec():
    spec = load_stage1_sphere_dimer_inside()
    candidates = generate_candidate_basis(
        spec,
        basis_types=["sphere_kelvin_ladder"],
        n_candidates=8,
        device="cpu",
        dtype=torch.float64,
    )
    assert 1 <= len(candidates) <= 8
    for elem in candidates:
        assert isinstance(elem, SphereKelvinImageBasis)
        pos = elem.params["position"]
        z = float(pos[2].item())
        # Distances to sphere centers
        d0 = torch.linalg.norm(pos - torch.tensor([0.0, 0.0, 0.0], dtype=pos.dtype)).item()
        d1 = torch.linalg.norm(pos - torch.tensor([0.0, 0.0, 2.4], dtype=pos.dtype)).item()
        assert math.isfinite(d0) and math.isfinite(d1)
        assert min(d0, d1) < 1.0 + 1e-6, f"Kelvin image not inside a sphere (z={z})"


def test_sphere_equatorial_ring_candidates():
    spec = load_stage1_sphere_dimer_inside()
    candidates = generate_candidate_basis(
        spec,
        basis_types=["sphere_equatorial_ring"],
        n_candidates=3,
        device="cpu",
        dtype=torch.float64,
    )
    # With n_candidates=3 we expect one ring per sphere.
    assert len(candidates) == 2
    for elem in candidates:
        assert isinstance(elem, SphereEquatorialRingBasis)
        center = elem.params["center"]
        radius = float(elem.params["radius"])
        assert math.isfinite(radius) and radius > 0.0
        # Check proximity to either sphere center in z.
        zc = float(center[2].item())
        assert abs(zc - 0.0) < 1.0 or abs(zc - 2.4) < 1.0


def test_kelvin_ladder_groups_split_by_conductor():
    spec = load_stage1_sphere_dimer_inside()
    candidates = generate_candidate_basis(
        spec,
        basis_types=["sphere_kelvin_ladder"],
        n_candidates=8,
        device="cpu",
        dtype=torch.float64,
    )
    kelvin = [c for c in candidates if isinstance(c, SphereKelvinImageBasis)]
    assert len(kelvin) >= 2

    op = BasisOperator(kelvin, device="cpu", dtype=torch.float64)
    groups = op.groups
    assert groups is not None

    conductor_ids = [int(getattr(elem, "_group_info", {}).get("conductor_id", -1)) for elem in kelvin]
    assert set(conductor_ids) >= {0, 1}

    groups_cpu = groups.cpu()
    family_code = BASIS_FAMILY_ENUM["sphere_kelvin_image"]
    expected_prefix = {cid: cid * 1000 + family_code * 10 for cid in set(conductor_ids)}

    assert torch.unique(groups_cpu).numel() == len(expected_prefix)
    for gid, cid in zip(groups_cpu.tolist(), conductor_ids):
        assert gid == expected_prefix[cid]


def test_ring_and_ladder_batched_matches_dense_dictionary():
    dtype = torch.float64
    pts = torch.tensor(
        [
            [0.1, 0.0, 0.2],
            [0.5, 0.1, -0.1],
            [-0.3, 0.2, 0.4],
        ],
        dtype=dtype,
    )
    center = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)

    ladder_inner = RingLadderBasis(
        {
            "center": center,
            "radius": torch.tensor(1.1, dtype=dtype),
            "minor_radius": torch.tensor(0.35, dtype=dtype),
            "variant": "inner",
            "n_quad": torch.tensor(48),
        }
    )
    ladder_outer = RingLadderBasis(
        {
            "center": center,
            "radius": torch.tensor(1.1, dtype=dtype),
            "minor_radius": torch.tensor(0.35, dtype=dtype),
            "variant": "outer",
            "n_quad": torch.tensor(48),
        }
    )
    ring = SphereEquatorialRingBasis(
        {
            "center": center,
            "radius": torch.tensor(0.8, dtype=dtype),
            "normal": torch.tensor([0.0, 0.0, 1.0], dtype=dtype),
            "n_quad": torch.tensor(64),
        }
    )

    basis = [ladder_inner, ladder_outer, ring]

    Phi_dense = build_dictionary(basis, pts, dtype=dtype, batched=False)
    Phi_batched = build_dictionary(basis, pts, dtype=dtype, batched=True)
    assert Phi_dense.shape == Phi_batched.shape
    assert torch.allclose(Phi_batched, Phi_dense, atol=1e-10, rtol=1e-7)

    w = torch.tensor([0.5, -0.3, 0.2], dtype=dtype)
    op = BasisOperator(basis, pts, dtype=dtype)
    mat_dense = Phi_dense @ w
    mat_op = op.matvec(w, pts)
    assert torch.allclose(mat_op, mat_dense, atol=1e-10, rtol=1e-7)
