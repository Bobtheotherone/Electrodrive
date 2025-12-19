from pathlib import Path

import torch

from electrodrive.images.basis import generate_candidate_basis
from electrodrive.orchestration.parser import parse_spec
from electrodrive.utils.device import ensure_cuda_available_or_skip


def test_three_layer_complex_candidates_cuda() -> None:
    ensure_cuda_available_or_skip("three_layer_complex basis requires CUDA")
    device = torch.device("cuda")
    dtype = torch.float32
    spec = parse_spec(Path("specs/planar_three_layer_eps2_80_sym_h04_region1.json"))
    basis_types = ["axis_point", "three_layer_images", "three_layer_complex"]

    candidates = generate_candidate_basis(
        spec,
        basis_types=basis_types,
        n_candidates=12,
        device=device,
        dtype=dtype,
    )

    layered = [c for c in candidates if c.type == "three_layer_images"]
    assert layered, "three_layer_complex should emit layered candidates on CUDA"

    families = [getattr(c, "_group_info", {}) or {} for c in layered]
    family_names = {f.get("family_name") for f in families if f.get("family_name")}
    assert "three_layer_complex_mirror" in family_names
    assert "three_layer_complex_tail" in family_names

    conductor_ids = {f.get("conductor_id") for f in families if f.get("conductor_id") is not None}
    assert 1 in conductor_ids  # slab interior images
    assert 2 in conductor_ids  # below-slab tails


def test_three_layer_complex_is_opt_in() -> None:
    ensure_cuda_available_or_skip("three_layer_complex opt-in requires CUDA")
    device = torch.device("cuda")
    dtype = torch.float32
    spec = parse_spec(Path("specs/planar_three_layer_eps2_80_sym_h04_region1.json"))

    candidates = generate_candidate_basis(
        spec,
        basis_types=["axis_point"],  # no three_layer_complex, no three_layer_images
        n_candidates=12,
        device=device,
        dtype=dtype,
    )
    layered = [c for c in candidates if c.type == "three_layer_images"]
    assert not layered, "three_layer_complex must remain opt-in; no layered candidates by default"
