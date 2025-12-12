import json
from pathlib import Path

import torch

from electrodrive.images.basis import BASIS_FAMILY_ENUM, compute_group_ids, generate_candidate_basis
from electrodrive.orchestration.parser import CanonicalSpec


def _load_three_layer_spec() -> CanonicalSpec:
    spec_path = Path("specs/planar_three_layer_eps2_80_sym_h04_region1.json")
    data = json.loads(spec_path.read_text())
    return CanonicalSpec.from_json(data)


def test_three_layer_basis_depths_and_groups():
    spec = _load_three_layer_spec()
    cands = generate_candidate_basis(
        spec,
        basis_types=["three_layer_images"],
        n_candidates=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    z_vals = [float(elem.params["position"][2]) for elem in cands]
    assert len(z_vals) == 6
    expected = [-0.2, -1.0, -0.1, -0.3, -0.6, -1.6]
    for z, z_exp in zip(z_vals, expected):
        assert abs(z - z_exp) < 1e-6
    families = [getattr(elem, "_group_info", {}).get("family_name") for elem in cands]
    assert families[:2] == ["three_layer_mirror", "three_layer_mirror"]
    assert families[2:4] == ["three_layer_slab", "three_layer_slab"]
    assert families[4:] == ["three_layer_tail", "three_layer_tail"]
    assert all(elem.type == "three_layer_images" for elem in cands)

    group_ids = compute_group_ids(cands, device=torch.device("cpu"), dtype=torch.long)
    assert len(group_ids) == len(cands)
    family_codes = {fam: BASIS_FAMILY_ENUM[fam] for fam in ("three_layer_mirror", "three_layer_slab", "three_layer_tail")}
    conductor_ids = [1, 2, 1, 1, 2, 2]
    motifs = [0, 1, 0, 1, 0, 1]
    expected_ids = []
    for cid, fam, motif in zip(conductor_ids, families, motifs):
        fam_code = family_codes[fam]
        expected_ids.append(cid * 1000 + fam_code * 10 + motif)
    assert group_ids.tolist() == expected_ids


def test_three_layer_basis_inactive_for_non_layered():
    spec = _load_three_layer_spec()
    data = spec.to_json()
    data["dielectrics"] = data["dielectrics"][:2]  # break the stack
    bad_spec = CanonicalSpec.from_json(data)
    cands = generate_candidate_basis(
        bad_spec,
        basis_types=["three_layer_images"],
        n_candidates=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert cands == []
