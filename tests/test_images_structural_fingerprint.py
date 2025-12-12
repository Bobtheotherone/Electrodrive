import json
from pathlib import Path

import torch

from electrodrive.images.basis import PointChargeBasis, annotate_group_info
from electrodrive.images.search import ImageSystem
from electrodrive.images.structural_features import structural_fingerprint
from electrodrive.discovery.novelty import novelty_score, compute_gate3_status
from electrodrive.orchestration.parser import CanonicalSpec


def _slab_spec() -> CanonicalSpec:
    data = {
        "domain": {"bbox": [[-1, -1, -2], [1, 1, 2]]},
        "dielectrics": [
            {"name": "region1", "epsilon": 1.0, "z_min": 0.5, "z_max": 2.0},
            {"name": "slab", "epsilon": 4.0, "z_min": 0.0, "z_max": 0.5},
            {"name": "region3", "epsilon": 1.0, "z_min": -2.0, "z_max": 0.0},
        ],
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.2]}],
        "BCs": "dielectric_interfaces",
    }
    return CanonicalSpec.from_json(json.loads(json.dumps(data)))


def test_structural_fingerprint_schema_and_defaults():
    spec = _slab_spec()
    elem_axis = PointChargeBasis({"position": torch.tensor([0.0, 0.0, 1.0])}, type_name="axis_point")
    annotate_group_info(elem_axis, conductor_id=0, family_name="axis_point", motif_index=0)
    elem_slab = PointChargeBasis({"position": torch.tensor([0.0, 0.0, 0.25])}, type_name="three_layer_images")
    annotate_group_info(elem_slab, conductor_id=1, family_name="three_layer_slab", motif_index=0)
    system = ImageSystem([elem_axis, elem_slab], torch.tensor([0.5, 1.0]))

    fp = structural_fingerprint(system, spec)
    assert set(fp["families"].keys()) == {
        "axis_point",
        "three_layer_mirror",
        "three_layer_slab",
        "three_layer_tail",
        "three_layer_diffusion",
    }
    assert fp["families"]["axis_point"]["count"] == 1
    assert fp["families"]["three_layer_slab"]["count"] == 1
    assert fp["ladder"]["three_layer_slab"]["b"] != 0.0
    assert "symmetry" in fp
    assert "axis_weight_l1_fraction" in fp
    assert abs(fp["axis_weight_l1_fraction"] - (0.5 / 1.5)) < 1e-6


def test_novelty_score_low_for_library_match_and_high_for_far_fp():
    spec = _slab_spec()
    elem = PointChargeBasis({"position": torch.tensor([0.0, 0.0, 0.25])}, type_name="three_layer_images")
    annotate_group_info(elem, conductor_id=1, family_name="three_layer_slab", motif_index=0)
    system = ImageSystem([elem], torch.tensor([1.0]))
    fp_match = structural_fingerprint(system, spec)
    score_match = novelty_score(fp_match)
    assert 0.0 <= score_match <= 1.0

    elem_far = PointChargeBasis({"position": torch.tensor([0.0, 0.0, 10.0])}, type_name="axis_point")
    annotate_group_info(elem_far, conductor_id=0, family_name="axis_point", motif_index=0)
    far_system = ImageSystem([elem_far], torch.tensor([1.0]))
    fp_far = structural_fingerprint(far_system, spec)
    score_far = novelty_score(fp_far)
    assert score_far >= score_match
    assert score_far <= 1.0


def test_gate3_status_logic():
    manifest = {
        "gate1_status": "pass",
        "numeric_status": "ok",
        "condition_status": "ok",
        "gate2_status": "pass",
    }
    status, novelty = compute_gate3_status(manifest, 0.8)
    assert status == "pass"
    status, novelty = compute_gate3_status(manifest, 0.3)
    assert status == "non_novel"
    manifest["numeric_status"] = "catastrophic"
    status, novelty = compute_gate3_status(manifest, 0.9)
    assert status == "n/a"
