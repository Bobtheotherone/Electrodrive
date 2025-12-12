import json
from pathlib import Path

import torch

from electrodrive.images.basis import PointChargeBasis, annotate_group_info
from electrodrive.images.search import ImageSystem
from tools.images_gate2 import compute_structural_summary
from electrodrive.orchestration.parser import CanonicalSpec


def _spec_three_layer() -> CanonicalSpec:
    data = {
        "domain": {"bbox": [[-1, -1, -1], [1, 1, 1]]},
        "conductors": [],
        "dielectrics": [
            {"name": "region1", "epsilon": 1.0, "z_min": 0.0, "z_max": 10.0},
            {"name": "slab", "epsilon": 5.0, "z_min": -0.5, "z_max": 0.0},
            {"name": "region3", "epsilon": 1.0, "z_min": -10.0, "z_max": -0.5},
        ],
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.2]}],
        "BCs": "dielectric_interfaces",
    }
    return CanonicalSpec.from_json(json.loads(json.dumps(data)))


def test_gate2_scoring_failures_due_to_duplicates_and_far_tails():
    spec = _spec_three_layer()
    elems = []
    weights = []
    pos_same = torch.tensor([0.0, 0.0, 0.1])

    e1 = PointChargeBasis({"position": pos_same.clone()}, type_name="axis_point")
    annotate_group_info(e1, conductor_id=0, family_name="axis_point", motif_index=0)
    elems.append(e1)
    weights.append(1.0)

    e2 = PointChargeBasis({"position": pos_same.clone()}, type_name="axis_point")
    annotate_group_info(e2, conductor_id=0, family_name="axis_point", motif_index=1)
    elems.append(e2)
    weights.append(0.99)

    far = PointChargeBasis({"position": torch.tensor([0.0, 0.0, 30.0])}, type_name="three_layer_images")
    annotate_group_info(far, conductor_id=2, family_name="three_layer_tail", motif_index=0)
    elems.append(far)
    weights.append(0.01)

    system = ImageSystem(elems, torch.tensor(weights, dtype=torch.float32))
    summary = compute_structural_summary(spec, system, numeric_status="ok", condition_status="ok")

    assert summary["gate2_status"] == "fail"
    assert summary["structure_score"] is not None
    assert summary["structure_score"] < 0.4
    assert summary["degeneracies"]["duplicate_physical_charges"] >= 1
    assert summary["degeneracies"]["far_tails_over_domain"] >= 1


def test_gate2_scoring_passes_for_healthy_slab_system():
    spec = _spec_three_layer()
    elems = []
    weights = []

    m = PointChargeBasis({"position": torch.tensor([0.0, 0.0, -0.1])}, type_name="three_layer_images")
    annotate_group_info(m, conductor_id=1, family_name="three_layer_mirror", motif_index=0)
    elems.append(m)
    weights.append(0.6)

    s = PointChargeBasis({"position": torch.tensor([0.0, 0.0, -0.2])}, type_name="three_layer_images")
    annotate_group_info(s, conductor_id=1, family_name="three_layer_slab", motif_index=0)
    elems.append(s)
    weights.append(0.5)

    t = PointChargeBasis({"position": torch.tensor([0.0, 0.0, -0.8])}, type_name="three_layer_images")
    annotate_group_info(t, conductor_id=2, family_name="three_layer_tail", motif_index=0)
    elems.append(t)
    weights.append(0.4)

    axis = PointChargeBasis({"position": torch.tensor([0.0, 0.0, 0.2])}, type_name="axis_point")
    annotate_group_info(axis, conductor_id=0, family_name="axis_point", motif_index=0)
    elems.append(axis)
    weights.append(0.05)

    system = ImageSystem(elems, torch.tensor(weights, dtype=torch.float32))
    summary = compute_structural_summary(spec, system, numeric_status="ok", condition_status="ok")

    assert summary["gate2_status"] == "pass"
    assert summary["structure_score"] is not None
    assert summary["structure_score"] > 0.7
    fams = summary["families"]
    assert "three_layer_slab" in fams and "three_layer_tail" in fams
