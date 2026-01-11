from __future__ import annotations

import json

import torch

from electrodrive.images.basis import DCIMPoleImageBasis, annotate_group_info
from electrodrive.images.search import ImageSystem
from electrodrive.images.structural_features import structural_fingerprint
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


def _dcim_pole(z: float, *, interface_id: int, schema_id: int) -> DCIMPoleImageBasis:
    elem = DCIMPoleImageBasis(
        {"position": torch.tensor([0.0, 0.0, z]), "z_imag": torch.tensor(0.1)}
    )
    annotate_group_info(elem, conductor_id=0, family_name="dcim_pole", motif_index=0)
    info = getattr(elem, "_group_info", {})
    if isinstance(info, dict):
        info["interface_id"] = int(interface_id)
        info["schema_id"] = int(schema_id)
        info["n_poles"] = 1
        setattr(elem, "_group_info", info)
    return elem


def test_structural_features_dcim_pole_count_changes() -> None:
    spec = _slab_spec()
    elem = _dcim_pole(0.25, interface_id=0, schema_id=3)
    system_one = ImageSystem([elem], torch.tensor([1.0]))
    system_two = ImageSystem([elem, _dcim_pole(0.3, interface_id=0, schema_id=3)], torch.tensor([1.0, 1.0]))

    fp_one = structural_fingerprint(system_one, spec)
    fp_two = structural_fingerprint(system_two, spec)
    assert fp_one["families"]["dcim_pole"]["count"] != fp_two["families"]["dcim_pole"]["count"]


def test_structural_features_interface_id_changes() -> None:
    spec = _slab_spec()
    system_a = ImageSystem([_dcim_pole(0.25, interface_id=0, schema_id=3)], torch.tensor([1.0]))
    system_b = ImageSystem([_dcim_pole(0.25, interface_id=1, schema_id=3)], torch.tensor([1.0]))

    fp_a = structural_fingerprint(system_a, spec)
    fp_b = structural_fingerprint(system_b, spec)
    assert fp_a["discrete_ids"]["interface_id"]["mean"] != fp_b["discrete_ids"]["interface_id"]["mean"]


def test_structural_features_schema_id_changes() -> None:
    spec = _slab_spec()
    system_a = ImageSystem([_dcim_pole(0.25, interface_id=0, schema_id=3)], torch.tensor([1.0]))
    system_b = ImageSystem([_dcim_pole(0.25, interface_id=0, schema_id=5)], torch.tensor([1.0]))

    fp_a = structural_fingerprint(system_a, spec)
    fp_b = structural_fingerprint(system_b, spec)
    assert fp_a["discrete_ids"]["schema_id"]["mean"] != fp_b["discrete_ids"]["schema_id"]["mean"]
