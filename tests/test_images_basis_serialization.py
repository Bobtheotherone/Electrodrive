import torch

from electrodrive.images.basis import (
    BASIS_FAMILY_ENUM,
    BASIS_FAMILY_NAMES,
    ImageBasisElement,
    PointChargeBasis,
    annotate_group_info,
)


def test_serialize_deserialize_round_trip_group_info() -> None:
    pos = torch.tensor([0.1, -0.2, 0.3])
    elem = PointChargeBasis({"position": pos})
    annotate_group_info(
        elem,
        conductor_id=2,
        family_name="axis_point",
        motif_index=5,
    )

    data = elem.serialize()
    assert "group_info" in data

    round_trip = ImageBasisElement.deserialize(data, device="cpu", dtype=torch.float32)
    assert isinstance(round_trip, PointChargeBasis)
    assert torch.allclose(round_trip.params["position"], pos)

    info = getattr(round_trip, "_group_info", {})
    assert info.get("conductor_id") == 2
    assert info.get("motif_index") == 5
    assert info.get("family_name") == "axis_point"
    assert info.get("family") == BASIS_FAMILY_ENUM["axis_point"]


def test_serialize_without_group_info_omits_field() -> None:
    elem = PointChargeBasis({"position": torch.tensor([0.0, 0.0, 0.0])})
    data = elem.serialize()
    assert "group_info" not in data

    round_trip = ImageBasisElement.deserialize(data, device="cpu", dtype=torch.float32)
    assert isinstance(round_trip, PointChargeBasis)
    assert getattr(round_trip, "_group_info", {}) == {}


def test_basis_family_enum_is_bijection() -> None:
    values = list(BASIS_FAMILY_ENUM.values())
    assert len(values) == len(set(values))
    assert len(BASIS_FAMILY_ENUM) == len(BASIS_FAMILY_NAMES)
    for name, code in BASIS_FAMILY_ENUM.items():
        assert BASIS_FAMILY_NAMES[code - 1] == name
    for fam in ("three_layer_mirror", "three_layer_slab", "three_layer_tail"):
        assert fam in BASIS_FAMILY_ENUM
