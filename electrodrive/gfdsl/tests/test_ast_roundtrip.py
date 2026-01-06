import json

import torch

from electrodrive.gfdsl.ast import (
    BranchCutApproxNode,
    ComplexImageChargeNode,
    ConjugatePairNode,
    DCIMBlockNode,
    DipoleNode,
    IntegerSoftRoundTransform,
    ImageLadderNode,
    InterfacePoleNode,
    MirrorAcrossPlaneNode,
    Param,
    RealImageChargeNode,
    SoftplusTransform,
    SumNode,
)
from electrodrive.gfdsl.compile import validate_program
from electrodrive.gfdsl.io import deserialize_program, serialize_program, serialize_program_json


def _make_roundtrip_program():
    complex_image = ComplexImageChargeNode(
        params={
            "x": Param(0.1),
            "y": Param(-0.2),
            "a": Param(0.4),
            "b": Param(raw=0.5, transform=SoftplusTransform(min=1e-3)),
        }
    )
    conj = ConjugatePairNode(children=(complex_image,))

    mirror = MirrorAcrossPlaneNode(
        params={"z0": Param(0.3)},
        children=(
            RealImageChargeNode(
                params={
                    "position": Param([0.0, 0.0, 1.0]),
                }
            ),
        ),
    )

    ladder = ImageLadderNode(
        params={
            "step": Param(0.1),
            "count": Param(raw=3.0, transform=IntegerSoftRoundTransform(min_value=1, max_value=5)),
            "decay": Param(0.9),
        },
        children=(
            DipoleNode(
                params={
                    "position": Param([0.0, 0.0, 1.0]),
                }
            ),
        ),
    )

    interface_pole = InterfacePoleNode(
        params={
            "mode_id": Param(1.0),
            "k_pole": Param(raw=[0.2, 0.3], transform=SoftplusTransform(min=1e-3)),
            "residue": Param(raw=[0.5, 0.6]),
        },
        meta={"region": "upper"},
    )

    branchcut = BranchCutApproxNode(
        params={
            "depths": Param(torch.tensor([0.4, 0.8])),
            "weights": Param(torch.tensor([0.5, 0.2])),
        },
        meta={"kind": "quadrature_hankel"},
    )

    dcim = DCIMBlockNode(
        poles=(interface_pole,),
        images=(
            ConjugatePairNode(
                children=(
                    ComplexImageChargeNode(
                        params={
                            "x": Param(0.05),
                            "y": Param(0.07),
                            "a": Param(0.2),
                            "b": Param(raw=0.3, transform=SoftplusTransform(min=1e-3)),
                        }
                    ),
                )
            ),
        ),
        branchcut=branchcut,
    )

    program = SumNode(children=(conj, mirror, ladder, dcim))
    return program


def test_roundtrip_canonical_dict_stable():
    program = _make_roundtrip_program()
    validate_program(program)

    payload = serialize_program(program, meta={"tag": "test"})
    json_payload = serialize_program_json(program, meta={"tag": "test"})
    deserialized = deserialize_program(payload)
    deserialized_from_json = deserialize_program(json_payload)

    assert program.canonical_dict(include_raw=True) == deserialized.canonical_dict(include_raw=True)
    assert program.canonical_dict(include_raw=True) == deserialized_from_json.canonical_dict(include_raw=True)
    assert program.full_hash() == deserialized.full_hash()
    assert json.loads(json_payload)["schema_name"] == payload["schema_name"]
