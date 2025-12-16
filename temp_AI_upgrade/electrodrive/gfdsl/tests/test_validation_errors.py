import pytest

from electrodrive.gfdsl.ast import (
    BranchCutApproxNode,
    ComplexImageChargeNode,
    DipoleNode,
    MultipoleNode,
    Param,
    RealImageChargeNode,
)
from electrodrive.gfdsl.compile import GFDSLValidationError, validate_program


def test_multipole_invalid_m():
    node = MultipoleNode(
        params={
            "L": Param(1.0),
            "m": Param(3.0),
            "coeff": Param(0.5),
            "position": Param([0.0, 0.0, 0.0]),
        }
    )
    with pytest.raises(GFDSLValidationError) as excinfo:
        validate_program(node)
    assert "multipole" in excinfo.value.message.lower()


def test_complex_image_missing_positive_b_transform():
    node = ComplexImageChargeNode(
        params={
            "x": Param(0.0),
            "y": Param(0.0),
            "a": Param(0.1),
            "b": Param(0.1),  # identity transform should fail validation
        }
    )
    with pytest.raises(GFDSLValidationError) as excinfo:
        validate_program(node)
    assert "compleximagecharge" in excinfo.value.message.replace("_", "").lower()


def test_branchcut_invalid_kind():
    node = BranchCutApproxNode(params={"weights": Param(0.1)}, meta={"kind": "unknown"})
    with pytest.raises(GFDSLValidationError) as excinfo:
        validate_program(node)
    assert "branchcutapprox" in excinfo.value.message.replace("_", "").lower()


def test_real_charge_and_dipole_reject_legacy_amplitudes():
    with pytest.raises(GFDSLValidationError):
        validate_program(
            RealImageChargeNode(
                params={"position": Param([0.0, 0.0, 0.0]), "charge": Param(1.0)}
            )
        )
    with pytest.raises(GFDSLValidationError):
        validate_program(
            DipoleNode(
                params={
                    "position": Param([0.0, 0.0, 0.0]),
                    "moment": Param([1.0, 0.0, 0.0]),
                }
            )
        )
