import electrodrive.gfdsl as gfdsl
from electrodrive.gfdsl.ast import Param, RealImageChargeNode
from electrodrive.gfdsl.compile import CompileContext, GFDSLOperator, lower_program, validate_program
from electrodrive.gfdsl.io import deserialize_program, serialize_program
from electrodrive.gfdsl.program_loader import load_gfdsl_programs


def test_import_contract():
    assert gfdsl is not None
    assert Param is not None
    assert RealImageChargeNode is not None
    assert CompileContext is not None
    assert GFDSLOperator is not None
    assert lower_program is not None
    assert validate_program is not None
    assert serialize_program is not None
    assert deserialize_program is not None
    assert load_gfdsl_programs is not None
