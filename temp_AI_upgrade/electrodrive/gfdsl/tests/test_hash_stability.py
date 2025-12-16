from electrodrive.gfdsl.ast import RealImageChargeNode, Param
from electrodrive.gfdsl.io import serialize_program, deserialize_program


def test_hash_stable_across_calls():
    node = RealImageChargeNode(
        params={
            "position": Param([0.0, 0.0, 0.5]),
        }
    )
    structure_1 = node.structure_hash()
    structure_2 = node.structure_hash()
    assert structure_1 == structure_2

    payload = serialize_program(node)
    roundtrip = deserialize_program(payload)
    assert node.structure_hash() == roundtrip.structure_hash()
    assert node.full_hash() == roundtrip.full_hash()
