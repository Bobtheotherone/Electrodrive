from electrodrive.gfdsl.ast import OpaqueNode
from electrodrive.gfdsl.io import deserialize_program, serialize_program
from electrodrive.gfdsl.io.schema import schema_header


def test_unknown_node_roundtrips_as_opaque():
    payload = {
        **schema_header(),
        "program": {
            "node_type": "unknown_future_node",
            "children": [],
            "params": {
                "mystery": {
                    "raw": 1.0,
                    "transform": {"type": "identity"},
                    "trainable": False,
                    "dtype_policy": "work",
                    "bounds_hint": None,
                }
            },
            "meta": {"note": "keep_me"},
        },
    }
    node = deserialize_program(payload)
    assert isinstance(node, OpaqueNode)

    serialized = serialize_program(node)
    assert serialized["program"]["node_type"] == "unknown_future_node"
    assert serialized["program"]["meta"]["note"] == "keep_me"

