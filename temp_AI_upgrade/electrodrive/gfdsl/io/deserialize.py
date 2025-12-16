"""Deserialization helpers for GFDSL programs."""

from __future__ import annotations

import json
from typing import Any, Dict

import electrodrive.gfdsl.ast.nodes as _nodes  # noqa: F401 - ensures node registration
from electrodrive.gfdsl.ast.registry import make_node
from electrodrive.gfdsl.io.schema import check_schema


def _deserialize_node(data: Dict[str, Any]):
    if "node_type" not in data:
        raise ValueError("Node JSON missing 'node_type'")
    node_type = data.get("node_type")
    payload = {k: v for k, v in data.items() if k != "node_type"}
    return make_node(node_type, **payload)


def deserialize_program(data_or_str: Dict[str, Any] | str):
    if isinstance(data_or_str, str):
        payload = json.loads(data_or_str)
    else:
        payload = data_or_str
    if not isinstance(payload, dict):
        raise ValueError("deserialize_program expects dict or JSON string")
    check_schema(payload)
    program_data = payload.get("program")
    if not isinstance(program_data, dict):
        raise ValueError("GFDSL payload missing 'program' dict")
    return _deserialize_node(program_data)
