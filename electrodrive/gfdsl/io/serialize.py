"""Serialization helpers for GFDSL programs."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from electrodrive.gfdsl.ast.nodes import GFNode
from electrodrive.gfdsl.io.schema import schema_header


def serialize_program(root: GFNode, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        **schema_header(),
        "program": root.to_json_dict(),
        "meta": meta or {},
    }
    return payload


def serialize_program_json(
    root: GFNode, meta: Optional[Dict[str, Any]] = None, **json_kwargs: Any
) -> str:
    payload = serialize_program(root, meta=meta)
    default_kwargs = {"sort_keys": True, "indent": 2}
    default_kwargs.update(json_kwargs)
    return json.dumps(payload, **default_kwargs)

