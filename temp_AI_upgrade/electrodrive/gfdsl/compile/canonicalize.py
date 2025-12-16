"""Canonicalization and hashing utilities for GFDSL nodes."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from electrodrive.gfdsl.ast.constraints import GroupInfo
from electrodrive.gfdsl.ast.nodes import (
    DCIMBlockNode,
    GFNode,
    OpaqueNode,
)


def canonicalize(node: GFNode) -> GFNode:
    """Placeholder for future rewrites; currently returns the node unchanged."""
    return node


def _canonicalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _canonicalize_value(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def canonical_node_dict(
    node: GFNode, include_raw: bool = False, quantization: float = 1e-12
) -> Dict[str, Any]:
    params_list = []
    for name in sorted(node.params):
        param = node.params[name]
        params_list.append({"name": name, **param.canonical_dict(include_raw, quantization)})

    child_dicts: List[Dict[str, Any]] = []
    for child in node.children:
        child_dicts.append(canonical_node_dict(child, include_raw, quantization))
    child_dicts = sorted(child_dicts, key=lambda d: json.dumps(d, sort_keys=True))

    base = {
        "node_type": node.node_type,
        "params": params_list,
        "children": child_dicts,
        "meta": _canonicalize_value(node.meta),
    }
    if node.group_info:
        if isinstance(node.group_info, GroupInfo):
            base["group_info"] = _canonicalize_value(node.group_info.to_json_dict())
        else:
            base["group_info"] = _canonicalize_value(node.group_info)

    if isinstance(node, DCIMBlockNode):
        base["poles"] = sorted(
            [canonical_node_dict(p, include_raw, quantization) for p in node.poles],
            key=lambda d: json.dumps(d, sort_keys=True),
        )
        base["images"] = sorted(
            [canonical_node_dict(p, include_raw, quantization) for p in node.images],
            key=lambda d: json.dumps(d, sort_keys=True),
        )
        base["has_branchcut"] = node.branchcut is not None
    if isinstance(node, OpaqueNode):
        base["original_node_type"] = node.original_node_type
    return base


def _hash_canonical_dict(data: Dict[str, Any]) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def structure_hash(node: GFNode) -> str:
    """Hash over structure only (no raw parameter values)."""
    canonical = canonical_node_dict(node, include_raw=False)
    return _hash_canonical_dict(canonical)


def full_hash(node: GFNode, quantization: float = 1e-12) -> str:
    """Hash including raw parameter values (quantized)."""
    canonical = canonical_node_dict(node, include_raw=True, quantization=quantization)
    return _hash_canonical_dict(canonical)

