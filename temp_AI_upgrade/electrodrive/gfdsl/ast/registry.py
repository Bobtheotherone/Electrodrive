"""Node registry utilities for GFDSL."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from .constraints import GroupInfo

try:
    from .nodes import GFNode  # type: ignore
except Exception:  # pragma: no cover - during type checking
    GFNode = None  # type: ignore

_NODE_REGISTRY: Dict[str, Type["GFNode"]] = {}


def register_node(cls: Type["GFNode"]) -> Type["GFNode"]:
    node_type = getattr(cls, "node_type", None)
    if not node_type:
        raise ValueError("register_node requires class with node_type attribute")
    if node_type in _NODE_REGISTRY:
        raise ValueError(f"Duplicate node_type registration: {node_type}")
    _NODE_REGISTRY[node_type] = cls
    return cls


def get_node_cls(node_type: str) -> Optional[Type["GFNode"]]:
    return _NODE_REGISTRY.get(node_type)


def list_node_types() -> Dict[str, Type["GFNode"]]:
    return dict(_NODE_REGISTRY)


def make_node(node_type: str, **payload: Any) -> "GFNode":
    cls = get_node_cls(node_type)
    if cls is None:
        from .nodes import OpaqueNode

        return OpaqueNode(original_payload={"node_type": node_type, **payload})
    if hasattr(cls, "from_json_dict"):
        return cls.from_json_dict({"node_type": node_type, **payload})
    return cls(**payload)

