"""Constraint and grouping helper types for GFDSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class GroupInfo:
    """Metadata for grouping coefficients (used for sparsity/regularization)."""

    conductor_id: int = -1
    family_name: str = ""
    motif_index: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "conductor_id": self.conductor_id,
            "family_name": self.family_name,
            "motif_index": self.motif_index,
            "extra": self.extra or {},
        }

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "GroupInfo":
        return cls(
            conductor_id=int(data.get("conductor_id", -1)),
            family_name=data.get("family_name", "") or "",
            motif_index=int(data.get("motif_index", 0)),
            extra=data.get("extra", data.get("extras", {})) or {},
        )

    @property
    def extras(self) -> Dict[str, Any]:
        """Backward-compatible alias for earlier 'extras' spelling."""
        return self.extra
