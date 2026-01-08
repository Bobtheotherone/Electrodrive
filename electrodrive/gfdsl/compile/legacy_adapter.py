"""Legacy adapter exposing GFDSL linear contributions as ImageBasisElement-like wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

from electrodrive.gfdsl.ast.constraints import GroupInfo
from electrodrive.gfdsl.compile.lower import ColumnEvaluator, LinearContribution


@dataclass
class LegacyBasisElement:
    """Minimal wrapper mimicking the legacy ImageBasisElement interface."""

    type: str
    params: Dict[str, Any]
    group_info: Dict[str, Any]
    _k: int
    _evaluator: ColumnEvaluator
    _cache: Dict[Tuple[int, int, torch.device, torch.dtype, Tuple[int, ...]], torch.Tensor]

    def __post_init__(self) -> None:
        if self.group_info:
            setattr(self, "_group_info", dict(self.group_info))

    def _columns_for(self, targets: torch.Tensor) -> torch.Tensor:
        key = (id(targets), targets.data_ptr(), targets.device, targets.dtype, tuple(targets.shape))
        if key not in self._cache:
            self._cache[key] = self._evaluator.eval_columns(targets)
        return self._cache[key]

    def potential(self, targets: torch.Tensor) -> torch.Tensor:
        cols = self._columns_for(targets)
        return cols[:, self._k]

    def description(self) -> str:
        slot_id = self.params.get("slot_id", "unknown")
        return f"GFDSL column {self._k} (slot_id={slot_id})"

    def serialize(self) -> Dict[str, Any]:
        base = {
            "type": self.type,
            "params": {k: (v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v) for k, v in self.params.items()},
        }
        if self.group_info:
            base["group_info"] = dict(self.group_info)
        return base


def linear_contribution_to_legacy_basis(contrib: LinearContribution) -> List[LegacyBasisElement]:
    """Convert a LinearContribution into legacy-compatible basis elements.

    Each column is exposed as a unit-coefficient basis element whose potential
    corresponds to the column evaluated at the provided targets. eval_columns
    calls are cached per input tensor to avoid recomputing columns when the
    legacy pipeline iterates over individual basis elements.
    """

    column_cache: Dict[
        Tuple[int, int, torch.device, torch.dtype, Tuple[int, ...]],
        torch.Tensor,
    ] = {}

    elems: List[LegacyBasisElement] = []
    for k, slot in enumerate(contrib.slots):
        group_dict: Dict[str, Any] = {}
        if isinstance(slot.group_info, GroupInfo):
            group_dict = slot.group_info.to_json_dict()
        elif slot.group_info is not None:
            group_dict = dict(slot.group_info)
        params = {"slot_id": slot.slot_id, **(slot.meta or {})}
        elems.append(
            LegacyBasisElement(
                type="gfdsl_column",
                params=params,
                group_info=group_dict,
                _k=k,
                _evaluator=contrib.evaluator,
                _cache=column_cache,
            )
        )
    return elems
