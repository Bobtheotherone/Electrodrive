from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Dict, List, Optional

from ..oracle_types import OracleQuery, OracleResult
from ..utils import require_cuda


def _assert_cuda_inputs(query: OracleQuery, result: Optional[OracleResult] = None) -> None:
    require_cuda(query.points, "points")
    if result is None:
        return
    require_cuda(result.valid_mask, "valid_mask")
    if result.V is not None:
        require_cuda(result.V, "V")
    if result.E is not None:
        require_cuda(result.E, "E")


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: str
    metrics: Dict[str, float] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    evidence: Dict[str, str] = field(default_factory=dict)
    oracle: Dict[str, object] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    config: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> Dict[str, object]:
        def _sanitize(val: object) -> object:
            if callable(val):
                return repr(val)
            try:
                json.dumps(val)  # type: ignore[arg-type]
                return val
            except Exception:
                return repr(val)

        return {
            "gate": self.gate,
            "status": self.status,
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "thresholds": {k: float(v) for k, v in self.thresholds.items()},
            "evidence": dict(self.evidence),
            "oracle": dict(self.oracle),
            "notes": [str(n) for n in self.notes],
            "config": {str(k): _sanitize(v) for k, v in self.config.items()},
        }

    @staticmethod
    def from_json(d: Dict[str, object]) -> "GateResult":
        return GateResult(
            gate=str(d.get("gate", "")),
            status=str(d.get("status", "")),
            metrics={str(k): float(v) for k, v in dict(d.get("metrics", {})).items()},
            thresholds={str(k): float(v) for k, v in dict(d.get("thresholds", {})).items()},
            evidence={str(k): str(v) for k, v in dict(d.get("evidence", {})).items()},
            oracle=dict(d.get("oracle", {})),
            notes=[str(n) for n in d.get("notes", [])],
            config=dict(d.get("config", {})),
        )
