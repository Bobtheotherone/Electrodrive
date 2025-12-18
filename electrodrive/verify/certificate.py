from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .utils import canonical_json_bytes, sha256_json


def canonical_hash(obj: Any) -> str:
    return sha256_json(obj)


@dataclass(frozen=True)
class DiscoveryCertificate:
    spec_digest: str
    candidate_digest: str
    git_sha: str
    hardware: Dict[str, object] = field(default_factory=dict)
    oracle_runs: List[Dict[str, object]] = field(default_factory=list)
    gates: Dict[str, object] = field(default_factory=dict)
    final_status: str = "unknown"
    reasons: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, object]:
        return {
            "spec_digest": self.spec_digest,
            "candidate_digest": self.candidate_digest,
            "git_sha": self.git_sha,
            "hardware": dict(self.hardware),
            "oracle_runs": [dict(x) for x in self.oracle_runs],
            "gates": dict(self.gates),
            "final_status": self.final_status,
            "reasons": [str(r) for r in self.reasons],
            "attachments": [str(a) for a in self.attachments],
        }

    @staticmethod
    def from_json(d: Dict[str, object]) -> "DiscoveryCertificate":
        return DiscoveryCertificate(
            spec_digest=str(d.get("spec_digest", "")),
            candidate_digest=str(d.get("candidate_digest", "")),
            git_sha=str(d.get("git_sha", "")),
            hardware=dict(d.get("hardware", {})),
            oracle_runs=[dict(x) for x in d.get("oracle_runs", [])],
            gates=dict(d.get("gates", {})),
            final_status=str(d.get("final_status", "unknown")),
            reasons=[str(r) for r in d.get("reasons", [])],
            attachments=[str(a) for a in d.get("attachments", [])],
        )

    def digest(self) -> str:
        return canonical_hash(self.to_json())

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_json())
