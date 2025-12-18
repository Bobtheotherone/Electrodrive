from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class PoleTerm:
    pole: complex
    residue: complex
    kind: str = "guided"
    meta: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> Dict[str, object]:
        return {
            "pole": [float(complex(self.pole).real), float(complex(self.pole).imag)],
            "residue": [float(complex(self.residue).real), float(complex(self.residue).imag)],
            "kind": str(self.kind),
            "meta": dict(self.meta),
        }

    @staticmethod
    def from_json(d: Dict[str, object]) -> "PoleTerm":
        def _complexify(val) -> complex:
            if isinstance(val, complex):
                return val
            if isinstance(val, (list, tuple)) and len(val) == 2:
                try:
                    return complex(float(val[0]), float(val[1]))
                except Exception:
                    pass
            return complex(val)

        return PoleTerm(
            pole=_complexify(d.get("pole", 0.0)),
            residue=_complexify(d.get("residue", 0.0)),
            kind=str(d.get("kind", "guided")),
            meta=dict(d.get("meta", {})),
        )
