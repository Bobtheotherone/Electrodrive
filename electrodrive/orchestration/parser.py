from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Union

try:  # YAML is optional; we fail gracefully if missing.
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


SpecLike = Union[str, Path, Mapping[str, Any]]


def _norm(x: Union[str, float, int]) -> float:
    """Coerce basic numeric / string values to float with a safe fallback."""
    if isinstance(x, (float, int)):
        return float(x)
    try:
        return float(x)
    except Exception:
        # Minimal symbolic fallback — keep deterministic but warn via print.
        print(f"WARNING: symbolic value '{x}' – substituting 1.0 for numeric path.")
        return 1.0


@dataclass
class CanonicalSpec:
    """
    Minimal canonical spec used by the solver stack.

    We deliberately keep this narrow: enough for planner + BEM/analytic to
    understand the geometry without over-encoding problem-specific details.
    """

    domain: Any
    conductors: List[Dict[str, Any]]
    dielectrics: List[Dict[str, Any]]
    charges: List[Dict[str, Any]]
    BCs: str
    symmetry: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    symbols: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "CanonicalSpec":
        """
        Construct CanonicalSpec from a raw dict parsed from JSON/YAML.

        This function is intentionally permissive:
        - If 'BCs' is missing, default to 'Dirichlet'.
        - 'domain' can be a dict (bbox, etc.) or a simple string tag.
        - We coerce basic numeric fields to float for downstream kernels.
        """
        if "domain" not in d:
            raise ValueError("Spec missing required key 'domain'.")

        if "BCs" not in d:
            d["BCs"] = "Dirichlet"

        # Normalize charges
        for ch in d.get("charges", []):
            if "q" in ch:
                ch["q"] = _norm(ch["q"])
            if "pos" in ch:
                ch["pos"] = [_norm(p) for p in ch["pos"]]

        # Normalize conductors
        for c in d.get("conductors", []):
            if "potential" in c:
                c["potential"] = _norm(c["potential"])
            t = c.get("type")
            if t == "plane" and "z" in c:
                c["z"] = _norm(c["z"])
            if t in ("sphere", "cylinder", "torus", "toroid"):
                if "radius" in c:
                    c["radius"] = _norm(c["radius"])
                if "center" in c:
                    c["center"] = [_norm(x) for x in c["center"]]
            if t in ("torus", "toroid"):
                if "major_radius" in c:
                    c["major_radius"] = _norm(c["major_radius"])
                if "minor_radius" in c:
                    c["minor_radius"] = _norm(c["minor_radius"])

        return CanonicalSpec(
            domain=d["domain"],
            conductors=d.get("conductors", []),
            dielectrics=d.get("dielectrics", []),
            charges=d.get("charges", []),
            BCs=d["BCs"],
            symmetry=d.get("symmetry", []),
            queries=d.get("queries", []),
            symbols=d.get("symbols", {}),
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "conductors_count": len(self.conductors),
            "charges_count": len(self.charges),
            "queries": self.queries,
        }

    def get_charge_locations(self) -> List[Tuple[float, float, float]]:
        return [
            tuple(c["pos"])
            for c in self.charges
            if c.get("type") == "point" and c.get("pos") is not None
        ]

    def to_json(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable dict representation of the spec.

        Deep copies are used to avoid accidental mutation by callers.
        """
        return {
            "domain": copy.deepcopy(self.domain),
            "conductors": copy.deepcopy(self.conductors),
            "dielectrics": copy.deepcopy(self.dielectrics),
            "charges": copy.deepcopy(self.charges),
            "BCs": self.BCs,
            "symmetry": copy.deepcopy(self.symmetry),
            "queries": copy.deepcopy(self.queries),
            "symbols": copy.deepcopy(self.symbols),
        }


# ---------------------------------------------------------------------------
# Public loader API
# ---------------------------------------------------------------------------


def _load_raw_spec(spec: SpecLike) -> Dict[str, Any]:
    """
    Load a raw spec dict from:
    - path to .json / .yaml / .yml
    - already-parsed dict-like object
    """
    if isinstance(spec, (str, Path)):
        path = Path(spec)
        if not path.exists():
            raise FileNotFoundError(f"Spec path does not exist: {path}")
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix == ".json":
            return json.loads(text)
        if suffix in (".yaml", ".yml"):
            if yaml is None:
                raise RuntimeError(
                    "YAML spec requested but PyYAML is not installed. "
                    "Install 'pyyaml' or use JSON."
                )
            return yaml.safe_load(text)  # type: ignore[arg-type]
        # Fallback: try JSON then YAML
        try:
            return json.loads(text)
        except Exception:
            if yaml is not None:
                return yaml.safe_load(text)  # type: ignore[arg-type]
            raise RuntimeError(f"Unrecognized spec format for path: {path}")
    else:
        # Mapping-like object; make a shallow copy to avoid side-effects.
        return dict(spec)


def parse_spec(spec: SpecLike) -> CanonicalSpec:
    """
    High-level helper used by orchestration/BEM probe.

    This mirrors the historical `parse_spec` entry point expected at
    `electrodrive.orchestration.parser.parse_spec` and wraps the result
    into a CanonicalSpec while also normalizing numerics.
    """
    raw = _load_raw_spec(spec)

    # Lightweight compatibility: if the incoming spec is "minimal"
    # (charges + conductors + domain only), we still accept it as-is.
    # Example: specs/plane_point.json in this repo.
    return CanonicalSpec.from_json(raw)
