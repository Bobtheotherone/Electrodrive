from __future__ import annotations

"""
ResearchED SpecInspector + stable spec_digest for manifests.

Design doc anchors (normative):
- FR-7 (Program/spec dashboards): provide a SpecInspector view + stable spec_digest suitable for comparisons.
- FR-8 (Cross-run comparison and trend analysis): spec_digest is used for run comparisons.
- §5.1 (Unified run manifest schema v1): manifests contain spec_digest and it must be a dict.

Repo integration points (normative):
- Specs are loaded/parsed via electrodrive/orchestration/parser.py:
  - CanonicalSpec dataclass fields: domain, conductors, dielectrics, charges, BCs, symmetry, queries, symbols
    (see electrodrive/orchestration/parser.py:L30-L46).
  - CanonicalSpec.from_json normalizes numeric fields and defaults BCs to "Dirichlet"
    (see electrodrive/orchestration/parser.py:L48-L98).
  - parse_spec supports JSON and optional YAML (PyYAML guarded) and returns CanonicalSpec
    (see electrodrive/orchestration/parser.py:L137-L183).
- ResearchED manifests must keep spec_digest type-stable as a dict:
  - contracts/manifest_schema.new_manifest sets a placeholder spec_digest={} (see electrodrive/researched/contracts/manifest_schema.py:L295-L310).
  - RunManager writes spec_digest: {} and notes it is added by SpecInspector (see electrodrive/researched/run_manager.py:L649-L651).
  - API shape enforcement also coerces spec_digest to {} if missing/wrong type (see electrodrive/researched/api.py:L130-L133).

This module is stdlib-only (plus imports from existing Electrodrive modules) and designed to be robust:
- It never raises for missing/partial specs; it returns best-effort digests with warnings.
"""

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, List, Union


# contracts.manifest_schema.json_sanitize should exist in a full ResearchED install.
# Keep a fail-safe to avoid hard import failures (SpecInspector must be robust).
try:
    from electrodrive.researched.contracts.manifest_schema import json_sanitize
except Exception:  # pragma: no cover
    def json_sanitize(obj: Any) -> Any:  # type: ignore
        return obj


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False)


def _as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, Mapping):
        return dict(obj)
    return {}


def _as_list(obj: Any) -> List[Any]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    return []


def _coerce_float(v: Any) -> Optional[float]:
    try:
        if v is None or isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            # tolerate repo NaN/Inf encodings
            if s in {"NaN", "Infinity", "-Infinity"}:
                return None
            return float(s)
    except Exception:
        return None
    return None


def _minmax(vals: List[float]) -> Optional[Dict[str, float]]:
    if not vals:
        return None
    try:
        return {"min": float(min(vals)), "max": float(max(vals))}
    except Exception:
        return None


def _find_repo_root(start: Path) -> Optional[Path]:
    try:
        start = start.resolve()
    except Exception:
        pass
    for p in [start, *start.parents]:
        try:
            if (p / ".git").exists():
                return p
        except Exception:
            continue
    return None


def resolve_spec_path(spec_path: str, *, repo_root_hint: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve a spec path string into a Path.

    - Expands ~
    - If relative, tries:
        1) relative to repo_root_hint (if provided)
        2) relative to repo root inferred from repo_root_hint or this file
        3) as-is relative to CWD

    This function does not raise if the file does not exist; callers can check .exists().
    """
    p = Path(str(spec_path)).expanduser()
    if p.is_absolute():
        return p

    hint = Path(repo_root_hint).expanduser() if repo_root_hint is not None else None
    if hint is not None:
        try:
            if hint.is_file():
                hint = hint.parent
        except Exception:
            pass
        try:
            cand = (hint / p)
            if cand.exists():
                return cand
        except Exception:
            pass

    repo_root = None
    if hint is not None:
        repo_root = _find_repo_root(hint)
    if repo_root is None:
        repo_root = _find_repo_root(Path(__file__).resolve())

    if repo_root is not None:
        try:
            cand2 = repo_root / p
            if cand2.exists():
                return cand2
        except Exception:
            pass

    return p


def _load_canonical_from_path(path: Path) -> Tuple[Optional[Any], Optional[Dict[str, Any]], List[str]]:
    """
    Try to parse a spec from disk via parse_spec (preferred), else fall back to raw JSON.

    Parser supports JSON + optional YAML. If YAML requested but PyYAML missing,
    parser raises RuntimeError (see electrodrive/orchestration/parser.py:L151-L156).
    """
    warnings: List[str] = []
    try:
        from electrodrive.orchestration.parser import parse_spec  # stdlib + optional PyYAML inside parser
    except Exception as exc:
        warnings.append(f"cannot import orchestration parser: {exc!r}")
        return None, None, warnings

    try:
        spec = parse_spec(path)  # may raise on missing file / YAML without PyYAML
        spec_json = getattr(spec, "to_json", None)
        if callable(spec_json):
            return spec, spec.to_json(), warnings
        return spec, None, warnings
    except FileNotFoundError:
        warnings.append(f"spec path not found: {path}")
    except Exception as exc:
        warnings.append(f"parse_spec failed: {exc!r}")

    # Best-effort raw JSON fallback (only if it looks like JSON).
    try:
        if path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                warnings.append("used raw JSON fallback (parse_spec failed)")
                return None, raw, warnings
    except Exception:
        pass

    return None, None, warnings


def _coerce_to_canonical(spec_like: Any) -> Tuple[Optional[Any], Optional[Dict[str, Any]], List[str]]:
    """
    Convert a spec_like object to CanonicalSpec when possible; otherwise provide a dict-like raw.

    Uses CanonicalSpec.from_json for dicts (see electrodrive/orchestration/parser.py:L48-L98).
    """
    warnings: List[str] = []

    try:
        from electrodrive.orchestration.parser import CanonicalSpec  # type: ignore
    except Exception as exc:
        warnings.append(f"cannot import CanonicalSpec: {exc!r}")
        # Without CanonicalSpec, try to coerce to dict only.
        d = None
        if isinstance(spec_like, Mapping):
            d = dict(spec_like)
        elif hasattr(spec_like, "to_json") and callable(getattr(spec_like, "to_json")):
            try:
                tj = spec_like.to_json()
                if isinstance(tj, dict):
                    d = tj
            except Exception:
                d = None
        return None, d, warnings

    # Already a CanonicalSpec-ish object.
    if hasattr(spec_like, "to_json") and callable(getattr(spec_like, "to_json")) and hasattr(spec_like, "conductors"):
        try:
            sj = spec_like.to_json()
            if isinstance(sj, dict):
                return spec_like, sj, warnings
        except Exception:
            pass

    if isinstance(spec_like, Mapping):
        try:
            raw = dict(spec_like)
            spec = CanonicalSpec.from_json(raw)  # type: ignore[attr-defined]
            return spec, spec.to_json(), warnings  # type: ignore[union-attr]
        except Exception as exc:
            warnings.append(f"CanonicalSpec.from_json failed: {exc!r}")
            return None, dict(spec_like), warnings

    # Unknown object: try to_json, else vars().
    if hasattr(spec_like, "to_json") and callable(getattr(spec_like, "to_json")):
        try:
            raw = spec_like.to_json()
            if isinstance(raw, dict):
                try:
                    spec = CanonicalSpec.from_json(raw)  # type: ignore[attr-defined]
                    return spec, spec.to_json(), warnings  # type: ignore[union-attr]
                except Exception:
                    return None, raw, warnings
        except Exception:
            pass

    try:
        raw2 = dict(vars(spec_like))
        return None, raw2, warnings
    except Exception:
        return None, None, warnings


def _best_effort_spec_to_dict(spec_obj: Any, warnings: List[str]) -> Dict[str, Any]:
    """
    Convert a CanonicalSpec-like object into a dict even if .to_json() is missing.

    This prevents spec_hash collapsing to sha1("{}") across many runs, which would break
    FR-8 comparisons and trend analysis.
    """
    # 1) Preferred: to_json()
    try:
        tj = getattr(spec_obj, "to_json", None)
        if callable(tj):
            out = tj()
            if isinstance(out, dict):
                return out
    except Exception as exc:
        warnings.append(f"spec.to_json failed: {exc!r}")

    # 2) Dataclass: dataclasses.asdict
    try:
        if dataclasses.is_dataclass(spec_obj):
            out2 = dataclasses.asdict(spec_obj)
            if isinstance(out2, dict):
                return out2
    except Exception as exc:
        warnings.append(f"dataclasses.asdict(spec) failed: {exc!r}")

    # 3) Generic: vars()
    try:
        out3 = dict(vars(spec_obj))
        if isinstance(out3, dict):
            return out3
    except Exception as exc:
        warnings.append(f"vars(spec) failed: {exc!r}")

    return {}


def _domain_digest(domain: Any, *, warnings: List[str]) -> Dict[str, Any]:
    kind: str
    bbox = None

    if isinstance(domain, str):
        kind = domain
    elif isinstance(domain, Mapping):
        d = dict(domain)
        kind = str(d.get("kind") or d.get("type") or ("bbox" if "bbox" in d else "dict"))
        bb = d.get("bbox")
        if isinstance(bb, (list, tuple)) and len(bb) == 2:
            a, b = bb[0], bb[1]
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) >= 3 and len(b) >= 3:
                try:
                    bbox = [
                        [float(a[0]), float(a[1]), float(a[2])],
                        [float(b[0]), float(b[1]), float(b[2])],
                    ]
                except Exception:
                    warnings.append("domain.bbox present but could not coerce to floats")
                    bbox = None
    else:
        kind = type(domain).__name__
        warnings.append(f"domain type is unusual: {kind}")

    # extent: derived from bbox when present; z-range refined later.
    extent: Dict[str, Any] = {}
    if bbox:
        extent["x_min"], extent["y_min"], extent["z_min"] = bbox[0]
        extent["x_max"], extent["y_max"], extent["z_max"] = bbox[1]

    return {"kind": kind, "bbox": bbox, "extent": extent or None}


def _counts_digest(spec: Any, raw: Optional[Dict[str, Any]]) -> Dict[str, int]:
    if spec is not None:
        try:
            return {
                "conductors": int(len(getattr(spec, "conductors", []) or [])),
                "dielectrics": int(len(getattr(spec, "dielectrics", []) or [])),
                "charges": int(len(getattr(spec, "charges", []) or [])),
                "queries": int(len(getattr(spec, "queries", []) or [])),
            }
        except Exception:
            pass
    d = raw or {}
    return {
        "conductors": int(len(_as_list(d.get("conductors")))),
        "dielectrics": int(len(_as_list(d.get("dielectrics")))),
        "charges": int(len(_as_list(d.get("charges")))),
        "queries": int(len(_as_list(d.get("queries")))),
    }


def _types_count(objs: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for o in objs:
        try:
            t = o.get("type")
            k = str(t) if t is not None and str(t).strip() else "unknown"
        except Exception:
            k = "unknown"
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _extract_conductor_potentials(conductors: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    vals: List[float] = []
    for c in conductors:
        v = _coerce_float(c.get("potential"))
        if v is not None:
            vals.append(v)
    mm = _minmax(vals)
    return mm if mm is not None else None


def _dielectric_ranges(dielectrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    z_vals: List[float] = []
    eps_vals: List[float] = []

    z_min = None
    z_max = None
    eps_min = None
    eps_max = None

    z_keys_lo = ("z_min", "z0", "z_lower", "z_bottom", "z_start", "zmin")
    z_keys_hi = ("z_max", "z1", "z_upper", "z_top", "z_end", "zmax")
    eps_keys = ("eps", "eps_r", "epsilon", "eps_rel", "permittivity", "eps_ratio")

    for d in dielectrics:
        lo = None
        hi = None
        for k in z_keys_lo:
            if k in d:
                lo = _coerce_float(d.get(k))
                if lo is not None:
                    break
        for k in z_keys_hi:
            if k in d:
                hi = _coerce_float(d.get(k))
                if hi is not None:
                    break
        if lo is not None:
            z_vals.append(lo)
        if hi is not None:
            z_vals.append(hi)

        for k in eps_keys:
            if k in d:
                ev = _coerce_float(d.get(k))
                if ev is not None:
                    eps_vals.append(ev)

    if z_vals:
        z_min = float(min(z_vals))
        z_max = float(max(z_vals))
    if eps_vals:
        eps_min = float(min(eps_vals))
        eps_max = float(max(eps_vals))

    return {
        "count": int(len(dielectrics)),
        "z_min": z_min,
        "z_max": z_max,
        "eps_min": eps_min,
        "eps_max": eps_max,
    }


def _charge_ranges(charges: List[Dict[str, Any]]) -> Dict[str, Any]:
    z_vals: List[float] = []
    point_count = 0
    for ch in charges:
        if str(ch.get("type") or "").strip() == "point":
            point_count += 1
        pos = ch.get("pos")
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            z = _coerce_float(pos[2])
            if z is not None:
                z_vals.append(z)

    z_min = float(min(z_vals)) if z_vals else None
    z_max = float(max(z_vals)) if z_vals else None

    return {
        "types": _types_count(charges),
        "z_min": z_min,
        "z_max": z_max,
        "point_charge_count": int(point_count),
    }


def _update_domain_extent_with_z(domain: Dict[str, Any], z_min: Optional[float], z_max: Optional[float]) -> None:
    try:
        ext = domain.get("extent") or {}
        if not isinstance(ext, dict):
            ext = {}
        if z_min is not None:
            ext["z_min"] = float(z_min)
        if z_max is not None:
            ext["z_max"] = float(z_max)
        domain["extent"] = ext or None
    except Exception:
        return


@dataclass
class SpecInspector:
    """
    SpecInspector: loads and summarizes CanonicalSpec inputs for FR-7 dashboards and §5.1 spec_digest.

    - from_path() uses parse_spec (electrodrive/orchestration/parser.py:L170-L183).
    - from_spec_like() accepts dicts or CanonicalSpec-like objects.
    """
    _spec: Optional[Any]
    _spec_json: Optional[Dict[str, Any]]
    _spec_path: Optional[str]
    _warnings: List[str]

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "SpecInspector":
        p = Path(path).expanduser()
        spec, spec_json, warnings = _load_canonical_from_path(p)
        return cls(_spec=spec, _spec_json=spec_json, _spec_path=str(p), _warnings=warnings)

    @classmethod
    def from_spec_like(cls, spec_like: Any, *, spec_path: Optional[str] = None) -> "SpecInspector":
        # If spec_like is path-like, delegate to from_path.
        if isinstance(spec_like, (str, Path)):
            try:
                return cls.from_path(spec_like)
            except Exception as exc:
                return cls(_spec=None, _spec_json=None, _spec_path=str(spec_like), _warnings=[f"from_path failed: {exc!r}"])

        spec, spec_json, warnings = _coerce_to_canonical(spec_like)
        return cls(_spec=spec, _spec_json=spec_json, _spec_path=spec_path, _warnings=warnings)

    def digest(self) -> Dict[str, Any]:
        """
        Stable, JSON-serializable digest (schema v1) suitable for manifest["spec_digest"] (§5.1) and run comparisons (FR-8).
        """
        warnings = list(self._warnings)

        # Determine a usable raw spec dict for hashing/digest.
        raw = self._spec_json
        spec_obj = self._spec

        if raw is None and spec_obj is not None:
            raw = _best_effort_spec_to_dict(spec_obj, warnings)

        if raw is None and spec_obj is not None and hasattr(spec_obj, "to_json") and callable(getattr(spec_obj, "to_json")):
            try:
                raw2 = spec_obj.to_json()
                if isinstance(raw2, dict):
                    raw = raw2
            except Exception:
                raw = None

        if raw is None:
            raw = {}
            warnings.append("spec could not be parsed; digest is minimal")

        # Compute stable hash from canonical JSON dict (sorted keys).
        # Hash the sanitized dict to avoid NaN/Inf/path weirdness producing unstable hashes.
        try:
            raw_for_hash = json_sanitize(raw)
        except Exception:
            raw_for_hash = raw
        spec_hash = _sha1_text(_safe_json_dumps(raw_for_hash))

        # Pull fields from CanonicalSpec if available; otherwise from raw dict.
        if spec_obj is not None:
            try:
                domain_val = getattr(spec_obj, "domain", None)
                conductors = list(getattr(spec_obj, "conductors", []) or [])
                dielectrics = list(getattr(spec_obj, "dielectrics", []) or [])
                charges = list(getattr(spec_obj, "charges", []) or [])
                bc = getattr(spec_obj, "BCs", None)
                symmetry = list(getattr(spec_obj, "symmetry", []) or [])
                queries = list(getattr(spec_obj, "queries", []) or [])
            except Exception:
                domain_val = raw.get("domain")
                conductors = _as_list(raw.get("conductors"))
                dielectrics = _as_list(raw.get("dielectrics"))
                charges = _as_list(raw.get("charges"))
                bc = raw.get("BCs")
                symmetry = _as_list(raw.get("symmetry"))
                queries = _as_list(raw.get("queries"))
        else:
            domain_val = raw.get("domain")
            conductors = _as_list(raw.get("conductors"))
            dielectrics = _as_list(raw.get("dielectrics"))
            charges = _as_list(raw.get("charges"))
            bc = raw.get("BCs")
            symmetry = _as_list(raw.get("symmetry"))
            queries = _as_list(raw.get("queries"))

        # Ensure dict element lists.
        conductors_d = [c for c in conductors if isinstance(c, dict)]
        dielectrics_d = [d for d in dielectrics if isinstance(d, dict)]
        charges_d = [c for c in charges if isinstance(c, dict)]

        domain = _domain_digest(domain_val, warnings=warnings)

        counts = _counts_digest(spec_obj, raw)

        cond_types = _types_count(conductors_d)
        potentials = _extract_conductor_potentials(conductors_d)

        diel = _dielectric_ranges(dielectrics_d)
        ch = _charge_ranges(charges_d)

        # Extent estimation (FR-7 guidance: stable summary; prompt suggests Gate2-like z-extents).
        z_candidates: List[float] = []
        if isinstance(domain.get("extent"), dict):
            try:
                if "z_min" in domain["extent"]:
                    z_candidates.append(float(domain["extent"]["z_min"]))
                if "z_max" in domain["extent"]:
                    z_candidates.append(float(domain["extent"]["z_max"]))
            except Exception:
                pass
        if isinstance(diel.get("z_min"), (int, float)):
            z_candidates.append(float(diel["z_min"]))
        if isinstance(diel.get("z_max"), (int, float)):
            z_candidates.append(float(diel["z_max"]))
        if isinstance(ch.get("z_min"), (int, float)):
            z_candidates.append(float(ch["z_min"]))
        if isinstance(ch.get("z_max"), (int, float)):
            z_candidates.append(float(ch["z_max"]))

        # Also include conductor z/center.z when present (best-effort).
        for c in conductors_d:
            t = str(c.get("type") or "")
            if t == "plane" and "z" in c:
                z = _coerce_float(c.get("z"))
                if z is not None:
                    z_candidates.append(z)
            ctr = c.get("center")
            if isinstance(ctr, (list, tuple)) and len(ctr) >= 3:
                z = _coerce_float(ctr[2])
                if z is not None:
                    z_candidates.append(z)

        z_min = float(min(z_candidates)) if z_candidates else None
        z_max = float(max(z_candidates)) if z_candidates else None
        _update_domain_extent_with_z(domain, z_min, z_max)

        sym_list = sorted({str(s) for s in symmetry if str(s).strip()})
        bc_s = str(bc) if bc is not None else None

        digest: Dict[str, Any] = {
            "schema_version": 1,
            "spec_hash": spec_hash,
            "spec_path": self._spec_path,
            "bc": bc_s,
            "domain": domain,
            "counts": {
                "conductors": int(counts.get("conductors", 0)),
                "dielectrics": int(counts.get("dielectrics", 0)),
                "charges": int(counts.get("charges", 0)),
                "queries": int(counts.get("queries", len(queries))),
            },
            "conductors": {
                "types": cond_types,
                "potentials": potentials,
            },
            "dielectrics": diel,
            "charges": ch,
            "symmetry": sym_list,
            "warnings": warnings,
        }

        # Ensure JSON-safe floats/paths; never raise (SpecInspector must be robust).
        try:
            out = json_sanitize(digest)
            return out if isinstance(out, dict) else digest
        except Exception:
            return digest

    def summary(self) -> Dict[str, Any]:
        """
        Human-friendly summary (superset of digest). Intended for FR-7 spec dashboards.
        """
        d = self.digest()
        s: Dict[str, Any] = dict(d)
        # Add a short textual blurb (stable, small).
        try:
            s["summary_text"] = (
                f"domain={d.get('domain', {}).get('kind')} "
                f"conductors={d.get('counts', {}).get('conductors')} "
                f"charges={d.get('counts', {}).get('charges')} "
                f"dielectrics={d.get('counts', {}).get('dielectrics')} "
                f"bc={d.get('bc')}"
            )
        except Exception:
            s["summary_text"] = "spec summary unavailable"
        return s


def compute_spec_digest(spec_like: Any, *, spec_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience helper: compute digest from a path, dict-like spec, or CanonicalSpec-like object.
    """
    try:
        insp = SpecInspector.from_spec_like(spec_like, spec_path=spec_path)
        return insp.digest()
    except Exception as exc:
        return {
            "schema_version": 1,
            "spec_hash": "",
            "spec_path": spec_path,
            "bc": None,
            "domain": {"kind": "unknown", "bbox": None, "extent": None},
            "counts": {"conductors": 0, "dielectrics": 0, "charges": 0, "queries": 0},
            "conductors": {"types": {}, "potentials": None},
            "dielectrics": {"count": 0, "z_min": None, "z_max": None, "eps_min": None, "eps_max": None},
            "charges": {"types": {}, "z_min": None, "z_max": None, "point_charge_count": 0},
            "symmetry": [],
            "warnings": [f"compute_spec_digest failed: {exc!r}"],
        }


def apply_spec_digest_to_manifest(manifest: Dict[str, Any], spec_digest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply a spec_digest into a manifest dict, ensuring manifest["spec_digest"] remains a dict (§5.1).

    This is consistent with ResearchED's type-stability requirements:
    - API enforces spec_digest must be a dict (electrodrive/researched/api.py:L130-L133).
    """
    if not isinstance(manifest, dict):
        return manifest  # type: ignore[return-value]
    if not isinstance(manifest.get("spec_digest"), dict):
        manifest["spec_digest"] = {}
    try:
        manifest["spec_digest"].update(dict(spec_digest))
    except Exception:
        manifest["spec_digest"] = dict(spec_digest)
    return manifest


def maybe_digest_from_manifest(
    manifest: Mapping[str, Any],
    run_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Best-effort helper for FR-7/FR-8: obtain a spec_digest from a manifest and run directory.

    Tries:
    1) manifest["inputs"]["spec_path"] (design doc §5.1)
    2) legacy keys ("spec_path", "problem", "spec")
    3) inline spec dict (manifest["inputs"]["spec"] / ["problem"]) when present
    4) if run_dir provided, resolve relative paths against it + repo root

    Returns (digest, warnings).
    """
    warnings: List[str] = []

    # If spec_digest already present and non-empty, return it as-is (but ensure dict).
    sd = manifest.get("spec_digest") if isinstance(manifest, Mapping) else None
    if isinstance(sd, dict) and sd:
        return dict(sd), warnings

    spec_path = None
    inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), Mapping) else {}

    # NEW: inline spec support (dict embedded directly in manifest inputs)
    try:
        if isinstance(inputs, Mapping):
            v = inputs.get("spec") or inputs.get("problem")
            if isinstance(v, Mapping):
                insp = SpecInspector.from_spec_like(dict(v), spec_path=None)
                d = insp.digest()
                w = list(d.get("warnings") or []) if isinstance(d.get("warnings"), list) else []
                w.extend(warnings)
                d["warnings"] = w
                return d, w
    except Exception as exc:
        warnings.append(f"inline spec digest failed: {exc!r}")

    try:
        if isinstance(inputs, Mapping):
            sp = inputs.get("spec_path") or inputs.get("problem") or inputs.get("spec")
            if isinstance(sp, str) and sp.strip():
                spec_path = sp.strip()
    except Exception:
        pass

    if spec_path is None:
        for k in ("spec_path", "problem", "spec"):
            v = manifest.get(k)
            if isinstance(v, str) and v.strip():
                spec_path = v.strip()
                break

    if not spec_path:
        warnings.append("no spec_path found in manifest")
        return {}, warnings

    # Resolve path relative to run_dir, then repo root.
    base_hint = Path(run_dir) if run_dir is not None else None
    resolved = resolve_spec_path(spec_path, repo_root_hint=base_hint) if base_hint is not None else resolve_spec_path(spec_path)

    if not resolved.exists():
        warnings.append(f"spec_path does not exist on disk: {resolved}")
        return {}, warnings

    try:
        insp = SpecInspector.from_path(resolved)
        d = insp.digest()
        # Keep warnings from inspector + this function.
        w = list(d.get("warnings") or []) if isinstance(d.get("warnings"), list) else []
        w.extend(warnings)
        d["warnings"] = w
        return d, w
    except Exception as exc:
        warnings.append(f"spec digest failed: {exc!r}")
        return {}, warnings
