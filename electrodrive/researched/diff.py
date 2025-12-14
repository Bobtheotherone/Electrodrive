from __future__ import annotations

"""
ResearchED diff utilities (“What changed?”) for FR-8 comparisons.

Design doc anchors (normative):
- FR-8 (Cross-run comparison and trend analysis): produce “What changed?” views (argv diff + config/spec diffs).
- FR-7 + §5.1: manifests include spec_digest (dict) and it is used for comparisons.

Repo grounding:
- ResearchED manifest contract requires spec_digest to be a dict placeholder (contracts/manifest_schema.py:L295-L310),
  and the API coerces spec_digest to {} if missing/wrong type (electrodrive/researched/api.py:L130-L133).
- This module is stdlib-only; YAML parsing is optional and guarded (consistent with orchestration/parser.py YAML guard,
  electrodrive/orchestration/parser.py:L9-L12 and L151-L156).

Notes:
- semantic diff targets JSON-like objects (dict/list/scalars).
- text diff fallback uses difflib unified diff.
- list alignment heuristic: if list elements are dicts with stable keys ("id","name","type"), align by that key.
"""

import dataclasses
import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union


@dataclass(frozen=True)
class DiffItem:
    path: str
    op: str  # {"add","remove","change","type_change"}
    before: Any
    after: Any


@dataclass(frozen=True)
class DiffResult:
    kind: str  # {"semantic","text","none"}
    items: list[DiffItem]
    unified_diff: str
    summary: dict[str, Any]


_MISSING = object()


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _float_equal(a: Any, b: Any, tol: float) -> bool:
    # Handle repo NaN/Inf string encodings as literal equality.
    if isinstance(a, str) and isinstance(b, str) and a in {"NaN", "Infinity", "-Infinity"}:
        return a == b
    if not (_is_number(a) and _is_number(b)):
        return False
    try:
        fa = float(a)
        fb = float(b)
        if tol <= 0.0:
            return fa == fb
        return abs(fa - fb) <= tol
    except Exception:
        return False


def _path_join(base: str, key: str) -> str:
    if not base:
        return key
    return f"{base}/{key}"


def _path_index(base: str, idx: str) -> str:
    if not base:
        return f"[{idx}]"
    return f"{base}[{idx}]"


def _is_ignored(path: str, ignore_paths: Optional[Set[str]]) -> bool:
    if not ignore_paths:
        return False
    for pref in ignore_paths:
        if not pref:
            continue
        if path == pref:
            return True
        if path.startswith(pref + "/") or path.startswith(pref + "["):
            return True
    return False


def _stable_key_for_list(xs: Sequence[Any]) -> Optional[str]:
    """
    Heuristic: if list elements are dicts and most have a unique stable key in {"id","name","type"}, use it.
    """
    candidates = ("id", "name", "type")
    dicts = [x for x in xs if isinstance(x, dict)]
    if len(dicts) < 2:
        return None

    for k in candidates:
        vals: List[str] = []
        ok = 0
        for d in dicts:
            if k in d and d.get(k) is not None:
                s = str(d.get(k))
                if s.strip():
                    ok += 1
                    vals.append(s)
        if ok >= max(2, int(0.6 * len(dicts))):
            # Unique values improve alignment.
            if len(set(vals)) == len(vals):
                return k
    return None


def _summarize(items: List[DiffItem], *, truncated: bool) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    top_paths: List[str] = []
    for it in items:
        counts[it.op] = counts.get(it.op, 0) + 1
        if len(top_paths) < 25:
            top_paths.append(it.path)
    return {
        "counts": dict(sorted(counts.items(), key=lambda kv: kv[0])),
        "total": int(len(items)),
        "truncated": bool(truncated),
        "top_paths": top_paths,
    }


def diff_semantic(
    a: Any,
    b: Any,
    *,
    ignore_paths: Optional[Set[str]] = None,
    float_tol: float = 0.0,
) -> DiffResult:
    """
    Semantic diff of JSON-like objects.

    - Dicts: key-wise diff
    - Lists: index-based diff, with optional alignment by stable key ("id","name","type")
    - Scalars: equality / float tolerance

    Important: We treat Mapping types as equivalent (dict vs OrderedDict, etc),
    and list/tuple as equivalent, so we don't incorrectly emit early "type_change"
    for common container representations.
    """
    items: List[DiffItem] = []
    max_items = 1000
    truncated = False

    def add_item(path: str, op: str, before: Any, after: Any) -> None:
        nonlocal truncated
        if len(items) >= max_items:
            truncated = True
            return
        items.append(DiffItem(path=path, op=op, before=before, after=after))

    def walk(x: Any, y: Any, path: str) -> None:
        nonlocal truncated
        if truncated:
            return
        if _is_ignored(path, ignore_paths):
            return

        # Missing handling.
        if x is _MISSING and y is _MISSING:
            return
        if x is _MISSING:
            add_item(path, "add", None, y)
            return
        if y is _MISSING:
            add_item(path, "remove", x, None)
            return

        # Dicts: treat any Mapping types as equivalent.
        if isinstance(x, Mapping) and isinstance(y, Mapping):
            kx = set(x.keys())
            ky = set(y.keys())
            for k in sorted(kx | ky, key=lambda z: str(z)):
                ks = str(k)
                px = _path_join(path, ks)
                xv = x.get(k, _MISSING)
                yv = y.get(k, _MISSING)
                walk(xv, yv, px)
                if truncated:
                    return
            return

        # Lists: treat list/tuple as equivalent.
        if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            lx = list(x)
            ly = list(y)
            key = _stable_key_for_list(lx) or _stable_key_for_list(ly)
            if key:
                mx: Dict[str, Any] = {}
                my: Dict[str, Any] = {}
                for e in lx:
                    if isinstance(e, dict) and key in e and e.get(key) is not None:
                        mx[str(e.get(key))] = e
                for e in ly:
                    if isinstance(e, dict) and key in e and e.get(key) is not None:
                        my[str(e.get(key))] = e
                for k in sorted(set(mx.keys()) | set(my.keys())):
                    label = k if len(k) <= 48 else (k[:45] + "…")
                    p2 = _path_index(path, f"{key}={label}")
                    walk(mx.get(k, _MISSING), my.get(k, _MISSING), p2)
                    if truncated:
                        return
                return

            # Index-based diff.
            n = max(len(lx), len(ly))
            for i in range(n):
                px = _path_index(path, str(i))
                xv = lx[i] if i < len(lx) else _MISSING
                yv = ly[i] if i < len(ly) else _MISSING
                walk(xv, yv, px)
                if truncated:
                    return
            return

        # Type handling (after compatible container handling).
        tx = type(x)
        ty = type(y)
        if tx != ty and not (_is_number(x) and _is_number(y)):
            add_item(path, "type_change", x, y)
            return

        # Floats with tolerance.
        if _is_number(x) and _is_number(y):
            if not _float_equal(x, y, float_tol):
                add_item(path, "change", x, y)
            return

        # Other scalars.
        if x != y:
            add_item(path, "change", x, y)

    walk(a, b, "")
    kind = "none" if not items and not truncated else "semantic"
    return DiffResult(kind=kind, items=items, unified_diff="", summary=_summarize(items, truncated=truncated))


def diff_text(a_text: str, b_text: str, *, fromfile: str = "a", tofile: str = "b") -> DiffResult:
    ud = "\n".join(
        difflib.unified_diff(
            (a_text or "").splitlines(),
            (b_text or "").splitlines(),
            fromfile=str(fromfile),
            tofile=str(tofile),
            lineterm="",
        )
    )
    kind = "none" if not ud.strip() else "text"
    return DiffResult(kind=kind, items=[], unified_diff=ud, summary={"lines": int(len(ud.splitlines()))})


def _try_parse_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _try_parse_yaml(text: str) -> Optional[Any]:
    # Optional dependency (PyYAML). We do not require it.
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    try:
        return yaml.safe_load(text)  # type: ignore[arg-type]
    except Exception:
        return None


def diff_files(a_path: Union[str, Path], b_path: Union[str, Path]) -> DiffResult:
    """
    Diff two files:
    - Try structured parse (JSON first; YAML if available),
    - Else unified text diff.
    """
    ap = Path(a_path)
    bp = Path(b_path)

    try:
        a_text = ap.read_text(encoding="utf-8", errors="ignore") if ap.is_file() else ""
    except Exception:
        a_text = ""
    try:
        b_text = bp.read_text(encoding="utf-8", errors="ignore") if bp.is_file() else ""
    except Exception:
        b_text = ""

    a_obj = _try_parse_json(a_text)
    b_obj = _try_parse_json(b_text)

    if a_obj is None or b_obj is None:
        ay = _try_parse_yaml(a_text)
        by = _try_parse_yaml(b_text)
        if ay is not None and by is not None:
            return diff_semantic(ay, by, float_tol=0.0)

    if a_obj is not None and b_obj is not None:
        return diff_semantic(a_obj, b_obj, float_tol=0.0)

    return diff_text(a_text, b_text, fromfile=str(ap), tofile=str(bp))


def diff_spec_digests(a: Mapping[str, Any], b: Mapping[str, Any]) -> DiffResult:
    """
    Semantic diff between two spec_digest dicts (no I/O).
    """
    return diff_semantic(dict(a or {}), dict(b or {}), ignore_paths=None, float_tol=0.0)


def diff_manifests(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """
    Produce a compact “What changed?” package for FR-8.

    Includes:
    - argv diff on inputs.command
    - spec pointer diff on inputs.spec_path (or legacy keys)
    - config pointer diff on inputs.config / inputs.config_path
    - semantic diff of manifests with noisy fields ignored (timestamps/pids/lifecycle)

    Note: file diffs are handled in compare_service where run_dir context exists.
    """
    a_in = a.get("inputs") if isinstance(a.get("inputs"), Mapping) else {}
    b_in = b.get("inputs") if isinstance(b.get("inputs"), Mapping) else {}

    a_cmd = a_in.get("command") if isinstance(a_in.get("command"), list) else []
    b_cmd = b_in.get("command") if isinstance(b_in.get("command"), list) else []
    argv_diff = diff_semantic(a_cmd, b_cmd, float_tol=0.0)

    def _pick(m: Mapping[str, Any], keys: Sequence[str]) -> Any:
        for k in keys:
            v = m.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if v is not None:
                return v
        return None

    a_spec = _pick(a_in, ("spec_path", "problem", "spec"))
    b_spec = _pick(b_in, ("spec_path", "problem", "spec"))

    a_cfg = _pick(a_in, ("config_path", "config"))
    b_cfg = _pick(b_in, ("config_path", "config"))

    # Ignore noisy keys and blocks.
    ignore = {
        "run_id",  # always unique; not meaningful in “what changed?”
        "started_at",
        "ended_at",
        "created_at",
        "created_at_epoch",
        "started_at_epoch",
        "ended_at_epoch",
        "pid",
        "returncode",
        # lifecycle block is useful in UI but noisy for “what changed”
        "researched",
    }

    manifest_sem = diff_semantic(dict(a or {}), dict(b or {}), ignore_paths=ignore, float_tol=0.0)

    return {
        "argv_diff": dataclasses.asdict(argv_diff),
        "spec_path": {"before": a_spec, "after": b_spec},
        "config_path": {"before": a_cfg, "after": b_cfg},
        "manifest_diff": dataclasses.asdict(manifest_sem),
    }
