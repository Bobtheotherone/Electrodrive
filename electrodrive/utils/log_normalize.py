from __future__ import annotations

"""
Stdlib-only log normalization for legacy viz/log consumers.

Rules (Design Doc Phase A):
- event = event or msg or message; if message/msg is JSON containing "event", parse it and merge fields.
- iter = iter or iters or k or step
- resid_precond = resid_precond or resid_precond_l2
- resid_true = resid_true or resid_true_l2
- resid = resid or resid_precond or resid_true
- ts: parse ISO-8601 or numeric into epoch seconds (t); missing -> None
- Merge/dedup across events.jsonl + evidence_log.jsonl.
"""

import json
import math
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


def _safe_float(v: Any) -> Optional[float]:
    try:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            f = float(v)
            return f / 1000.0 if f > 1e12 else f
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            f = float(s)
            return f / 1000.0 if f > 1e12 else f
    except Exception:
        return None
    return None


def _parse_ts(ts: Any) -> Optional[float]:
    # datetime or numeric handled in _safe_float
    if isinstance(ts, datetime):
        dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        try:
            return float(dt.timestamp())
        except Exception:
            return None
    if isinstance(ts, (int, float, str)):
        f = _safe_float(ts)
        if f is not None:
            return f
    try:
        s = str(ts).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    except Exception:
        return None


def _try_parse_embedded_json(msg: Any) -> Tuple[Optional[str], Dict[str, Any]]:
    if not isinstance(msg, str):
        return None, {}
    s = msg.strip()
    if not s or "{" not in s or "}" not in s:
        return None, {}
    try:
        obj = json.loads(s)
    except Exception:
        return None, {}
    if not isinstance(obj, dict):
        return None, {}
    ev = obj.get("event")
    ev_str = str(ev).strip() if ev is not None else None
    return (ev_str or None), {str(k): v for k, v in obj.items()}


def _iter_variant(rec: Dict[str, Any]) -> Optional[int]:
    for k in ("iter", "iters", "k", "step"):
        if k in rec:
            v = _safe_float(rec.get(k))
            if v is not None:
                try:
                    return int(v)
                except Exception:
                    return None
    return None


def _residuals(rec: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    resid_precond = None
    for k in ("resid_precond", "resid_precond_l2"):
        if k in rec:
            rv = _safe_float(rec.get(k))
            if rv is not None:
                resid_precond = rv
                break

    resid_true = None
    for k in ("resid_true", "resid_true_l2"):
        if k in rec:
            rv = _safe_float(rec.get(k))
            if rv is not None:
                resid_true = rv
                break

    resid = None
    if "resid" in rec:
        rv = _safe_float(rec.get("resid"))
        if rv is not None:
            resid = rv
    if resid is None and resid_precond is not None:
        resid = resid_precond
    if resid is None and resid_true is not None:
        resid = resid_true

    return resid, resid_precond, resid_true


def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    raw: Dict[str, Any] = dict(rec or {})

    msg_val = raw.get("event") or raw.get("msg") or raw.get("message")
    embedded_event, embedded_fields = _try_parse_embedded_json(msg_val if isinstance(msg_val, str) else raw.get("message"))
    event_name = embedded_event or (str(msg_val).strip() if msg_val is not None else "") or "(unknown)"

    # Merge structured fields, dropping envelope keys.
    fields: Dict[str, Any] = {}
    drop = {
        "ts",
        "t",
        "level",
        "msg",
        "message",
        "event",
        "iter",
        "iters",
        "k",
        "step",
        "resid",
        "resid_precond",
        "resid_true",
        "resid_precond_l2",
        "resid_true_l2",
    }
    for k, v in raw.items():
        ks = str(k)
        if ks in drop:
            continue
        fields[ks] = v
    for k, v in embedded_fields.items():
        if k in drop or k == "event":
            continue
        fields.setdefault(k, v)

    t_val = _parse_ts(raw.get("ts"))
    level = None
    try:
        if raw.get("level") is not None:
            level = str(raw.get("level")).lower()
    except Exception:
        level = None

    it = _iter_variant({**raw, **embedded_fields})
    resid, resid_precond, resid_true = _residuals({**raw, **embedded_fields})

    return {
        "event": event_name,
        "iter": it,
        "resid": resid,
        "resid_precond": resid_precond,
        "resid_true": resid_true,
        "t": t_val,
        "level": level,
        "fields": fields,
        "raw": raw,
    }


def _sanitize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        if math.isnan(obj):
            return "NaN"
        return "Infinity" if obj > 0 else "-Infinity"
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_sanitize(x) for x in obj)
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def stable_hash_fields(fields: Dict[str, Any]) -> str:
    try:
        blob = json.dumps(_sanitize(fields), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        blob = str(fields)
    return sha1(blob.encode("utf-8", errors="ignore")).hexdigest()


def _dedup_key(ev: Dict[str, Any]) -> Tuple[Any, str, str]:
    t = ev.get("t")
    t_r = round(float(t), 3) if isinstance(t, (int, float)) else None
    level = str(ev.get("level") or "").lower()
    event = str(ev.get("event") or "").strip().lower()
    h = stable_hash_fields(ev.get("fields") or {})
    return (t_r, level, event + "|" + h)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists() or not path.is_file():
        return rows
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = (line or "").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except Exception:
        return rows
    return rows


def iter_merged_events(run_dir: Path) -> Iterator[Dict[str, Any]]:
    """
    Yield normalized, deduplicated events from events.jsonl and evidence_log.jsonl.
    """
    rd = Path(run_dir)
    sources: List[Tuple[Path, List[Dict[str, Any]]]] = []
    for name in ("events.jsonl", "evidence_log.jsonl"):
        p = rd / name
        rows = _load_jsonl(p)
        if rows:
            sources.append((p, rows))

    # If both empty, nothing to do.
    if not sources:
        return iter(())

    normalized: List[Dict[str, Any]] = []
    for _, rows in sources:
        for rec in rows:
            try:
                norm = normalize_record(rec)
                normalized.append(norm)
            except Exception:
                continue

    # Dedup
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for ev in normalized:
        k = _dedup_key(ev)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(ev)

    # Sort by time when available.
    def sort_key(e: Dict[str, Any]) -> Tuple[int, float, str]:
        has_t = 0 if e.get("t") is None else 1
        t = float(e.get("t") or 0.0)
        return (has_t, t, str(e.get("event") or ""))

    deduped.sort(key=sort_key)
    for ev in deduped:
        yield ev
