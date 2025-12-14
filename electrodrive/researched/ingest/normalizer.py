from __future__ import annotations

"""
Record normalizer for ResearchED log ingestion.

Design Doc anchors:
* FR-4: Robust log ingestion and normalization
* 1.3 A critical known mismatch: logs vs viz parsers (must be fixed)
* FR-9.6 Visualization + log consumer audit panel (coverage tracks which fields were used)

This module is stdlib-only and defensive: normalize_record never raises.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .learn_message_json import extract_event_from_message_json

# Keys that should never be merged into "fields" from message-JSON payloads.
_ENVELOPE_KEYS = {"ts", "t", "level", "msg", "message", "event"}


@dataclass(frozen=True)
class NormalizedEvent:
    ts: Optional[str]
    t: float
    level: Optional[str]
    event: str
    iter: Optional[int]
    resid: Optional[float]
    resid_precond: Optional[float]
    resid_true: Optional[float]
    fields: Dict[str, Any]
    raw: Dict[str, Any]
    source: Optional[str]
    event_name_source: str  # "event"|"msg"|"message"|"message_json"|"unknown"
    residual_sources: Dict[str, str]  # canonical->source_key (e.g., {"resid":"resid_true_l2"})


def parse_ts_to_epoch_seconds(ts: Any) -> Optional[float]:
    """
    Parse ISO-8601 or numeric timestamps into epoch seconds.

    Design Doc FR-4 timestamp rules:
    - accept ISO strings and numeric timestamps
    - treat large numeric values as milliseconds

    Never raises.
    """
    try:
        if ts is None:
            return None

        if isinstance(ts, datetime):
            dt = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
            return float(dt.timestamp())

        # Numeric timestamp (seconds or milliseconds).
        if isinstance(ts, (int, float)) and not isinstance(ts, bool):
            x = float(ts)
            return x / 1000.0 if x > 1e12 else x

        if isinstance(ts, str):
            s = ts.strip()
            if not s:
                return None

            # Numeric string?
            try:
                x = float(s)
                return x / 1000.0 if x > 1e12 else x
            except Exception:
                pass

            # ISO-ish string.
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"

            dt = datetime.fromisoformat(s)
            dt = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
            return float(dt.timestamp())
    except Exception:
        return None
    return None


def _safe_float(v: Any) -> Optional[float]:
    try:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            return float(s)
    except Exception:
        return None
    return None


def _safe_int(v: Any) -> Optional[int]:
    f = _safe_float(v)
    if f is None:
        return None
    try:
        return int(f)
    except Exception:
        return None


def _normalize_level(level: Any) -> Optional[str]:
    try:
        if level is None:
            return None
        s = str(level).strip()
        if not s:
            return None
        return s.lower()
    except Exception:
        return None


_DROP_KEYS = {
    # common envelope
    "ts",
    "t",
    "level",
    "msg",
    "message",
    "event",
    # iter aliases
    "iter",
    "iters",
    "step",
    "k",
    # residual variants
    "resid",
    "resid_precond",
    "resid_true",
    "resid_precond_l2",
    "resid_true_l2",
}


def extract_fields(rec: Mapping[str, Any], *, drop_keys: Sequence[str] = ()) -> Dict[str, Any]:
    """
    Extract a "fields" dict by removing known envelope keys.

    Design Doc FR-4: canonical event record retains remaining structured kvs as fields.
    Never raises.
    """
    try:
        drop = set(_DROP_KEYS)
        drop.update(drop_keys)
        out: Dict[str, Any] = {}
        for k, v in rec.items():
            ks = str(k)
            if ks in drop:
                continue
            out[ks] = v
        return out
    except Exception:
        return {}


def _extract_iter(rec: Mapping[str, Any]) -> Optional[int]:
    # Design Doc FR-4 iteration key: iter/iters/step/k
    for k in ("iter", "iters", "k", "step"):
        if k in rec:
            it = _safe_int(rec.get(k))
            if it is not None:
                return it
    return None


def _extract_residuals(
    rec: Mapping[str, Any]
) -> Tuple[Optional[float], Optional[float], Optional[float], Dict[str, str]]:
    """
    Design Doc FR-4 residual normalization:
        resid_precond = resid_precond or resid_precond_l2
        resid_true    = resid_true or resid_true_l2
        resid         = resid or resid_precond or resid_true
    """
    resid_sources: Dict[str, str] = {}

    resid_precond: Optional[float] = None
    for k in ("resid_precond", "resid_precond_l2"):
        if k in rec:
            v = _safe_float(rec.get(k))
            if v is not None:
                resid_precond = v
                resid_sources["resid_precond"] = k
                break

    resid_true: Optional[float] = None
    for k in ("resid_true", "resid_true_l2"):
        if k in rec:
            v = _safe_float(rec.get(k))
            if v is not None:
                resid_true = v
                resid_sources["resid_true"] = k
                break

    resid: Optional[float] = None
    if "resid" in rec:
        v = _safe_float(rec.get("resid"))
        if v is not None:
            resid = v
            resid_sources["resid"] = "resid"

    if resid is None and resid_precond is not None:
        resid = resid_precond
        resid_sources["resid"] = resid_sources.get("resid_precond", "resid_precond")

    if resid is None and resid_true is not None:
        resid = resid_true
        resid_sources["resid"] = resid_sources.get("resid_true", "resid_true")

    return resid, resid_precond, resid_true, resid_sources


def normalize_record(
    rec: Mapping[str, Any],
    *,
    source: Optional[str] = None,
    default_time: Optional[float] = None,
) -> NormalizedEvent:
    """
    Normalize a raw record into ResearchED's canonical internal event.

    Design Doc FR-4 normalization rules implemented:
    A) event name:
       event = rec.get("event") or rec.get("msg") or rec.get("message")
       If the chosen message is a JSON string containing "event", parse it and treat that as the event.
    B) iter: iter/iters/step/k  (extracted from both top-level and message-json payload)
    C) residuals: resid + precond/true variants (extracted from both top-level and message-json payload)
    D) timestamps: parse ts as ISO or numeric; fallback to default_time or now

    Robustness:
    - Never raises.
    """
    try:
        raw_orig: Dict[str, Any] = dict(rec)
        raw: Dict[str, Any] = {str(k): v for k, v in raw_orig.items()}

        # Timestamp.
        ts_raw = raw.get("ts")
        ts_str = ts_raw if isinstance(ts_raw, str) else (str(ts_raw) if ts_raw is not None else None)
        t = parse_ts_to_epoch_seconds(ts_raw)
        if t is None:
            t = float(default_time) if default_time is not None else float(datetime.now(timezone.utc).timestamp())

        # Level.
        level = _normalize_level(raw.get("level"))

        def _nonempty(x: Any) -> bool:
            try:
                return x is not None and str(x).strip() != ""
            except Exception:
                return False

        # Event name with schema mismatch handling.
        event_name_source = "unknown"
        msg_val: Any = None

        if _nonempty(raw.get("event")):
            msg_val = raw.get("event")
            event_name_source = "event"
        elif _nonempty(raw.get("msg")):
            msg_val = raw.get("msg")
            event_name_source = "msg"
        elif _nonempty(raw.get("message")):
            msg_val = raw.get("message")
            event_name_source = "message"

        # Learn/train message JSON parsing (only when the selected message is a string).
        message_json_fields: Dict[str, Any] = {}
        if isinstance(msg_val, str):
            ev_from_json, fields_from_json = extract_event_from_message_json(msg_val)
            if fields_from_json:
                message_json_fields = {str(k): v for k, v in fields_from_json.items()}
            if ev_from_json is not None:
                msg_val = ev_from_json
                event_name_source = "message_json"

        # Final event name.
        try:
            event_str = str(msg_val).strip() if msg_val is not None else ""
        except Exception:
            event_str = ""
        if not event_str:
            event_str = "(unknown)"
            event_name_source = "unknown"

        # Combine message-json + raw so we can extract iter/resid from either place.
        combo: Dict[str, Any] = {}
        if message_json_fields:
            combo.update(message_json_fields)
        combo.update(raw)  # raw overrides message-json

        # Iter/residuals.
        it = _extract_iter(combo)
        resid, resid_precond, resid_true, resid_sources = _extract_residuals(combo)

        # Remaining fields (structured kvs).
        fields = extract_fields(raw)

        # Merge in parsed message-json structured kvs (excluding envelope keys + "event").
        # IMPORTANT: do NOT drop step/iter/resid keys from message-json payloads (learn/train encodes them there).
        if message_json_fields:
            for k, v in message_json_fields.items():
                if k == "event":
                    continue
                if k in _ENVELOPE_KEYS:
                    continue
                fields.setdefault(str(k), v)

        return NormalizedEvent(
            ts=ts_str,
            t=float(t),
            level=level,
            event=event_str,
            iter=it,
            resid=resid,
            resid_precond=resid_precond,
            resid_true=resid_true,
            fields=fields,
            raw=raw_orig,
            source=source,
            event_name_source=event_name_source,
            residual_sources=resid_sources,
        )
    except Exception:
        # Extreme fallback; never raise.
        now = float(datetime.now(timezone.utc).timestamp())
        return NormalizedEvent(
            ts=None,
            t=float(default_time) if default_time is not None else now,
            level=None,
            event="(unknown)",
            iter=None,
            resid=None,
            resid_precond=None,
            resid_true=None,
            fields={},
            raw=dict(rec) if isinstance(rec, Mapping) else {},
            source=source,
            event_name_source="unknown",
            residual_sources={},
        )
