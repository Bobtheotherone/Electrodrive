from __future__ import annotations

"""
Merge + deduplicate normalized events across multiple JSONL sources.

Design Doc anchors:
* FR-4: Robust log ingestion and normalization (multi-file merge + dedup)
* 1.4 Explicit compatibility policy: events.jsonl vs evidence_log.jsonl (ingest both)
* FR-9.6 Visualization + log consumer audit panel

This module is stdlib-only and defensive: merge_streams never raises.
"""

import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .normalizer import NormalizedEvent, normalize_record


def _sanitize(obj: Any) -> Any:
    """
    Deterministic JSON-safe sanitization for stable hashing.

    - dict keys -> str (json.dumps uses sort_keys=True)
    - NaN/Inf floats -> strings
    - Paths -> str
    - sets -> sorted list
    - unknown objects -> str
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        if math.isnan(obj):
            return "NaN"
        return "Infinity" if obj > 0 else "-Infinity"
    if isinstance(obj, Path):
        return str(obj)
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


def stable_hash_fields(fields: Mapping[str, Any]) -> str:
    """
    Stable hash of fields dict (order-independent).

    Design Doc FR-4 dedup requires a stable_hash(fields) stable across dict ordering.
    """
    try:
        blob = json.dumps(_sanitize(dict(fields)), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        blob = str(fields)
    return sha1(blob.encode("utf-8", errors="ignore")).hexdigest()


def dedup_key(ev: NormalizedEvent) -> Tuple[float, str, str, str]:
    """
    Dedup key per Design Doc FR-4:
        (normalized_time, level, normalized_event_name, stable_hash(fields))
    """
    t = round(float(ev.t), 3)  # millisecond-ish stability
    level = (ev.level or "").lower()
    event = str(ev.event or "").strip().lower()
    h = stable_hash_fields(ev.fields)
    return (t, level, event, h)


class DedupCache:
    """
    Bounded dedup cache with max-items and TTL to prevent unbounded growth.
    """

    def __init__(self, max_items: int = 50_000, ttl_seconds: float = 300.0) -> None:
        self.max_items = max(1, int(max_items))
        self.ttl_seconds = float(ttl_seconds)
        self._od: "OrderedDict[Tuple[float, str, str, str], float]" = OrderedDict()
        self._lock = Lock()

    def _prune(self, now: float) -> None:
        # TTL prune (oldest-first).
        if self.ttl_seconds > 0:
            cutoff = now - self.ttl_seconds
            while self._od:
                k, ts = next(iter(self._od.items()))
                if ts >= cutoff:
                    break
                self._od.popitem(last=False)

        # Size prune.
        while len(self._od) > self.max_items:
            self._od.popitem(last=False)

    def seen(self, key: Tuple[float, str, str, str]) -> bool:
        """
        Returns True if key already present (and refreshes its TTL), else inserts and returns False.
        """
        now = time.time()
        with self._lock:
            self._prune(now)
            if key in self._od:
                self._od.move_to_end(key)
                self._od[key] = now
                return True
            self._od[key] = now
            self._prune(now)
            return False


def _call_normalizer(
    normalizer: Callable[..., NormalizedEvent],
    rec: Mapping[str, Any],
    *,
    source: Optional[str],
    default_time: Optional[float],
) -> NormalizedEvent:
    """
    Call a custom normalizer if provided.

    Supports both:
    - normalize_record(rec, *, source=..., default_time=...)
    - custom_normalizer(rec)
    """
    try:
        return normalizer(rec, source=source, default_time=default_time)
    except TypeError:
        return normalizer(rec)


def merge_streams(
    batch: Sequence[Tuple[Path, Dict[str, Any]]],
    *,
    normalizer: Optional[Callable[..., NormalizedEvent]] = None,
    dedup_cache: Optional[DedupCache] = None,
    coverage: Any = None,
) -> List[NormalizedEvent]:
    """
    Normalize + merge + deduplicate a batch of (path, raw_record) pairs from JsonlTailer.poll().

    Returns normalized events sorted by timestamp ascending.

    Coverage integration (FR-9.6):
    - marks ingested files
    - counts parsed/emitted
    - counts dedup drops

    Never raises.
    """
    norm = normalizer or normalize_record
    cache = dedup_cache or DedupCache()

    out: List[NormalizedEvent] = []

    # Per-path file mtime snapshot (used as a fallback default_time when record has no ts).
    mtime: Dict[Path, float] = {}
    for p, _ in batch:
        if p not in mtime:
            try:
                mtime[p] = float(p.stat().st_mtime)
            except Exception:
                mtime[p] = time.time()

    for idx, (path, rec) in enumerate(batch):
        try:
            if coverage is not None:
                try:
                    coverage.note_file(path)
                except Exception:
                    pass

            # Parsed record count (dict records coming from tailer).
            if coverage is not None:
                try:
                    coverage.total_records_parsed += 1
                except Exception:
                    pass

            # Default time fallback: if no ts present, use file mtime; also ensure monotonicity in-batch.
            ts_raw = rec.get("ts")
            default_time = None
            if ts_raw is None or (isinstance(ts_raw, str) and not ts_raw.strip()):
                default_time = mtime.get(path, time.time()) + (idx * 1e-6)

            ev = _call_normalizer(norm, rec, source=path.name, default_time=default_time)

            k = dedup_key(ev)
            if cache.seen(k):
                if coverage is not None:
                    try:
                        coverage.note_dedup_drop()
                    except Exception:
                        pass
                continue

            out.append(ev)
            if coverage is not None:
                try:
                    coverage.note_emitted(ev)
                except Exception:
                    pass
        except Exception:
            # Defensive: never raise in ingestion.
            continue

    # Sort by t, with stable tie-breakers.
    try:
        out.sort(key=lambda e: (float(e.t), str(e.source or ""), str(e.event), stable_hash_fields(e.fields)))
    except Exception:
        pass
    return out
