from __future__ import annotations

"""
Log consumer coverage / audit metrics accumulator for ResearchED.

Design Doc anchors:
* FR-9.6 Visualization + log consumer audit panel
* FR-4: Robust log ingestion and normalization (malformed/dropped stats, residual/key coverage)
* 1.3 A critical known mismatch: logs vs viz parsers (must be fixed)
* 1.4 Explicit compatibility policy: events.jsonl vs evidence_log.jsonl (which files were ingested)

This module is stdlib-only and defensive.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .normalizer import NormalizedEvent


@dataclass
class LogCoverage:
    total_lines_seen: int = 0
    total_records_parsed: int = 0
    total_records_emitted: int = 0
    total_json_errors: int = 0
    total_non_dict_records: int = 0
    dropped_by_dedup: int = 0

    # "unknown" event names are not dropped, but should be counted for FR-9.6 fix-it checks.
    records_missing_event_name: int = 0

    event_name_source_counts: Dict[str, int] = field(default_factory=dict)
    residual_field_detection_counts: Dict[str, int] = field(default_factory=dict)
    ingested_files: Set[str] = field(default_factory=set)

    last_event_t: float = 0.0

    def note_file(self, path: Path) -> None:
        try:
            self.ingested_files.add(Path(path).name)
        except Exception:
            pass

    def note_dedup_drop(self) -> None:
        try:
            self.dropped_by_dedup += 1
        except Exception:
            pass

    def note_emitted(self, ev: "NormalizedEvent") -> None:
        """
        Record an emitted normalized event for audit reporting (FR-9.6).
        """
        try:
            self.total_records_emitted += 1
        except Exception:
            pass

        try:
            self.last_event_t = max(float(self.last_event_t), float(ev.t))
        except Exception:
            pass

        # Which event-name field was used? ("event"/"msg"/"message"/"message_json"/"unknown")
        src = "unknown"
        try:
            src = str(getattr(ev, "event_name_source", None) or "unknown")
        except Exception:
            src = "unknown"
        self.event_name_source_counts[src] = self.event_name_source_counts.get(src, 0) + 1

        # Count unknown/missing event names (for “Fix-it checklist” panel).
        try:
            ev_name = str(getattr(ev, "event", "") or "").strip()
            if src == "unknown" or ev_name in ("", "(unknown)"):
                self.records_missing_event_name += 1
        except Exception:
            pass

        # Residual variants detected (track raw-source key names).
        try:
            rs = dict(getattr(ev, "residual_sources", None) or {})
        except Exception:
            rs = {}
        for canonical in ("resid", "resid_precond", "resid_true"):
            key = rs.get(canonical)
            if not key:
                continue
            k = str(key)
            self.residual_field_detection_counts[k] = self.residual_field_detection_counts.get(k, 0) + 1

    def note_tailer_delta(self, delta: Dict[str, Any]) -> None:
        """
        Apply JsonlTailer.drain_stats() output into coverage counters.

        Important: we intentionally do NOT add records_parsed here if merge_streams already increments it,
        to avoid double counting. This method is for line-level and malformed-line stats.
        """
        try:
            tot = delta.get("total") or {}
            self.total_lines_seen += int(tot.get("lines_seen", 0) or 0)
            self.total_json_errors += int(tot.get("json_errors", 0) or 0)
            self.total_non_dict_records += int(tot.get("non_dict_records", 0) or 0)
        except Exception:
            pass

    def snapshot(self) -> Dict[str, Any]:
        """
        JSON-serializable snapshot for UI.

        Includes FR-9.6 required fields:
        - parsed count
        - ingested filenames
        - event-name-source breakdown
        - residual variants breakdown
        - malformed/dropped counts
        """
        return {
            "total_lines_seen": int(self.total_lines_seen),
            "total_records_parsed": int(self.total_records_parsed),
            "total_records_emitted": int(self.total_records_emitted),
            "total_json_errors": int(self.total_json_errors),
            "total_non_dict_records": int(self.total_non_dict_records),
            "dropped_by_dedup": int(self.dropped_by_dedup),
            "records_missing_event_name": int(self.records_missing_event_name),
            "event_name_source_counts": dict(sorted(self.event_name_source_counts.items(), key=lambda kv: kv[0])),
            "residual_field_detection_counts": dict(
                sorted(self.residual_field_detection_counts.items(), key=lambda kv: kv[0])
            ),
            "ingested_files": sorted(self.ingested_files),
            "last_event_t": float(self.last_event_t) if self.last_event_t else None,
        }
