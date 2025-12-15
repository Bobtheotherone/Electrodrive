from __future__ import annotations

"""
ResearchED ingestion layer (stdlib-only): tail JSONL logs, normalize schema mismatches, merge streams,
and compute coverage metrics for the UI “log consumer audit panel”.

Design Doc anchors:
* FR-4: Robust log ingestion and normalization
* 1.3 A critical known mismatch: logs vs viz parsers (must be fixed)
* 1.4 Explicit compatibility policy: events.jsonl vs evidence_log.jsonl
* FR-9.6 Visualization + log consumer audit panel

This package is deliberately stdlib-only and defensive: public functions should not raise.
"""

from .learn_message_json import extract_event_from_message_json, try_parse_message_json
from .log_coverage import LogCoverage
from .merge import DedupCache, dedup_key, merge_event_files, merge_streams, stable_hash_fields
from .normalizer import NormalizedEvent, normalize_record, parse_ts_to_epoch_seconds
from .tailer import JsonlTailer

__all__ = [
    # message JSON helpers
    "try_parse_message_json",
    "extract_event_from_message_json",
    # normalization
    "NormalizedEvent",
    "normalize_record",
    "parse_ts_to_epoch_seconds",
    # tailer
    "JsonlTailer",
    # merge + dedup
    "stable_hash_fields",
    "dedup_key",
    "DedupCache",
    "merge_streams",
    "merge_event_files",
    # coverage
    "LogCoverage",
]
