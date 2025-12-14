from __future__ import annotations

"""
Helpers for parsing learn/train-style JSON embedded in the stdlib logger message string.

Design Doc anchors:
* FR-4: Robust log ingestion and normalization (rule: if message is JSON containing "event", parse it)
* 1.3 A critical known mismatch: logs vs viz parsers (must be fixed)

This module is intentionally defensive and never raises.
"""

import json
from typing import Any, Dict, Optional, Tuple

_MAX_MESSAGE_BYTES = 1_000_000  # defensive cap (1 MB)


def try_parse_message_json(msg: Any) -> Optional[Dict[str, Any]]:
    """
    Try to parse `msg` as a JSON object dict.

    Returns:
        dict if msg is a JSON string that parses to an object; otherwise None.

    Defensive notes:
    - Only accepts strings.
    - Refuses extremely large strings to avoid pathological memory/time.
    - Never raises.
    """
    try:
        if not isinstance(msg, str):
            return None

        s = msg.strip()
        if not s:
            return None

        # Cheap size check, then accurate byte check.
        if len(s) > _MAX_MESSAGE_BYTES:
            return None
        if len(s.encode("utf-8", errors="ignore")) > _MAX_MESSAGE_BYTES:
            return None

        # Quick shape check (learn/train emits a JSON object).
        if not (s.startswith("{") and s.endswith("}")):
            return None

        # Cheap pre-check: must mention "event" to be worth parsing for our use-case.
        if '"event"' not in s:
            return None

        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def extract_event_from_message_json(msg: Any) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Parse learn/train message JSON and return (event_name, parsed_fields).

    Returns:
        (event_name_or_None, parsed_dict_fields)

    Never raises.
    """
    d = try_parse_message_json(msg)
    if not d:
        return None, {}

    try:
        ev = d.get("event")
        if isinstance(ev, str) and ev.strip():
            return ev.strip(), d
        if ev is not None:
            # Non-string event; coerce to string for downstream consistency.
            s = str(ev).strip()
            return (s if s else None), d
        return None, d
    except Exception:
        return None, d
