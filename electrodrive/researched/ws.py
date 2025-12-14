"""
WebSocket endpoints for ResearchED.

Design Doc references:

- FR-4: robust log ingestion + normalization (msg/event mismatch; resid/iter variants)
- §1.4: events.jsonl vs evidence_log.jsonl filename drift (must ingest both)
- FR-5: live monitor: logs + frames (viz/*.png watcher)
- §5.2: canonical event record shape (ts, t, level, event, fields, iter, resid, ...)

Dependency policy: FastAPI is optional extra; this module avoids importing it at
import time. Use get_ws_router() to construct the router.
"""

RESEARCHED_EVENTS_JSONL = "researched_events.jsonl"


import asyncio
import base64
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Robust JSONL tailer (inode/offset handling)
# ---------------------------------------------------------------------------

# When a new websocket client connects, start near the end of the file so we don't
# flood the UI with huge historical logs.
_DEFAULT_BACKLOG_BYTES = 2_000_000  # ~2MB


@dataclass
class _JsonlFollowState:
    path: Path
    name: str
    fh: Optional[Any] = None
    inode: Optional[int] = None
    offset: int = 0
    buffer: str = ""
    initialized: bool = False


def _open_if_needed(st: _JsonlFollowState) -> None:
    if st.fh is not None:
        return
    try:
        st.fh = st.path.open("r", encoding="utf-8", errors="ignore")
        info = st.path.stat()
        st.inode = getattr(info, "st_ino", None)
        size = int(getattr(info, "st_size", 0))
        # On first open, start from near the end (bounded backlog).
        if not st.initialized:
            st.offset = max(0, size - _DEFAULT_BACKLOG_BYTES)
            st.initialized = True
        else:
            st.offset = 0
        st.buffer = ""
    except FileNotFoundError:
        st.fh = None
    except Exception:
        st.fh = None


def _close(st: _JsonlFollowState) -> None:
    try:
        if st.fh is not None:
            st.fh.close()
    except Exception:
        pass
    st.fh = None


def _detect_rotation_or_truncation(st: _JsonlFollowState) -> None:
    if st.fh is None:
        return
    try:
        info = st.path.stat()
    except FileNotFoundError:
        _close(st)
        st.inode = None
        st.offset = 0
        st.buffer = ""
        st.initialized = False
        return
    except Exception:
        return

    cur_ino = getattr(info, "st_ino", None)
    size = int(getattr(info, "st_size", 0))

    if st.inode is not None and cur_ino is not None and cur_ino != st.inode:
        # rotation
        _close(st)
        st.inode = cur_ino
        st.offset = 0
        st.buffer = ""
        st.initialized = False
        _open_if_needed(st)
    elif st.offset > size:
        # truncation
        try:
            st.fh.seek(0)
        except Exception:
            _close(st)
            _open_if_needed(st)
        st.offset = 0
        st.buffer = ""
        st.initialized = True


def _poll_jsonl(st: _JsonlFollowState, *, max_bytes: int = 1_000_000) -> List[Dict[str, Any]]:
    """
    Read newly appended JSONL records from a file.

    - Robust to malformed JSON and partial writes (keeps incomplete trailing line in buffer).
    - Never raises.
    """
    _open_if_needed(st)
    if st.fh is None:
        return []

    _detect_rotation_or_truncation(st)
    if st.fh is None:
        return []

    try:
        st.fh.seek(st.offset)
    except Exception:
        _close(st)
        _open_if_needed(st)
        if st.fh is None:
            return []

    try:
        chunk = st.fh.read(max_bytes)
    except Exception:
        return []
    if not chunk:
        return []

    try:
        st.offset = int(st.fh.tell())
    except Exception:
        # If tell() fails, we conservatively stop advancing.
        return []

    st.buffer += chunk
    if "\n" not in st.buffer:
        # No complete line yet.
        return []

    parts = st.buffer.split("\n")
    if st.buffer.endswith("\n"):
        complete, st.buffer = parts[:-1], ""
    else:
        complete, st.buffer = parts[:-1], parts[-1]

    out: List[Dict[str, Any]] = []
    for line in complete:
        line = (line or "").strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _poll_lines(st: _JsonlFollowState, *, max_bytes: int = 1_000_000) -> List[str]:
    """
    Read newly appended text lines from a file.

    Used for "raw" stdout/stderr streaming (Design Doc FR-5: structured + raw logs).

    - Robust to partial writes (keeps incomplete trailing line in buffer).
    - Never raises.
    """
    _open_if_needed(st)
    if st.fh is None:
        return []

    _detect_rotation_or_truncation(st)
    if st.fh is None:
        return []

    try:
        st.fh.seek(st.offset)
    except Exception:
        _close(st)
        _open_if_needed(st)
        if st.fh is None:
            return []

    try:
        chunk = st.fh.read(max_bytes)
    except Exception:
        return []
    if not chunk:
        return []

    try:
        st.offset = int(st.fh.tell())
    except Exception:
        # If tell() fails, we conservatively stop advancing.
        return []

    st.buffer += chunk
    if "\n" not in st.buffer:
        return []

    parts = st.buffer.split("\n")
    if st.buffer.endswith("\n"):
        complete, st.buffer = parts[:-1], ""
    else:
        complete, st.buffer = parts[:-1], parts[-1]

    out: List[str] = []
    for line in complete:
        # Preserve content; normalize CRLF.
        line = (line or "").rstrip("\r")
        if not line:
            continue
        out.append(line)
    return out


# ---------------------------------------------------------------------------
# Normalization (Design Doc FR-4 + §5.2)
# ---------------------------------------------------------------------------


def _parse_iso_ts(ts: str) -> Optional[float]:
    try:
        s = ts.strip()
        if not s:
            return None
        # Handle "Z" suffix.
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    except Exception:
        return None


def _parse_numeric_ts(v: Any) -> Optional[float]:
    try:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            x = float(v)
        elif isinstance(v, str):
            x = float(v.strip())
        else:
            return None
        # Heuristic: treat very large as ms.
        if x > 1e12:
            return x / 1000.0
        return x
    except Exception:
        return None


def _first_number(rec: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for k in keys:
        if k not in rec:
            continue
        v = rec.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v.strip())
            except Exception:
                continue
    return None


def _first_int(rec: Mapping[str, Any], keys: Sequence[str]) -> Optional[int]:
    v = _first_number(rec, keys)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _maybe_parse_message_json(msg: Any) -> Optional[Dict[str, Any]]:
    """
    If msg looks like an embedded JSON dict string (learn/train style), parse it.

    Design Doc §1.3 / FR-4: stdlib logging may embed {"event": ...} inside message.
    """
    if not isinstance(msg, str):
        return None
    s = msg.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    # Cheap pre-check to avoid parsing arbitrary large strings.
    if '"event"' not in s and '"msg"' not in s and '"iter"' not in s and '"step"' not in s:
        return None
    try:
        obj = json.loads(s)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def normalize_event(rec: Mapping[str, Any], *, ingest_time: float, source: str) -> Dict[str, Any]:
    """
    Produce canonical normalized record for the UI.

    Design Doc FR-4 normalization rules:
      - event = rec.get("event") or rec.get("msg") or rec.get("message")
      - if message is JSON string containing "event", parse it and use its fields
      - iter = first present of ["iter","iters","step","k"]
      - resid = accept variants ["resid","resid_precond","resid_true","resid_precond_l2","resid_true_l2"]
      - timestamps: parse ISO ts to numeric epoch seconds 't'; fallback to ingest time
    """
    # Snapshot raw fields before we mutate/merge anything (for accurate provenance).
    raw_event = rec.get("event")
    raw_msg = rec.get("msg")
    raw_message = rec.get("message")

    # learn/train style: JSON embedded in the message string.
    msg_val = raw_msg if isinstance(raw_msg, str) else (raw_message if isinstance(raw_message, str) else None)
    embedded = _maybe_parse_message_json(msg_val)

    # Determine event name source per Design Doc FR-4 / FR-9.6.
    event_source = "missing"
    event_name: Any = None
    if isinstance(raw_event, str) and raw_event.strip():
        event_name = raw_event
        event_source = "event"
    elif embedded and isinstance(embedded.get("event"), str) and str(embedded.get("event")).strip():
        event_name = embedded.get("event")
        event_source = "embedded_json"
    elif isinstance(raw_msg, str) and raw_msg.strip():
        event_name = raw_msg
        event_source = "msg"
    elif isinstance(raw_message, str) and raw_message.strip():
        event_name = raw_message
        event_source = "message"
    else:
        event_name = ""

    base: Dict[str, Any] = dict(rec)  # shallow copy for safe access

    if embedded:
        # Merge embedded keys (learn/train style) while preserving core logger fields.
        for k, v in embedded.items():
            if k in {"ts", "level"}:
                continue
            # Preserve the original msg string for debugging.
            if k in {"msg", "message"} and ("msg" in base or "message" in base):
                continue
            base[k] = v

    if event_name is None:
        event_name = ""
    try:
        event_str = str(event_name)
    except Exception:
        event_str = ""

    level = base.get("level", "INFO")
    try:
        level_str = str(level).lower()
    except Exception:
        level_str = "info"

    # Timestamp parsing.
    ts_raw = base.get("ts")
    t: Optional[float] = None
    ts_str: Optional[str] = None
    if isinstance(ts_raw, str):
        ts_str = ts_raw
        t = _parse_iso_ts(ts_raw)
        if t is None:
            t = _parse_numeric_ts(ts_raw)
    else:
        t = _parse_numeric_ts(ts_raw)
        if isinstance(ts_raw, (int, float)):
            ts_str = str(ts_raw)

    if t is None:
        t = float(ingest_time)

    # Iter/residual extraction.
    it = _first_int(base, ["iter", "iters", "step", "k"])
    resid_precond = _first_number(base, ["resid_precond", "resid_precond_l2"])
    resid_true = _first_number(base, ["resid_true", "resid_true_l2"])
    resid = _first_number(base, ["resid"])
    if resid is None:
        resid = resid_precond if resid_precond is not None else resid_true

    # Build fields dict (everything except canonical keys).
    skip = {
        "ts",
        "t",
        "level",
        "event",
        "msg",
        "message",
        # iter variants
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
    fields: Dict[str, Any] = {}
    for k, v in base.items():
        if k in skip:
            continue
        fields[k] = v

    # For records that truly have no event-ish field (common in metrics.jsonl),
    # provide a stable fallback so filtering/search doesn't break.
    if not event_str.strip():
        event_str = f"{source}:record"
        event_source = "fallback"

    return {
        "ts": ts_str,
        "t": float(t),
        "level": level_str,
        "event": event_str,
        "event_source": event_source,
        "iter": it,
        "resid": resid,
        "resid_precond": resid_precond,
        "resid_true": resid_true,
        "fields": fields,
        "source": source,
    }


def _fingerprint(ev: Mapping[str, Any]) -> str:
    """
    Conservative dedup key.

    Design Doc FR-4: deduplicate identical records when merging events.jsonl and evidence_log.jsonl.
    """
    try:
        core = {
            "t": round(float(ev.get("t", 0.0)), 3),
            "level": str(ev.get("level", "")),
            "event": str(ev.get("event", "")),
            "iter": ev.get("iter"),
            "resid": ev.get("resid"),
            "fields": ev.get("fields", {}),
        }
        blob = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        blob = str(ev)
    return sha1(blob.encode("utf-8", errors="ignore")).hexdigest()


def _resolve_run_dir(runs_root: Path, run_id: str) -> Optional[Path]:
    # Quick path: direct folder name.
    cand = (runs_root / run_id).expanduser()
    try:
        if cand.is_dir():
            return cand
    except Exception:
        pass

    # Scan a few levels (best-effort) to match by dir name or manifest run_id.
    def is_run_dir(p: Path) -> bool:
        markers = (
            "manifest.researched.json",
            "manifest.json",
            "metrics.json",
            "events.jsonl",
            "evidence_log.jsonl",
            RESEARCHED_EVENTS_JSONL,
            "train_log.jsonl",
            "metrics.jsonl",
        )
        try:
            return any((p / m).is_file() for m in markers) or (p / "viz").is_dir()
        except Exception:
            return False

    skip = {".git", "__pycache__", ".pytest_cache", "viz", "plots", "artifacts", "node_modules"}
    q: List[Tuple[Path, int]] = [(runs_root, 0)]
    seen: set[str] = set()

    while q:
        cur, d = q.pop(0)
        key = str(cur)
        if key in seen:
            continue
        seen.add(key)

        if d > 0 and is_run_dir(cur):
            if cur.name == run_id:
                return cur
            m = None
            for name in ("manifest.researched.json", "manifest.json"):
                try:
                    m = json.loads((cur / name).read_text(encoding="utf-8"))
                except Exception:
                    m = None
                if isinstance(m, dict):
                    break
            if isinstance(m, dict):
                rid = m.get("run_id")
                if isinstance(rid, str) and rid == run_id:
                    return cur
            continue

        if d >= 4:
            continue
        try:
            for child in cur.iterdir():
                if child.is_dir() and child.name not in skip:
                    q.append((child, d + 1))
        except Exception:
            continue

    return None


def _require_fastapi() -> None:
    try:
        from fastapi import APIRouter  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "ResearchED WebSocket endpoints require FastAPI (optional extra). Install with: pip install fastapi uvicorn"
        ) from exc


def get_ws_router():
    """Construct and return the WebSocket router (lazy FastAPI import)."""
    _require_fastapi()
    from fastapi import APIRouter, WebSocket
    from starlette.websockets import WebSocketDisconnect

    router = APIRouter()

    @router.websocket("/runs/{run_id}/events")
    async def ws_events(websocket: WebSocket, run_id: str) -> None:
        """
        Stream normalized events for a run.

        Design Doc FR-4 + §1.4: merge/tail multiple JSONL files:
          - events.jsonl
          - evidence_log.jsonl
          - researched_events.jsonl
          - train_log.jsonl
          - metrics.jsonl
        """
        await websocket.accept()

        runs_root = Path(getattr(websocket.app.state, "runs_root", "runs"))
        run_dir = _resolve_run_dir(runs_root, run_id)
        if run_dir is None:
            await websocket.send_json({"type": "error", "error": "run_not_found", "run_id": run_id})
            await websocket.close(code=1008)
            return

        followers = [
            _JsonlFollowState(path=run_dir / "events.jsonl", name="events.jsonl"),
            _JsonlFollowState(path=run_dir / "evidence_log.jsonl", name="evidence_log.jsonl"),
            _JsonlFollowState(path=run_dir / RESEARCHED_EVENTS_JSONL, name=RESEARCHED_EVENTS_JSONL),
            _JsonlFollowState(path=run_dir / "train_log.jsonl", name="train_log.jsonl"),
            _JsonlFollowState(path=run_dir / "metrics.jsonl", name="metrics.jsonl"),
        ]

        recent: List[str] = []
        recent_set: set[str] = set()
        ping_interval = 10.0
        poll_interval = 0.25
        last_ping = 0.0

        try:
            while True:
                now = time.time()
                if now - last_ping >= ping_interval:
                    last_ping = now
                    try:
                        await websocket.send_json({"type": "ping", "t": now})
                    except WebSocketDisconnect:
                        break
                    except Exception:
                        pass

                new_recs: List[Dict[str, Any]] = []
                for st in followers:
                    for rec in _poll_jsonl(st):
                        ingest = time.time()
                        ev = normalize_event(rec, ingest_time=ingest, source=st.name)
                        fp = _fingerprint(ev)
                        if fp in recent_set:
                            continue
                        # Maintain bounded dedup window.
                        recent.append(fp)
                        recent_set.add(fp)
                        if len(recent) > 5000:
                            old = recent.pop(0)
                            recent_set.discard(old)
                        new_recs.append(ev)

                # Sort by time for nicer UI ordering (best-effort).
                try:
                    new_recs.sort(key=lambda e: float(e.get("t", 0.0)))
                except Exception:
                    pass

                for ev in new_recs:
                    try:
                        await websocket.send_json({"type": "event", **ev})
                    except WebSocketDisconnect:
                        return
                    except Exception:
                        continue

                await asyncio.sleep(poll_interval)
        except WebSocketDisconnect:
            return
        except asyncio.CancelledError:
            return
        finally:
            for st in followers:
                _close(st)

    async def _ws_tail_raw(websocket: WebSocket, run_id: str, *, filename: str, stream: str) -> None:
        """
        Stream raw (unstructured) logs for a run.

        Design Doc FR-5: live monitor must support streaming logs (structured + raw).
        """
        await websocket.accept()

        runs_root = Path(getattr(websocket.app.state, "runs_root", "runs"))
        run_dir = _resolve_run_dir(runs_root, run_id)
        if run_dir is None:
            await websocket.send_json({"type": "error", "error": "run_not_found", "run_id": run_id})
            await websocket.close(code=1008)
            return

        target_path = run_dir / filename

        # Compatibility: some runners merge stderr into stdout and never create stderr.log.
        if filename == "stderr.log":
            try:
                missing = (not target_path.is_file())
                empty = (target_path.is_file() and target_path.stat().st_size == 0)
                if missing or empty:
                    fallback = run_dir / "stdout.log"
                    if fallback.is_file():
                        await websocket.send_json(
                            {
                                "type": "raw",
                                "stream": stream,
                                "t": time.time(),
                                "text": "[ResearchED] stderr.log missing/empty; streaming stdout.log instead",
                            }
                        )
                        target_path = fallback
                        filename = "stdout.log"
            except Exception:
                pass

        follower = _JsonlFollowState(path=target_path, name=filename)

        ping_interval = 10.0
        poll_interval = 0.25
        last_ping = 0.0

        try:
            while True:
                now = time.time()
                if now - last_ping >= ping_interval:
                    last_ping = now
                    try:
                        await websocket.send_json({"type": "ping", "t": now})
                    except WebSocketDisconnect:
                        break
                    except Exception:
                        pass

                for line in _poll_lines(follower):
                    try:
                        await websocket.send_json({"type": "raw", "stream": stream, "t": time.time(), "text": line})
                    except WebSocketDisconnect:
                        return
                    except Exception:
                        continue

                await asyncio.sleep(poll_interval)
        except WebSocketDisconnect:
            return
        except asyncio.CancelledError:
            return
        finally:
            _close(follower)

    @router.websocket("/runs/{run_id}/stdout")
    async def ws_stdout(websocket: WebSocket, run_id: str) -> None:
        await _ws_tail_raw(websocket, run_id, filename="stdout.log", stream="stdout")

    @router.websocket("/runs/{run_id}/stderr")
    async def ws_stderr(websocket: WebSocket, run_id: str) -> None:
        await _ws_tail_raw(websocket, run_id, filename="stderr.log", stream="stderr")

    @router.websocket("/runs/{run_id}/frames")
    async def ws_frames(websocket: WebSocket, run_id: str) -> None:
        """
        Stream viz frame notifications (and bytes) for a run.

        Design Doc FR-5: watch run_dir/viz/*.png and stream new frames.
        """
        await websocket.accept()

        runs_root = Path(getattr(websocket.app.state, "runs_root", "runs"))
        run_dir = _resolve_run_dir(runs_root, run_id)
        if run_dir is None:
            await websocket.send_json({"type": "error", "error": "run_not_found", "run_id": run_id})
            await websocket.close(code=1008)
            return

        viz_dir = run_dir / "viz"
        seen: Dict[str, Tuple[float, int]] = {}
        ping_interval = 10.0
        poll_interval = 0.25
        last_ping = 0.0

        def parse_index(name: str) -> Optional[int]:
            # Expect viz_0001.png style; otherwise None.
            stem = Path(name).stem
            if stem.startswith("viz_"):
                suf = stem[4:]
                try:
                    return int(suf)
                except Exception:
                    return None
            return None

        try:
            while True:
                now = time.time()
                if now - last_ping >= ping_interval:
                    last_ping = now
                    try:
                        await websocket.send_json({"type": "ping", "t": now})
                    except WebSocketDisconnect:
                        break
                    except Exception:
                        pass

                if viz_dir.is_dir():
                    try:
                        frames = sorted(viz_dir.glob("*.png"))
                    except Exception:
                        frames = []

                    for p in frames:
                        try:
                            st = p.stat()
                            key = p.name
                            mtime = float(st.st_mtime)
                            size = int(st.st_size)
                        except Exception:
                            continue

                        prev = seen.get(key)
                        if prev is not None and prev == (mtime, size):
                            continue  # unchanged

                        # Read bytes (best-effort).
                        try:
                            data = p.read_bytes()
                        except Exception:
                            continue

                        seen[key] = (mtime, size)
                        msg = {
                            "type": "frame",
                            "name": key,
                            "index": parse_index(key),
                            "mtime": mtime,
                            "bytes_b64": base64.b64encode(data).decode("ascii"),
                        }
                        try:
                            await websocket.send_json(msg)
                        except WebSocketDisconnect:
                            return
                        except Exception:
                            continue

                await asyncio.sleep(poll_interval)
        except WebSocketDisconnect:
            return
        except asyncio.CancelledError:
            return

    return router

