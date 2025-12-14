from __future__ import annotations

"""
Robust JSONL tailer for ResearchED.

Design Doc anchors:
* FR-4: Robust log ingestion and normalization (tail JSONL, tolerate partial writes, never raise)
* 1.4 Explicit compatibility policy: events.jsonl vs evidence_log.jsonl (tail multiple files concurrently)

This tailer is stdlib-only, cross-platform, and defensive:
* missing files are tolerated (keeps polling)
* malformed JSON lines are skipped and counted
* partial lines are buffered until newline arrives
* buffer size is capped to prevent unbounded growth when a newline never arrives

Implementation note:
- Files are opened in *binary* mode to keep offsets comparable to st_size and to make truncation detection reliable.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class TailerFileStats:
    lines_seen: int = 0
    records_parsed: int = 0
    json_errors: int = 0
    non_dict_records: int = 0


@dataclass
class _FollowState:
    path: Path
    fh: Optional[Any] = None
    inode: Optional[int] = None
    offset: int = 0  # byte offset
    buffer: bytes = b""
    initialized: bool = False
    stats_total: TailerFileStats = field(default_factory=TailerFileStats)
    stats_delta: TailerFileStats = field(default_factory=TailerFileStats)


class JsonlTailer:
    """
    Follow multiple JSONL files concurrently and return newly appended dict records.

    API:
        JsonlTailer(paths, start_at_end=False, encoding="utf-8", poll_interval=0.25)
        .poll() -> list[(Path, dict)]
        .close()

    Notes:
    - `poll_interval` is advisory; the caller controls scheduling.
    - Stats are exposed via .stats_snapshot() and .drain_stats().
    - Never raises.
    """

    def __init__(
        self,
        paths: Sequence[Path],
        *,
        start_at_end: bool = False,
        encoding: str = "utf-8",
        poll_interval: float = 0.25,
        max_read_bytes: int = 1_000_000,
        max_buffer_bytes: int = 8_000_000,
    ) -> None:
        self._encoding = str(encoding)
        self.poll_interval = float(poll_interval)
        self._start_at_end = bool(start_at_end)
        self._max_read_bytes = int(max(4096, max_read_bytes))
        self._max_buffer_bytes = int(max(4096, max_buffer_bytes))

        self._states: Dict[Path, _FollowState] = {}
        for p in paths:
            pp = Path(p)
            self._states[pp] = _FollowState(path=pp)

        self._closed = False

    def close(self) -> None:
        """
        Close all open file handles. Best-effort; never raises.
        Also attempts to parse any buffered final partial line if it is valid JSON.
        """
        if self._closed:
            return
        self._closed = True
        for st in self._states.values():
            self._flush_buffer_as_line(st)
            self._close_state(st)

    def _close_state(self, st: _FollowState) -> None:
        try:
            if st.fh is not None:
                st.fh.close()
        except Exception:
            pass
        st.fh = None

    def _open_if_needed(self, st: _FollowState) -> None:
        if st.fh is not None:
            return
        try:
            if not st.path.exists():
                return
        except Exception:
            return
        try:
            st.fh = st.path.open("rb")
            info = st.path.stat()
            st.inode = getattr(info, "st_ino", None)
            if self._start_at_end and not st.initialized:
                try:
                    st.fh.seek(0, 2)  # SEEK_END
                    st.offset = int(st.fh.tell())
                except Exception:
                    st.offset = 0
            st.initialized = True
        except FileNotFoundError:
            st.fh = None
        except Exception:
            st.fh = None

    def _detect_rotation_or_truncation(self, st: _FollowState) -> None:
        """
        Best-effort rotation/truncation detection.
        - Uses POSIX inode when available.
        - Uses file size to detect truncation.

        Never raises.
        """
        if st.fh is None:
            return

        try:
            info = st.path.stat()
        except FileNotFoundError:
            # File removed; close and reset.
            self._close_state(st)
            st.inode = None
            st.offset = 0
            st.buffer = b""
            return
        except Exception:
            return

        cur_ino = getattr(info, "st_ino", None)
        size = int(getattr(info, "st_size", 0))

        if st.inode is not None and cur_ino is not None and cur_ino != st.inode:
            # Rotation / replacement.
            self._close_state(st)
            st.inode = cur_ino
            st.offset = 0
            st.buffer = b""
            # After rotation, we generally want to read from the beginning of the new file.
            st.initialized = True
            self._open_if_needed(st)
        elif st.offset > size:
            # Truncation.
            try:
                if st.fh is not None:
                    st.fh.seek(0)
            except Exception:
                self._close_state(st)
                self._open_if_needed(st)
            st.offset = 0
            st.buffer = b""

    def _read_new_bytes(self, st: _FollowState) -> bytes:
        self._open_if_needed(st)
        if st.fh is None:
            return b""
        self._detect_rotation_or_truncation(st)
        if st.fh is None:
            return b""

        try:
            st.fh.seek(st.offset)
        except Exception:
            # Try reopen.
            self._close_state(st)
            self._open_if_needed(st)
            if st.fh is None:
                return b""
            try:
                st.fh.seek(st.offset)
            except Exception:
                return b""

        try:
            chunk = st.fh.read(self._max_read_bytes)
        except Exception:
            return b""
        if not chunk:
            return b""

        try:
            st.offset = int(st.fh.tell())
        except Exception:
            # If tell fails, we keep offset unchanged to avoid skipping.
            pass

        return chunk

    def _count_line_seen(self, st: _FollowState) -> None:
        st.stats_total.lines_seen += 1
        st.stats_delta.lines_seen += 1

    def _count_json_error(self, st: _FollowState) -> None:
        st.stats_total.json_errors += 1
        st.stats_delta.json_errors += 1

    def _count_non_dict(self, st: _FollowState) -> None:
        st.stats_total.non_dict_records += 1
        st.stats_delta.non_dict_records += 1

    def _count_parsed(self, st: _FollowState) -> None:
        st.stats_total.records_parsed += 1
        st.stats_delta.records_parsed += 1

    def _decode_line(self, b: bytes) -> str:
        try:
            return b.decode(self._encoding, errors="ignore")
        except Exception:
            try:
                return b.decode("utf-8", errors="ignore")
            except Exception:
                return ""

    def _flush_buffer_as_line(self, st: _FollowState) -> None:
        """
        Attempt to parse the current buffer as a final line (if valid JSON).

        This is best-effort and only used on close. Never raises.
        """
        try:
            b = (st.buffer or b"").strip()
            if not b:
                return

            s = self._decode_line(b).strip()
            if not s:
                return

            # Only attempt if it looks like a JSON object/array.
            if not ((s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))):
                return

            self._count_line_seen(st)
            try:
                obj = json.loads(s)
            except Exception:
                self._count_json_error(st)
                return

            if not isinstance(obj, dict):
                self._count_non_dict(st)
                return

            self._count_parsed(st)
        finally:
            st.buffer = b""

    def poll(self) -> List[Tuple[Path, Dict[str, Any]]]:
        """
        Return newly parsed dict records since the last poll, across all tracked paths.

        Never raises; malformed lines are skipped and counted.
        """
        out: List[Tuple[Path, Dict[str, Any]]] = []
        if self._closed:
            return out

        for st in self._states.values():
            try:
                chunk = self._read_new_bytes(st)
                if not chunk:
                    continue

                st.buffer += chunk

                # Defensive cap: if we never see a newline, buffer could grow forever.
                if b"\n" not in st.buffer and len(st.buffer) > self._max_buffer_bytes:
                    self._count_line_seen(st)
                    self._count_json_error(st)
                    st.buffer = b""
                    continue

                if b"\n" not in st.buffer:
                    continue

                parts = st.buffer.split(b"\n")
                if st.buffer.endswith(b"\n"):
                    complete, st.buffer = parts[:-1], b""
                else:
                    complete, st.buffer = parts[:-1], parts[-1]

                for line_b in complete:
                    line_b = (line_b or b"").strip()
                    if not line_b:
                        continue
                    self._count_line_seen(st)
                    line = self._decode_line(line_b).strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        self._count_json_error(st)
                        continue
                    if not isinstance(obj, dict):
                        self._count_non_dict(st)
                        continue
                    self._count_parsed(st)
                    out.append((st.path, obj))
            except Exception:
                # Ultimate guard: never crash caller.
                continue

        return out

    def stats_snapshot(self) -> Dict[str, Any]:
        """
        Return total stats for all files (non-resetting).
        """
        per: Dict[str, Any] = {}
        tot = TailerFileStats()
        for p, st in self._states.items():
            s = st.stats_total
            per[str(p)] = {
                "lines_seen": s.lines_seen,
                "records_parsed": s.records_parsed,
                "json_errors": s.json_errors,
                "non_dict_records": s.non_dict_records,
            }
            tot.lines_seen += s.lines_seen
            tot.records_parsed += s.records_parsed
            tot.json_errors += s.json_errors
            tot.non_dict_records += s.non_dict_records

        return {
            "total": {
                "lines_seen": tot.lines_seen,
                "records_parsed": tot.records_parsed,
                "json_errors": tot.json_errors,
                "non_dict_records": tot.non_dict_records,
            },
            "per_file": per,
        }

    def drain_stats(self) -> Dict[str, Any]:
        """
        Return delta stats since last drain and reset deltas to zero.

        This is useful for feeding LogCoverage without double counting.
        """
        per: Dict[str, Any] = {}
        tot = TailerFileStats()
        for p, st in self._states.items():
            s = st.stats_delta
            per[str(p)] = {
                "lines_seen": s.lines_seen,
                "records_parsed": s.records_parsed,
                "json_errors": s.json_errors,
                "non_dict_records": s.non_dict_records,
            }
            tot.lines_seen += s.lines_seen
            tot.records_parsed += s.records_parsed
            tot.json_errors += s.json_errors
            tot.non_dict_records += s.non_dict_records
            st.stats_delta = TailerFileStats()  # reset

        return {
            "total": {
                "lines_seen": tot.lines_seen,
                "records_parsed": tot.records_parsed,
                "json_errors": tot.json_errors,
                "non_dict_records": tot.non_dict_records,
            },
            "per_file": per,
        }
