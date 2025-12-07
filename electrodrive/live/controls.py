from __future__ import annotations

"""
Electrodrive live controls.

This module implements a robust, JSON-based control channel for long-running
solver jobs (e.g., BEM GMRES iterations). It is intentionally:

- stdlib-only (no new dependencies),
- cross-platform (Linux/macOS/Windows),
- robust against partial writes, concurrent writers, and CI environments,
- deterministic when files are absent or malformed,
- friendly to external tooling (simple JSON schema & helpers).

Control file:
    control.json  (in a run directory)

Core fields (all optional on disk; defaults are applied on load):

    pause: bool           # request solver to pause (polling, cooperative)
    terminate: bool       # request solver to finish early (cooperative)
    write_every: int|null # suggested cadence for heavy outputs; >=1 or null
    snapshot: str|null    # one-shot marker for next snapshot/frame
    ts: float             # update timestamp (epoch seconds; set by writers)
    version: int          # schema version (default: 1)
    seq: int              # monotonically increasing update sequence
    ack_seq: int|null     # last seq acknowledged by solver

Unknown/extra keys must be preserved when using merge=True so that external
processes can add their own metadata without breaking the solver.

Example (doctest-style):

    >>> from pathlib import Path
    >>> from electrodrive.live.controls import (
    ...     DEFAULT_STATE, write_controls, read_controls, update_controls
    ... )
    >>> tmp = Path("tmp_run")
    >>> tmp.mkdir(exist_ok=True)
    >>> # Initial read when file is missing:
    >>> read_controls(tmp) == DEFAULT_STATE
    True
    >>> # Write a pause request:
    >>> st = write_controls(tmp, {"pause": True})
    >>> st.pause
    True
    >>> # Merge an update without clobbering pause:
    >>> st2 = update_controls(tmp, terminate=True)
    >>> (st2.pause, st2.terminate)
    (True, True)
    >>> # Invalid values are normalized on read:
    >>> (tmp / "control.json").write_text('{"write_every": -5}', encoding="utf-8")
    >>> read_controls(tmp).write_every is None
    True

"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
from typing import Dict, Mapping, MutableMapping, Optional, Union

logger = logging.getLogger(__name__)

_JSONDict = Dict[str, object]
_PathLike = Union[str, Path]


@dataclass(frozen=True)
class ControlState:
    """
    Immutable snapshot of control.json.

    Fields:
        pause: whether the solver should pause cooperatively.
        terminate: whether the solver should terminate early.
        write_every: suggested iteration cadence for heavy outputs.
            - None: no override.
            - int: must be >= 1; invalid values are normalized to None.
        snapshot: optional one-shot snapshot marker.
        ts: last update time (epoch seconds); set by writer.
        version: schema version (currently 1).
        seq: monotonically increasing sequence number for updates.
        ack_seq: last seq acknowledged by the solver (optional).

    Unknown fields in the underlying JSON are preserved only when merging
    through write_controls() with merge=True; they are not stored here.

    Use DEFAULT_STATE as the baseline.
    """

    pause: bool = False
    terminate: bool = False
    write_every: Optional[int] = None
    snapshot: Optional[str] = None
    ts: float = 0.0
    version: int = 1
    seq: int = 0
    ack_seq: Optional[int] = None

    # Internal: any extra keys preserved across merges (not part of public API).
    _extra: Mapping[str, object] = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #

    def to_dict(self) -> _JSONDict:
        """
        Convert to a JSON-ready dict, including any preserved extra keys.

        Only public keys are guaranteed; extra keys round-trip unknown fields.

        >>> DEFAULT_STATE.to_dict()["pause"]
        False
        """
        data: _JSONDict = {
            "pause": bool(self.pause),
            "terminate": bool(self.terminate),
            "write_every": int(self.write_every) if self.write_every is not None else None,
            "snapshot": self.snapshot,
            "ts": float(self.ts),
            "version": int(self.version),
            "seq": int(self.seq),
            "ack_seq": int(self.ack_seq) if self.ack_seq is not None else None,
        }
        # Merge extras without clobbering core fields
        for k, v in (self._extra or {}).items():
            if k not in data:
                data[k] = v
        return data

    @staticmethod
    def from_dict(data: Mapping[str, object]) -> "ControlState":
        """
        Parse and validate a raw mapping into ControlState.

        - Missing/invalid fields fall back to DEFAULT_STATE values.
        - write_every is normalized: <1 or non-int -> None.
        - seq/version negative -> 0 (clipped).
        - Extra keys are preserved on the resulting instance.

        Invalid structures (non-mapping) should be handled by callers by
        falling back to DEFAULT_STATE.

        >>> ControlState.from_dict({"pause": True}).pause
        True
        >>> ControlState.from_dict({"write_every": -5}).write_every is None
        True
        """
        if not isinstance(data, Mapping):
            return DEFAULT_STATE

        base = DEFAULT_STATE

        def _get_bool(key: str, default: bool) -> bool:
            v = data.get(key, default)
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            return default

        def _get_int_nonneg(key: str, default: int) -> int:
            v = data.get(key, default)
            try:
                iv = int(v)
                return iv if iv >= 0 else default
            except Exception:
                return default

        def _get_int_opt_pos(key: str) -> Optional[int]:
            v = data.get(key, None)
            if v is None:
                return None
            try:
                iv = int(v)
            except Exception:
                return None
            return iv if iv >= 1 else None

        def _get_str_opt(key: str) -> Optional[str]:
            v = data.get(key, None)
            if v is None:
                return None
            try:
                s = str(v)
            except Exception:
                return None
            return s

        pause = _get_bool("pause", base.pause)
        terminate = _get_bool("terminate", base.terminate)
        write_every = _get_int_opt_pos("write_every")
        snapshot = _get_str_opt("snapshot")
        ts_raw = data.get("ts", base.ts)
        try:
            ts = float(ts_raw)
            if not (ts == ts and ts >= 0.0):  # NaN or negative
                ts = base.ts
        except Exception:
            ts = base.ts
        version = _get_int_nonneg("version", base.version)
        seq = _get_int_nonneg("seq", base.seq)

        ack_seq_val: Optional[int]
        ack_raw = data.get("ack_seq", None)
        if ack_raw is None:
            ack_seq_val = None
        else:
            try:
                ai = int(ack_raw)
                ack_seq_val = ai if ai >= 0 else None
            except Exception:
                ack_seq_val = None

        # Preserve extra keys (for round-trip friendliness)
        extra: Dict[str, object] = {}
        for k, v in data.items():
            if k not in {
                "pause",
                "terminate",
                "write_every",
                "snapshot",
                "ts",
                "version",
                "seq",
                "ack_seq",
            }:
                extra[k] = v

        return ControlState(
            pause=pause,
            terminate=terminate,
            write_every=write_every,
            snapshot=snapshot,
            ts=ts,
            version=version,
            seq=seq,
            ack_seq=ack_seq_val,
            _extra=extra,
        )

    # ------------------------------------------------------------------ #
    # Merge helpers
    # ------------------------------------------------------------------ #

    def merged(self, updates: Mapping[str, object]) -> "ControlState":
        """
        Return a new ControlState with given keys updated, preserving extras.

        Core keys are applied through from_dict-style validation.

        Unknown keys in updates are merged into _extra.

        >>> s = DEFAULT_STATE.merged({"pause": True, "write_every": 5})
        >>> (s.pause, s.write_every)
        (True, 5)
        """
        if not isinstance(updates, Mapping):
            return self

        # Start from current dict (including extras), overlay updates, then parse.
        base_dict = self.to_dict()
        merged_dict: Dict[str, object] = dict(base_dict)
        for k, v in updates.items():
            merged_dict[k] = v
        # Preserve current seq unless caller explicitly sets it; seq_increment is
        # handled by write_controls().
        return ControlState.from_dict(merged_dict)


# Default immutable state (no extras).
DEFAULT_STATE = ControlState()


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #


def _resolve_run_dir(run_dir: Optional[_PathLike]) -> Optional[Path]:
    if run_dir is None:
        env = os.getenv("EDE_RUN_DIR")
        if not env:
            return None
        run_dir = env
    try:
        p = Path(run_dir).expanduser()
    except Exception:
        return None
    return p


def controls_path(run_dir: _PathLike) -> Path:
    """
    Return the path to control.json under run_dir.

    >>> from pathlib import Path
    >>> controls_path("runs/x") == Path("runs/x") / "control.json"
    True
    """
    return Path(run_dir).expanduser() / "control.json"


def _lock_path(run_dir: Path) -> Path:
    return run_dir / "control.lock"


def _temp_path(run_dir: Path) -> Path:
    ns = time.time_ns()
    pid = os.getpid()
    return run_dir / f"control.json.tmp-{pid}-{ns}"


# --------------------------------------------------------------------------- #
# Core file I/O
# --------------------------------------------------------------------------- #


def read_controls(run_dir: Optional[_PathLike]) -> ControlState:
    """
    Read and parse control.json from run_dir.

    Behavior:
    - If run_dir is None, uses $EDE_RUN_DIR when set; otherwise returns defaults.
    - If the file is missing, unreadable, or malformed, returns DEFAULT_STATE.
    - Never raises on common I/O/JSON errors.
    - Invalid fields are normalized; extra fields ignored (but preserved when
      later merged via write_controls).

    Logging:
    - DEBUG for non-fatal issues.
    - WARNING only when encountering repeated malformed content is likely helpful.
    """
    rdir = _resolve_run_dir(run_dir)
    if rdir is None:
        return DEFAULT_STATE

    path = controls_path(rdir)
    try:
        st = path.stat()
    except FileNotFoundError:
        return DEFAULT_STATE
    except OSError as exc:
        logger.debug("read_controls: stat() failed.", exc_info=False)
        logger.debug("read_controls: error=%s", exc)
        return DEFAULT_STATE

    if not path.is_file():
        return DEFAULT_STATE

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = f.read()
    except (OSError, UnicodeError) as exc:
        logger.debug("read_controls: failed to read file.", exc_info=False)
        logger.debug("read_controls: error=%s", exc)
        return DEFAULT_STATE

    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        logger.debug("read_controls: JSON decode error.", exc_info=False)
        logger.debug("read_controls: error=%s", exc)
        return DEFAULT_STATE

    try:
        return ControlState.from_dict(data)
    except Exception as exc:  # extremely defensive
        logger.warning("read_controls: unexpected error in from_dict: %s", exc)
        return DEFAULT_STATE


# --------------------------------------------------------------------------- #
# Atomic write with advisory lock
# --------------------------------------------------------------------------- #


def _acquire_lock(lock_file: Path, timeout: float = 0.25) -> bool:
    """
    Best-effort advisory lock via O_CREAT|O_EXCL.

    - Waits up to timeout seconds, sleeping briefly between attempts.
    - If lock is stale (mtime older than timeout*8), it may be removed.
    - Never raises; on failure, returns False (caller may still write without lock).
    """
    deadline = time.time() + max(timeout, 0.0)
    sleep_interval = 0.01
    while True:
        try:
            # Use low-level os.open for atomic create.
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
            return True
        except FileExistsError:
            # Maybe stale?
            try:
                st = lock_file.stat()
                # If lock file is very old, consider it stale and remove.
                if (time.time() - st.st_mtime) > (timeout * 8.0 + 1.0):
                    try:
                        lock_file.unlink()
                    except OSError:
                        pass
            except OSError:
                # If stat fails, try to remove or ignore.
                try:
                    lock_file.unlink()
                except OSError:
                    pass
        except OSError:
            # Unexpected I/O; don't spin.
            break

        if time.time() >= deadline:
            break
        time.sleep(sleep_interval)

    return False


def _release_lock(lock_file: Path) -> None:
    try:
        lock_file.unlink()
    except OSError:
        pass


def _fsync_path(path: Path) -> None:
    """
    Best-effort fsync helper (file or directory).

    Errors are swallowed; this is advisory for durability.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def write_controls(
    run_dir: Optional[_PathLike],
    updates: Optional[Mapping[str, object]] = None,
    *,
    merge: bool = True,
    seq_increment: bool = True,
) -> ControlState:
    """
    Atomically write control.json for run_dir.

    Args:
        run_dir:
            Target run directory. If None, use $EDE_RUN_DIR. If neither is set,
            this is a no-op returning DEFAULT_STATE.
        updates:
            Mapping of fields to update. Common keys:
                pause, terminate, write_every, snapshot, ack_seq, ...
            Unknown keys are preserved (on merge) as extra metadata.
        merge:
            - True (default): read current state (best-effort), merge updates,
              and preserve unknown fields.
            - False: treat updates as authoritative, applied over DEFAULT_STATE.
        seq_increment:
            When True (default), seq is incremented by 1 over the existing seq.
            When False, seq is left unchanged unless explicitly set in updates.

    Behavior:
        - Reads current control.json (if present & valid) for merge/seq.
        - Applies updates via ControlState.merged().
        - Sets ts = time.time().
        - Adjusts seq according to seq_increment.
        - Writes using a temp file + fsync + os.replace (atomic).
        - Uses a best-effort advisory lock (control.lock) around the write.
        - Cleans up temp and lock files best-effort.
        - Never raises on expected I/O races; logs at DEBUG/WARNING.

    Returns:
        The ControlState that was attempted to be written (even on partial errors).

    Note:
        If run_dir cannot be resolved, or directory creation fails, this returns
        DEFAULT_STATE without modifying the filesystem.
    """
    rdir = _resolve_run_dir(run_dir)
    if rdir is None:
        logger.debug("write_controls: no run_dir/EDE_RUN_DIR; skipping write.")
        return DEFAULT_STATE

    try:
        rdir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("write_controls: failed to ensure run_dir: %s", exc)
        return DEFAULT_STATE

    cur_state = read_controls(rdir) if merge else DEFAULT_STATE

    updates = updates or {}
    if not isinstance(updates, Mapping):
        logger.debug("write_controls: non-mapping updates ignored.")
        updates = {}

    # Apply updates on top of current.
    new_state = cur_state.merged(updates)

    # Handle seq increment/override.
    if "seq" in updates:
        # If user explicitly set seq, respect validated value from merged().
        pass
    elif seq_increment:
        new_seq = max(cur_state.seq, new_state.seq) + 1
        new_state = replace(new_state, seq=new_seq)

    # Always update timestamp to "now".
    now = time.time()
    new_state = replace(new_state, ts=now)

    # Prepare JSON payload.
    payload = new_state.to_dict()
    path = controls_path(rdir)
    tmp_path = _temp_path(rdir)
    lock_file = _lock_path(rdir)

    locked = _acquire_lock(lock_file)
    try:
        # Atomic write via temp file -> flush+fsync -> replace
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, separators=(",", ":"), sort_keys=False)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    # Not all platforms support fsync; ignore.
                    pass

            # Replace target atomically (POSIX + Windows semantics).
            os.replace(str(tmp_path), str(path))

            # Fsync directory for durability (best-effort).
            _fsync_path(rdir)
        except Exception as exc:
            # Clean up temp and never raise.
            logger.warning("write_controls: failed to write control.json: %s", exc)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
    finally:
        if locked:
            _release_lock(lock_file)

    return new_state


def update_controls(run_dir: Optional[_PathLike], **kwargs: object) -> ControlState:
    """
    Convenience wrapper around write_controls with keyword updates.

    Examples:
        - update_controls(run_dir, pause=True)
        - update_controls(run_dir, terminate=True, snapshot="final")

    Returns:
        The ControlState that was attempted to be written.
    """
    return write_controls(run_dir, updates=kwargs, merge=True, seq_increment=True)


def ack_controls(run_dir: Optional[_PathLike], ack_seq: int) -> ControlState:
    """
    Set ack_seq in control.json safely (merge-only).

    Intended for the solver to acknowledge having processed all commands
    up to a given seq.

    Behavior:
        - If ack_seq < 0, treated as no-op (returns current state).
        - Otherwise, merges {"ack_seq": ack_seq} over existing state,
          with seq_increment=False (acks do not create new commands).
    """
    if ack_seq is None:
        return read_controls(run_dir)
    try:
        iv = int(ack_seq)
    except Exception:
        return read_controls(run_dir)
    if iv < 0:
        return read_controls(run_dir)
    return write_controls(
        run_dir,
        updates={"ack_seq": iv},
        merge=True,
        seq_increment=False,
    )


def schema() -> _JSONDict:
    """
    Return a minimal JSON-schema-like description of control.json.

    This is intended for external tooling / UIs.

    >>> s = schema()
    >>> sorted(k for k in s["properties"].keys())[:3]
    ['ack_seq', 'pause', 'seq']
    """
    return {
        "type": "object",
        "properties": {
            "pause": {"type": "boolean", "default": False},
            "terminate": {"type": "boolean", "default": False},
            "write_every": {
                "type": ["integer", "null"],
                "minimum": 1,
                "default": None,
            },
            "snapshot": {"type": ["string", "null"], "default": None},
            "ts": {"type": "number", "default": 0.0},
            "version": {"type": "integer", "minimum": 1, "default": 1},
            "seq": {"type": "integer", "minimum": 0, "default": 0},
            "ack_seq": {
                "type": ["integer", "null"],
                "minimum": 0,
                "default": None,
            },
        },
        "additionalProperties": True,
    }


# --------------------------------------------------------------------------- #
# Watcher utility
# --------------------------------------------------------------------------- #


class ControlWatcher:
    """
    Lightweight polling watcher for control.json.

    Features:
        - Caches last (mtime_ns, size) to avoid re-reading unchanged files.
        - Returns ControlState on change, or None if unchanged.
        - Tolerates partial/corrupt writes:
            * On JSON error, backs off for backoff_bad_json seconds and
              preserves last good state.
        - Logs at most once per error type (rate-limited).
        - Thread-safe for concurrent peek() via an internal Lock.

    Parameters:
        run_dir:
            Run directory or None (uses $EDE_RUN_DIR).
        poll_interval:
            Minimum time between successful re-reads of unchanged files.
            (Used as a soft guard against busy-wait loops.)
        backoff_bad_json:
            Cooldown interval after a JSON parse/validation error before
            attempting to read again.

    Typical usage in a solver loop (doctest-style):

        >>> from pathlib import Path
        >>> rd = Path("watcher_demo")
        >>> rd.mkdir(exist_ok=True)
        >>> w = ControlWatcher(rd, poll_interval=0.1)
        >>> update_controls(rd, pause=True)
        >>> st = w.peek()
        >>> bool(st.pause) if st else False
        True
        >>> # Subsequent peek without changes returns None:
        >>> w.peek() is None
        True
    """

    def __init__(
        self,
        run_dir: Optional[_PathLike],
        poll_interval: float = 0.2,
        backoff_bad_json: float = 0.5,
    ) -> None:
        self._run_dir = _resolve_run_dir(run_dir)
        self._poll_interval = max(float(poll_interval), 0.0)
        self._backoff_bad_json = max(float(backoff_bad_json), 0.0)

        self._last_state: Optional[ControlState] = None
        self._last_mtime_ns: int = -1
        self._last_size: int = -1
        self._last_read_time: float = 0.0
        self._bad_json_until: float = 0.0

        self._lock = Lock()
        self._logged_bad_json = False
        self._logged_io_error = False

    def _path(self) -> Optional[Path]:
        if self._run_dir is None:
            self._run_dir = _resolve_run_dir(self._run_dir)
        if self._run_dir is None:
            return None
        return controls_path(self._run_dir)

    def peek(self) -> Optional[ControlState]:
        """
        Return a new ControlState if control.json changed since last read.

        Returns:
            - ControlState: when file changed and was parsed successfully.
            - None: when file unchanged, unreadable, or malformed (while
              preserving the last good state).

        This method never raises. It uses a small internal lock, so it is safe
        to call from multiple threads.
        """
        with self._lock:
            path = self._path()
            if path is None:
                return None

            now = time.time()

            # Backoff after bad JSON/validation.
            if now < self._bad_json_until:
                return None

            try:
                st = path.stat()
            except FileNotFoundError:
                # Treat as no controls; reset cached markers to detect reappearance.
                self._last_mtime_ns = -1
                self._last_size = -1
                self._last_read_time = now
                # No new state; do not wipe last_state (sticky).
                return None
            except OSError as exc:
                if not self._logged_io_error:
                    logger.debug("ControlWatcher: stat() failed: %s", exc)
                    self._logged_io_error = True
                return None

            mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
            size = int(st.st_size)

            # If unchanged and we've recently polled, return None.
            if (
                mtime_ns == self._last_mtime_ns
                and size == self._last_size
                and (now - self._last_read_time) < self._poll_interval
            ):
                return None

            # Attempt to read/parse new content.
            try:
                with path.open("r", encoding="utf-8") as f:
                    raw = f.read()
                data = json.loads(raw or "{}")
                state = ControlState.from_dict(data)
            except json.JSONDecodeError as exc:
                # Likely partial write; back off briefly.
                if not self._logged_bad_json:
                    logger.debug("ControlWatcher: JSON decode error: %s", exc)
                    self._logged_bad_json = True
                self._bad_json_until = now + self._backoff_bad_json
                return None
            except (OSError, UnicodeError) as exc:
                if not self._logged_io_error:
                    logger.debug("ControlWatcher: read error: %s", exc)
                    self._logged_io_error = True
                return None
            except Exception as exc:
                # Unexpected validation/other errors; back off briefly.
                if not self._logged_bad_json:
                    logger.warning("ControlWatcher: unexpected error: %s", exc)
                    self._logged_bad_json = True
                self._bad_json_until = now + self._backoff_bad_json
                return None

            # Successful read: update cache and return new state iff changed.
            self._last_read_time = now
            self._bad_json_until = 0.0
            self._logged_bad_json = False
            self._logged_io_error = False

            if (
                mtime_ns == self._last_mtime_ns
                and size == self._last_size
                and self._last_state is not None
                and state == self._last_state
            ):
                # Spurious wake-up; treat as unchanged.
                return None

            self._last_mtime_ns = mtime_ns
            self._last_size = size
            self._last_state = state
            return state