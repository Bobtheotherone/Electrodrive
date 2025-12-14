"""
VizWatcher: robust, stdlib-only watcher for `run_dir/viz/*.png`.

Design Doc alignment:
- §3.2 “VizWatcher”: watch `viz/` for new PNGs; provide “latest frame” + timeline/slider; MP4 optional (not required).
- FR-5 acceptance: frames appear as they are written when `viz/` + `*.png` observed; detect quickly (~1–2 seconds locally).
- FR-3: frames live under `run_dir/viz/*.png`.

Repo alignment:
- Defensive I/O: swallow filesystem errors; never crash caller.
- Prefer `viz_*.png` if present; otherwise allow `viz.png` (iter_viz-style).
- Keep dependencies minimal (no inotify requirement; polling-based).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, List, Literal, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameInfo:
    """
    Metadata about a single visualization frame.
    """
    path: Path
    name: str
    index: int
    mtime: float
    size_bytes: int


@dataclass(frozen=True)
class VizEvent:
    """
    Event emitted by VizWatcher.
    """
    kind: Literal["frame_added", "frame_updated", "error"]
    frame: Optional[FrameInfo]
    message: Optional[str]
    t: float


_VIZ_INDEX_RE = re.compile(r"^viz_(\d+)(?:_overlay)?\.png$", re.IGNORECASE)


def _parse_frame_index(name: str) -> int:
    """
    Best-effort frame index extraction:
    - viz_####.png -> ####
    - viz.png / viz_overlay.png -> 0 (singleton frame)
    - viz_####_overlay.png -> ####
    - otherwise -> -1
    """
    try:
        n = str(name)
    except Exception:
        return -1

    if n in ("viz.png", "viz_overlay.png"):
        return 0

    m = _VIZ_INDEX_RE.match(n)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def _is_overlay_name(name: str) -> bool:
    try:
        return str(name).endswith("_overlay.png")
    except Exception:
        return False


def _safe_stat(path: Path) -> Optional[Tuple[float, int]]:
    """
    Return (mtime, size) or None. Never raises.
    """
    try:
        st = path.stat()
        return (float(st.st_mtime), int(st.st_size))
    except Exception:
        return None


def _stable_stat(path: Path, *, checks: int, sleep_s: float) -> Optional[Tuple[float, int]]:
    """
    Confirm file size is stable across `checks` reads separated by `sleep_s`.

    Returns the final (mtime, size) if stable; otherwise None.
    Never raises.
    """
    if checks <= 1:
        return _safe_stat(path)

    first = _safe_stat(path)
    if first is None:
        return None
    prev_mtime, prev_size = first

    for _ in range(checks - 1):
        try:
            time.sleep(max(0.0, float(sleep_s)))
        except Exception:
            pass
        cur = _safe_stat(path)
        if cur is None:
            return None
        cur_mtime, cur_size = cur
        # Primary stability signal is file size; mtime may tick even after final write on some FS.
        if cur_size != prev_size:
            return None
        prev_mtime, prev_size = cur_mtime, cur_size

    return (prev_mtime, prev_size)


def _sorted_viz_frames_numeric(paths: List[Path]) -> List[Path]:
    """
    Sort candidate PNG paths with numeric preference when possible; otherwise lexicographic.
    """
    def key(p: Path) -> Tuple[int, int, str]:
        n = p.name
        idx = _parse_frame_index(n)
        # idx>=0 means parsed numeric; sort those first by idx, then by name.
        if idx >= 0:
            return (0, idx, n)
        return (1, 0, n)

    try:
        return sorted(paths, key=key)
    except Exception:
        try:
            return sorted(paths, key=lambda p: p.name)
        except Exception:
            return paths


def _list_frame_paths_iter_viz_policy(
    viz_dir: Path,
    *,
    include_singleton_viz_png: bool,
    include_overlay_png: bool,
) -> List[Path]:
    """
    Frame discovery aligned with iter_viz-style preference ordering, plus a design-doc-safe fallback.

    Preference order:
    1) If any `viz_*.png` exist, return those (optionally including overlay variants).
    2) Else, fall back to `viz_overlay.png` (if enabled) and/or `viz.png` (if enabled).
    3) Else, final fallback: any `*.png` in viz_dir (design doc: watch viz/*.png).
    """
    try:
        if not viz_dir.is_dir():
            return []
    except Exception:
        return []

    # 1) Preferred: viz_*.png (iter_viz policy)
    try:
        frames = list(viz_dir.glob("viz_*.png"))
    except Exception:
        frames = []

    if not include_overlay_png:
        frames = [p for p in frames if not _is_overlay_name(p.name)]

    frames = _sorted_viz_frames_numeric(frames)
    if frames:
        return frames

    # 2) Singleton fallbacks when no viz_*.png exists
    fallbacks: List[Path] = []

    if include_overlay_png:
        try:
            p = viz_dir / "viz_overlay.png"
            if p.is_file():
                fallbacks.append(p)
        except Exception:
            pass

    if include_singleton_viz_png:
        try:
            p = viz_dir / "viz.png"
            if p.is_file():
                fallbacks.append(p)
        except Exception:
            pass

    if fallbacks:
        return fallbacks

    # 3) Final fallback: any .png (filtered)
    try:
        any_png = list(viz_dir.glob("*.png"))
    except Exception:
        return []

    out: List[Path] = []
    for p in any_png:
        try:
            if not include_overlay_png and _is_overlay_name(p.name):
                continue
            if not include_singleton_viz_png and p.name == "viz.png":
                continue
            # viz_overlay.png is considered "overlay"
            if not include_overlay_png and p.name == "viz_overlay.png":
                continue
            if p.is_file():
                out.append(p)
        except Exception:
            continue

    return _sorted_viz_frames_numeric(out)


class VizWatcher:
    """
    Polling-based watcher for `run_dir/viz/*.png` that maintains a timeline and emits events.

    Notes:
    - Threaded polling (cross-platform); no inotify requirement.
    - Defensive: filesystem errors are swallowed; the watcher never raises to callers.
    - Partial writes are mitigated via a stable-size check before emitting or reading bytes.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        viz_subdir: str = "viz",
        poll_interval_s: float = 0.5,
        include_singleton_viz_png: bool = True,
        include_overlay_png: bool = False,
        stable_size_checks: int = 2,
        stable_size_sleep_s: float = 0.05,
        max_events: int = 1000,
        on_event: Optional[Callable[[VizEvent], None]] = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.viz_subdir = str(viz_subdir)
        self.poll_interval_s = float(poll_interval_s)
        self.include_singleton_viz_png = bool(include_singleton_viz_png)
        self.include_overlay_png = bool(include_overlay_png)
        self.stable_size_checks = int(stable_size_checks)
        self.stable_size_sleep_s = float(stable_size_sleep_s)
        self.max_events = int(max_events)
        self.on_event = on_event

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Internal state.
        self._frames_by_path: Dict[Path, FrameInfo] = {}
        self._timeline: List[FrameInfo] = []
        self._last_seen: Dict[Path, Tuple[float, int]] = {}  # path -> (mtime, size)
        self._events: Deque[VizEvent] = deque(maxlen=max(1, self.max_events))

        self._running = False
        self._initialized = False  # baseline scan completed (suppresses initial event storm)
        self._last_error_emit_t = 0.0

    def start(self) -> None:
        """
        Start background polling thread (daemon ok).
        Idempotent.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._running = True
            t = threading.Thread(
                target=self._loop,
                name=f"VizWatcher[{self.run_dir}]",
                daemon=True,
            )
            self._thread = t
            t.start()

    def stop(self) -> None:
        """
        Request stop. Idempotent.
        """
        self._stop.set()

    def join(self, timeout: Optional[float] = None) -> None:
        t = None
        with self._lock:
            t = self._thread
        if t is None:
            return
        try:
            t.join(timeout=timeout)
        except Exception:
            return

    def running(self) -> bool:
        """
        True if the background thread is alive and stop has not been requested.
        """
        t = None
        with self._lock:
            t = self._thread
        return bool(t is not None and t.is_alive() and not self._stop.is_set())

    def frames(self) -> List[FrameInfo]:
        """
        Snapshot of current timeline, in deterministic order.
        """
        with self._lock:
            return list(self._timeline)

    def latest(self) -> Optional[FrameInfo]:
        """
        Latest frame (by timeline order).
        """
        with self._lock:
            if not self._timeline:
                return None
            return self._timeline[-1]

    def drain_events(self, max_n: int = 100) -> List[VizEvent]:
        """
        Drain up to max_n queued events (FIFO order).
        """
        n = max(0, int(max_n))
        out: List[VizEvent] = []
        with self._lock:
            while self._events and len(out) < n:
                try:
                    out.append(self._events.popleft())
                except Exception:
                    break
        return out

    def read_bytes(self, frame: FrameInfo, max_bytes: int = 50_000_000) -> Optional[bytes]:
        """
        Safe read with a size guard + stable-size check to avoid partial PNG reads.
        Returns bytes or None.
        """
        try:
            p = Path(frame.path)
            st = _stable_stat(
                p,
                checks=max(1, self.stable_size_checks),
                sleep_s=max(0.0, self.stable_size_sleep_s),
            )
            if st is None:
                return None
            _mtime, size = st
            if int(size) > int(max_bytes):
                return None
            return p.read_bytes()
        except Exception:
            return None

    # -------------------------
    # Internal helpers
    # -------------------------

    def _emit(self, ev: VizEvent) -> None:
        """
        Enqueue event and call callback (best-effort; callback errors swallowed).
        """
        cb = None
        with self._lock:
            self._events.append(ev)
            cb = self.on_event
        if cb is not None:
            try:
                cb(ev)
            except Exception:
                # Never let callback exceptions crash watcher.
                return

    def _emit_error_throttled(self, msg: str) -> None:
        """
        Emit an 'error' event, throttled to avoid spamming.
        """
        now = time.time()
        # Throttle to at most once per 2 seconds.
        with self._lock:
            if now - self._last_error_emit_t < 2.0:
                return
            self._last_error_emit_t = now

        try:
            log.debug("VizWatcher error: %s", msg)
        except Exception:
            pass

        self._emit(VizEvent(kind="error", frame=None, message=msg, t=now))

    def _scan(self, *, emit_events: bool = True) -> None:
        viz_dir = self.run_dir / self.viz_subdir

        paths = _list_frame_paths_iter_viz_policy(
            viz_dir,
            include_singleton_viz_png=self.include_singleton_viz_png,
            include_overlay_png=self.include_overlay_png,
        )

        current_set = set(paths)

        # Process additions/updates.
        for p in paths:
            try:
                st = _safe_stat(p)
                if st is None:
                    continue
                mtime, size = st

                prev = self._last_seen.get(p)
                is_new = prev is None
                is_changed = (prev is not None and (prev[0] != mtime or prev[1] != size))

                if not is_new and not is_changed:
                    continue

                # Partial write handling: ensure stable size before emitting.
                stable = _stable_stat(
                    p,
                    checks=max(1, self.stable_size_checks),
                    sleep_s=max(0.0, self.stable_size_sleep_s),
                )
                if stable is None:
                    # Not stable yet; skip this cycle.
                    continue
                mtime2, size2 = stable

                fi = FrameInfo(
                    path=p,
                    name=p.name,
                    index=_parse_frame_index(p.name),
                    mtime=float(mtime2),
                    size_bytes=int(size2),
                )

                with self._lock:
                    self._last_seen[p] = (float(mtime2), int(size2))
                    self._frames_by_path[p] = fi

                if emit_events:
                    kind: Literal["frame_added", "frame_updated"] = "frame_added" if is_new else "frame_updated"
                    self._emit(VizEvent(kind=kind, frame=fi, message=None, t=time.time()))
            except Exception as exc:
                self._emit_error_throttled(f"viz scan error: {exc}")
                continue

        # Handle removals and rebuild timeline (best-effort).
        with self._lock:
            removed = [p for p in list(self._frames_by_path.keys()) if p not in current_set]
            for p in removed:
                self._frames_by_path.pop(p, None)
                self._last_seen.pop(p, None)

            # Rebuild timeline in deterministic order based on current `paths`.
            self._timeline = [self._frames_by_path[p] for p in paths if p in self._frames_by_path]

    def _loop(self) -> None:
        """
        Background polling loop.
        """
        try:
            # Baseline population: avoid flooding the UI with old frames on startup.
            try:
                self._scan(emit_events=False)
            except Exception as exc:
                self._emit_error_throttled(f"viz watcher baseline scan error: {exc}")

            with self._lock:
                self._initialized = True

            while not self._stop.is_set():
                try:
                    self._scan(emit_events=True)
                except Exception as exc:
                    self._emit_error_throttled(f"viz watcher loop error: {exc}")

                try:
                    time.sleep(max(0.05, self.poll_interval_s))
                except Exception:
                    time.sleep(0.05)
        finally:
            with self._lock:
                self._running = False
