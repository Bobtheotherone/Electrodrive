from __future__ import annotations

"""
Lightweight, cross-platform live residual console for running Electrodrive solves.

Usage:
    python -m electrodrive.viz.live_console --run RUN_DIR [--hz 4] [--win 200]

Features:
    - Stdlib only, no extra dependencies.
    - Tails evidence_log.jsonl incrementally (JSONL, robust to malformed lines).
    - Extracts GMRES iteration/residual from log lines and shows:
        iter, last residual, residual trend sparkline, pause state, write_every,
        log lag, last update age, optional GPU metrics.
    - Non-blocking key commands via file-based control.json:
        p: toggle pause       (pause = !pause)
        r: resume             (pause = false)
        s: stop/terminate     (terminate = true)
        +: faster frames      (write_every = max(1, write_every // 2) or 1)
        -: slower frames      (write_every = write_every * 2 or 5)
        m: mark snapshot      (snapshot label; one-shot)
        q: quit console       (no control change)
    - If electrodrive.live.controls is importable, delegates to write_controls.
      Otherwise falls back to atomic temp-file write+os.replace to control.json.
    - Robust:
        * Handles missing/rotating logs.
        * Ignores malformed JSON.
        * Never raises to top-level; prints warnings and continues.
    - TTY aware:
        * TTY: single dynamic status line at given refresh Hz.
        * Non-TTY (e.g., CI): prints summary once per ~5s, no key handling,
          auto-exits after extended inactivity if no progress.
    - GPU-friendly:
        * If torch+CUDA available, calls torch.cuda.synchronize() once per loop.
        * Optionally displays gpu_mem_peak_mb.* from metrics.json when present.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

# -----------------------------------------------------------------------------
# TTY / encoding helpers
# -----------------------------------------------------------------------------


def _is_tty(stream: object = sys.stdout) -> bool:
    try:
        return bool(getattr(stream, "isatty", lambda: False)())
    except Exception:
        return False


def _supports_utf8() -> bool:
    try:
        enc = getattr(sys.stdout, "encoding", None) or ""
        return "utf" in enc.lower()
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Tailer state
# -----------------------------------------------------------------------------


@dataclass
class TailState:
    path: Path
    fh: Optional[object] = None
    inode: Optional[int] = None
    offset: int = 0
    last_iter: int = -1
    last_resid: float = math.nan
    last_ts: float = 0.0  # timestamp when last progress parsed

    def close(self) -> None:
        try:
            if self.fh is not None:
                self.fh.close()
        except Exception:
            pass
        self.fh = None


@dataclass
class ResidualWindow:
    maxlen: int
    values: Deque[float]

    @classmethod
    def create(cls, maxlen: int) -> "ResidualWindow":
        return cls(maxlen=maxlen, values=deque(maxlen=maxlen))

    def append(self, v: float) -> None:
        self.values.append(float(v))

    def as_list(self) -> List[float]:
        return list(self.values)

    def last(self) -> float:
        if not self.values:
            return math.nan
        return float(self.values[-1])


# -----------------------------------------------------------------------------
# Control I/O (with optional electrodrive.live.controls)
# -----------------------------------------------------------------------------


def _load_existing_controls(path: Path) -> Dict[str, object]:
    try:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _merge_controls(existing: Dict[str, object], updates: Dict[str, object]) -> Dict[str, object]:
    merged = dict(existing) if isinstance(existing, dict) else {}
    for k, v in updates.items():
        merged[k] = v
    # Always stamp ts
    merged["ts"] = float(time.time())
    return merged


def _atomic_write_controls(path: Path, data: Dict[str, object]) -> None:
    try:
        run_dir = path.parent
        run_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    tmp = run_dir / f".control.json.tmp-{os.getpid()}-{time.time_ns()}"
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"), sort_keys=False)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(str(tmp), str(path))
        # Best-effort directory fsync
        try:
            dfd = os.open(str(run_dir), os.O_RDONLY)
            try:
                os.fsync(dfd)
            except Exception:
                pass
            finally:
                os.close(dfd)
        except Exception:
            pass
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


class ControlWriter:
    """
    Unified control writer.

    - If electrodrive.live.controls is importable, uses write_controls(run_dir, **updates).
    - Otherwise falls back to atomic control.json merge/write in run_dir.
    - Rate-limits writes to avoid thrashing.
    """

    def __init__(self, run_dir: Path, min_interval: float = 0.1) -> None:
        self.run_dir = run_dir
        self.min_interval = float(min_interval)
        self.last_write: float = 0.0
        self._impl = self._detect_impl()

    def _detect_impl(self):
        try:
            from electrodrive.live.controls import write_controls as wc  # type: ignore

            def _wrap(upd: Dict[str, object]) -> None:
                if not upd:
                    return
                # electrodrive.live.controls.write_controls merges and stamps ts by itself
                wc(str(self.run_dir), updates=upd, merge=True, seq_increment=True)

            return _wrap
        except Exception:
            return None

    def _write_fallback(self, updates: Dict[str, object]) -> None:
        if not updates:
            return
        path = self.run_dir / "control.json"
        existing = _load_existing_controls(path)
        merged = _merge_controls(existing, updates)
        _atomic_write_controls(path, merged)

    def write(self, updates: Dict[str, object]) -> None:
        now = time.time()
        if now - self.last_write < self.min_interval:
            return
        self.last_write = now
        try:
            if self._impl is not None:
                self._impl(updates)
            else:
                self._write_fallback(updates)
        except Exception:
            # Never propagate control write failures.
            pass


# -----------------------------------------------------------------------------
# Non-blocking key input
# -----------------------------------------------------------------------------


def _read_keys() -> List[str]:
    """
    Non-blocking single-character key reader.

    - On Windows, uses msvcrt.kbhit/getwch.
    - On POSIX TTY, uses select.select on sys.stdin.
    - Returns a list of characters (no special decoding beyond basic mapping).
    - Never blocks; never raises.
    """
    keys: List[str] = []
    try:
        if os.name == "nt":
            import msvcrt  # type: ignore

            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch == "\x03":  # Ctrl-C
                    keys.append("q")
                elif ch:
                    keys.append(ch)
        else:
            # POSIX
            if not _is_tty(sys.stdin):
                return keys
            import select

            r, _, _ = select.select([sys.stdin], [], [], 0.0)
            if r:
                data = sys.stdin.read(1)
                if data:
                    if data == "\x03":  # Ctrl-C
                        keys.append("q")
                    else:
                        keys.append(data)
    except Exception:
        # Input is best-effort.
        return []
    return keys


def _prompt_label(timeout: float = 10.0) -> str:
    """
    Prompt for a short snapshot label.

    - Intended for interactive use only; if stdin is not a TTY or read fails,
      falls back to an empty label.
    - To avoid hanging, applies a simple timeout where possible.
    """
    if not _is_tty(sys.stdin):
        # Non-interactive; auto-generate from timestamp.
        return time.strftime("%H:%M:%S")

    sys.stdout.write("\nlabel: ")
    sys.stdout.flush()
    start = time.time()
    buf = ""
    try:
        if os.name == "nt":
            import msvcrt  # type: ignore

            while True:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ("\r", "\n"):
                        break
                    if ch in ("\x08", "\x7f"):  # backspace
                        buf = buf[:-1]
                        continue
                    buf += ch
                if time.time() - start > timeout:
                    break
                time.sleep(0.05)
        else:
            import select

            while True:
                if time.time() - start > timeout:
                    break
                r, _, _ = select.select([sys.stdin], [], [], 0.2)
                if r:
                    chunk = sys.stdin.readline()
                    if not chunk:
                        break
                    buf = chunk.strip()
                    break
    except Exception:
        buf = ""

    if not buf:
        buf = time.strftime("%H:%M:%S")
    return buf.strip()[:64]


# -----------------------------------------------------------------------------
# JSONL tailer
# -----------------------------------------------------------------------------


def _open_log_if_needed(state: TailState) -> None:
    if state.fh is not None:
        return
    try:
        fh = state.path.open("r", encoding="utf-8", errors="ignore")
        st = state.path.stat()
        state.fh = fh
        state.inode = getattr(st, "st_ino", None)
        state.offset = 0
    except FileNotFoundError:
        # Caller handles "waiting for log".
        pass
    except Exception:
        # Best-effort; ignore.
        pass


def _detect_rotation_or_truncation(state: TailState) -> None:
    if state.fh is None:
        return
    try:
        st = state.path.stat()
    except FileNotFoundError:
        # Log removed -> close & reset.
        state.close()
        state.inode = None
        state.offset = 0
        return
    except Exception:
        return

    cur_ino = getattr(st, "st_ino", None)
    size = st.st_size
    if state.inode is not None and cur_ino is not None and cur_ino != state.inode:
        # File replaced (rotation).
        state.close()
        state.inode = cur_ino
        state.offset = 0
        _open_log_if_needed(state)
    elif state.offset > size:
        # Truncated.
        try:
            state.fh.seek(0)
        except Exception:
            state.close()
            _open_log_if_needed(state)
        state.offset = 0


def _first_present(obj: Dict[str, object], keys) -> object:
    for k in keys:
        if k in obj:
            v = obj.get(k)
            if v is not None:
                return v
    return None


def _event_name(obj: Dict[str, object]) -> str:
    return str(obj.get("event") or obj.get("msg") or obj.get("message") or "")


def _safe_float(v: object) -> Optional[float]:
    try:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str) and v.strip():
            return float(v.strip())
    except Exception:
        return None
    return None


def _parse_progress_line(obj: Dict[str, object]) -> Optional[Tuple[int, float]]:
    """
    Extract (iter, resid) from a JSON object if it looks like GMRES progress.
    """
    if not isinstance(obj, dict):
        return None

    msg = _event_name(obj).lower()
    if msg and "gmres" not in msg:
        return None

    it_raw = _first_present(obj, ("iter", "iters", "step", "k"))
    resid_raw = _first_present(
        obj,
        ("resid", "resid_true", "resid_precond", "resid_true_l2", "resid_precond_l2"),
    )
    it_val = _safe_float(it_raw)
    resid_val = _safe_float(resid_raw)
    if it_val is None or resid_val is None:
        return None

    try:
        return int(it_val), float(resid_val)
    except Exception:
        return None

    return None


def _tail_jsonl(state: TailState, window: ResidualWindow) -> None:
    """
    Read newly appended lines from evidence_log.jsonl, updating:

        - window.values with residuals
        - state.last_iter
        - state.last_resid
        - state.last_ts (monotonic: time of last parsed progress)

    Robust to:
        - missing/truncated/rotated logs
        - malformed JSON lines
        - unrelated events in the JSONL

    Never raises.
    """
    _open_log_if_needed(state)
    if state.fh is None:
        return

    _detect_rotation_or_truncation(state)
    if state.fh is None:
        return

    try:
        state.fh.seek(state.offset)
    except Exception:
        # Re-open if seek fails.
        state.close()
        _open_log_if_needed(state)
        if state.fh is None:
            return

    while True:
        try:
            line = state.fh.readline()
        except Exception:
            break

        if not line:
            break  # EOF

        state.offset = state.fh.tell()
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except Exception:
            continue

        parsed = _parse_progress_line(obj)
        if parsed is None:
            continue

        it, resid = parsed
        state.last_iter = max(state.last_iter, it)
        state.last_resid = resid
        state.last_ts = time.time()
        window.append(resid)


# -----------------------------------------------------------------------------
# Sparkline rendering
# -----------------------------------------------------------------------------


_SPARK_UTF8 = "▁▂▃▄▅▆▇█"
_SPARK_ASCII = "._-~=*#@"


def _make_sparkline(values: Iterable[float], width: int, use_utf8: bool) -> str:
    vals = [v for v in values if math.isfinite(v)]
    if not vals or width <= 0:
        return " " * max(width, 0)

    n = len(vals)
    if n > width:
        # Downsample to width via simple stride.
        step = n / float(width)
        sampled = []
        for i in range(width):
            idx = int(i * step)
            if idx >= n:
                idx = n - 1
            sampled.append(vals[idx])
        vals = sampled
        n = width

    vmin = min(vals)
    vmax = max(vals)
    if vmax <= vmin:
        # Flat line
        return (use_utf8 and "▁" or "-") * n

    chars = _SPARK_UTF8 if use_utf8 else _SPARK_ASCII
    m = len(chars) - 1
    out = []
    for v in vals:
        if not math.isfinite(v):
            idx = 0
        else:
            t = (v - vmin) / (vmax - vmin)
            idx = int(t * m)
            if idx < 0:
                idx = 0
            elif idx > m:
                idx = m
        out.append(chars[idx])
    return "".join(out).ljust(width)


# -----------------------------------------------------------------------------
# GPU / metrics helpers (best-effort)
# -----------------------------------------------------------------------------


def _maybe_cuda_sync() -> None:
    try:
        import torch

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
    except Exception:
        # torch not importable or other failure; ignore.
        pass


def _read_gpu_metrics(run_dir: Path) -> str:
    """
    Best-effort read of gpu_mem_peak_mb.allocated/reserved from metrics.json.

    Returns a short string like "gpu=123/456MB" or "".
    """
    path = run_dir / "metrics.json"
    if not path.is_file():
        return ""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            metrics = data.get("metrics", data)
        else:
            return ""
        gpu = metrics.get("gpu_mem_peak_mb", {})
        if isinstance(gpu, (int, float)):
            # legacy single value
            return f"gpu={float(gpu):.0f}MB"
        if isinstance(gpu, dict):
            a = gpu.get("allocated") or gpu.get("alloc") or gpu.get("used")
            r = gpu.get("reserved") or gpu.get("total")
            if isinstance(a, (int, float)) and isinstance(r, (int, float)):
                return f"gpu={float(a):.0f}/{float(r):.0f}MB"
            if isinstance(a, (int, float)):
                return f"gpu={float(a):.0f}MB"
    except Exception:
        return ""
    return ""


# -----------------------------------------------------------------------------
# Status line rendering
# -----------------------------------------------------------------------------


def _colorize(text: str, color: str, enable: bool) -> str:
    if not enable:
        return text
    codes = {"red": "31", "green": "32", "yellow": "33"}
    code = codes.get(color)
    if not code:
        return text
    return f"\033[{code}m{text}\033[0m"


def _render_line(
    iter_idx: int,
    resid: float,
    window_vals: List[float],
    paused: bool,
    write_every: Optional[int],
    lag_s: float,
    age_s: float,
    gpu_str: str,
    tty: bool,
    use_utf8: bool,
) -> str:
    # Trend color: based on last few residuals.
    trend_color = "yellow"
    if len(window_vals) >= 4 and all(math.isfinite(v) for v in window_vals[-4:]):
        first = window_vals[-4]
        last = window_vals[-1]
        tol = max(1e-12, abs(first) * 1e-3)
        if last < first - tol:
            trend_color = "green"
        elif last > first + tol:
            trend_color = "red"

    spark = _make_sparkline(window_vals, width=20, use_utf8=use_utf8)

    if math.isfinite(resid):
        resid_str = f"{resid: .3e}"
    else:
        resid_str = "   nan   "

    iter_str = f"{iter_idx:5d}" if iter_idx >= 0 else "    ?"

    # Write_every: None => "auto"
    if write_every is None or write_every <= 0:
        we_str = "auto"
    else:
        we_str = str(write_every)

    paused_str = "True" if paused else "False"

    lag_str = f"{lag_s:.1f}s" if lag_s >= 0.0 and math.isfinite(lag_s) else "n/a"
    age_str = f"{age_s:.1f}s" if age_s >= 0.0 and math.isfinite(age_s) else "n/a"

    trend_colored = _colorize(spark, trend_color, tty)

    parts = [
        f"iter={iter_str}",
        f"resid={resid_str}",
        f"trend=[{trend_colored}]",
        f"paused={paused_str}",
        f"every={we_str}",
        f"lag={lag_str}",
        f"updated={age_str}",
    ]
    if gpu_str:
        parts.append(gpu_str)

    line = "  ".join(parts)
    return line


# -----------------------------------------------------------------------------
# Main live console loop
# -----------------------------------------------------------------------------


def live_console(run_dir: str, refresh_hz: float = 4.0, window: int = 200) -> int:
    """
    Run a live residual console for a given run directory.

    Returns:
        0 on clean exit,
        1 if run_dir is invalid (non-directory).
    """
    try:
        refresh_hz = float(refresh_hz)
    except Exception:
        refresh_hz = 4.0
    if refresh_hz <= 0.1:
        refresh_hz = 0.1
    interval = 1.0 / refresh_hz

    run_path = Path(run_dir).expanduser()
    if not run_path.exists() or not run_path.is_dir():
        print(f"[live] Invalid run dir: {run_path}", file=sys.stderr)
        return 1

    events_log = run_path / "events.jsonl"
    evidence_log = run_path / "evidence_log.jsonl"
    log_path = events_log if events_log.exists() else evidence_log
    tail = TailState(path=log_path)
    win = ResidualWindow.create(maxlen=max(10, int(window) if window > 0 else 200))
    ctrl = ControlWriter(run_path, min_interval=0.1)

    is_tty = _is_tty(sys.stdout)
    use_utf8 = _supports_utf8()
    warned_waiting = False

    paused = False
    write_every: Optional[int] = None  # unknown; we only adjust via controls

    last_print = 0.0
    quiet_start = time.time()
    last_progress_ts = 0.0

    # Non-interactive: only status at ~1 Hz; exit after long inactivity.
    non_tty_print_interval = 5.0
    non_tty_quiet_timeout = 15 * 60.0  # 15 minutes

    # Instruction header (TTY only)
    if is_tty:
        msg = (
            "Live console keys: "
            "[p]ause  [r]esume  [s]top  [+]/[-] write_every  [m]ark  [q]uit"
        )
        print(msg, flush=True)

    try:
        while True:
            loop_start = time.time()

            # GPU sync (best-effort)
            _maybe_cuda_sync()

            preferred_log = events_log if events_log.exists() else evidence_log
            if preferred_log != tail.path:
                tail.close()
                tail.path = preferred_log
                tail.inode = None
                tail.offset = 0
            log_path = tail.path

            # Tail log for new residuals
            _tail_jsonl(tail, win)
            if tail.last_ts > 0:
                last_progress_ts = tail.last_ts
                quiet_start = loop_start  # reset quiet timer on progress

            # If log missing, print once and keep waiting
            if tail.fh is None and not log_path.exists():
                if not warned_waiting:
                    print(f"[live] Waiting for log at {log_path} ...", file=sys.stderr)
                    warned_waiting = True
            elif tail.fh is None and log_path.exists():
                # Try reopen soon; message once
                if not warned_waiting:
                    print(f"[live] Opening log {log_path} ...", file=sys.stderr)
                    warned_waiting = True

            # Compute lag and age
            now = loop_start
            if last_progress_ts > 0:
                lag_s = max(0.0, now - last_progress_ts)
                age_s = lag_s
            else:
                lag_s = float("nan")
                age_s = float("nan")

            # GPU metrics (cached occasionally)
            gpu_str = _read_gpu_metrics(run_path)

            # Render status line
            vals = win.as_list()
            resid = tail.last_resid if math.isfinite(tail.last_resid) else (vals[-1] if vals else math.nan)
            status = _render_line(
                iter_idx=tail.last_iter,
                resid=resid,
                window_vals=vals,
                paused=paused,
                write_every=write_every,
                lag_s=lag_s,
                age_s=age_s,
                gpu_str=gpu_str,
                tty=is_tty,
                use_utf8=use_utf8,
            )

            if is_tty:
                # Dynamic single-line output.
                try:
                    sys.stdout.write("\r" + status[: max(0, os.get_terminal_size().columns - 1)])
                except Exception:
                    sys.stdout.write("\r" + status)
                sys.stdout.flush()
            else:
                # Non-TTY: periodic summaries.
                if now - last_print >= non_tty_print_interval:
                    print(status, flush=True)
                    last_print = now

            # Key handling (TTY only)
            if is_tty:
                keys = _read_keys()
                for key in keys:
                    if key == "q":
                        if is_tty:
                            sys.stdout.write("\n[live] Quit.\n")
                            sys.stdout.flush()
                        return 0
                    elif key == "p":
                        paused = not paused
                        ctrl.write({"pause": bool(paused)})
                    elif key == "r":
                        paused = False
                        ctrl.write({"pause": False})
                    elif key == "s":
                        ctrl.write({"terminate": True})
                        if is_tty:
                            sys.stdout.write("\n[live] Terminate requested.\n")
                            sys.stdout.flush()
                    elif key == "+":
                        if write_every is None or write_every <= 1:
                            write_every = 1
                        else:
                            write_every = max(1, write_every // 2)
                        ctrl.write({"write_every": int(write_every)})
                    elif key == "-":
                        if write_every is None:
                            write_every = 5
                        else:
                            write_every = max(1, write_every * 2)
                        ctrl.write({"write_every": int(write_every)})
                    elif key == "m":
                        label = _prompt_label()
                        ctrl.write({"snapshot": label})
                    # ignore other keys

            # Non-interactive auto-exit on long inactivity
            if not is_tty:
                if (now - quiet_start) > non_tty_quiet_timeout:
                    print("[live] No progress for a while; exiting.", file=sys.stderr)
                    return 0

            # Sleep remaining time in this refresh interval
            elapsed = time.time() - loop_start
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        if is_tty:
            print("\n[live] Interrupted.", file=sys.stderr)
        return 0
    except Exception as exc:
        # Defensive: never crash; just report once and exit.
        try:
            print(f"\n[live] Error: {exc}", file=sys.stderr)
        except Exception:
            pass
        return 0
    finally:
        tail.close()
        if is_tty:
            try:
                sys.stdout.write("\n")
                sys.stdout.flush()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Electrodrive live residual console (JSONL tail + controls)."
    )
    parser.add_argument(
        "--run",
        required=True,
        help="Path to run directory (containing evidence_log.jsonl and control.json).",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=4.0,
        help="Refresh frequency in Hz (default: 4.0).",
    )
    parser.add_argument(
        "--win",
        type=int,
        default=200,
        help="Rolling window length for residual trend sparkline (default: 200).",
    )

    args = parser.parse_args()
    try:
        return live_console(args.run, refresh_hz=args.hz, window=args.win)
    except Exception as exc:
        # Ultimate guard: never crash when invoked as CLI.
        try:
            print(f"[live] Fatal error: {exc}", file=sys.stderr)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
