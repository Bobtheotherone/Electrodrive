"""
REST API routes for ResearchED.

Design Doc references (see "ResearchED GUI Design Document (Updated)"):

- FR-1: workflow launch (server must be able to start subprocess workflows)
- FR-2: presets persisted to disk (~/.researched/presets/*.json)
- FR-3: run directory contract + artifact browsing
- FR-4: robust log ingestion must handle events.jsonl vs evidence_log.jsonl drift
- FR-6: control panel uses control.json protocol (must call write_controls + schema)

Dependency policy: FastAPI is an optional extra; this module avoids importing it
at import time. Use get_api_router() to construct the router.
"""

import asyncio
import difflib
import json
import inspect
import os
import platform
import re
import socket
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import __version__

# Preset names should be filesystem-safe on Linux/macOS/Windows (no '*' etc.).
_PRESET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Manifest filenames:
# - manifest.json may be produced/overwritten by existing Electrodrive CLIs (e.g. electrodrive.cli solve).
# - We keep a ResearchED-owned copy so UI-required fields (started_at/status/inputs/outputs) are never lost.
_RESEARCHED_MANIFEST_NAME = "manifest.researched.json"

# GUI-generated events (kept separate to avoid concurrent writes with subprocess logs).
_RESEARCHED_EVENTS_JSONL = "researched_events.jsonl"



def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_iso_from_epoch(ts: float) -> str:
    """Convert epoch seconds to an ISO-8601 UTC timestamp string."""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return _utc_iso_now()


def _write_controls_compat(run_dir: Path, updates: Dict[str, Any] | None = None, *, merge: bool = True) -> Any:
    """
    Call electrodrive.live.controls.write_controls across Electrodrive versions.

    Some repo versions accept `seq_increment`, others don't. This wrapper tries a
    signature-aware call first, then falls back through common call patterns.
    """
    from electrodrive.live.controls import write_controls  # type: ignore

    u: Dict[str, Any] = dict(updates or {})

    # Signature-aware best effort.
    try:
        sig = inspect.signature(write_controls)
        params = sig.parameters
        args: list[Any] = [run_dir]
        kwargs: Dict[str, Any] = {}

        if "updates" in params:
            kwargs["updates"] = u
        else:
            args.append(u)

        if "merge" in params:
            kwargs["merge"] = merge

        if "seq_increment" in params:
            kwargs["seq_increment"] = True

        return write_controls(*args, **kwargs)
    except TypeError:
        pass
    except Exception:
        pass

    # Brute-force fallbacks.
    attempts: list[tuple[tuple[Any, ...], Dict[str, Any]]] = [
        ((run_dir,), {"updates": u, "merge": merge, "seq_increment": True}),
        ((run_dir,), {"updates": u, "merge": merge}),
        ((run_dir,), {"updates": u}),
        ((run_dir, u), {"merge": merge, "seq_increment": True}),
        ((run_dir, u), {"merge": merge}),
        ((run_dir, u), {}),
        ((run_dir,), {}),
    ]
    last: Exception | None = None
    for args, kwargs in attempts:
        try:
            return write_controls(*args, **kwargs)
        except TypeError as exc:
            last = exc
            continue
        except Exception as exc:
            last = exc
            break
    if last is not None:
        raise last
    return write_controls(run_dir, u)
def _presets_dir() -> Path:
    # FR-2: presets saved to disk.
    env = os.getenv("RESEARCHED_PRESETS_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".researched" / "presets"


def _safe_json_load(path: Path) -> Any:
    try:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_json_dump(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
    except Exception:
        try:
            return json.dumps(str(obj), ensure_ascii=False)
        except Exception:
            return "{}"



def _load_manifest_any(run_dir: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Load a manifest dict from a run directory.

    Preference order:
      1) manifest.researched.json (ResearchED-owned, never overwritten by workflows)
      2) manifest.json (may be produced/overwritten by existing Electrodrive CLIs)

    Returns (manifest_dict_or_none, filename_used).
    """
    for name in (_RESEARCHED_MANIFEST_NAME, "manifest.json"):
        obj = _safe_json_load(run_dir / name)
        if isinstance(obj, dict):
            return obj, name
    return None, ""


def _ensure_manifest_v1_shape(man: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce minimal manifest v1 *shape* (types + required keys) without clobbering
    workflow-owned extra fields.
    """
    if not isinstance(man.get("schema_version"), int):
        man["schema_version"] = 1

    # Required mapping blocks must never be null/non-dict.
    for k in ("git", "env", "inputs", "outputs", "gate"):
        if not isinstance(man.get(k), dict):
            man[k] = {}

    # Required scalars.
    if not isinstance(man.get("run_id"), str) or not str(man.get("run_id") or "").strip():
        man["run_id"] = str(man.get("run_id") or "")
    if not isinstance(man.get("workflow"), str) or not str(man.get("workflow") or "").strip():
        man["workflow"] = str(man.get("workflow") or "unknown")

    if not isinstance(man.get("started_at"), str) or not str(man.get("started_at") or "").strip():
        man["started_at"] = _utc_iso_now()
    if "ended_at" not in man:
        man["ended_at"] = None

    if not isinstance(man.get("status"), str) or not str(man.get("status") or "").strip():
        man["status"] = "running"

    # spec_digest must be a dict (not list/null).
    if not isinstance(man.get("spec_digest"), dict):
        man["spec_digest"] = {}

    # Fill git defaults (prevents frontend key errors).
    git = man["git"]
    git.setdefault("sha", None)
    git.setdefault("branch", None)
    git.setdefault("dirty", None)
    git.setdefault("diff_summary", None)

    # Fill gate defaults (design doc says keys always exist).
    gate = man["gate"]
    gate.setdefault("gate1_status", None)
    gate.setdefault("gate2_status", None)
    gate.setdefault("gate3_status", None)
    gate.setdefault("structure_score", None)
    gate.setdefault("novelty_score", None)

    # Ensure researched is dict if present.
    if "researched" in man and not isinstance(man.get("researched"), dict):
        man["researched"] = {"_coerced_from": type(man.get("researched")).__name__}

    return man



def _atomic_write_text(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}-{time.time_ns()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _atomic_write_json(path: Path, obj: Any) -> None:
    _atomic_write_text(path, _safe_json_dump(obj))


def _touch(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8"):
            pass
    except Exception:
        pass


def _ensure_events_evidence_bridge(run_dir: Path) -> None:
    """
    Best-effort compatibility bridge: ensure both events.jsonl and evidence_log.jsonl exist.

    Design Doc §1.4 (recommended): create a symlink or link so legacy tools that tail
    evidence_log.jsonl don't go blank when JsonlLogger writes events.jsonl.
    """
    events = run_dir / "events.jsonl"
    evidence = run_dir / "evidence_log.jsonl"

    _touch(events)
    if evidence.exists():
        return

    # Try: relative symlink -> hardlink -> (last resort) empty file.
    try:
        os.symlink("events.jsonl", evidence)  # relative target
        return
    except Exception:
        pass
    try:
        os.link(events, evidence)
        return
    except Exception:
        pass

    # Last resort: create an empty file so callers at least see it.
    _touch(evidence)


def _sync_events_evidence_bridge(run_dir: Path) -> None:
    """
    Best-effort post-run sync for legacy filename drift.

    If exactly one of events.jsonl / evidence_log.jsonl has content and the other is
    missing or empty, copy the non-empty file to the empty/missing one.

    We only overwrite when the destination is empty, to avoid clobbering workflows
    that genuinely write distinct streams to the legacy filename.

    Design Doc §1.4 (recommended): bridge at run start/end.
    """
    events = run_dir / "events.jsonl"
    evidence = run_dir / "evidence_log.jsonl"

    # Ensure both exist at least as empty files.
    if not events.exists():
        _touch(events)
    if not evidence.exists():
        _ensure_events_evidence_bridge(run_dir)

    try:
        events_size = int(events.stat().st_size) if events.is_file() else 0
    except Exception:
        events_size = 0
    try:
        evidence_size = int(evidence.stat().st_size) if evidence.is_file() else 0
    except Exception:
        evidence_size = 0

    # Copy only into an empty destination.
    try:
        if events_size == 0 and evidence_size > 0 and evidence.is_file():
            shutil.copyfile(str(evidence), str(events))
        elif evidence_size == 0 and events_size > 0 and events.is_file():
            # If evidence is a symlink/hardlink to events, this is a no-op; safe.
            shutil.copyfile(str(events), str(evidence))
    except Exception:
        return

def _safe_jsonl_iter(path: Path, *, limit: int = 5000) -> Iterable[Dict[str, Any]]:
    """
    Robust JSONL reader.

    - Ignores malformed lines.
    - Tolerates partial writes.
    - Never raises to caller.

    Design Doc FR-4.
    """
    if not path.is_file():
        return
    n = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if n >= limit:
                    return
                line = (line or "").strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict):
                    n += 1
                    yield rec
    except Exception:
        return


def _is_run_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    markers = [
        _RESEARCHED_MANIFEST_NAME,
        "manifest.json",
        "metrics.json",
        "events.jsonl",
        "evidence_log.jsonl",
        "train_log.jsonl",
        "metrics.jsonl",
        "discovery_manifest.json",
        "discovered_system.json",
        "control.json",
    ]
    try:
        if any((p / m).is_file() for m in markers):
            return True
        viz = p / "viz"
        if viz.is_dir():
            # Any PNG indicates a viz-producing run.
            try:
                next(viz.glob("*.png"))
                return True
            except StopIteration:
                pass
        return False
    except Exception:
        return False


def _iter_run_dirs(runs_root: Path, *, max_depth: int = 4) -> List[Path]:
    """
    Find run directories under runs_root.

    We scan breadth-first to a bounded depth and stop descending once a run dir
    is identified (to avoid treating artifacts/ subfolders as runs).

    Design Doc FR-3.
    """
    runs: List[Path] = []
    root = runs_root.expanduser()
    try:
        root = root.resolve()
    except Exception:
        pass

    if not root.exists() or not root.is_dir():
        return runs

    # Directories we never descend into.
    skip_names = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "viz",
        "plots",
        "artifacts",
        "node_modules",
    }

    queue: List[Tuple[Path, int]] = [(root, 0)]
    seen: set[str] = set()

    while queue:
        cur, depth = queue.pop(0)
        key = str(cur)
        if key in seen:
            continue
        seen.add(key)

        if depth > 0 and _is_run_dir(cur):
            runs.append(cur)
            continue  # do not descend further

        if depth >= max_depth:
            continue

        try:
            for child in cur.iterdir():
                if not child.is_dir():
                    continue
                if child.name in skip_names:
                    continue
                queue.append((child, depth + 1))
        except Exception:
            continue

    return runs


def _classify_workflow(run_dir: Path, manifest: Mapping[str, Any] | None) -> str:
    # Heuristic classification aligned to Design Doc FR-1/FR-3; best-effort.
    if (run_dir / "discovered_system.json").is_file() or (run_dir / "discovery_manifest.json").is_file():
        return "images_discover"
    if (run_dir / "train_log.jsonl").is_file() or (run_dir / "metrics.jsonl").is_file():
        return "learn_train"
    if isinstance(manifest, Mapping):
        wf = manifest.get("workflow")
        if isinstance(wf, str) and wf.strip():
            return wf.strip()
        # Older solve manifests often have planner or solver_mode_effective keys.
        if "planner" in manifest or "solver_mode_effective" in manifest:
            return "solve"
    # FMM runs often emit events.jsonl but no manifest; leave unknown.
    return "unknown"


def _run_id_from_manifest_or_dir(run_dir: Path, manifest: Mapping[str, Any] | None) -> str:
    if manifest:
        for k in ("run_id", "id", "uuid"):
            v = manifest.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return run_dir.name


def _run_status_from_manifest(manifest: Mapping[str, Any] | None) -> Optional[str]:
    if not manifest:
        return None

    # Prefer ResearchED lifecycle when present (queued/starting/running/success/error/killed/canceled).
    try:
        r = manifest.get("researched")
        if isinstance(r, Mapping):
            v = r.get("internal_status") or r.get("phase")
            if isinstance(v, str) and v.strip():
                return v.strip()
    except Exception:
        pass

    for k in ("status", "run_status", "state"):
        v = manifest.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def _ts_from_manifest(manifest: Mapping[str, Any] | None, key: str) -> Optional[str]:
    if not manifest:
        return None
    v = manifest.get(key)
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def _run_summary(run_dir: Path) -> Dict[str, Any]:
    manifest, manifest_src = _load_manifest_any(run_dir)

    workflow = _classify_workflow(run_dir, manifest)
    run_id = _run_id_from_manifest_or_dir(run_dir, manifest)
    status = _run_status_from_manifest(manifest)

    started_at = _ts_from_manifest(manifest, "started_at")
    ended_at = _ts_from_manifest(manifest, "ended_at")

    # Fall back to filesystem times if manifest timestamps missing.
    if started_at is None:
        try:
            started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_dir.stat().st_mtime))
        except Exception:
            started_at = None

    has_viz = False
    try:
        viz_dir = run_dir / "viz"
        if viz_dir.is_dir():
            has_viz = any(viz_dir.glob("*.png"))
    except Exception:
        has_viz = False

    has_events = (run_dir / "events.jsonl").is_file()
    has_evidence = (run_dir / "evidence_log.jsonl").is_file()
    has_train = (run_dir / "train_log.jsonl").is_file()
    has_metrics_jsonl = (run_dir / "metrics.jsonl").is_file()
    has_metrics = (run_dir / "metrics.json").is_file()

    return {
        "run_id": run_id,
        "workflow": workflow,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "path": str(run_dir),
        "manifest_file": manifest_src,
        "has_viz": bool(has_viz),
        "has_events": bool(has_events),
        "has_evidence": bool(has_evidence),
        "has_train_log": bool(has_train),
        "has_metrics_jsonl": bool(has_metrics_jsonl),
        "has_metrics": bool(has_metrics),
    }


def _index_runs(runs_root: Path) -> List[Dict[str, Any]]:
    runs = [_run_summary(rd) for rd in _iter_run_dirs(runs_root)]
    # Most recent first (best-effort).
    try:
        runs.sort(key=lambda r: str(r.get("started_at") or ""), reverse=True)
    except Exception:
        pass
    return runs


def _resolve_run_dir(runs_root: Path, run_id: str) -> Optional[Path]:
    # Fast path: directory name matches.
    candidate = (runs_root / run_id).expanduser()
    try:
        if candidate.is_dir() and _is_run_dir(candidate):
            return candidate
    except Exception:
        pass

    # Slow path: scan and match manifest run_id.
    for rd in _iter_run_dirs(runs_root):
        manifest, _ = _load_manifest_any(rd)
        rid = _run_id_from_manifest_or_dir(rd, manifest) if manifest else rd.name
        if rid == run_id or rd.name == run_id:
            return rd
    return None


def _safe_relpath(relpath: str) -> bool:
    """
    Basic guard: no absolute paths, no parent traversal, no drive letters.

    Used for artifact download paths.
    """
    if not relpath:
        return False
    # Block Windows drive letters and other colon usages for safety.
    if ":" in relpath:
        return False

    p = Path(relpath)
    try:
        if p.is_absolute():
            return False
    except Exception:
        # In rare cases Path.is_absolute can raise for weird paths; treat as unsafe.
        return False

    for part in p.parts:
        if part in ("", ".", ".."):
            return False
    return True


def _list_artifacts(run_dir: Path, *, recursive: bool, max_depth: int = 6) -> List[Dict[str, Any]]:
    """
    List files under run_dir, returning relpath/size/mtime/is_dir.

    Robust, never raises (Design Doc FR-3).
    """
    out: List[Dict[str, Any]] = []

    def add_path(p: Path) -> None:
        try:
            st = p.stat()
            rel = str(p.relative_to(run_dir))
            out.append(
                {
                    "path": rel,
                    "is_dir": p.is_dir(),
                    "size": int(st.st_size) if p.is_file() else 0,
                    "mtime": float(st.st_mtime),
                }
            )
        except Exception:
            return

    if not recursive:
        try:
            for child in sorted(run_dir.iterdir(), key=lambda x: x.name):
                add_path(child)
        except Exception:
            return out
        return out

    # Recursive (bounded)
    queue: List[Tuple[Path, int]] = [(run_dir, 0)]
    while queue:
        cur, depth = queue.pop(0)
        if depth > max_depth:
            continue
        try:
            for child in sorted(cur.iterdir(), key=lambda x: x.name):
                add_path(child)
                if child.is_dir():
                    # Avoid descending into extremely large node_modules directories.
                    if child.name == "node_modules":
                        continue
                    queue.append((child, depth + 1))
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Cross-process safe JSONL append helpers
# ---------------------------------------------------------------------------

def _lock_file_handle(fh) -> object | None:
    """Best-effort cross-platform advisory lock for append operations."""
    try:
        import fcntl  # type: ignore

        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        return ("fcntl", fcntl)
    except Exception:
        pass

    try:
        import msvcrt  # type: ignore

        # Lock first byte of the file. All writers must lock the same region.
        try:
            fh.seek(0)
        except Exception:
            pass
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            return ("msvcrt", msvcrt)
        except Exception:
            return None
    except Exception:
        return None


def _unlock_file_handle(fh, token: object | None) -> None:
    if token is None:
        return
    try:
        kind, mod = token  # type: ignore[misc]
    except Exception:
        return
    try:
        if kind == "fcntl":
            mod.flock(fh.fileno(), mod.LOCK_UN)  # type: ignore[attr-defined]
            return
    except Exception:
        pass
    try:
        if kind == "msvcrt":
            try:
                fh.seek(0)
            except Exception:
                pass
            mod.locking(fh.fileno(), mod.LK_UNLCK, 1)  # type: ignore[attr-defined]
    except Exception:
        return


def _locked_append_bytes(path: Path, data: bytes) -> None:
    """Append bytes to a file with best-effort locking (cross-platform)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        with path.open("ab") as f:
            token = _lock_file_handle(f)
            try:
                f.write(data)
                try:
                    f.flush()
                except Exception:
                    pass
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
            finally:
                _unlock_file_handle(f, token)
    except Exception:
        return


def _append_jsonl_record(path: Path, rec: Dict[str, Any]) -> None:
    """Append a single JSON record as one JSONL line (best-effort)."""
    try:
        line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8") + b"\n"
    except Exception:
        return
    _locked_append_bytes(path, line)


def _log_gui_event(run_dir: Path, action: str, payload: Mapping[str, Any]) -> None:
    """
    Best-effort append a GUI-generated event into the run's structured log.

    Design Doc FR-6 acceptance: "Control actions are logged into the run’s events stream".

    We avoid importing heavy logging utilities here; instead we append a JSONL record
    directly to researched_events.jsonl.

    IMPORTANT: we do not write to events.jsonl, because the subprocess workflow
    may also write there, which risks corrupting JSONL lines with concurrent appends.
    """
    try:
        # Ensure both filenames exist for downstream tailers.
        _ensure_events_evidence_bridge(run_dir)

        rec: Dict[str, Any] = {
            "ts": _utc_iso_now(),
            "level": "info",
            "event": "researched_gui",
            "action": str(action),
            "payload": dict(payload),
        }
        _append_jsonl_record(run_dir / _RESEARCHED_EVENTS_JSONL, rec)
    except Exception:
        # Never propagate logging failures.
        return


def _upgrade_summary(run_dir: Path) -> Dict[str, Any]:
    """
    Minimal "Experimental Upgrades" summary for UI stability.

    Design Doc FR-9 (minimal v1): return structured JSON even if empty.
    """
    notes: List[str] = []
    ingested: List[str] = []
    cond_min: Optional[float] = None
    cond_max: Optional[float] = None
    has_dist = False

    # Probe both log filenames (Design Doc §1.4 + FR-4).
    for name in ("events.jsonl", "evidence_log.jsonl"):
        p = run_dir / name
        if not p.is_file():
            continue
        ingested.append(name)
        for rec in _safe_jsonl_iter(p, limit=2000):
            # Look for conditioning telemetry keys.
            for key in ("col_norm_min", "col_norm_max"):
                if key in rec and isinstance(rec.get(key), (int, float)):
                    if key == "col_norm_min":
                        cond_min = float(rec[key])
                    else:
                        cond_max = float(rec[key])
            # More detailed telemetry indicates "not limited".
            if any(k in rec for k in ("col_norm_p50", "col_norm_p95", "col_norm_p99", "col_norm_hist", "hist_bins", "hist_counts")):
                has_dist = True

    if has_dist:
        limited = False
    elif cond_min is not None or cond_max is not None:
        notes.append("Conditioning telemetry limited to min/max (no distribution/histogram).")
        limited = True
    else:
        notes.append("No conditioning telemetry detected in logs.")
        limited = True

    return {
        "limited_telemetry": bool(limited),
        "notes": notes,
        "logs_ingested": ingested,
        "conditioning": {
            "col_norm_min": cond_min,
            "col_norm_max": cond_max,
            "has_distribution": bool(has_dist),
        },
    }


# ---------------------------------------------------------------------------
# Process registry / RunManager-lite (Design Doc FR-1 / FR-3)
# ---------------------------------------------------------------------------


@dataclass
class _ProcRecord:
    run_id: str
    workflow: str
    run_dir: Path
    argv: List[str]
    env: Dict[str, str]
    cwd: Optional[str]
    started_at: float
    proc: Any
    stdout_path: Optional[Path] = None
    stderr_path: Optional[Path] = None
    exit_code: Optional[int] = None
    ended_at: Optional[float] = None
    task: Optional[asyncio.Task] = None
    requested_terminate: bool = False
    requested_kill: bool = False


def _get_proc_registry(app) -> Tuple[Dict[str, _ProcRecord], threading.Lock]:
    reg = getattr(app.state, "_researched_proc_registry", None)
    lock = getattr(app.state, "_researched_proc_lock", None)
    if not isinstance(reg, dict) or lock is None:
        reg = {}
        lock = threading.Lock()
        app.state._researched_proc_registry = reg
        app.state._researched_proc_lock = lock
    return reg, lock


def _get_run_manager(app) -> Any:
    """Return the shared RunManager instance from app.state (or None if unavailable)."""
    rm = getattr(app.state, "run_manager", None)
    if rm is not None:
        return rm
    # Lazy fallback (should normally be created by app.create_app()).
    try:
        from .run_manager import RunManager  # stdlib-only
        runs_root = Path(getattr(app.state, "runs_root", "runs"))
        rm = RunManager(runs_root, max_parallel=1)
    except Exception:
        rm = None
    try:
        app.state.run_manager = rm
    except Exception:
        pass
    return rm


def _run_manager_record_to_process(rec: Any) -> Dict[str, Any]:
    """Convert a RunManager RunRecord into a UI-friendly process dict."""
    try:
        pid = int(getattr(rec, "pid", None)) if getattr(rec, "pid", None) is not None else None
    except Exception:
        pid = None
    status = getattr(rec, "status", None)
    try:
        status_s = str(status.value) if hasattr(status, "value") else str(status)
    except Exception:
        status_s = None
    try:
        returncode = int(getattr(rec, "returncode", None)) if getattr(rec, "returncode", None) is not None else None
    except Exception:
        returncode = None
    return {
        "run_id": getattr(rec, "run_id", None),
        "workflow": getattr(rec, "workflow", None),
        "pid": pid,
        "status": status_s,
        "running": bool(status_s in {"queued", "starting", "running"}),
        "returncode": returncode,
        "created_at": getattr(rec, "created_at", None),
        "started_at": getattr(rec, "started_at", None),
        "ended_at": getattr(rec, "ended_at", None),
        "command": getattr(rec, "command", None),
        "env": getattr(rec, "env", None),
        "error": getattr(rec, "error", None),
        "out_dir": str(getattr(rec, "out_dir", "")),
    }


def _find_repo_root(start: Path) -> Optional[Path]:
    try:
        start = start.resolve()
    except Exception:
        pass
    for p in [start, *start.parents]:
        try:
            if (p / ".git").exists():
                return p
        except Exception:
            continue
    return None


def _git_info(repo_root: Optional[Path]) -> Optional[Dict[str, Any]]:
    if repo_root is None:
        return None

    def run_git(args: List[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(
                ["git", *args],
                cwd=str(repo_root),
                stderr=subprocess.DEVNULL,
                text=True,
            )
            s = (out or "").strip()
            return s or None
        except Exception:
            return None

    sha = run_git(["rev-parse", "HEAD"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    dirty = False
    try:
        por = run_git(["status", "--porcelain"])
        dirty = bool(por)
    except Exception:
        dirty = False

    diff_summary: Optional[str] = None
    # Keep this intentionally small (avoid huge diffs in manifest).
    stat = run_git(["diff", "--stat"])
    if stat:
        diff_summary = "\n".join(stat.splitlines()[:50])

    if not sha and not branch:
        return None
    return {"sha": sha, "branch": branch, "dirty": bool(dirty), "diff_summary": diff_summary}


def _env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "host": socket.gethostname(),
    }
    # Best-effort torch info (repo already depends on torch; but do not hard-fail).
    try:
        import torch  # type: ignore

        info["torch_version"] = getattr(torch, "__version__", "unknown")
        try:
            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False
        info["device"] = "cuda" if cuda_ok else "cpu"
        try:
            info["dtype"] = str(torch.get_default_dtype())
        except Exception:
            pass
    except Exception:
        info["torch_version"] = "unavailable"
        info["device"] = "unknown"
    return info


def _normalize_launch_argv(argv: Sequence[Any]) -> List[str]:
    if not isinstance(argv, (list, tuple)):
        raise ValueError("argv must be a list of strings")
    out: List[str] = []
    for x in argv:
        if x is None:
            continue
        s = str(x)
        if "\x00" in s:
            raise ValueError("argv contains NUL byte")
        out.append(s)
    if not out:
        raise ValueError("argv must be non-empty")
    return out


def _prefix_python_if_needed(argv: List[str]) -> List[str]:
    """
    Allow callers to send either:
      - ["-m", "electrodrive.cli", "solve", ...]  (preferred)
      - [sys.executable, "-m", ...]              (also accepted)

    We always execute without shell=True.
    """
    first = argv[0]
    if os.path.basename(first).lower().startswith("python"):
        return argv
    if os.path.abspath(first) == os.path.abspath(sys.executable):
        return argv
    return [sys.executable, *argv]


def _enforce_out_flag(argv: List[str], out_flag: str, out_value: str) -> List[str]:
    """
    Ensure an argv list contains out_flag set to out_value.

    Supports:
      --out PATH
      --out=PATH
    """
    out_flag = str(out_flag).strip()
    if not out_flag:
        return argv

    out: List[str] = list(argv)
    # Exact token form
    for i, tok in enumerate(out):
        if tok == out_flag and i + 1 < len(out):
            out[i + 1] = out_value
            return out
        if tok.startswith(out_flag + "="):
            out[i] = out_flag + "=" + out_value
            return out

    # Not present: append
    out.extend([out_flag, out_value])
    return out


def _write_command_txt(run_dir: Path, argv: List[str], env: Mapping[str, str], cwd: Optional[str]) -> None:
    lines: List[str] = []
    lines.append("# ResearchED command (exact argv):")
    lines.append(" ".join(json.dumps(x) for x in argv))
    lines.append("")
    lines.append("# cwd:")
    lines.append(str(cwd or os.getcwd()))
    lines.append("")
    lines.append("# env overrides:")
    for k in sorted(env.keys()):
        v = env.get(k)
        lines.append(f"{k}={v}")
    lines.append("")
    _atomic_write_text(run_dir / "command.txt", "\n".join(lines))


def _init_run_dir(runs_root: Path, run_id: str) -> Path:
    run_dir = (runs_root / run_id).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    # Contract folders (FR-3)
    try:
        (run_dir / "artifacts").mkdir(exist_ok=True)
    except Exception:
        pass
    try:
        (run_dir / "plots").mkdir(exist_ok=True)
    except Exception:
        pass
    return run_dir


def _write_manifest_running(
    run_dir: Path,
    *,
    run_id: str,
    workflow: str,
    argv: List[str],
    env_overrides: Mapping[str, str],
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    git: Optional[Mapping[str, Any]],
    env_info: Mapping[str, Any],
) -> None:
    """
    Write the initial manifest(s) for a running subprocess-run.

    Alignment fixes:
    - Include schema_version=1 and spec_digest placeholders per §5.1.
    - Keep inputs.command and inputs.env_overrides stable for reproducibility.
    - Include ResearchED lifecycle fields in manifest['researched'] so the UI can
      always show a rich phase/status even if workflows overwrite manifest.json.
    """
    # Build a stable inputs block, allowing caller-provided inputs to extend it.
    inputs_block: Dict[str, Any] = {"command": list(argv), "env_overrides": dict(env_overrides)}
    try:
        for k, v in dict(inputs).items():
            if k in {"command", "env_overrides"}:
                continue
            inputs_block[k] = v
    except Exception:
        pass

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "workflow": workflow,
        "started_at": _utc_iso_now(),
        "ended_at": None,
        "status": "running",
        "error": None,
        "git": dict(git or {}),
        "env": dict(env_info),
        "inputs": inputs_block,
        "outputs": dict(outputs),
        "gate": {
            "gate1_status": None,
            "gate2_status": None,
            "gate3_status": None,
            "structure_score": None,
            "novelty_score": None,
        },
        "spec_digest": {},
        "researched": {
            "version": __version__,
            "schema_version": 1,
            "phase": "running",
            "internal_status": "running",
        },
    }
    manifest = _ensure_manifest_v1_shape(manifest)
    _atomic_write_json(run_dir / _RESEARCHED_MANIFEST_NAME, manifest)
    _atomic_write_json(run_dir / "manifest.json", manifest)


def _update_manifest_terminal(
    run_dir: Path,
    *,
    status: str,
    exit_code: Optional[int],
    run_id: Optional[str] = None,
    workflow: Optional[str] = None,
    started_at_iso: Optional[str] = None,
) -> None:
    """
    Update terminal fields in the run manifest(s).

    Important integration note:
    electrodrive CLI workflows may overwrite manifest.json with their own schema.
    We therefore:
      - always update the ResearchED-owned manifest.researched.json, and
      - merge terminal fields into manifest.json without removing existing keys.

    This keeps existing tooling compatible while satisfying the ResearchED UI contract.
    """
    ended_at = _utc_iso_now()

    for name in (_RESEARCHED_MANIFEST_NAME, "manifest.json"):
        path = run_dir / name
        man = _safe_json_load(path)
        if not isinstance(man, dict):
            man = {}

        # Populate stable identifiers if missing.
        if run_id and (not isinstance(man.get("run_id"), str) or not str(man.get("run_id", "")).strip()):
            man["run_id"] = str(run_id)
        if workflow and (not isinstance(man.get("workflow"), str) or not str(man.get("workflow", "")).strip()):
            man["workflow"] = str(workflow)

        # Ensure started_at is present (FR-3).
        if started_at_iso and (not isinstance(man.get("started_at"), str) or not str(man.get("started_at", "")).strip()):
            man["started_at"] = str(started_at_iso)

        # Terminal fields (always authoritative from the runner).
        man["ended_at"] = ended_at
        man["status"] = str(status)
        if exit_code is not None:
            try:
                man["exit_code"] = int(exit_code)
            except Exception:
                man["exit_code"] = None

        # Terminal stabilization: ensure v1 required keys exist so the UI contract is satisfied
        # even if a workflow overwrote manifest.json with a partial schema.
        if "schema_version" not in man:
            man["schema_version"] = 1

        # Ensure researched lifecycle block exists and reflects terminal state.
        r = man.get("researched")
        if not isinstance(r, dict):
            r = {}
        r.setdefault("version", __version__)
        r.setdefault("schema_version", 1)
        r["internal_status"] = str(status)
        r["phase"] = "finished"
        r["finalized_at"] = ended_at
        man["researched"] = r

        man = _ensure_manifest_v1_shape(man)
        _atomic_write_json(path, man)





def _safe_read_last_jsonl_record(path: Path, *, max_bytes: int = 2_000_000) -> Optional[Dict[str, Any]]:
    """
    Best-effort: read the last valid JSON object from a JSONL file.

    We tail a bounded byte window to avoid loading huge files.
    Returns None on any error.
    """
    try:
        if not path.is_file():
            return None
        size = int(path.stat().st_size)
        start = max(0, size - max_bytes)
        with path.open("rb") as f:
            f.seek(start)
            data = f.read(max_bytes + 1)
        text = data.decode("utf-8", errors="ignore")
        # Iterate lines from the end for "last record" semantics.
        for line in reversed(text.splitlines()):
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                return obj
        return None
    except Exception:
        return None


def _maybe_finalize_metrics_json(
    run_dir: Path,
    *,
    status: str,
    exit_code: Optional[int],
    started_at_epoch: Optional[float],
    ended_at_epoch: Optional[float],
) -> None:
    """
    Ensure metrics.json is non-empty when possible (Design Doc FR-3/FR-7).

    - If a workflow already wrote a non-empty metrics.json, we leave it untouched.
    - Otherwise, we write a small metrics.json with runtime/status/exit_code and
      best-effort scalar fields from the tail of metrics.jsonl/train_log.jsonl.
    """
    metrics_path = run_dir / "metrics.json"
    existing = _safe_json_load(metrics_path)
    if isinstance(existing, dict) and existing:
        return  # workflow produced metrics

    out: Dict[str, Any] = {}
    if isinstance(existing, dict):
        out.update(existing)  # likely empty

    # Basic terminal scalars.
    out.setdefault("status", str(status))
    if exit_code is not None:
        try:
            out.setdefault("exit_code", int(exit_code))
        except Exception:
            out.setdefault("exit_code", None)

    # Runtime (seconds).
    if isinstance(started_at_epoch, (int, float)) and isinstance(ended_at_epoch, (int, float)):
        try:
            rt = float(ended_at_epoch) - float(started_at_epoch)
            if rt == rt and rt >= 0.0:  # not NaN, not negative
                out.setdefault("runtime_s", rt)
        except Exception:
            pass

    # Best-effort: pull last JSONL record.
    for name in ("metrics.jsonl", "train_log.jsonl", "events.jsonl", "evidence_log.jsonl"):
        rec = _safe_read_last_jsonl_record(run_dir / name)
        if not rec:
            continue

        # Try to extract a few useful numeric scalars.
        for k, v in rec.items():
            if k in {"ts", "t", "level", "msg", "message"}:
                continue
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                out.setdefault(k, float(v) if isinstance(v, float) else v)
            elif isinstance(v, str) and len(v) <= 256:
                # occasionally useful: e.g. best_ckpt, phase labels
                if k.lower() in {"best_ckpt", "phase", "event", "type"}:
                    out.setdefault(k, v)

        # Add a last_event hint for dashboards.
        last_event = rec.get("event") or rec.get("msg") or rec.get("message")
        if isinstance(last_event, str) and last_event.strip():
            out.setdefault("last_log_event", last_event.strip())

        break  # first file with a record wins

    # Only write if we have something beyond an empty dict.
    try:
        _atomic_write_json(metrics_path, out)
    except Exception:
        return



async def _monitor_process(app, run_id: str) -> None:
    reg, lock = _get_proc_registry(app)
    with lock:
        rec = reg.get(run_id)
    if rec is None:
        return

    proc = rec.proc
    try:
        # Wait in a thread so we don't block the event loop.
        code = await asyncio.to_thread(proc.wait)
    except Exception:
        code = None

    now = time.time()
    with lock:
        rec.exit_code = code if isinstance(code, int) else None
        rec.ended_at = now

    # Update manifest.
    if code == 0:
        status = "success"
    else:
        status = "killed" if (rec.requested_terminate or rec.requested_kill) else "error"
    _update_manifest_terminal(
        rec.run_dir,
        status=status,
        exit_code=code if isinstance(code, int) else None,
        run_id=rec.run_id,
        workflow=rec.workflow,
        started_at_iso=_utc_iso_from_epoch(rec.started_at),
    )
    _sync_events_evidence_bridge(rec.run_dir)
    _maybe_finalize_metrics_json(
        rec.run_dir,
        status=status,
        exit_code=code if isinstance(code, int) else None,
        started_at_epoch=rec.started_at,
        ended_at_epoch=rec.ended_at,
    )

    # Close raw log handles if any were used.
    for attr in ("_stdout_fh", "_stderr_fh"):
        fh = getattr(rec, attr, None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
            try:
                setattr(rec, attr, None)
            except Exception:
                pass


def _refresh_proc_record(rec: _ProcRecord) -> None:
    """
    Poll the subprocess and, if it has exited, update the in-memory record + manifest.

    This makes the API robust even if background monitor tasks were missed (e.g. server restart).
    """
    if rec.exit_code is not None:
        return
    try:
        code = rec.proc.poll()
    except Exception:
        return
    if code is None:
        return

    try:
        rec.exit_code = int(code)
    except Exception:
        rec.exit_code = None
    rec.ended_at = time.time()

    if rec.exit_code == 0:
        status = "success"
    else:
        status = "killed" if (rec.requested_terminate or rec.requested_kill) else "error"
    _update_manifest_terminal(
        rec.run_dir,
        status=status,
        exit_code=rec.exit_code,
        run_id=rec.run_id,
        workflow=rec.workflow,
        started_at_iso=_utc_iso_from_epoch(rec.started_at),
    )
    _sync_events_evidence_bridge(rec.run_dir)
    _maybe_finalize_metrics_json(
        rec.run_dir,
        status=status,
        exit_code=rec.exit_code,
        started_at_epoch=rec.started_at,
        ended_at_epoch=rec.ended_at,
    )

    # Close raw log handles if any were used.
    for attr in ("_stdout_fh", "_stderr_fh"):
        fh = getattr(rec, attr, None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
            try:
                setattr(rec, attr, None)
            except Exception:
                pass


def _proc_info_dict(rec: _ProcRecord) -> Dict[str, Any]:
    pid: Optional[int]
    try:
        pid = int(getattr(rec.proc, "pid", None))
    except Exception:
        pid = None
    running = False
    try:
        running = rec.exit_code is None and rec.proc.poll() is None
    except Exception:
        running = rec.exit_code is None

    return {
        "run_id": rec.run_id,
        "workflow": rec.workflow,
        "pid": pid,
        "running": bool(running),
        "exit_code": rec.exit_code,
        "started_at": rec.started_at,
        "ended_at": rec.ended_at,
        "stdout_path": str(rec.stdout_path) if rec.stdout_path else None,
        "stderr_path": str(rec.stderr_path) if rec.stderr_path else None,
    }


# ---------------------------------------------------------------------------
# Log coverage (Design Doc FR-9.6)
# ---------------------------------------------------------------------------


def _maybe_parse_message_json(msg: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(msg, str):
        return None
    s = msg.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    if '"event"' not in s and '"msg"' not in s and '"iter"' not in s and '"step"' not in s:
        return None
    try:
        obj = json.loads(s)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _compute_log_coverage(run_dir: Path, *, limit_per_file: int = 5000) -> Dict[str, Any]:
    files = ["events.jsonl", "evidence_log.jsonl", _RESEARCHED_EVENTS_JSONL, "train_log.jsonl", "metrics.jsonl"]
    present = [f for f in files if (run_dir / f).is_file()]

    total_parsed = 0
    parse_by_file: Dict[str, int] = {}
    event_source_counts = {"event": 0, "msg": 0, "message": 0, "embedded_json": 0, "missing": 0}
    residual_field_counts: Dict[str, int] = {
        "resid": 0,
        "resid_precond": 0,
        "resid_true": 0,
        "resid_precond_l2": 0,
        "resid_true_l2": 0,
    }

    for fname in present:
        p = run_dir / fname
        n = 0
        for rec in _safe_jsonl_iter(p, limit=limit_per_file):
            n += 1
            total_parsed += 1

            # Event source accounting (FR-9.6)
            embedded = _maybe_parse_message_json(rec.get("msg", rec.get("message")))
            if isinstance(rec.get("event"), str) and rec.get("event"):
                event_source_counts["event"] += 1
            elif embedded and isinstance(embedded.get("event"), str) and embedded.get("event"):
                event_source_counts["embedded_json"] += 1
            elif isinstance(rec.get("msg"), str) and rec.get("msg"):
                event_source_counts["msg"] += 1
            elif isinstance(rec.get("message"), str) and rec.get("message"):
                event_source_counts["message"] += 1
            else:
                event_source_counts["missing"] += 1

            # Residual field detection (FR-9.6)
            for k in list(residual_field_counts.keys()):
                if k in rec and isinstance(rec.get(k), (int, float, str)):
                    residual_field_counts[k] += 1

        parse_by_file[fname] = n

    # Fix-it checklist (FR-9.6)
    warnings: List[str] = []
    if total_parsed == 0:
        warnings.append("No JSONL records parsed (logs may be missing or malformed).")

    if event_source_counts["event"] == 0 and (event_source_counts["msg"] > 0 or event_source_counts["embedded_json"] > 0):
        warnings.append("Records use msg/embedded-json for event names; older parsers expecting `event` may fail.")

    resid = residual_field_counts.get("resid", 0)
    pre = residual_field_counts.get("resid_precond", 0) + residual_field_counts.get("resid_precond_l2", 0)
    tru = residual_field_counts.get("resid_true", 0) + residual_field_counts.get("resid_true_l2", 0)
    if resid == 0 and (pre > 0 or tru > 0):
        warnings.append("Residuals are only in resid_precond/resid_true variants; tools expecting `resid` may miss curves.")

    if ("events.jsonl" in present) and ("evidence_log.jsonl" not in present):
        warnings.append("Only events.jsonl present; legacy tools tailing evidence_log.jsonl may miss events unless bridged.")
    if ("evidence_log.jsonl" in present) and ("events.jsonl" not in present):
        warnings.append("Only evidence_log.jsonl present; tools expecting events.jsonl may miss events unless bridged.")

    return {
        "logs_present": present,
        "total_records_parsed": int(total_parsed),
        "records_parsed_by_file": parse_by_file,
        "event_name_field_used": event_source_counts,
        "residual_fields_detected": residual_field_counts,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# FastAPI glue
# ---------------------------------------------------------------------------


def _require_fastapi() -> None:
    try:
        from fastapi import APIRouter  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "ResearchED API requires FastAPI (optional extra). Install with: pip install fastapi uvicorn"
        ) from exc


def get_api_router():
    """
    Construct and return the API router.

    Keeping this lazy ensures importing electrodrive.researched does not require FastAPI.
    """
    _require_fastapi()
    from fastapi import APIRouter, Body, HTTPException, Query, Request
    from fastapi.responses import FileResponse

    router = APIRouter()

    @router.get("/health")
    def health(request: Request) -> Dict[str, Any]:
        runs_root = getattr(request.app.state, "runs_root", None)
        db_path = getattr(request.app.state, "db_path", None)
        return {
            "ok": True,
            "runs_root": str(runs_root) if runs_root is not None else None,
            "db_path": str(db_path) if db_path is not None else None,
            "version": __version__,
        }


    # ------------------------------------------------------------------
    # Workflows registry (FR-1/FR-2)
    # ------------------------------------------------------------------

    @router.get("/workflows")
    def list_workflows() -> Dict[str, Any]:
        """List workflow templates that the UI can launch."""
        try:
            from .workflows import WORKFLOWS  # lazy import

            items = []
            for name in sorted(WORKFLOWS.keys()):
                try:
                    items.append(WORKFLOWS[name].describe())
                except Exception:
                    items.append({"name": name, "supports_controls": bool(getattr(WORKFLOWS[name], "supports_controls", False))})
            return {"ok": True, "workflows": items}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to enumerate workflows: {exc}")

    @router.post("/workflows/{name}/preview")
    async def preview_workflow(name: str, request: Request, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Preview the exact argv/env that would be used for a workflow launch (FR-2 Explain panel)."""
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="payload must be object")
        req = payload.get("request", payload)
        if not isinstance(req, dict):
            raise HTTPException(status_code=400, detail="request must be object")

        # Choose an out_dir placeholder (caller may override).
        runs_root = Path(getattr(request.app.state, "runs_root", "runs"))
        rid = payload.get("run_id")
        if isinstance(rid, str) and rid.strip():
            rid_s = rid.strip()
        else:
            rid_s = "RUN_ID"
        out_dir = runs_root / rid_s

        try:
            from .workflows import get_workflow  # lazy import

            wf = get_workflow(name)
            wf.validate_request(req)
            argv = wf.build_command(req, out_dir)
            env_overrides = wf.build_env(req, out_dir, rid_s)
            return {"ok": True, "workflow": name, "run_id": rid_s, "out_dir": str(out_dir), "argv": argv, "env": env_overrides}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown workflow: {name}")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # Runs library (FR-3)
    # ------------------------------------------------------------------

    @router.get("/runs")
    def list_runs(request: Request) -> List[Dict[str, Any]]:
        runs_root = Path(getattr(request.app.state, "runs_root", "runs"))
        return _index_runs(runs_root)

    @router.post("/runs")
    async def launch_run(request: Request, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """
        Launch a workflow subprocess and create a run directory.

        This is a minimal "RunManager-lite" endpoint (Design Doc FR-1/FR-3).
        The UI should pass a module argv list (no shell).

        Expected payload (minimal):
          {
            "workflow": "solve" | "images_discover" | "learn_train" | "fmm_suite" | "custom",
            "argv": ["-m", "electrodrive.cli", "solve", ...],   # preferred (server prefixes sys.executable)
            "env": {"EXTRA_ENV": "1"},                          # optional
            "cwd": "/path/to/cwd",                              # optional
            "run_id": "uuid-or-name",                           # optional (else auto uuid4)
            "out_flag": "--out",                                # optional; if provided, enforced to point at run_dir
            "inputs": {...}                                     # optional: spec_path/config for manifest
          }
        """
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="payload must be JSON object")

        workflow = payload.get("workflow", "custom")
        if not isinstance(workflow, str) or not workflow.strip():
            raise HTTPException(status_code=400, detail="workflow must be a non-empty string")
        workflow = workflow.strip()

        # Preferred path: RunManager-backed launches (Design Doc §3.2 RunManager).
        # If the client provides argv explicitly, we fall back to the legacy raw-subprocess launcher below.
        if "argv" not in payload:
            rm = _get_run_manager(request.app)
            if rm is None:
                raise HTTPException(status_code=500, detail="RunManager is not initialized")

            req = payload.get("request")
            if req is None:
                # Back-compat alias.
                req = payload.get("inputs", {})
            if not isinstance(req, dict):
                raise HTTPException(status_code=400, detail="request must be a JSON object")

            # Optional client-supplied run_id; otherwise generate.
            rid = payload.get("run_id")
            if isinstance(rid, str) and rid.strip():
                rid = rid.strip()
            else:
                rid = str(uuid.uuid4())

            # Optional explicit run_dir; default to <runs_root>/<run_id> for fast lookup.
            run_dir_raw = payload.get("run_dir")
            if isinstance(run_dir_raw, str) and run_dir_raw.strip():
                out_dir = Path(run_dir_raw).expanduser()
                if not out_dir.is_absolute():
                    out_dir = Path(getattr(request.app.state, "runs_root", "runs")) / out_dir
            else:
                runs_root = Path(getattr(request.app.state, "runs_root", "runs"))
                out_dir = runs_root / str(rid)

            try:
                run_id2 = rm.submit(workflow, req, run_dir=out_dir, run_id=str(rid))
                try:
                    rec = rm.get(run_id2)
                    proc_info = _run_manager_record_to_process(rec)
                except Exception:
                    proc_info = None
            except KeyError:
                raise HTTPException(status_code=400, detail=f"unknown workflow: {workflow}")
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc))

            return {
                "ok": True,
                "run_id": str(run_id2),
                "workflow": workflow,
                "path": str(out_dir),
                "process": proc_info,
            }


        argv_raw = payload.get("argv")
        if argv_raw is None:
            raise HTTPException(status_code=400, detail="argv is required (list of strings)")
        try:
            argv = _normalize_launch_argv(argv_raw)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        runs_root = Path(getattr(request.app.state, "runs_root", "runs"))
        try:
            runs_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to ensure runs_root: {exc}")

        run_id = payload.get("run_id")
        if isinstance(run_id, str) and run_id.strip():
            run_id = run_id.strip()
        else:
            run_id = str(uuid.uuid4())

        run_dir = _init_run_dir(runs_root, run_id)

        # Ensure log compatibility files exist early.
        _ensure_events_evidence_bridge(run_dir)

        # Run dir contract placeholders (FR-3): create early so the UI can rely on them.
        if not (run_dir / "metrics.json").is_file():
            _atomic_write_json(run_dir / "metrics.json", {})
        if not (run_dir / "report.html").is_file():
            _atomic_write_text(
                run_dir / "report.html",
                "<html><head><meta charset='utf-8'/><title>ResearchED Report</title></head>"
                "<body><h1>Report not generated yet</h1><p>This placeholder will be overwritten by ReportService.</p></body></html>",
            )

        # Environment variables required by repo integrations.
        env_overrides: Dict[str, str] = {}
        raw_env = payload.get("env", {})
        if isinstance(raw_env, dict):
            for k, v in raw_env.items():
                if k is None:
                    continue
                env_overrides[str(k)] = "" if v is None else str(v)

        env_overrides.setdefault("EDE_RUN_DIR", str(run_dir))
        env_overrides.setdefault("EDE_RUN_ID", str(run_id))
        env_overrides.setdefault("PYTHONUNBUFFERED", "1")

        # Optional: enforce output flag.
        out_flag = payload.get("out_flag")
        if isinstance(out_flag, str) and out_flag.strip():
            argv = _enforce_out_flag(argv, out_flag.strip(), str(run_dir))
        elif workflow == "solve":
            # Design Doc baseline: solve requires --out; enforce by default.
            argv = _enforce_out_flag(argv, "--out", str(run_dir))

        full_argv = _prefix_python_if_needed(argv)

        # Resolve working directory for the subprocess (and git capture).
        cwd_val = payload.get("cwd")
        cwd = str(cwd_val) if isinstance(cwd_val, str) and cwd_val.strip() else None

        # Manifest + command.txt at start (FR-3 acceptance criteria).
        repo_root = _find_repo_root(Path(cwd) if cwd else Path.cwd())
        git = _git_info(repo_root)
        env_info = _env_info()

        inputs = payload.get("inputs", {})
        if not isinstance(inputs, dict):
            inputs = {}

        outputs = {
            # Canonical v1 contract keys (Design Doc §5.1)
            "manifest_json": "manifest.json",
            "researched_manifest_json": _RESEARCHED_MANIFEST_NAME,
            "metrics_json": "metrics.json",
            "events_jsonl": "events.jsonl",
            "evidence_log_jsonl": "evidence_log.jsonl",
            "command_txt": "command.txt",
            "report_html": "report.html",
            "artifacts_dir": "artifacts",
            "plots_dir": "plots",
            "viz_dir": "viz",
            "stdout_log": "stdout.log",
            "stderr_log": "stderr.log",
            # Back-compat aliases used by some existing UI/prototypes:
            "manifest": "manifest.json",
            "researched_manifest": _RESEARCHED_MANIFEST_NAME,
            "metrics": "metrics.json",
            "events": "events.jsonl",
            "evidence": "evidence_log.jsonl",
            "stdout": "stdout.log",
            "stderr": "stderr.log",
        }

        _write_manifest_running(
            run_dir,
            run_id=str(run_id),
            workflow=workflow,
            argv=list(full_argv),
            env_overrides=env_overrides,
            inputs=inputs,
            outputs=outputs,
            git=git,
            env_info=env_info,
        )
        _write_command_txt(run_dir, list(full_argv), env_overrides, cwd)

        # Start process; capture raw stdout/stderr into run_dir for "raw logs" (FR-5).
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"

        stdout_opened = False
        stderr_opened = False
        try:
            stdout_fh = stdout_path.open("ab")
            stdout_opened = True
        except Exception:
            stdout_fh = subprocess.DEVNULL  # type: ignore[assignment]
        try:
            stderr_fh = stderr_path.open("ab")
            stderr_opened = True
        except Exception:
            stderr_fh = subprocess.DEVNULL  # type: ignore[assignment]

        env = os.environ.copy()
        env.update(env_overrides)

        try:
            proc = subprocess.Popen(
                full_argv,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_fh,
                stderr=stderr_fh,
            )
        except Exception as exc:
            # Close raw log handles we may have opened.
            for fh in (stdout_fh, stderr_fh):
                try:
                    fh.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
            # Mark run as error and return failure.
            _update_manifest_terminal(run_dir, status="error", exit_code=None, run_id=str(run_id), workflow=workflow)
            raise HTTPException(status_code=500, detail=f"failed to start process: {exc}")

        rec = _ProcRecord(
            run_id=str(run_id),
            workflow=workflow,
            run_dir=run_dir,
            argv=list(full_argv),
            env=dict(env_overrides),
            cwd=cwd,
            started_at=time.time(),
            proc=proc,
            stdout_path=stdout_path if stdout_opened else None,
            stderr_path=stderr_path if stderr_opened else None,
        )
        # Keep handles for later close (best-effort).
        setattr(rec, "_stdout_fh", stdout_fh)
        setattr(rec, "_stderr_fh", stderr_fh)

        reg, lock = _get_proc_registry(request.app)
        with lock:
            reg[str(run_id)] = rec

        # Monitor in background to update manifest on exit.
        rec.task = asyncio.create_task(_monitor_process(request.app, str(run_id)))

        return {
            "ok": True,
            "run_id": str(run_id),
            "workflow": workflow,
            "path": str(run_dir),
            "pid": int(proc.pid) if getattr(proc, "pid", None) is not None else None,
            "argv": list(full_argv),
        }

    @router.get("/runs/{run_id}")
    def get_run(run_id: str, request: Request) -> Dict[str, Any]:
        # Prefer in-memory RunManager tracking when available (faster + includes queued state).
        run_dir = None
        rm_proc_info = None
        rm = _get_run_manager(request.app)
        if rm is not None:
            try:
                rec_rm = rm.get(run_id)
                run_dir = Path(getattr(rec_rm, "out_dir"))
                rm_proc_info = _run_manager_record_to_process(rec_rm)
            except Exception:
                run_dir = None
                rm_proc_info = None

        runs_root = Path(getattr(request.app.state, "runs_root", "runs"))
        if run_dir is None:
            run_dir = _resolve_run_dir(runs_root, run_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")

        # Best-effort: ensure legacy log filenames are bridged for this run.
        _sync_events_evidence_bridge(run_dir)

        manifest, manifest_file = _load_manifest_any(run_dir)
        metrics = _safe_json_load(run_dir / "metrics.json")

        # Key artifacts: stable, UI-friendly list (FR-3).
        key_names = [
            "manifest.json",
            "metrics.json",
            "events.jsonl",
            "evidence_log.jsonl",
            "train_log.jsonl",
            "metrics.jsonl",
            "control.json",
            "command.txt",
            "stdout.log",
            "stderr.log",
            "report.html",
            "discovered_system.json",
            "discovery_manifest.json",
        ]
        artifacts = []
        for name in key_names:
            p = run_dir / name
            if p.is_file():
                try:
                    artifacts.append(
                        {
                            "path": name,
                            "size": int(p.stat().st_size),
                            "mtime": float(p.stat().st_mtime),
                        }
                    )
                except Exception:
                    artifacts.append({"path": name})

        # Viz gallery hint
        viz_dir = run_dir / "viz"
        viz = []
        if viz_dir.is_dir():
            try:
                for png in sorted(viz_dir.glob("*.png"))[:200]:
                    viz.append({"path": str(png.relative_to(run_dir))})
            except Exception:
                pass

        # Include live process info if available.
        proc_info = rm_proc_info
        if proc_info is None:
            reg, lock = _get_proc_registry(request.app)
            with lock:
                rec = reg.get(run_id)
                if rec is not None:
                    _refresh_proc_record(rec)
                    proc_info = _proc_info_dict(rec)

        return {
            **_run_summary(run_dir),
            "manifest": manifest if isinstance(manifest, (dict, list)) else None,
            "metrics": metrics if isinstance(metrics, (dict, list)) else None,
            "artifacts": artifacts,
            "viz": viz,
            "process": proc_info,
        }

    @router.get("/runs/{run_id}/process")
    def get_run_process(run_id: str, request: Request) -> Dict[str, Any]:
        rm = _get_run_manager(request.app)
        if rm is not None:
            try:
                rec_rm = rm.get(run_id)
                return {"ok": True, "tracked": True, "process": _run_manager_record_to_process(rec_rm)}
            except KeyError:
                pass
            except Exception:
                pass

        reg, lock = _get_proc_registry(request.app)
        with lock:
            rec = reg.get(run_id)
            if rec is not None:
                _refresh_proc_record(rec)
        if rec is None:
            # Best-effort: unknown to this server; return manifest status only.
            runs_root = Path(getattr(request.app.state, "runs_root", "runs"))
            rd = _resolve_run_dir(runs_root, run_id)
            if rd is None:
                raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
            man, _ = _load_manifest_any(rd)
            status = _run_status_from_manifest(man)
            return {"ok": True, "run_id": run_id, "tracked": False, "status": status}
        return {"ok": True, "tracked": True, "process": _proc_info_dict(rec)}

    @router.post("/runs/{run_id}/terminate")
    async def terminate_run(run_id: str, request: Request, payload: Optional[Dict[str, Any]] = Body(None)) -> Dict[str, Any]:
        """
        Terminate a run.

        Strategy:
          1) Always attempt graceful terminate via control.json (FR-6).
          2) If the server has a tracked subprocess, send SIGTERM/terminate().
          3) Optionally force kill if requested.
        """
        if not isinstance(payload, dict):
            payload = {}
        force = bool(payload.get("force", False))
        try:
            timeout_s = float(payload.get("timeout_s", 2.0))
        except Exception:
            timeout_s = 2.0
        timeout_s = max(0.0, min(timeout_s, 30.0))

        runs_root = Path(getattr(request.app.state, "runs_root", "runs"))
        run_dir = _resolve_run_dir(runs_root, run_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")

        # 1) Control-aware graceful terminate.
        try:
            _write_controls_compat(run_dir, {"terminate": True}, merge=True)
            _log_gui_event(run_dir, "terminate_requested", {"terminate": True})
        except Exception:
            pass

        # 2) Process terminate if tracked.
        # Prefer RunManager cancellation when available (supports queued runs + serial queue semantics).
        rm = _get_run_manager(request.app)
        if rm is not None:
            try:
                rm.cancel(run_id, force=bool(force))
                try:
                    rec_rm = rm.get(run_id)
                    proc_info = _run_manager_record_to_process(rec_rm)
                except Exception:
                    proc_info = None
                return {"ok": True, "run_id": run_id, "tracked": True, "force": bool(force), "via": "run_manager", "process": proc_info}
            except KeyError:
                # Not a RunManager-tracked run; fall back to legacy proc registry.
                pass
            except Exception:
                # Fall through to legacy logic for best-effort termination.
                pass

        reg, lock = _get_proc_registry(request.app)
        with lock:
            rec = reg.get(run_id)
        if rec is None:
            return {"ok": True, "run_id": run_id, "tracked": False, "terminate_signal_sent": True}


        # Mark intent so manifest status can be "killed" when appropriate.
        try:
            rec.requested_terminate = True
        except Exception:
            pass

        try:
            rec.proc.terminate()
        except Exception:
            pass

        # Wait briefly
        try:
            await asyncio.to_thread(rec.proc.wait, timeout_s)
        except Exception:
            pass

        # 3) Force kill if needed
        still_running = False
        try:
            still_running = rec.proc.poll() is None
        except Exception:
            still_running = rec.exit_code is None

        if still_running and force:
            try:
                rec.requested_kill = True
            except Exception:
                pass
            try:
                rec.proc.kill()
            except Exception:
                pass

        _refresh_proc_record(rec)

        return {"ok": True, "run_id": run_id, "tracked": True, "force": bool(force)}

    def _serve_artifact(run_dir: Path, relpath: str):
        if not _safe_relpath(relpath):
            raise HTTPException(status_code=400, detail="invalid relpath")

        try:
            base = run_dir.resolve()
            target = (run_dir / relpath).resolve()
            if base != target and base not in target.parents:
                raise HTTPException(status_code=400, detail="path traversal blocked")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=400, detail="invalid path")

        if not target.is_file():
            raise HTTPException(status_code=404, detail="file not found")

        return FileResponse(str(target), filename=target.name)

    @router.get("/runs/{run_id}/artifacts")
    def list_artifacts(
        run_id: str,
        request: Request,
        recursive: bool = Query(False, description="Recurse into subdirectories."),
    ) -> List[Dict[str, Any]]:
        runs_root = Path(getattr(request.app.state, "runs_root", "runs"))
        run_dir = _resolve_run_dir(runs_root, run_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        return _list_artifacts(run_dir, recursive=bool(recursive))

    @router.get("/runs/{run_id}/artifact")
    def get_artifact_query(
        run_id: str,
        request: Request,
        relpath: str = Query(..., alias="path", description="Relative path within the run directory."),
    ):
        runs_root = Path(getattr(request.app.state, "runs_root", "runs"))
        run_dir = _resolve_run_dir(runs_root, run_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        return _serve_artifact(run_dir, relpath)

    @router.get("/runs/{run_id}/artifacts/{relpath:path}")
    def get_artifact(run_id: str, relpath: str, request: Request):
        runs_root = Path(getattr(request.app.state, "runs_root", "runs"))
        run_dir = _resolve_run_dir(runs_root, run_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        return _serve_artifact(run_dir, relpath)

    # ------------------------------------------------------------------
    # Controls (FR-6)
    # ------------------------------------------------------------------

    @router.get("/runs/{run_id}/control")
    def get_control(run_id: str, request: Request) -> Dict[str, Any]:
        """
        Read the current control.json state for a run.

        This complements POST /runs/{run_id}/control so the UI can render current
        pause/terminate/write_every/snapshot/seq/ack_seq values (Design Doc FR-6).
        """
        runs_root = Path(getattr(request.app.state, "runs_root", "runs"))
        run_dir = _resolve_run_dir(runs_root, run_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")

        try:
            from electrodrive.live.controls import read_controls  # type: ignore

            st = read_controls(run_dir)
            return {"ok": True, "run_id": run_id, "control": st.to_dict()}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"control_read_failed: {exc}")

    @router.get("/control/schema")
    def control_schema() -> Dict[str, Any]:
        """
        Expose the repo's control schema so the UI can render protocol-correct controls.

        Design Doc FR-6: render controls from schema() to avoid drift.
        """
        try:
            from electrodrive.live.controls import schema  # type: ignore

            return {"ok": True, "schema": schema()}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to load control schema: {exc}")

    @router.post("/runs/{run_id}/control")
    async def post_control(run_id: str, request: Request, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """
        Update control.json for a run.

        Design Doc FR-6: must use repo control protocol and `write_controls(...)`.
        """
        runs_root = Path(getattr(request.app.state, "runs_root", "runs"))
        run_dir = _resolve_run_dir(runs_root, run_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")

        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="payload must be a JSON object")

        # Validate core keys; pass-through extra keys (merge=True preserves extras).
        updates: Dict[str, Any] = {}

        if "pause" in payload:
            v = payload.get("pause")
            if isinstance(v, bool):
                updates["pause"] = v
            elif v is None:
                pass
            else:
                raise HTTPException(status_code=400, detail="pause must be boolean")

        if "terminate" in payload:
            v = payload.get("terminate")
            if isinstance(v, bool):
                updates["terminate"] = v
            elif v is None:
                pass
            else:
                raise HTTPException(status_code=400, detail="terminate must be boolean")

        if "write_every" in payload:
            v = payload.get("write_every")
            if v is None:
                updates["write_every"] = None
            elif isinstance(v, bool):
                raise HTTPException(status_code=400, detail="write_every must be int|null (not bool)")
            else:
                try:
                    iv = int(v)
                except Exception:
                    raise HTTPException(status_code=400, detail="write_every must be int|null")
                updates["write_every"] = iv

        if "snapshot" in payload:
            # Design Doc FR-6: snapshot is a string token (one-shot request) or null.
            # For robustness, allow snapshot=True as "generate a token" (we still write a string token to disk).
            v = payload.get("snapshot")
            if v is None:
                updates["snapshot"] = None
            elif isinstance(v, bool):
                if v:
                    updates["snapshot"] = f"{_utc_iso_now()}-{uuid.uuid4().hex}"
                else:
                    updates["snapshot"] = None
            else:
                s = str(v).strip()
                if not s:
                    raise HTTPException(status_code=400, detail="snapshot token must be non-empty string")
                updates["snapshot"] = s

        # Pass-through unknown keys so external tools can extend controls without breaking (Design Doc §1.2).
        for k, v in payload.items():
            if k in {"pause", "terminate", "write_every", "snapshot"}:
                continue
            # Keep keys JSON-safe and reasonably small
            try:
                ks = str(k)
            except Exception:
                continue
            if not ks:
                continue
            updates[ks] = v

        # Protocol-correct write (atomic + seq/ts + merge) via repo helper.
        try:
            st = _write_controls_compat(run_dir, updates, merge=True)
            control_dict = st.to_dict()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"control_write_failed: {exc}")

        # Best-effort: log GUI action into run log stream (FR-6 acceptance).
        _log_gui_event(run_dir, "control", updates)

        return {"ok": True, "control": control_dict}

    # ------------------------------------------------------------------
    # Presets (FR-2)
    # ------------------------------------------------------------------

    @router.get("/presets")
    def list_presets() -> Dict[str, Any]:
        root = _presets_dir()
        items: List[str] = []
        try:
            if root.is_dir():
                for p in sorted(root.glob("*.json")):
                    items.append(p.stem)
        except Exception:
            items = []
        return {"ok": True, "presets": items, "dir": str(root)}

    @router.get("/presets/{name}")
    def get_preset(name: str) -> Dict[str, Any]:
        if not _PRESET_NAME_RE.match(name or ""):
            raise HTTPException(status_code=400, detail="invalid preset name")
        path = _presets_dir() / f"{name}.json"
        data = _safe_json_load(path)
        if data is None:
            raise HTTPException(status_code=404, detail="preset not found")
        return {"ok": True, "name": name, "preset": data}

    @router.post("/presets/{name}")
    async def put_preset(name: str, payload: Any = Body(...)) -> Dict[str, Any]:
        if not _PRESET_NAME_RE.match(name or ""):
            raise HTTPException(status_code=400, detail="invalid preset name")
        root = _presets_dir()
        path = root / f"{name}.json"
        # Accept any JSON-serializable payload; store verbatim.
        _atomic_write_text(path, _safe_json_dump(payload))
        return {"ok": True, "name": name, "path": str(path)}

    @router.delete("/presets/{name}")
    def delete_preset(name: str) -> Dict[str, Any]:
        if not _PRESET_NAME_RE.match(name or ""):
            raise HTTPException(status_code=400, detail="invalid preset name")
        path = _presets_dir() / f"{name}.json"
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass
        return {"ok": True, "name": name}

    # ------------------------------------------------------------------
    # Compare (FR-8, minimal v1)
    # ------------------------------------------------------------------

    @router.post("/compare")
    async def compare_runs(request: Request, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="payload must be object")
        run_ids = payload.get("run_ids")
        fields = payload.get("fields", None)
        if not isinstance(run_ids, list) or not all(isinstance(x, str) for x in run_ids):
            raise HTTPException(status_code=400, detail="run_ids must be list[str]")
        if fields is not None and not isinstance(fields, list):
            raise HTTPException(status_code=400, detail="fields must be list or omitted")

        runs_root = Path(getattr(request.app.state, "runs_root", "runs"))
        out: Dict[str, Any] = {"ok": True, "runs": {}, "missing": [], "diff": None}

        for rid in run_ids:
            rd = _resolve_run_dir(runs_root, rid)
            if rd is None:
                out["missing"].append(rid)
                continue
            manifest, _ = _load_manifest_any(rd)
            metrics = _safe_json_load(rd / "metrics.json")
            data = {
                "summary": _run_summary(rd),
                "manifest": manifest if isinstance(manifest, (dict, list)) else None,
                "metrics": metrics if isinstance(metrics, (dict, list)) else None,
            }
            if fields:
                # Best-effort field filtering (flat keys only).
                filt: Dict[str, Any] = {}
                for key in fields:
                    if not isinstance(key, str):
                        continue
                    for section in ("summary", "manifest", "metrics"):
                        src = data.get(section)
                        if isinstance(src, dict) and key in src:
                            filt[f"{section}.{key}"] = src.get(key)
                data["selected_fields"] = filt
            out["runs"][rid] = data

        # Basic diff: if exactly two runs are present, provide unified diff of JSON.
        present = [rid for rid in run_ids if rid in out["runs"]]
        if len(present) == 2:
            a = _safe_json_dump(out["runs"][present[0]])
            b = _safe_json_dump(out["runs"][present[1]])
            diff_lines = difflib.unified_diff(
                a.splitlines(),
                b.splitlines(),
                fromfile=present[0],
                tofile=present[1],
                lineterm="",
            )
            out["diff"] = "\n".join(diff_lines)

        return out

    # ------------------------------------------------------------------
    # Upgrades (FR-9, minimal v1)
    # ------------------------------------------------------------------

    @router.get("/runs/{run_id}/upgrades/summary")
    def upgrades_summary(run_id: str, request: Request) -> Dict[str, Any]:
        runs_root = Path(getattr(request.app.state, "runs_root", "runs"))
        run_dir = _resolve_run_dir(runs_root, run_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        return {"ok": True, "run_id": run_id, "summary": _upgrade_summary(run_dir)}

    # ------------------------------------------------------------------
    # Log coverage / audit panel (FR-9.6)
    # ------------------------------------------------------------------

    @router.get("/runs/{run_id}/logs/coverage")
    def log_coverage(run_id: str, request: Request) -> Dict[str, Any]:
        runs_root = Path(getattr(request.app.state, "runs_root", "runs"))
        run_dir = _resolve_run_dir(runs_root, run_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        cov = _compute_log_coverage(run_dir)
        return {"ok": True, "run_id": run_id, "coverage": cov}

    return router

