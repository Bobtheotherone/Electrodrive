from __future__ import annotations

"""
ResearchED RunManager: local, file-backed run launcher + controller.

Design Doc mapping:
- §3.2 RunManager: create run dirs, launch subprocesses, track state, cancel/kill
- FR-1: workflow launch + queueing (serial by default; optional parallel)
- FR-3: run directory contract (manifest.json, command.txt, metrics.json, logs)
- §1.4: compatibility policy: events.jsonl vs evidence_log.jsonl (bridge best-effort)
- FR-6 + §1.2: reuse control.json protocol; snapshot is a string token (not boolean)

Hard constraints (prompt):
- stdlib-only in this file (imports from existing electrodrive modules are OK)
- Windows-compatible process control and atomic file ops
- Never write concurrently to events.jsonl if the subprocess also writes it:
  we only tee stdout -> events.jsonl for learn_train *when events.jsonl is empty*.
"""

import dataclasses
import enum
import getpass
import json
import os
import platform
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from electrodrive.researched.workflows import get_workflow


class RunStatus(str, enum.Enum):
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    KILLED = "killed"
    CANCELED = "canceled"


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    workflow: str
    out_dir: Path
    status: RunStatus
    created_at: float
    started_at: float | None
    ended_at: float | None
    pid: int | None
    returncode: int | None
    command: list[str]
    env: dict[str, str]  # only env vars we set/override
    error: str | None


def _utc_iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(ts)))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Manifest / lifecycle helpers (Design Doc §5.1)
# --------------------------------------------------------------------------- #

_GIT_INFO_CACHE: Dict[str, Any] | None = None
_TORCH_VERSION_CACHE: str | None = None

# ResearchED-owned manifest (never overwritten by subprocess workflows).
_RESEARCHED_MANIFEST_NAME = "manifest.researched.json"


def _find_repo_root(start: Path) -> Path | None:
    """Walk upward to locate repo root (identified by .git). Returns None if not found."""
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


# Used to make git capture more reliable when the server is started from a non-repo CWD.
_RESEARCHED_REPO_ROOT: Path | None = _find_repo_root(Path(__file__).resolve()) or _find_repo_root(Path.cwd())


def _json_sanitize(obj: Any, *, max_depth: int = 6, max_items: int = 200) -> Any:
    """Best-effort JSON-serializable conversion for manifest/request fields."""
    if max_depth <= 0:
        return repr(obj)

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Common path-like types.
    try:
        if isinstance(obj, Path):
            return str(obj)
    except Exception:
        pass

    # Mappings.
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        n = 0
        for k, v in obj.items():
            if n >= max_items:
                out["_truncated"] = True
                break
            n += 1
            out[str(k)] = _json_sanitize(v, max_depth=max_depth - 1, max_items=max_items)
        return out

    # Sequences.
    if isinstance(obj, (list, tuple)):
        items = []
        for i, v in enumerate(obj):
            if i >= max_items:
                items.append("...")
                break
            items.append(_json_sanitize(v, max_depth=max_depth - 1, max_items=max_items))
        return items

    # Fallback.
    return repr(obj)


def _run_git(args: list[str], *, cwd: str | None = None, timeout_s: float = 1.5) -> str | None:
    try:
        cp = subprocess.run(
            ["git", *args],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=float(timeout_s),
        )
        if cp.returncode != 0:
            return None
        out = (cp.stdout or "").strip()
        return out or None
    except Exception:
        return None


def _collect_git_info(*, cwd: str | None = None) -> Dict[str, Any]:
    """Best-effort git capture for manifest.json (Design Doc §5.1)."""
    global _GIT_INFO_CACHE

    if _GIT_INFO_CACHE is not None:
        return dict(_GIT_INFO_CACHE)

    info: Dict[str, Any] = {}
    sha = _run_git(["rev-parse", "HEAD"], cwd=cwd)
    if not sha:
        _GIT_INFO_CACHE = {}
        return {}

    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd) or "unknown"
    status = _run_git(["status", "--porcelain"], cwd=cwd, timeout_s=2.0) or ""
    dirty = bool(status.strip())
    diff_summary = None
    if dirty:
        # Keep this small: a few porcelain lines are enough for audit.
        lines = status.splitlines()[:25]
        diff_summary = "\n".join(lines)

    info = {
        "sha": sha,
        "branch": branch,
        "dirty": bool(dirty),
        "diff_summary": diff_summary,
    }
    _GIT_INFO_CACHE = dict(info)
    return info


def _torch_version() -> str:
    global _TORCH_VERSION_CACHE
    if _TORCH_VERSION_CACHE is not None:
        return _TORCH_VERSION_CACHE
    try:
        import torch  # type: ignore

        _TORCH_VERSION_CACHE = str(getattr(torch, "__version__", "unknown"))
    except Exception:
        _TORCH_VERSION_CACHE = "unavailable"
    return _TORCH_VERSION_CACHE


def _collect_env_info(request: Dict[str, Any] | None, env_overrides: Dict[str, str]) -> Dict[str, Any]:
    """Best-effort runtime env capture for manifest.json (Design Doc §5.1)."""
    host = "unknown"
    try:
        host = socket.gethostname() or platform.node() or "unknown"
    except Exception:
        host = "unknown"

    user = None
    try:
        user = getpass.getuser()
    except Exception:
        user = None

    device = None
    dtype = None
    if isinstance(request, dict):
        for k in ("device", "dtype"):
            v = request.get(k)
            if isinstance(v, str) and v.strip():
                if k == "device":
                    device = v.strip()
                elif k == "dtype":
                    dtype = v.strip()

    # Allow env overrides to win (if caller provided them).
    if isinstance(env_overrides, dict):
        if env_overrides.get("EDE_DEVICE") and not device:
            device = str(env_overrides.get("EDE_DEVICE"))
        if env_overrides.get("EDE_DTYPE") and not dtype:
            dtype = str(env_overrides.get("EDE_DTYPE"))

    return {
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "torch_version": _torch_version(),
        "host": host,
        "user": user,
        "device": device,
        "dtype": dtype,
        "platform": platform.platform(),
    }


def _extract_inputs(workflow: str, request: Dict[str, Any] | None) -> Dict[str, Any]:
    """Pull stable input pointers into manifest.inputs for comparisons."""
    if not isinstance(request, dict):
        return {}

    if workflow == "solve":
        spec = request.get("spec_path") or request.get("problem") or request.get("spec")
        return {
            "spec_path": spec,
            "mode": request.get("mode"),
            "cert": request.get("cert"),
            "fast": request.get("fast") or request.get("cert_fast") or request.get("cert-fast"),
            "viz_enable": request.get("viz_enable"),
        }

    if workflow == "images_discover":
        spec = request.get("spec_path") or request.get("spec")
        return {
            "spec_path": spec,
            "basis": request.get("basis"),
            "nmax": request.get("nmax"),
            "reg_l1": request.get("reg_l1") if request.get("reg_l1") is not None else request.get("reg-l1"),
            "solver": request.get("solver"),
            "operator_mode": request.get("operator_mode")
            if request.get("operator_mode") is not None
            else request.get("operator-mode"),
            "subtract_physical": request.get("subtract_physical")
            if request.get("subtract_physical") is not None
            else request.get("subtract-physical"),
            "intensive": request.get("intensive"),
        }

    if workflow == "learn_train":
        cfg = request.get("config_path") or request.get("config")
        return {"config_path": cfg}

    if workflow == "fmm_suite":
        keys = [
            "device",
            "dtype",
            "n_points",
            "tol_p2p",
            "tol_fmm",
            "tol_bem",
            "expansion_order",
            "mac_theta",
            "leaf_size",
        ]
        out: Dict[str, Any] = {}
        for k in keys:
            if k in request:
                out[k] = request.get(k)
        return out

    return {}


def _atomic_write_text(path: Path, text: str) -> None:
    """
    Atomic file write (temp + os.replace), cross-platform safe.

    Important: temp filename must be unique to avoid collisions with other writers
    that use a fixed ".tmp" pattern.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}-{time.time_ns()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        # Best-effort fallback.
        try:
            path.write_text(text, encoding="utf-8")
        except Exception:
            pass
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    try:
        txt = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    except Exception:
        try:
            txt = json.dumps({"_error": "json_dump_failed", "repr": repr(data)}, indent=2)
        except Exception:
            return
    _atomic_write_text(path, txt)


def _safe_read_json(path: Path) -> Dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _touch(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    except Exception:
        return


def _try_symlink(dst: Path, target_name: str) -> bool:
    """
    Best-effort symlink creation.

    Note: On Windows, symlink creation may require privileges; failures are swallowed.
    """
    try:
        if dst.is_symlink():
            return True
    except Exception:
        pass
    try:
        if dst.exists():
            return True
    except Exception:
        pass
    try:
        os.symlink(target_name, dst)
        return True
    except Exception:
        return False


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return -1


def _is_symlink_to(src: Path, target: Path) -> bool:
    try:
        if not src.is_symlink():
            return False
    except Exception:
        return False
    try:
        return src.resolve() == target.resolve()
    except Exception:
        return False


def _bridge_events_evidence_start(out_dir: Path) -> None:
    """
    Design Doc §1.4: events.jsonl vs evidence_log.jsonl compatibility bridge.

    Best-effort at run start:
    - Ensure events.jsonl exists (empty OK; writers will append).
    - Try to make evidence_log.jsonl a symlink to events.jsonl.
      If symlink fails, create an empty evidence_log.jsonl so legacy readers don't
      crash on FileNotFoundError (content may be filled by end-of-run copy).
    """
    events = out_dir / "events.jsonl"
    evidence = out_dir / "evidence_log.jsonl"
    _touch(events)

    # Prefer evidence -> events (legacy readers).
    if not _try_symlink(evidence, "events.jsonl"):
        _touch(evidence)


def _bridge_events_evidence_finalize(out_dir: Path) -> None:
    """
    Design Doc §1.4: ensure both names exist after the run.

    Important nuance:
    - If evidence_log.jsonl was created as an empty placeholder (e.g., symlink failed),
      we must still copy events.jsonl into it after the run so legacy tooling sees logs.
    - Likewise, if only evidence_log.jsonl was written by a workflow, copy into events.jsonl.

    Policy:
    - If one is missing: create via symlink (best-effort) or copy.
    - If both exist but one is empty and the other is non-empty: copy non-empty -> empty.
    - If both non-empty: leave as-is (ResearchED merges them at ingest time).
    """
    events = out_dir / "events.jsonl"
    evidence = out_dir / "evidence_log.jsonl"

    try:
        ev_exists = events.exists()
    except Exception:
        ev_exists = False
    try:
        ed_exists = evidence.exists()
    except Exception:
        ed_exists = False

    if not ev_exists and not ed_exists:
        return

    if ev_exists and ed_exists:
        if _is_symlink_to(evidence, events) or _is_symlink_to(events, evidence):
            return

        size_events = _file_size(events)
        size_evidence = _file_size(evidence)

        if size_events > 0 and size_evidence == 0:
            try:
                shutil.copyfile(events, evidence)
            except Exception:
                pass
        elif size_evidence > 0 and size_events == 0:
            try:
                shutil.copyfile(evidence, events)
            except Exception:
                pass
        return

    if ev_exists and not ed_exists:
        if not _try_symlink(evidence, "events.jsonl"):
            try:
                shutil.copyfile(events, evidence)
            except Exception:
                pass
        return

    if ed_exists and not ev_exists:
        if not _try_symlink(events, "evidence_log.jsonl"):
            try:
                shutil.copyfile(evidence, events)
            except Exception:
                pass


def _write_command_txt(out_dir: Path, command: List[str], env_overrides: Dict[str, str]) -> None:
    lines: List[str] = []
    lines.append("# ResearchED command.txt")
    lines.append(f"# timestamp_utc: {_utc_iso(time.time())}")
    lines.append("")
    lines.append("argv:")
    lines.append("  " + " ".join(command))
    lines.append("")
    lines.append("env_overrides:")
    for k in sorted(env_overrides.keys()):
        v = env_overrides.get(k, "")
        v1 = str(v).replace("\n", "\\n")
        lines.append(f"  {k}={v1}")
    lines.append("")
    _atomic_write_text(out_dir / "command.txt", "\n".join(lines))


def _merge_manifest(existing: Dict[str, Any] | None, updates: Dict[str, Any]) -> Dict[str, Any]:
    base: Dict[str, Any] = dict(existing or {})
    merge_keys = {"inputs", "outputs", "env", "git", "gate", "spec_digest", "researched"}
    for k, v in updates.items():
        if k in merge_keys and isinstance(base.get(k), dict) and isinstance(v, dict):
            merged = dict(base.get(k) or {})
            merged.update(v)
            base[k] = merged
        else:
            base[k] = v
    return base


def _write_manifest_minimal(
    out_dir: Path,
    *,
    run_id: str,
    workflow: str,
    status: str,
    created_at: float,
    started_at: float | None,
    ended_at: float | None,
    command: List[str],
    env_overrides: Dict[str, str],
    pid: int | None,
    returncode: int | None,
    error: str | None,
    phase: str | None = None,
    request: Dict[str, Any] | None = None,
    cancel_requested: bool | None = None,
) -> None:
    """
    Write/refresh run manifests.

    Design Doc FR-3 / §5.1:
    - manifest.json must exist in each run dir and be updated at terminal.
    - Workflows may overwrite manifest.json (e.g., electrodrive CLI solve).
      To avoid losing UI-required fields, ResearchED also writes an owned copy:
        manifest.researched.json
      which the UI prefers when present.

    Policy:
    - Always write manifest.researched.json (authoritative for UI-required fields).
    - For manifest.json:
        * create it if missing
        * update it at terminal (success/error/killed) by merging terminal fields,
          preserving any keys written by subprocess workflows.
        * avoid repeated writes during RUNNING to reduce clobber risk.
    """
    out_dir = Path(out_dir)
    researched_path = out_dir / _RESEARCHED_MANIFEST_NAME
    workflow_path = out_dir / "manifest.json"

    # Read existing manifests (best-effort, never raises).
    existing_researched = _safe_read_json(researched_path) or {}
    existing_workflow = _safe_read_json(workflow_path) or {}

    # Merge base as: workflow first, then researched (so our prior fields win),
    # then overlay the new updates below.
    base = _merge_manifest(
        existing_workflow if isinstance(existing_workflow, dict) else {},
        existing_researched if isinstance(existing_researched, dict) else {},
    )

    # Keep run_id stable even if a subprocess wrote a different run_id.
    extra_fields: Dict[str, Any] = {}
    try:
        if isinstance(existing_workflow, dict):
            existing_run_id = existing_workflow.get("run_id")
            if existing_run_id is not None and str(existing_run_id) and str(existing_run_id) != str(run_id):
                extra_fields["subprocess_run_id"] = str(existing_run_id)
    except Exception:
        pass

    # Normalize status to the Design Doc manifest enum, while preserving a richer
    # lifecycle in researched.internal_status / researched.phase.
    st = str(status or "").lower().strip()
    terminal = st in {"success", "error", "killed"}
    if st not in {"running", "success", "error", "killed"}:
        # queued/starting/canceled/etc -> running in the top-level manifest status
        # (UI should use researched.internal_status for the richer lifecycle).
        st = "running"

    phase_norm = str(phase or status or "unknown")

    # Inputs/env/git per Design Doc §5.1 (best-effort, small + stable).
    inputs_core = _extract_inputs(str(workflow), request)
    env_info = _collect_env_info(request, env_overrides)
    git_info = _collect_git_info(cwd=str(_RESEARCHED_REPO_ROOT) if _RESEARCHED_REPO_ROOT else None)

    updates: Dict[str, Any] = {
        "schema_version": 1,
        "run_id": str(run_id),
        "workflow": str(workflow),
        "started_at": _utc_iso(started_at),
        "ended_at": _utc_iso(ended_at),
        "status": st,
        "created_at": _utc_iso(created_at),
        # Helpful for plotting/sorting without parsing ISO strings:
        "created_at_epoch": float(created_at),
        "started_at_epoch": float(started_at) if started_at is not None else None,
        "ended_at_epoch": float(ended_at) if ended_at is not None else None,
        "pid": int(pid) if pid is not None else None,
        "returncode": int(returncode) if returncode is not None else None,
        "error": error,
        "git": git_info,
        "env": env_info,
        "inputs": {
            **inputs_core,
            "command": list(command),
            "env_overrides": dict(env_overrides),
            # Keep the full request for reproducibility (sanitized).
            "request": _json_sanitize(request or {}),
        },
        "outputs": {
            "manifest_json": "manifest.json",
            "researched_manifest_json": _RESEARCHED_MANIFEST_NAME,
            "metrics_json": "metrics.json",
            "events_jsonl": "events.jsonl",
            "evidence_log_jsonl": "evidence_log.jsonl",
            "command_txt": "command.txt",
            "stdout_log": "stdout.log",
            "stderr_log": "stderr.log",
            # Optional; will exist for solve when viz is enabled.
            "viz_dir": "viz",
            "plots_dir": "plots",
            "artifacts_dir": "artifacts",
            "report_html": "report.html",
        },
        # Design Doc §5.1: gate fields are always present (may be null).
        "gate": {
            "gate1_status": None,
            "gate2_status": None,
            "gate3_status": None,
            "structure_score": None,
            "novelty_score": None,
        },
        # Design Doc §5.1: spec_digest is added by SpecInspector when available.
        # Keep type-stable (dict) for the UI.
        "spec_digest": {},
        "researched": {
            "manager": "RunManager",
            "schema_version": 1,
            "phase": phase_norm,
            "internal_status": str(status),
            "cancel_requested": bool(cancel_requested) if cancel_requested is not None else False,
        },
    }
    updates.update(extra_fields)

    merged = _merge_manifest(base, updates)

    # Always write ResearchED-owned manifest.
    _atomic_write_json(researched_path, merged)

    # Write/refresh workflow manifest.json only when safe/necessary.
    try:
        workflow_exists = workflow_path.exists()
    except Exception:
        workflow_exists = False

    if (not workflow_exists) or terminal:
        if workflow_exists:
            # Merge terminal fields into whatever the workflow wrote.
            existing = _safe_read_json(workflow_path) or {}
            term_updates = {
                "run_id": str(run_id) if not isinstance(existing.get("run_id"), str) else existing.get("run_id"),
                "workflow": str(workflow)
                if not isinstance(existing.get("workflow"), str)
                else existing.get("workflow"),
                "started_at": existing.get("started_at") or _utc_iso(started_at),
                "ended_at": _utc_iso(ended_at),
                "status": st,
                "pid": int(pid) if pid is not None else existing.get("pid"),
                "returncode": int(returncode) if returncode is not None else existing.get("returncode"),
                "error": error,
                "researched": merged.get("researched", {}),
            }
            _atomic_write_json(workflow_path, _merge_manifest(existing if isinstance(existing, dict) else {}, term_updates))
        else:
            # Create a complete manifest.json if absent.
            _atomic_write_json(workflow_path, merged)


def _sanitize_env_overrides(env: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in env.items():
        try:
            if v is None:
                continue
            out[str(k)] = str(v)
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------- #
# Cross-process safe JSONL append helpers
# --------------------------------------------------------------------------- #

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
    """Append a single JSON record as one JSONL line."""
    try:
        line = json.dumps(rec, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    except Exception:
        return
    _locked_append_bytes(path, line)


def _stdout_pump(
    *,
    proc: subprocess.Popen[str],
    stdout_path: Path,
    tee_events_path: Path | None,
) -> None:
    """
    Dedicated stdout reader thread: prevents subprocess pipe deadlocks.

    - Always appends combined stdout+stderr to stdout.log.
    - Optionally tees each stdout line into structured JSONL records for workflows
      that don't produce structured JSONL themselves (learn_train).
    """
    try:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    try:
        with stdout_path.open("a", encoding="utf-8") as out_fh:
            stream = proc.stdout
            if stream is None:
                return

            for line in stream:
                try:
                    out_fh.write(line)
                    out_fh.flush()
                except Exception:
                    pass

                if tee_events_path is not None:
                    msg = (line or "").rstrip("\n")
                    rec = {
                        "ts": _utc_iso(time.time()) or "",
                        "level": "info",
                        "msg": msg,
                        # Provide event for legacy consumers; normalizers still prefer msg.
                        "event": msg,
                    }
                    _append_jsonl_record(tee_events_path, rec)
    except Exception:
        return


def _safe_jsonl_tail(path: Path, *, max_bytes: int = 2_000_000, max_lines: int = 5000) -> List[str]:
    """Return up to max_lines lines from the tail of a file, best-effort."""
    try:
        if not path.is_file():
            return []
    except Exception:
        return []
    try:
        size = path.stat().st_size
    except Exception:
        size = 0
    try:
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(max(0, size - max_bytes))
            data = f.read()
    except Exception:
        return []
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return []
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return lines


def _ensure_metrics_json(out_dir: Path, workflow: str, *, run_status: str, returncode: int | None) -> None:
    """
    Enforce FR-3/FR-7: metrics.json should be non-empty when possible.

    Fixes:
    - Do NOT bail out just because metrics.json exists (we often create {} at run start).
      Only skip if metrics.json is already a non-empty dict.
    - Write a flat dict of scalar metrics for compatibility, storing metadata under "_meta".
    """
    metrics_path = out_dir / "metrics.json"

    existing = _safe_read_json(metrics_path)
    if isinstance(existing, dict) and existing:
        return  # workflow produced real metrics

    out: Dict[str, Any] = dict(existing or {})

    meta_existing = out.get("_meta")
    meta: Dict[str, Any] = dict(meta_existing) if isinstance(meta_existing, dict) else {}

    now = time.time()
    meta.update(
        {
            "workflow": workflow,
            "generated_by": "ResearchED RunManager",
            "generated_at": _utc_iso(now),
            "returncode": int(returncode) if returncode is not None else None,
            "run_status": run_status,
        }
    )

    # Terminal scalars (safe defaults).
    out.setdefault("status", str(run_status))
    out.setdefault("ok", bool(run_status == "success"))
    if returncode is not None:
        out.setdefault("returncode", int(returncode))

    # Workflow-specific best-effort scalars.
    if workflow == "images_discover":
        m = _safe_read_json(out_dir / "discovery_manifest.json") or {}
        if isinstance(m, dict):
            for k in ("rel_resid", "max_abs_weight", "min_nonzero_weight"):
                v = m.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.setdefault(k, float(v))
            for k in ("numeric_status", "condition_status", "gate1_status", "solver", "spec_path"):
                if k in m and k not in meta:
                    meta[k] = m.get(k)

    elif workflow == "learn_train":
        mjl = out_dir / "metrics.jsonl"
        last_rec: Dict[str, Any] | None = None
        lines = _safe_jsonl_tail(mjl)
        for line in reversed(lines):
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                last_rec = obj
                break
        if last_rec:
            for k, v in list(last_rec.items()):
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.setdefault(k, float(v))
            meta["metrics_jsonl"] = "metrics.jsonl"

    elif workflow == "fmm_suite":
        events_path = out_dir / "events.jsonl"
        lines = _safe_jsonl_tail(events_path, max_bytes=2_000_000, max_lines=20000)
        total = 0
        ok_n = 0
        max_abs_err = None
        rel_l2_err = None
        worst_name = None
        for line in lines:
            s = (line or "").strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            if str(rec.get("event") or rec.get("msg") or rec.get("message")) != "fmm_test_result":
                continue
            total += 1
            ok = bool(rec.get("ok", False))
            if ok:
                ok_n += 1
            try:
                mae = float(rec.get("max_abs_err", float("nan")))
            except Exception:
                mae = float("nan")
            try:
                rle = float(rec.get("rel_l2_err", float("nan")))
            except Exception:
                rle = float("nan")
            if mae == mae:
                if max_abs_err is None or mae > max_abs_err:
                    max_abs_err = mae
                    worst_name = rec.get("name")
            if rle == rle:
                if rel_l2_err is None or rle > rel_l2_err:
                    rel_l2_err = rle

        out.setdefault("tests_total", int(total))
        out.setdefault("tests_ok", int(ok_n))
        if max_abs_err is not None:
            out.setdefault("max_abs_err_max", float(max_abs_err))
        if rel_l2_err is not None:
            out.setdefault("rel_l2_err_max", float(rel_l2_err))
        if worst_name is not None:
            meta["worst_test"] = worst_name

    out["_meta"] = meta
    _atomic_write_json(metrics_path, out)


# --------------------------------------------------------------------------- #
# JSONL event helpers (for lifecycle + GUI controls) and live log bridging
# --------------------------------------------------------------------------- #


def _append_event(out_dir: Path, *, level: str = "info", msg: str, **fields: Any) -> None:
    """Append a single JSONL record to events.jsonl (best-effort)."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    base_msg = str(msg)
    rec: Dict[str, Any] = {
        "ts": _utc_iso(time.time()) or "",
        "level": str(level).lower(),
        "msg": base_msg,
        # Provide event for legacy consumers; normalizers still prefer msg.
        "event": base_msg,
    }

    # Avoid clobbering core keys.
    for k, v in (fields or {}).items():
        if k in {"ts", "t", "level", "msg", "event"}:
            rec[f"field_{k}"] = v
        else:
            rec[str(k)] = v

    _append_jsonl_record(out_dir / "events.jsonl", rec)


def _ensure_report_html(out_dir: Path) -> None:
    """Create a minimal report.html if none exists (Design Doc FR-10 required output)."""
    path = out_dir / "report.html"
    try:
        if path.is_file() and path.stat().st_size > 0:
            return
    except Exception:
        pass

    manifest = _safe_read_json(out_dir / "manifest.json") or {}
    metrics = _safe_read_json(out_dir / "metrics.json") or {}

    title = str(manifest.get("workflow") or "run")
    run_id = str(manifest.get("run_id") or "")
    status = str(manifest.get("status") or "unknown")

    def _link(name: str) -> str:
        return f'<li><a href="{name}">{name}</a></li>'

    body = "\n".join(
        [
            "<!doctype html>",
            "<html><head>",
            '<meta charset="utf-8"/>',
            f"<title>ResearchED report - {title}</title>",
            "<style>body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;} code{background:#f4f4f4;padding:2px 4px;border-radius:4px;}</style>",
            "</head><body>",
            f"<h1>ResearchED report: {title}</h1>",
            f"<p><b>run_id:</b> <code>{run_id}</code><br/><b>status:</b> <code>{status}</code></p>",
            "<h2>Artifacts</h2>",
            "<ul>",
            _link("manifest.json"),
            _link("command.txt"),
            _link("metrics.json"),
            _link("events.jsonl"),
            _link("evidence_log.jsonl"),
            _link("stdout.log"),
            "</ul>",
            "<h2>Metrics</h2>",
            "<pre>" + json.dumps(metrics, indent=2, ensure_ascii=False) + "</pre>",
            "</body></html>",
        ]
    )
    _atomic_write_text(path, body)


def _start_live_log_bridge(out_dir: Path, stop_event: threading.Event) -> threading.Thread | None:
    """
    Best-effort live bridge between events.jsonl and evidence_log.jsonl when symlinks are unavailable.

    Goal (Design Doc §1.4): keep legacy tailers happy by mirroring whichever file
    is being appended by the workflow into the other filename.
    """
    events = out_dir / "events.jsonl"
    evidence = out_dir / "evidence_log.jsonl"

    # If either is already a symlink to the other, nothing to do.
    try:
        if _is_symlink_to(evidence, events) or _is_symlink_to(events, evidence):
            return None
    except Exception:
        pass

    def _copy_append(src: Path, dst: Path, start: int, end: int) -> int:
        if end <= start:
            return start
        try:
            with src.open("rb") as fsrc:
                fsrc.seek(max(0, start))
                data = fsrc.read(max(0, end - start))
            if not data:
                return end
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            with dst.open("ab") as fdst:
                fdst.write(data)
                try:
                    fdst.flush()
                except Exception:
                    pass
            return end
        except Exception:
            return start

    def _loop() -> None:
        # Default direction: events -> evidence (events.jsonl is the canonical writer for most workflows).
        src = events
        dst = evidence
        offset = 0

        while not stop_event.is_set():
            se = _file_size(events)
            sd = _file_size(evidence)

            if se < 0 and sd < 0:
                time.sleep(0.2)
                continue

            # If events is empty/missing but evidence has data, assume a workflow is writing evidence instead.
            if (se <= 0) and (sd > 0):
                src = evidence
                dst = events
            else:
                src = events
                dst = evidence

            src_size = _file_size(src)
            if src_size < 0:
                time.sleep(0.2)
                continue

            # If the source file truncated, restart offset.
            if src_size < offset:
                offset = 0

            offset = _copy_append(src, dst, offset, src_size)
            time.sleep(0.2)

    t = threading.Thread(target=_loop, name=f"ResearchED-LogBridge-{out_dir.name}", daemon=True)
    t.start()
    return t


class RunManager:
    """
    Threaded run queue manager.

    - Serial-by-default queue (FR-1) with max_parallel workers.
    - Run directory + artifact contract enforcement (FR-3).
    - Control protocol reuse via electrodrive.live.controls (FR-6, §1.2).
    - Log filename drift bridging: events.jsonl <-> evidence_log.jsonl (Design Doc §1.4).
    """

    def __init__(self, runs_root: Path, *, max_parallel: int = 1):
        self.runs_root = Path(runs_root)
        self.max_parallel = max(1, int(max_parallel))

        try:
            self.runs_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        self._q: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()

        self._records: Dict[str, RunRecord] = {}
        self._requests: Dict[str, Dict[str, Any]] = {}
        self._procs: Dict[str, subprocess.Popen[str]] = {}
        self._pump_threads: Dict[str, threading.Thread] = {}

        self._bridge_threads: Dict[str, threading.Thread] = {}
        self._bridge_stops: Dict[str, threading.Event] = {}

        self._cancel_requested: set[str] = set()
        self._workers: List[threading.Thread] = []

        for i in range(self.max_parallel):
            t = threading.Thread(target=self._worker_loop, name=f"ResearchED-RunWorker-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    def submit(
        self,
        workflow_name: str,
        request: dict,
        *,
        run_dir: Path | None = None,
        run_id: str | None = None,
    ) -> str:
        """
        Enqueue a run.

        FR-1 alignment:
        - Create the run directory immediately.
        - Write a manifest immediately so the UI can navigate to a monitor view.
        - Queue execution (serial by default; configurable via max_parallel).

        Notes:
        - If run_id is provided, it is used verbatim (after stripping).
        - If run_dir is provided and is relative, it is interpreted relative to runs_root.
        """
        wf = get_workflow(workflow_name)
        wf.validate_request(request)

        rid = str(run_id).strip() if run_id is not None else ""
        if not rid:
            rid = str(uuid.uuid4())

        created_at = time.time()

        # Resolve out_dir.
        if run_dir is not None:
            out_dir = Path(run_dir).expanduser()
            if not out_dir.is_absolute():
                out_dir = (self.runs_root / out_dir).expanduser()
        else:
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(created_at))
            out_dir = self.runs_root / f"{ts}_{workflow_name}_{rid}"

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)
            (out_dir / "plots").mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Ensure key contract files exist early.
        _bridge_events_evidence_start(out_dir)
        try:
            if not (out_dir / "metrics.json").is_file():
                _atomic_write_json(out_dir / "metrics.json", {})
        except Exception:
            pass
        # Ensure core raw-log placeholders exist immediately (UI/WS won't 404).
        _touch(out_dir / "stdout.log")
        _touch(out_dir / "stderr.log")
        _touch(out_dir / "command.txt")

        # Ensure control.json exists early for control-aware workflows (FR-6).
        if wf.supports_controls:
            try:
                from electrodrive.live.controls import write_controls  # type: ignore

                write_controls(out_dir, {})
            except Exception:
                pass

        rec = RunRecord(
            run_id=rid,
            workflow=str(workflow_name),
            out_dir=out_dir,
            status=RunStatus.QUEUED,
            created_at=float(created_at),
            started_at=None,
            ended_at=None,
            pid=None,
            returncode=None,
            command=[],
            env={},
            error=None,
        )

        with self._lock:
            self._records[rid] = rec
            self._requests[rid] = dict(request)

        _write_manifest_minimal(
            out_dir,
            run_id=rid,
            workflow=str(workflow_name),
            status="queued",
            created_at=created_at,
            started_at=None,
            ended_at=None,
            command=[],
            env_overrides={},
            pid=None,
            returncode=None,
            error=None,
            phase="queued",
            request=dict(request),
            cancel_requested=False,
        )
        _append_event(out_dir, msg="researched_run_queued", run_id=rid, workflow=str(workflow_name))

        # Ensure required report output exists even before ReportService runs.
        _ensure_report_html(out_dir)

        self._q.put(rid)
        return rid

    def get(self, run_id: str) -> RunRecord:
        with self._lock:
            rec = self._records.get(run_id)
            if rec is None:
                raise KeyError(f"Unknown run_id: {run_id}")
            return dataclasses.replace(rec)

    def list(self) -> list[RunRecord]:
        with self._lock:
            items = list(self._records.values())
        items.sort(key=lambda r: float(r.created_at), reverse=True)
        return [dataclasses.replace(r) for r in items]

    def cancel(self, run_id: str, *, force: bool = False) -> None:
        """
        Cancel a queued or running run.

        Design Doc FR-6 + §1.2:
        - If workflow supports controls, prefer control.json terminate=true.
        - Fall back to process terminate/kill if the run does not exit cooperatively.

        Notes:
        - For queued runs, this is a pure lifecycle cancellation (no process exists).
        - For running runs, this requests cooperative terminate and escalates if needed.
        """
        with self._lock:
            rec = self._records.get(run_id)
            if rec is None:
                raise KeyError(f"Unknown run_id: {run_id}")

            # Fast-path: cancel before start.
            if rec.status == RunStatus.QUEUED:
                ended = time.time()
                self._records[run_id] = dataclasses.replace(
                    rec,
                    status=RunStatus.CANCELED,
                    ended_at=ended,
                    error="canceled_before_start",
                )
                _append_event(rec.out_dir, msg="researched_run_canceled", run_id=run_id, workflow=rec.workflow)
                _bridge_events_evidence_finalize(rec.out_dir)
                _ensure_metrics_json(rec.out_dir, rec.workflow, run_status="killed", returncode=None)
                _ensure_report_html(rec.out_dir)
                _write_manifest_minimal(
                    rec.out_dir,
                    run_id=run_id,
                    workflow=rec.workflow,
                    status="killed",
                    created_at=rec.created_at,
                    started_at=None,
                    ended_at=ended,
                    command=rec.command,
                    env_overrides=rec.env,
                    pid=None,
                    returncode=None,
                    error="canceled_before_start",
                    phase="canceled",
                    request=dict(self._requests.get(run_id, {})),
                    cancel_requested=True,
                )
                self._requests.pop(run_id, None)
                return

            # For STARTING/RUNNING: record cancel intent.
            self._cancel_requested.add(run_id)
            req = dict(self._requests.get(run_id, {}))

        _append_event(rec.out_dir, msg="researched_cancel_requested", run_id=run_id, workflow=rec.workflow, force=bool(force))

        wf = get_workflow(rec.workflow)

        # Primary path for control-aware workflows: cooperative terminate via control.json.
        if wf.supports_controls:
            try:
                from electrodrive.live.controls import write_controls  # type: ignore
                write_controls(rec.out_dir, updates={"terminate": True}, merge=True, seq_increment=True)
            except Exception:
                pass

        with self._lock:
            proc = self._procs.get(run_id)

        # If the process isn't alive yet (race), _run_one will observe _cancel_requested
        # and avoid launching.
        if proc is None:
            return

        if force:
            try:
                proc.kill()
            except Exception:
                pass
            return

        # Escalation strategy:
        # 1) If controls are supported, give the solver a short grace window to
        #    observe terminate=true and exit cleanly.
        # 2) If still alive, send process.terminate().
        # 3) If still alive after a longer grace, send process.kill().
        def _escalate() -> None:
            poll = 0.2
            grace_controls = 2.5 if wf.supports_controls else 0.0
            grace_terminate = 5.0

            deadline = time.time() + max(0.0, grace_controls)
            while time.time() < deadline:
                try:
                    if proc.poll() is not None:
                        return
                except Exception:
                    return
                time.sleep(poll)

            try:
                proc.terminate()
            except Exception:
                pass

            deadline2 = time.time() + grace_terminate
            while time.time() < deadline2:
                try:
                    if proc.poll() is not None:
                        return
                except Exception:
                    return
                time.sleep(poll)

            try:
                proc.kill()
            except Exception:
                return

        threading.Thread(target=_escalate, name=f"ResearchED-CancelEscalate-{run_id}", daemon=True).start()

    def send_control(self, run_id: str, updates: dict) -> None:
        """
        Write control.json updates using repo helper.

        Snapshot semantics:
        - snapshot is str|null
        - If caller provides snapshot=True, convert to a unique string token.
        - Never write snapshot=true.
        """
        with self._lock:
            rec = self._records.get(run_id)
            if rec is None:
                raise KeyError(f"Unknown run_id: {run_id}")

        wf = get_workflow(rec.workflow)
        if not wf.supports_controls:
            raise ValueError(f"Workflow '{rec.workflow}' does not support live controls.")

        safe_updates: Dict[str, object] = dict(updates or {})

        if "snapshot" in safe_updates:
            snap = safe_updates.get("snapshot")
            if isinstance(snap, bool):
                safe_updates["snapshot"] = str(uuid.uuid4()) if snap else None
            elif snap is None:
                safe_updates["snapshot"] = None
            else:
                s = str(snap).strip()
                safe_updates["snapshot"] = s if s else None

        try:
            from electrodrive.live.controls import write_controls  # type: ignore

            st = write_controls(rec.out_dir, updates=safe_updates, merge=True, seq_increment=True)
            # FR-6: Control actions must be visible in the run's event stream.
            try:
                _append_event(
                    rec.out_dir,
                    msg="researched_control",
                    run_id=run_id,
                    workflow=rec.workflow,
                    control_updates=_json_sanitize(safe_updates),
                    seq=getattr(st, "seq", None),
                )
            except Exception:
                pass
        except Exception as exc:
            raise ValueError(f"Failed to write controls: {exc}") from exc

    # ----------------- internals -----------------

    def _worker_loop(self) -> None:
        while True:
            run_id = self._q.get()
            try:
                self._run_one(run_id)
            except Exception:
                try:
                    with self._lock:
                        rec = self._records.get(run_id)
                        if rec is not None and rec.status not in {
                            RunStatus.SUCCESS,
                            RunStatus.ERROR,
                            RunStatus.KILLED,
                            RunStatus.CANCELED,
                        }:
                            ended = time.time()
                            self._records[run_id] = dataclasses.replace(
                                rec,
                                status=RunStatus.ERROR,
                                ended_at=ended,
                                error="worker_exception",
                            )
                            _write_manifest_minimal(
                                rec.out_dir,
                                run_id=run_id,
                                workflow=rec.workflow,
                                status="error",
                                created_at=rec.created_at,
                                started_at=rec.started_at,
                                ended_at=ended,
                                command=rec.command,
                                env_overrides=rec.env,
                                pid=rec.pid,
                                returncode=rec.returncode,
                                error="worker_exception",
                            )
                            _ensure_metrics_json(rec.out_dir, rec.workflow, run_status="error", returncode=rec.returncode)
                            _bridge_events_evidence_finalize(rec.out_dir)
                            _ensure_report_html(rec.out_dir)
                except Exception:
                    pass
            finally:
                try:
                    self._q.task_done()
                except Exception:
                    pass

    def _run_one(self, run_id: str) -> None:
        # Snapshot record + request (best-effort; request may be dropped on completion).
        with self._lock:
            rec = self._records.get(run_id)
            req = dict(self._requests.get(run_id, {}))
            cancel_requested = run_id in self._cancel_requested

        if rec is None:
            return

        # If canceled before the worker picked it up, do nothing.
        if rec.status == RunStatus.CANCELED:
            return

        out_dir = rec.out_dir

        # Cancellation race: if cancel() happened while the job was still queued/starting,
        # avoid launching anything.
        if cancel_requested and rec.status in {RunStatus.QUEUED, RunStatus.STARTING}:
            ended = time.time()
            with self._lock:
                cur = self._records.get(run_id)
                if cur is not None and cur.status != RunStatus.CANCELED:
                    self._records[run_id] = dataclasses.replace(
                        cur,
                        status=RunStatus.KILLED,
                        ended_at=ended,
                        error="canceled_before_start",
                    )
                self._cancel_requested.discard(run_id)
                self._requests.pop(run_id, None)

            _append_event(out_dir, msg="researched_run_canceled", run_id=run_id, workflow=rec.workflow)
            _bridge_events_evidence_finalize(out_dir)
            _ensure_metrics_json(out_dir, rec.workflow, run_status="killed", returncode=None)
            _ensure_report_html(out_dir)
            _write_manifest_minimal(
                out_dir,
                run_id=run_id,
                workflow=rec.workflow,
                status="killed",
                created_at=rec.created_at,
                started_at=rec.started_at,
                ended_at=ended,
                command=rec.command,
                env_overrides=rec.env,
                pid=rec.pid,
                returncode=None,
                error="canceled_before_start",
                phase="canceled",
                request=req,
                cancel_requested=True,
            )
            return

        wf = get_workflow(rec.workflow)

        started = time.time()
        with self._lock:
            rec0 = self._records.get(run_id)
            if rec0 is None:
                return
            if rec0.status == RunStatus.CANCELED:
                return
            self._records[run_id] = dataclasses.replace(rec0, status=RunStatus.STARTING, started_at=started)

        _append_event(out_dir, msg="researched_run_starting", run_id=run_id, workflow=rec.workflow)

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)
            (out_dir / "plots").mkdir(parents=True, exist_ok=True)
        except Exception:
            ended = time.time()
            with self._lock:
                rec1 = self._records.get(run_id)
                if rec1 is not None:
                    self._records[run_id] = dataclasses.replace(
                        rec1,
                        status=RunStatus.ERROR,
                        ended_at=ended,
                        error="failed_to_create_out_dir",
                    )
                self._cancel_requested.discard(run_id)

            _append_event(out_dir, msg="researched_run_error", run_id=run_id, workflow=rec.workflow, error="failed_to_create_out_dir")
            _write_manifest_minimal(
                out_dir,
                run_id=run_id,
                workflow=rec.workflow,
                status="error",
                created_at=rec.created_at,
                started_at=started,
                ended_at=ended,
                command=[],
                env_overrides={},
                pid=None,
                returncode=None,
                error="failed_to_create_out_dir",
                phase="error",
                request=req,
                cancel_requested=False,
            )
            _ensure_metrics_json(out_dir, rec.workflow, run_status="error", returncode=None)
            _ensure_report_html(out_dir)
            return

        _bridge_events_evidence_start(out_dir)

        # Ensure control.json exists for control-aware workflows (FR-6).
        if wf.supports_controls:
            try:
                from electrodrive.live.controls import write_controls  # type: ignore

                write_controls(out_dir, {})
            except Exception:
                pass

        # If cancel was requested while we were preparing the run, abort before Popen.
        with self._lock:
            if run_id in self._cancel_requested:
                ended = time.time()
                cur = self._records.get(run_id)
                if cur is not None:
                    self._records[run_id] = dataclasses.replace(cur, status=RunStatus.KILLED, ended_at=ended, error="canceled_before_start")
                self._cancel_requested.discard(run_id)
                self._requests.pop(run_id, None)
            else:
                ended = None

        if ended is not None:
            _append_event(out_dir, msg="researched_run_canceled", run_id=run_id, workflow=rec.workflow)
            _bridge_events_evidence_finalize(out_dir)
            _ensure_metrics_json(out_dir, rec.workflow, run_status="killed", returncode=None)
            _ensure_report_html(out_dir)
            _write_manifest_minimal(
                out_dir,
                run_id=run_id,
                workflow=rec.workflow,
                status="killed",
                created_at=rec.created_at,
                started_at=started,
                ended_at=ended,
                command=[],
                env_overrides={},
                pid=None,
                returncode=None,
                error="canceled_before_start",
                phase="canceled",
                request=req,
                cancel_requested=True,
            )
            return

        try:
            command = wf.build_command(req, out_dir)
            env_overrides = _sanitize_env_overrides(wf.build_env(req, out_dir, run_id))
        except Exception as exc:
            ended = time.time()
            with self._lock:
                rec1 = self._records.get(run_id)
                if rec1 is not None:
                    self._records[run_id] = dataclasses.replace(
                        rec1,
                        status=RunStatus.ERROR,
                        ended_at=ended,
                        error=f"build_command_env_failed: {exc}",
                        command=[],
                        env={},
                    )
                self._cancel_requested.discard(run_id)

            _append_event(out_dir, msg="researched_run_error", run_id=run_id, workflow=rec.workflow, error=str(exc))
            _write_manifest_minimal(
                out_dir,
                run_id=run_id,
                workflow=rec.workflow,
                status="error",
                created_at=rec.created_at,
                started_at=started,
                ended_at=ended,
                command=[],
                env_overrides={},
                pid=None,
                returncode=None,
                error=str(exc),
                phase="error",
                request=req,
                cancel_requested=False,
            )
            _ensure_metrics_json(out_dir, rec.workflow, run_status="error", returncode=None)
            _ensure_report_html(out_dir)
            return

        _write_command_txt(out_dir, command, env_overrides)

        _write_manifest_minimal(
            out_dir,
            run_id=run_id,
            workflow=rec.workflow,
            status="starting",
            created_at=rec.created_at,
            started_at=started,
            ended_at=None,
            command=command,
            env_overrides=env_overrides,
            pid=None,
            returncode=None,
            error=None,
            phase="starting",
            request=req,
            cancel_requested=False,
        )

        stderr_path = out_dir / "stderr.log"
        try:
            if not stderr_path.exists():
                _atomic_write_text(
                    stderr_path,
                    "\n".join(
                        [
                            "# ResearchED stderr.log",
                            "# NOTE: this workflow runner merges stderr into stdout.log (stderr=STDOUT).",
                            f"# run_id: {run_id}",
                            f"# workflow: {rec.workflow}",
                            "",
                        ]
                    )
                    + "\n",
                )
        except Exception:
            pass

        stdout_path = out_dir / "stdout.log"
        try:
            _atomic_write_text(
                stdout_path,
                "\n".join(
                    [
                        "# ResearchED stdout.log",
                        f"# started_at_utc: {_utc_iso(started)}",
                        f"# run_id: {run_id}",
                        f"# workflow: {rec.workflow}",
                        f"# argv: {' '.join(command)}",
                        "",
                    ]
                )
                + "\n",
            )
        except Exception:
            pass

        tee_events: Path | None = None
        if rec.workflow == "learn_train":
            events_path = out_dir / "events.jsonl"
            try:
                if events_path.is_file() and events_path.stat().st_size == 0:
                    tee_events = events_path
            except Exception:
                tee_events = None

        env = os.environ.copy()
        env.update(env_overrides)

        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
            )
        except Exception as exc:
            ended = time.time()
            with self._lock:
                rec2 = self._records.get(run_id)
                if rec2 is not None:
                    self._records[run_id] = dataclasses.replace(
                        rec2,
                        status=RunStatus.ERROR,
                        ended_at=ended,
                        error=f"popen_failed: {exc}",
                        command=list(command),
                        env=dict(env_overrides),
                    )
                self._cancel_requested.discard(run_id)

            _append_event(out_dir, msg="researched_run_error", run_id=run_id, workflow=rec.workflow, error=f"popen_failed: {exc}")
            _write_manifest_minimal(
                out_dir,
                run_id=run_id,
                workflow=rec.workflow,
                status="error",
                created_at=rec.created_at,
                started_at=started,
                ended_at=ended,
                command=command,
                env_overrides=env_overrides,
                pid=None,
                returncode=None,
                error=str(exc),
                phase="error",
                request=req,
                cancel_requested=False,
            )
            _ensure_metrics_json(out_dir, rec.workflow, run_status="error", returncode=None)
            _ensure_report_html(out_dir)
            return

        try:
            pid = int(proc.pid)
        except Exception:
            pid = None

        with self._lock:
            rec3 = self._records.get(run_id)
            if rec3 is not None:
                self._records[run_id] = dataclasses.replace(
                    rec3,
                    status=RunStatus.RUNNING,
                    pid=pid,
                    command=list(command),
                    env=dict(env_overrides),
                )
            self._procs[run_id] = proc

        # Start stdout pump to avoid pipe deadlocks.
        pump = threading.Thread(
            target=_stdout_pump,
            kwargs={
                "proc": proc,
                "stdout_path": stdout_path,
                "tee_events_path": tee_events,
            },
            name=f"ResearchED-StdoutPump-{run_id}",
            daemon=True,
        )
        pump.start()
        with self._lock:
            self._pump_threads[run_id] = pump

        # If symlinks are unavailable, keep events/evidence mirrored live so legacy tailers work.
        stop_bridge = threading.Event()
        bridge_thread = _start_live_log_bridge(out_dir, stop_bridge)
        with self._lock:
            if bridge_thread is not None:
                self._bridge_threads[run_id] = bridge_thread
                self._bridge_stops[run_id] = stop_bridge

        _append_event(out_dir, msg="researched_run_started", run_id=run_id, workflow=rec.workflow, pid=pid)

        _write_manifest_minimal(
            out_dir,
            run_id=run_id,
            workflow=rec.workflow,
            status="running",
            created_at=rec.created_at,
            started_at=started,
            ended_at=None,
            command=command,
            env_overrides=env_overrides,
            pid=pid,
            returncode=None,
            error=None,
            phase="running",
            request=req,
            cancel_requested=False,
        )

        try:
            returncode = int(proc.wait())
        except Exception:
            returncode = None

        try:
            pump.join(timeout=2.0)
        except Exception:
            pass

        # Stop bridge thread (best-effort).
        with self._lock:
            ev = self._bridge_stops.get(run_id)
        if ev is not None:
            try:
                ev.set()
            except Exception:
                pass
        with self._lock:
            t = self._bridge_threads.get(run_id)
        if t is not None:
            try:
                t.join(timeout=1.0)
            except Exception:
                pass

        ended = time.time()

        with self._lock:
            cancel_requested = run_id in self._cancel_requested

        # Final status:
        # - If cancel requested and returncode != 0/None: killed
        # - If cancel requested but returncode == 0: success (but flagged in manifest)
        if cancel_requested and (returncode is None or returncode != 0):
            final_status = RunStatus.KILLED
            status_str = "killed"
            err_str = "terminated_by_user"
        else:
            if returncode == 0:
                final_status = RunStatus.SUCCESS
                status_str = "success"
                err_str = None
            else:
                final_status = RunStatus.ERROR
                status_str = "error"
                err_str = f"nonzero_returncode:{returncode}"

        _append_event(
            out_dir,
            msg="researched_run_exit",
            run_id=run_id,
            workflow=rec.workflow,
            returncode=returncode,
            status=status_str,
            cancel_requested=bool(cancel_requested),
        )

        _bridge_events_evidence_finalize(out_dir)
        _ensure_metrics_json(out_dir, rec.workflow, run_status=status_str, returncode=returncode)
        _ensure_report_html(out_dir)

        _write_manifest_minimal(
            out_dir,
            run_id=run_id,
            workflow=rec.workflow,
            status=status_str,
            created_at=rec.created_at,
            started_at=started,
            ended_at=ended,
            command=command,
            env_overrides=env_overrides,
            pid=pid,
            returncode=returncode,
            error=err_str,
            phase="finished",
            request=req,
            cancel_requested=bool(cancel_requested),
        )

        with self._lock:
            rec4 = self._records.get(run_id)
            if rec4 is not None:
                self._records[run_id] = dataclasses.replace(
                    rec4,
                    status=final_status,
                    ended_at=ended,
                    returncode=returncode,
                    error=err_str,
                )
            self._procs.pop(run_id, None)
            self._pump_threads.pop(run_id, None)
            self._bridge_threads.pop(run_id, None)
            self._bridge_stops.pop(run_id, None)
            self._requests.pop(run_id, None)
            self._cancel_requested.discard(run_id)
