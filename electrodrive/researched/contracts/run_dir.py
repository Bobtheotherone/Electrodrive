from __future__ import annotations

"""Run directory contract utilities for ResearchED.

Design Doc anchors (from “ResearchED GUI Design Document (Updated)” in this chat):
- FR-3: Run directory creation and artifact contract enforcement
- §1.4: events.jsonl vs evidence_log.jsonl compatibility policy (optionally bridge)
- §5.1: Unified run manifest schema (v1)

Notes:
- Cross-platform: symlink may fail on Windows; we fall back to hardlink or (as last resort)
  ensuring both files exist and copying at finalize.
- Safe with concurrent readers: we avoid truncation and prefer linking over copying.
- Subprocess workflows (e.g., electrodrive.cli solve) may overwrite manifest.json with
  their own schema. To keep UI-required fields stable, ResearchED also writes an owned
  manifest copy: manifest.researched.json.
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .manifest_schema import (
    MANIFEST_JSON_NAME,
    RESEARCHED_MANIFEST_NAME,
    collect_env_info,
    collect_git_info,
    infer_outputs,
    json_sanitize,
    new_manifest,
    utc_now_iso,
    validate_manifest,
)

# ----------------------------
# Atomic write helpers
# ----------------------------

def _atomic_write_text(path: Path, text: str) -> None:
    """Atomic file write: write to unique temp then os.replace.

    Using a unique temp name avoids collisions if multiple writers write the same path.
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
        # Best-effort fallback (non-atomic).
        try:
            path.write_text(text, encoding="utf-8")
        except Exception:
            pass
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _atomic_write_json(path: Path, data: Any) -> None:
    """Atomic JSON write with strict-JSON sanitization (NaN/Inf -> strings)."""
    safe = json_sanitize(data)
    try:
        txt = json.dumps(safe, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str)
    except Exception:
        # If allow_nan=False trips due to unexpected values, fall back to non-strict
        # but still sanitized.
        try:
            txt = json.dumps(safe, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            txt = "{}"
    _atomic_write_text(path, txt)


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# ----------------------------
# Run dir structure
# ----------------------------

def ensure_run_dir(run_dir: Path) -> None:
    """Ensure required run_dir structure exists (FR-3)."""
    rd = Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "artifacts").mkdir(parents=True, exist_ok=True)
    (rd / "plots").mkdir(parents=True, exist_ok=True)

    # Minimum contract: metrics.json exists (may be empty until finalization).
    metrics = rd / "metrics.json"
    if not metrics.exists():
        try:
            _atomic_write_json(metrics, {})
        except Exception:
            try:
                metrics.write_text("{}\n", encoding="utf-8")
            except Exception:
                pass

    # Minimum contract: report.html exists (stub; later overwritten).
    report = rd / "report.html"
    if not report.exists():
        stub = "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8'/>",
                "<title>ResearchED report (stub)</title>",
                "<style>body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:2rem}</style>",
                "</head><body>",
                "<h1>ResearchED report</h1>",
                "<p>This is a placeholder <code>report.html</code>. ResearchED will overwrite it.</p>",
                "</body></html>",
                "",
            ]
        )
        try:
            _atomic_write_text(report, stub)
        except Exception:
            try:
                report.write_text(stub, encoding="utf-8")
            except Exception:
                pass


def write_command_txt(run_dir: Path, argv: Sequence[str], env: Mapping[str, str] | None = None) -> Path:
    """Write command.txt for reproducibility (FR-3)."""
    rd = Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    path = rd / "command.txt"

    lines: list[str] = []
    lines.append("# ResearchED command.txt")
    lines.append(f"# timestamp_utc: {utc_now_iso()}")
    try:
        lines.append(f"# cwd: {os.getcwd()}")
    except Exception:
        lines.append("# cwd: <unavailable>")
    lines.append("")
    lines.append("argv:")
    lines.append("  " + " ".join(str(x) for x in argv))
    lines.append("")
    if env:
        lines.append("env_overrides:")
        for k in sorted(env.keys()):
            v = str(env.get(k, "")).replace("\n", "\\n")
            lines.append(f"  {k}={v}")
        lines.append("")
    text = "\n".join(lines) + "\n"

    try:
        _atomic_write_text(path, text)
    except Exception:
        try:
            path.write_text(text, encoding="utf-8")
        except Exception:
            pass

    return path


# ----------------------------
# Log filename compatibility (events.jsonl vs evidence_log.jsonl)
# ----------------------------

def _try_symlink(link_path: Path, target_name: str) -> bool:
    """Try to create a relative symlink link_path -> target_name (same directory)."""
    try:
        if link_path.exists() or link_path.is_symlink():
            return True
    except Exception:
        pass
    try:
        os.symlink(target_name, link_path)
        return True
    except Exception:
        return False


def _try_hardlink(link_path: Path, target_path: Path) -> bool:
    """Try to create hard link link_path -> target_path."""
    try:
        if link_path.exists():
            return True
    except Exception:
        pass
    try:
        os.link(target_path, link_path)
        return True
    except Exception:
        return False


def ensure_log_compat(run_dir: Path, *, primary: str = "events.jsonl", legacy: str = "evidence_log.jsonl") -> Dict[str, Any]:
    """Ensure log filename compatibility (Design Doc §1.4 + FR-3)."""
    rd = Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    p = rd / primary
    l = rd / legacy

    def touch(path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
        except Exception:
            return

    p_exists = p.exists()
    l_exists = l.exists()

    if not p_exists and not l_exists:
        touch(p)
        p_exists = True

    if p_exists and not l_exists:
        if _try_symlink(l, p.name):
            return {"mode": "symlink", "target": p.name, "link": l.name}
        if _try_hardlink(l, p):
            return {"mode": "hardlink", "target": p.name, "link": l.name}
        touch(l)
        return {"mode": "copy_on_finalize", "source": p.name, "dest": l.name}

    if l_exists and not p_exists:
        if _try_symlink(p, l.name):
            return {"mode": "symlink", "target": l.name, "link": p.name}
        if _try_hardlink(p, l):
            return {"mode": "hardlink", "target": l.name, "link": p.name}
        touch(p)
        return {"mode": "copy_on_finalize", "source": l.name, "dest": p.name}

    # Both exist. If one is empty and the other isn't, note for potential finalize copy.
    try:
        ps = p.stat().st_size if p.exists() else 0
        ls = l.stat().st_size if l.exists() else 0
        if ps > 0 and ls == 0:
            return {"mode": "both_present_copy_suggested", "source": p.name, "dest": l.name}
        if ls > 0 and ps == 0:
            return {"mode": "both_present_copy_suggested", "source": l.name, "dest": p.name}
    except Exception:
        pass
    return {"mode": "both_present", "primary": p.name, "legacy": l.name}


def _finalize_log_copy(run_dir: Path, primary: str, legacy: str) -> None:
    """Copy content at finalize when linking was impossible and one file is empty/missing."""
    rd = Path(run_dir)
    p = rd / primary
    l = rd / legacy

    try:
        p_exists = p.exists()
        l_exists = l.exists()
    except Exception:
        return

    if p_exists and not l_exists:
        try:
            shutil.copyfile(p, l)
        except Exception:
            return
        return

    if l_exists and not p_exists:
        try:
            shutil.copyfile(l, p)
        except Exception:
            return
        return

    if p_exists and l_exists:
        try:
            ps = p.stat().st_size
            ls = l.stat().st_size
        except Exception:
            return
        if ps > 0 and ls == 0:
            try:
                shutil.copyfile(p, l)
            except Exception:
                return
        elif ls > 0 and ps == 0:
            try:
                shutil.copyfile(l, p)
            except Exception:
                return


# ----------------------------
# Manifest writing policy (pair: manifest.researched.json + manifest.json)
# ----------------------------

def _ensure_manifest_v1_shape(man: Dict[str, Any]) -> Dict[str, Any]:
    # schema_version
    if not isinstance(man.get("schema_version"), int):
        man["schema_version"] = 1

    # required mapping blocks
    for k in ("git", "env", "inputs", "outputs", "gate"):
        if not isinstance(man.get(k), dict):
            man[k] = {}

    # spec_digest must be dict
    if not isinstance(man.get("spec_digest"), dict):
        man["spec_digest"] = {}

    # timestamps/status minimal safety
    if not isinstance(man.get("started_at"), str) or not str(man.get("started_at") or "").strip():
        man["started_at"] = utc_now_iso()
    if "ended_at" not in man:
        man["ended_at"] = None
    if not isinstance(man.get("status"), str) or not str(man.get("status") or "").strip():
        man["status"] = "running"

    # git defaults
    git = man["git"]
    git.setdefault("sha", None)
    git.setdefault("branch", None)
    git.setdefault("dirty", None)
    git.setdefault("diff_summary", None)

    # gate defaults
    gate = man["gate"]
    gate.setdefault("gate1_status", None)
    gate.setdefault("gate2_status", None)
    gate.setdefault("gate3_status", None)
    gate.setdefault("structure_score", None)
    gate.setdefault("novelty_score", None)

    return man


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


def _write_manifest_pair(run_dir: Path, manifest: Dict[str, Any], *, terminal: bool = False) -> None:
    """Write the ResearchED-owned manifest and best-effort sync manifest.json.

    - Always writes manifest.researched.json (authoritative for UI-required fields).
    - Writes manifest.json if missing.
    - If manifest.json exists and terminal=True, merges terminal fields into it
      (preserving workflow-written keys).
    - If terminal=False and manifest.json exists, avoids overwriting to prevent
      clobbering concurrent workflow writers; only creates if absent.
    """
    manifest = _ensure_manifest_v1_shape(dict(manifest))
    rd = Path(run_dir)
    researched_path = rd / RESEARCHED_MANIFEST_NAME
    workflow_path = rd / MANIFEST_JSON_NAME

    # Always write ResearchED manifest.
    try:
        _atomic_write_json(researched_path, manifest)
    except Exception:
        pass

    # For manifest.json: create if absent.
    if not workflow_path.exists():
        try:
            _atomic_write_json(workflow_path, manifest)
        except Exception:
            pass
        return

    if not terminal:
        return

    # Terminal merge into existing manifest.json (preserve workflow keys).
    existing = _safe_read_json(workflow_path)
    if not isinstance(existing, dict):
        existing = {}
    term_updates = {
        "run_id": manifest.get("run_id"),
        "workflow": manifest.get("workflow"),
        "started_at": existing.get("started_at") or manifest.get("started_at"),
        "ended_at": manifest.get("ended_at"),
        "status": manifest.get("status"),
        "error": manifest.get("error"),
        "outputs": manifest.get("outputs", {}),
        "researched": manifest.get("researched", {}),
    }
    try:
        merged2 = _merge_manifest(existing, json_sanitize(term_updates))
        merged2 = _ensure_manifest_v1_shape(dict(merged2))
        _atomic_write_json(workflow_path, merged2)
    except Exception:
        pass


# ----------------------------
# Public API: init/finalize
# ----------------------------

def init_run(
    run_dir: Path,
    *,
    workflow: str,
    argv: Sequence[str],
    inputs: dict,
    repo_root: Path | None = None,
    env: dict | None = None,
    git: dict | None = None,
) -> Dict[str, Any]:
    """Initialize a run directory and write initial manifests (status="running") (FR-3)."""
    rd = Path(run_dir)
    ensure_run_dir(rd)

    env_overrides = dict(env or {})
    write_command_txt(rd, argv=argv, env=env_overrides)

    log_compat_info = ensure_log_compat(rd, primary="events.jsonl", legacy="evidence_log.jsonl")

    run_id = str(inputs.get("run_id") or "")
    if not run_id:
        import uuid
        run_id = str(uuid.uuid4())

    # Collect git/env info best-effort.
    git_info = dict(git or {})
    if not git_info and repo_root is not None:
        try:
            git_info = dict(collect_git_info(Path(repo_root)))
        except Exception:
            git_info = {}

    env_info = collect_env_info()

    # Build schema-compliant inputs block.
    inputs_block = dict(inputs or {})
    inputs_block.setdefault("command", list(argv))
    inputs_block.setdefault("env_overrides", env_overrides)

    # Fill outputs pointers (stable keys).
    outputs_block = infer_outputs(rd)
    outputs_block["events_jsonl"] = "events.jsonl"
    outputs_block["evidence_log_jsonl"] = "evidence_log.jsonl"
    outputs_block["manifest_json"] = MANIFEST_JSON_NAME
    outputs_block["researched_manifest_json"] = RESEARCHED_MANIFEST_NAME

    manifest = new_manifest(
        run_id=run_id,
        workflow=str(workflow),
        status="running",
        started_at=utc_now_iso(),
        ended_at=None,
        inputs=inputs_block,
        outputs=outputs_block,
        git=git_info,
        env=env_info,
        gate=None,
        spec_digest=None,
        error=None,
    )

    # Include compatibility action details (audit/debug).
    try:
        r = manifest.get("researched") if isinstance(manifest.get("researched"), dict) else {}
        r = dict(r)
        r.update({"schema_version": 1, "log_compat": log_compat_info})
        manifest["researched"] = r
    except Exception:
        manifest["researched"] = {"schema_version": 1, "log_compat": log_compat_info}

    # Validate before writing.
    try:
        validate_manifest(manifest)
    except Exception:
        # Do not hard-fail init on schema validation; still write best-effort.
        pass

    manifest = _ensure_manifest_v1_shape(dict(manifest))

    _write_manifest_pair(rd, dict(manifest), terminal=False)
    return dict(manifest)


def finalize_run(
    run_dir: Path,
    *,
    status: str,
    error: str | None = None,
    extra_outputs: dict | None = None,
) -> Dict[str, Any]:
    """Finalize a run by updating manifests atomically (FR-3)."""
    rd = Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)

    # Finalize log compat: ensure both names exist and copy if needed.
    ensure_log_compat(rd, primary="events.jsonl", legacy="evidence_log.jsonl")
    _finalize_log_copy(rd, primary="events.jsonl", legacy="evidence_log.jsonl")

    # Coerce status to the design-doc enum.
    st = str(status).strip().lower()
    if st not in {"running", "success", "error", "killed"}:
        # Preserve caller's intent as an error string; keep status valid.
        if error is None:
            error = f"invalid_status:{status}"
        st = "error"

    # Load existing ResearchED manifest if present, else synthesize from manifest.json.
    existing_researched = _safe_read_json(rd / RESEARCHED_MANIFEST_NAME)
    existing_workflow = _safe_read_json(rd / MANIFEST_JSON_NAME)
    base = existing_researched if isinstance(existing_researched, dict) else (existing_workflow if isinstance(existing_workflow, dict) else {})

    # Identify stable fields.
    run_id = base.get("run_id") or rd.name
    workflow = base.get("workflow") or "unknown"
    started_at = base.get("started_at") or utc_now_iso()

    # Merge outputs (prefer current filesystem reality).
    outputs_block = dict(infer_outputs(rd))
    if extra_outputs:
        for k, v in extra_outputs.items():
            outputs_block[str(k)] = v
    outputs_block.setdefault("events_jsonl", "events.jsonl")
    outputs_block.setdefault("evidence_log_jsonl", "evidence_log.jsonl")
    outputs_block.setdefault("manifest_json", MANIFEST_JSON_NAME)
    outputs_block.setdefault("researched_manifest_json", RESEARCHED_MANIFEST_NAME)

    ended_at = utc_now_iso()

    # Build a v1-compliant manifest (this guarantees validate_manifest can pass for the ResearchED copy).
    # We preserve git/env/inputs/gate/spec_digest where possible.
    manifest_out = new_manifest(
        run_id=str(run_id),
        workflow=str(workflow),
        status=st,
        started_at=str(started_at),
        ended_at=ended_at,
        inputs=base.get("inputs") if isinstance(base.get("inputs"), Mapping) else {},
        outputs=outputs_block,
        git=base.get("git") if isinstance(base.get("git"), Mapping) else {},
        env=base.get("env") if isinstance(base.get("env"), Mapping) else collect_env_info(),
        gate=base.get("gate") if isinstance(base.get("gate"), Mapping) else None,
        spec_digest=base.get("spec_digest") if isinstance(base.get("spec_digest"), Mapping) else None,
        error=str(error) if error is not None else None,
    )

    # Preserve extra keys under researched, if present.
    try:
        prev_r = base.get("researched") if isinstance(base.get("researched"), Mapping) else {}
        manifest_out["researched"] = dict(prev_r)
        manifest_out["researched"].update({"finalized_at": ended_at})
    except Exception:
        pass

    # Validate ResearchED manifest (should pass, but keep best-effort).
    try:
        validate_manifest(manifest_out)
    except Exception:
        pass

    manifest_out = _ensure_manifest_v1_shape(dict(manifest_out))

    # Write both manifests (terminal=True merges into manifest.json).
    _write_manifest_pair(rd, dict(manifest_out), terminal=True)
    return dict(manifest_out)
