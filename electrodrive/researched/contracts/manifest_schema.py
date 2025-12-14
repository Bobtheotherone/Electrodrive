from __future__ import annotations

"""Unified manifest schema (v1) helpers for ResearchED.

Design Doc anchors (from “ResearchED GUI Design Document (Updated)” in this chat):
- FR-3: Run directory creation and artifact contract enforcement
- §5.1: Unified run manifest schema (v1)
- §1.4: Compatibility policy: events.jsonl vs evidence_log.jsonl

This module is stdlib-only by design (GUI is an optional extra; no new heavy deps).

Notes:
- ResearchED may additionally write an owned manifest copy (manifest.researched.json) to
  avoid losing UI-required fields when subprocess workflows overwrite manifest.json.
  The design doc does not forbid extra manifest files; validate_manifest allows extras.
"""

import math
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, TypedDict

# Manifest filenames (used by run_dir helpers and UI).
MANIFEST_JSON_NAME = "manifest.json"
RESEARCHED_MANIFEST_NAME = "manifest.researched.json"

# ----------------------------
# Types (lightweight; forward-compatible)
# ----------------------------

class GitInfo(TypedDict, total=False):
    sha: Optional[str]
    branch: Optional[str]
    dirty: Optional[bool]
    diff_summary: Optional[str]


class EnvInfo(TypedDict, total=False):
    python_version: Optional[str]
    torch_version: Optional[str]
    device: Optional[str]
    dtype: Optional[str]
    host: Optional[str]
    platform: Optional[str]


class InputsInfo(TypedDict, total=False):
    spec_path: Optional[str]
    config: Optional[str]
    command: List[str]
    env_overrides: Dict[str, str]


class OutputsInfo(TypedDict, total=False):
    # core contract files
    metrics_json: Optional[str]
    events_jsonl: Optional[str]
    evidence_log_jsonl: Optional[str]
    command_txt: Optional[str]
    report_html: Optional[str]
    artifacts_dir: Optional[str]
    plots_dir: Optional[str]
    viz_dir: Optional[str]

    # optional helpful extras (raw logs)
    stdout_log: Optional[str]
    stderr_log: Optional[str]

    # optional pointers to manifest files
    manifest_json: Optional[str]
    researched_manifest_json: Optional[str]


class GateInfo(TypedDict, total=False):
    gate1_status: Optional[str]
    gate2_status: Optional[str]
    gate3_status: Optional[str]
    structure_score: Optional[float]
    novelty_score: Optional[float]


class ManifestV1(TypedDict, total=False):
    run_id: str
    workflow: str
    started_at: Optional[str]
    ended_at: Optional[str]
    status: str  # running|success|error|killed
    error: Optional[str]
    git: GitInfo
    env: EnvInfo
    inputs: InputsInfo
    outputs: OutputsInfo
    gate: GateInfo
    spec_digest: Optional[Dict[str, Any]]
    schema_version: int


_ALLOWED_STATUS = {"running", "success", "error", "killed"}


# ----------------------------
# JSON safety helpers
# ----------------------------

def _json_serialize_float_like(v: Any) -> Any:
    """Sanitize floats (NaN/Inf) for strict JSON.

    Repo precedent (electrodrive/cli.py) serializes NaN/Inf as strings.
    """
    try:
        f = float(v)
    except Exception:
        return v
    if math.isfinite(f):
        return f
    if math.isnan(f):
        return "NaN"
    return "Infinity" if f > 0 else "-Infinity"


def _json_sanitize(obj: Any) -> Any:
    """Recursively sanitize for strict JSON.

    - NaN/Inf floats -> strings
    - Paths -> str
    - sets -> sorted list
    - dict keys -> str
    - unknown objects -> str/float-like fallback
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return _json_serialize_float_like(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            out[str(k)] = _json_sanitize(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_json_sanitize(x) for x in obj)
    # last resort: try float conversion, else string
    try:
        return _json_serialize_float_like(obj)
    except Exception:
        return str(obj)


def json_sanitize(obj: Any) -> Any:
    """Public wrapper for JSON sanitization (used by run_dir contract writers)."""
    return _json_sanitize(obj)


# ----------------------------
# Public helpers
# ----------------------------

def utc_now_iso() -> str:
    """UTC timestamp in ISO-8601 (Z-suffixed) form."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_git(repo_root: Path, args: Sequence[str], *, timeout_s: float = 2.0) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout_s,
        ).strip()
        return out or None
    except Exception:
        return None


def collect_git_info(repo_root: Path) -> GitInfo:
    """Best-effort git info collection."""
    root = Path(repo_root)
    sha = _run_git(root, ["rev-parse", "HEAD"])
    branch = _run_git(root, ["rev-parse", "--abbrev-ref", "HEAD"])

    dirty: Optional[bool]
    try:
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
        dirty = bool(porcelain.strip())
    except Exception:
        dirty = None

    diff_summary: Optional[str] = None
    if dirty:
        # Keep this lightweight: list a few changed files.
        try:
            names = subprocess.check_output(
                ["git", "diff", "--name-only"],
                cwd=str(root),
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.0,
            )
            changed = [ln.strip() for ln in names.splitlines() if ln.strip()]
            if changed:
                head = changed[:50]
                more = len(changed) - len(head)
                diff_summary = ", ".join(head) + (f" (+{more} more)" if more > 0 else "")
        except Exception:
            diff_summary = None

    return {
        "sha": sha,
        "branch": branch,
        "dirty": dirty,
        "diff_summary": diff_summary,
    }


def collect_env_info() -> EnvInfo:
    """Collect a best-effort environment summary for manifest.env."""
    host = None
    try:
        host = socket.gethostname()
    except Exception:
        host = None

    torch_version: Optional[str] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    try:
        import importlib

        torch = importlib.import_module("torch")
        torch_version = getattr(torch, "__version__", None) or "unknown"
        try:
            cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        except Exception:
            cuda_ok = False
        device = "cuda" if cuda_ok else "cpu"
        try:
            dtype = str(torch.get_default_dtype())
        except Exception:
            dtype = None
    except Exception:
        torch_version = None
        device = None
        dtype = None

    return {
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch_version,
        "device": device,
        "dtype": dtype,
        "host": host,
        "platform": platform.platform(),
    }


def new_manifest(
    *,
    run_id: str,
    workflow: str,
    status: str = "running",
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
    inputs: Optional[Mapping[str, Any]] = None,
    outputs: Optional[Mapping[str, Any]] = None,
    git: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, Any]] = None,
    gate: Optional[Mapping[str, Any]] = None,
    spec_digest: Optional[Mapping[str, Any]] = None,
    error: Optional[str] = None,
) -> ManifestV1:
    """Create a schema-compliant manifest dict with sensible defaults."""
    st = str(status)
    if st not in _ALLOWED_STATUS:
        raise ValueError(f"Invalid status {st!r}. Allowed: {sorted(_ALLOWED_STATUS)}")

    if started_at is None:
        started_at = utc_now_iso()

    manifest: ManifestV1 = {
        "schema_version": 1,
        "run_id": str(run_id),
        "workflow": str(workflow),
        "started_at": started_at,
        "ended_at": ended_at,
        "status": st,
        "error": error,
        "git": dict(git or {}),
        "env": dict(env or {}),
        "inputs": dict(inputs or {}),
        "outputs": dict(outputs or {}),
        "gate": dict(gate or {}),
        # placeholder: may be replaced by SpecInspector
        "spec_digest": dict(spec_digest) if spec_digest is not None else {},
    }

    # Ensure required gate keys exist (may be None).
    g = manifest["gate"]
    g.setdefault("gate1_status", None)
    g.setdefault("gate2_status", None)
    g.setdefault("gate3_status", None)
    g.setdefault("structure_score", None)
    g.setdefault("novelty_score", None)

    # Ensure input/output keys exist (schema stability).
    ins = manifest["inputs"]
    ins.setdefault("spec_path", None)
    ins.setdefault("config", None)
    ins.setdefault("command", [])
    ins.setdefault("env_overrides", {})

    outs = manifest["outputs"]
    outs.setdefault("metrics_json", "metrics.json")
    outs.setdefault("events_jsonl", "events.jsonl")
    outs.setdefault("evidence_log_jsonl", "evidence_log.jsonl")
    outs.setdefault("viz_dir", "viz")
    outs.setdefault("artifacts_dir", "artifacts")
    outs.setdefault("plots_dir", "plots")
    outs.setdefault("report_html", "report.html")
    outs.setdefault("command_txt", "command.txt")
    outs.setdefault("stdout_log", "stdout.log")
    outs.setdefault("stderr_log", "stderr.log")
    outs.setdefault("manifest_json", MANIFEST_JSON_NAME)
    outs.setdefault("researched_manifest_json", RESEARCHED_MANIFEST_NAME)

    # Ensure git/env keys exist.
    gi = manifest["git"]
    gi.setdefault("sha", None)
    gi.setdefault("branch", None)
    gi.setdefault("dirty", None)
    gi.setdefault("diff_summary", None)

    ei = manifest["env"]
    ei.setdefault("python_version", None)
    ei.setdefault("torch_version", None)
    ei.setdefault("device", None)
    ei.setdefault("dtype", None)
    ei.setdefault("host", None)
    ei.setdefault("platform", None)

    return _json_sanitize(manifest)  # type: ignore[return-value]


def update_manifest_status(
    manifest: Mapping[str, Any],
    status: str,
    *,
    ended_at: Optional[str] = None,
    error: Optional[str] = None,
) -> ManifestV1:
    """Return a new dict with updated status/ended_at/error fields."""
    st = str(status)
    if st not in _ALLOWED_STATUS:
        raise ValueError(f"Invalid status {st!r}. Allowed: {sorted(_ALLOWED_STATUS)}")
    out: Dict[str, Any] = dict(manifest)
    out["status"] = st
    if ended_at is not None:
        out["ended_at"] = ended_at
    if error is not None:
        out["error"] = error
    return _json_sanitize(out)  # type: ignore[return-value]


def infer_outputs(run_dir: Path) -> OutputsInfo:
    """Populate outputs pointers to known artifacts if present."""
    rd = Path(run_dir)
    outs: OutputsInfo = {
        "metrics_json": "metrics.json" if (rd / "metrics.json").is_file() else None,
        "events_jsonl": "events.jsonl" if (rd / "events.jsonl").exists() else None,
        "evidence_log_jsonl": "evidence_log.jsonl" if (rd / "evidence_log.jsonl").exists() else None,
        "viz_dir": "viz" if (rd / "viz").is_dir() else None,
        "artifacts_dir": "artifacts" if (rd / "artifacts").is_dir() else None,
        "plots_dir": "plots" if (rd / "plots").is_dir() else None,
        "report_html": "report.html" if (rd / "report.html").is_file() else None,
        "command_txt": "command.txt" if (rd / "command.txt").is_file() else None,
        "stdout_log": "stdout.log" if (rd / "stdout.log").is_file() else None,
        "stderr_log": "stderr.log" if (rd / "stderr.log").is_file() else None,
        "manifest_json": MANIFEST_JSON_NAME if (rd / MANIFEST_JSON_NAME).is_file() else None,
        "researched_manifest_json": RESEARCHED_MANIFEST_NAME if (rd / RESEARCHED_MANIFEST_NAME).is_file() else None,
    }
    return outs


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Lightweight structural checks (no pydantic)."""
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")

    def req_str(k: str) -> str:
        v = manifest.get(k)
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"manifest[{k!r}] must be a non-empty string")
        return v

    req_str("run_id")
    req_str("workflow")

    status = manifest.get("status")
    if status not in _ALLOWED_STATUS:
        raise ValueError(f"manifest['status'] must be one of {sorted(_ALLOWED_STATUS)}")

    for k in ("git", "env", "inputs", "outputs", "gate"):
        v = manifest.get(k)
        if not isinstance(v, Mapping):
            raise ValueError(f"manifest[{k!r}] must be an object")

    gate = manifest.get("gate", {})
    if not isinstance(gate, Mapping):
        raise ValueError("manifest['gate'] must be an object")
    for k in ("gate1_status", "gate2_status", "gate3_status", "structure_score", "novelty_score"):
        if k not in gate:
            raise ValueError(f"manifest['gate'] missing required key: {k!r}")

    if "spec_digest" not in manifest:
        raise ValueError("manifest missing required key: 'spec_digest'")

    if "started_at" not in manifest:
        raise ValueError("manifest missing required key: 'started_at'")
