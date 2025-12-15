from __future__ import annotations

"""ResearchED contracts package: run directory contract + unified manifest helpers.

Design Doc anchors (from “ResearchED GUI Design Document (Updated)” in this chat):
- FR-3: Run directory creation and artifact contract enforcement
- §1.4: Compatibility policy: events.jsonl vs evidence_log.jsonl
- §5.1: Unified run manifest schema (v1)

This package is stdlib-only and safe to import without GUI dependencies.
"""

from .manifest_schema import (
    MANIFEST_JSON_NAME,
    RESEARCHED_MANIFEST_NAME,
    collect_env_info,
    collect_git_info,
    infer_outputs,
    json_sanitize,
    new_manifest,
    update_manifest_status,
    utc_now_iso,
    validate_manifest,
)
from .run_dir import (
    ensure_log_compat,
    ensure_run_dir,
    finalize_run,
    init_run,
    init_run_dir,
    create_run_dir,
    write_command_txt,
)

__all__ = [
    # constants
    "MANIFEST_JSON_NAME",
    "RESEARCHED_MANIFEST_NAME",
    # manifest_schema
    "utc_now_iso",
    "collect_git_info",
    "collect_env_info",
    "json_sanitize",
    "new_manifest",
    "update_manifest_status",
    "infer_outputs",
    "validate_manifest",
    # run_dir
    "ensure_run_dir",
    "write_command_txt",
    "ensure_log_compat",
    "init_run",
    "init_run_dir",
    "create_run_dir",
    "finalize_run",
]
