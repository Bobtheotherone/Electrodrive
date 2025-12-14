from __future__ import annotations

"""
FMM sanity-suite workflow launcher for ResearchED.

Design Doc mapping:
- FR-1: workflow discovery/launch
- §3.2 RunManager -> subprocess spawns electrodrive.fmm3d.sanity_suite
- FR-3: run dir contract -> events.jsonl under out_dir; RunManager bridges evidence name

Repo behavior (current):
- FMM JSONL emission is controlled via env vars (see electrodrive.fmm3d.logging_utils):
    - EDE_FMM_ENABLE_JSONL
    - EDE_FMM_JSONL_PATH
    - EDE_FMM_JSONL_NO_STDOUT
"""

import functools
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional


class FmmSuiteWorkflow:
    name = "fmm_suite"
    supports_controls = False

    _HELP_TIMEOUT_S = 3.0

    def validate_request(self, request: dict) -> None:
        if not isinstance(request, dict):
            raise ValueError("request must be a dict")

        extra = request.get("extra_args", [])
        if extra is not None and not (isinstance(extra, list) and all(isinstance(x, str) for x in extra)):
            raise ValueError("'extra_args' must be a list[str] if provided")

        if "jsonl" in request and request.get("jsonl") is not None and not isinstance(request.get("jsonl"), bool):
            raise ValueError("'jsonl' must be boolean if provided")

        for k in ("device", "dtype"):
            if k in request and request.get(k) is not None and not isinstance(request.get(k), str):
                raise ValueError(f"'{k}' must be a string if provided")

        # Numeric knobs: reject booleans; allow int/float/str.
        for k in ("n_points", "tol_p2p", "tol_fmm", "tol_bem", "expansion_order", "mac_theta", "leaf_size"):
            if k not in request:
                continue
            v = request.get(k)
            if v is None:
                continue
            if isinstance(v, bool):
                raise ValueError(f"'{k}' must not be boolean")

    @functools.lru_cache(maxsize=1)
    def _suite_help_text(self) -> str:
        """Best-effort probe of sanity_suite CLI help (flags drift across versions)."""
        try:
            cp = subprocess.run(
                [sys.executable, "-m", "electrodrive.fmm3d.sanity_suite", "-h"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=float(self._HELP_TIMEOUT_S),
            )
            return cp.stdout or ""
        except Exception:
            return ""

    def _flag_supported(self, flag: str) -> bool:
        return flag in self._suite_help_text()

    def _opt_if_supported(self, cmd: List[str], flag: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, bool):
            raise ValueError(f"{flag} must not be boolean")
        if self._flag_supported(flag):
            cmd.extend([flag, str(value)])

    def build_command(self, request: dict, out_dir: Path) -> list[str]:
        self.validate_request(request)

        cmd: List[str] = [
            sys.executable,
            "-m",
            "electrodrive.fmm3d.sanity_suite",
        ]

        # Optional CLI flags: only add if the CLI advertises them.
        # (JSONL is env-controlled; we do NOT pass a --jsonl flag.)
        self._opt_if_supported(cmd, "--device", request.get("device"))
        self._opt_if_supported(cmd, "--dtype", request.get("dtype"))

        # Common knobs (may or may not exist on the CLI depending on version).
        self._opt_if_supported(cmd, "--n-points", request.get("n_points"))
        self._opt_if_supported(cmd, "--tol-p2p", request.get("tol_p2p"))
        self._opt_if_supported(cmd, "--tol-fmm", request.get("tol_fmm"))
        self._opt_if_supported(cmd, "--tol-bem", request.get("tol_bem"))

        # Sweep-friendly knobs (if supported by the CLI in your branch).
        self._opt_if_supported(cmd, "--expansion-order", request.get("expansion_order"))
        self._opt_if_supported(cmd, "--mac-theta", request.get("mac_theta"))
        self._opt_if_supported(cmd, "--leaf-size", request.get("leaf_size"))

        extra_args = request.get("extra_args") or []
        if isinstance(extra_args, list):
            cmd.extend([str(x) for x in extra_args if isinstance(x, str)])

        return cmd

    def build_env(self, request: dict, out_dir: Path, run_id: str) -> dict[str, str]:
        env: dict[str, str] = {
            "EDE_RUN_DIR": str(out_dir),
            "EDE_RUN_ID": str(run_id),
            "PYTHONUNBUFFERED": "1",
        }

        # Design Doc FR-3/FR-4: for GUI runs we always want a machine-readable event stream.
        # The underlying suite uses env vars for JSONL emission (logging_utils.want_jsonl).
        env.update(
            {
                "EDE_FMM_ENABLE_JSONL": "1",
                "EDE_FMM_JSONL_PATH": str((out_dir / "events.jsonl").resolve()),
                "EDE_FMM_JSONL_NO_STDOUT": "1",
            }
        )
        return env

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "supports_controls": False,
            "description": "FMM3D sanity tests via `python -m electrodrive.fmm3d.sanity_suite`",
            "request_template": {
                "device": "cpu",
                "dtype": "float64",
                "n_points": 512,
                "tol_p2p": 1e-10,
                "tol_fmm": 1e-2,
                "tol_bem": 1e-1,
                "expansion_order": None,
                "mac_theta": None,
                "leaf_size": None,
                "jsonl": True,
                "extra_args": [],
            },
        }
