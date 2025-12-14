from __future__ import annotations

"""
Solve workflow launcher for ResearchED.

Design Doc mapping:
- FR-1: workflow discovery/launch -> build exact CLI argv
- §3.2 RunManager -> subprocess spawns electrodrive CLI
- FR-6: solve supports control.json protocol

Notes:
- The Electrodrive solve CLI has evolved across versions (flag names may drift).
  We avoid guessing optional flags by probing `python -m electrodrive.cli solve -h`.
- Some branches treat `--viz` as a boolean flag, others may treat it as a path.
  We detect whether `--viz` takes a value from the help text to avoid launching
  a broken command.
"""

import functools
import re
import subprocess
import sys
from pathlib import Path
from typing import List


class SolveWorkflow:
    name = "solve"
    supports_controls = True

    _HELP_TIMEOUT_S = 3.0

    def validate_request(self, request: dict) -> None:
        if not isinstance(request, dict):
            raise ValueError("request must be a dict")

        spec = request.get("spec_path") or request.get("problem") or request.get("spec")
        if not spec or not isinstance(spec, str):
            raise ValueError("solve requires 'spec_path' (or 'problem'/'spec') as a string path")

        mode = request.get("mode", "auto")
        if mode is not None and not isinstance(mode, str):
            raise ValueError("'mode' must be a string if provided")

        extra = request.get("extra_args", [])
        if extra is not None and not (isinstance(extra, list) and all(isinstance(x, str) for x in extra)):
            raise ValueError("'extra_args' must be a list[str] if provided")

        # Boolean knobs (accept both snake_case and dash-case inputs).
        bool_keys = {
            "cert",
            "fast",
            "cert_fast",
            "cert-fast",
            "viz_enable",
        }
        for k in bool_keys:
            if k in request and request.get(k) is not None and not isinstance(request.get(k), bool):
                raise ValueError(f"'{k}' must be boolean if provided")

        viz_dir = request.get("viz_dir") or request.get("viz-dir")
        if viz_dir is not None and not isinstance(viz_dir, str):
            raise ValueError("'viz_dir' must be a string path if provided")

    @functools.lru_cache(maxsize=1)
    def _solve_help_text(self) -> str:
        """Best-effort probe of solve CLI help to avoid guessing optional flags."""
        try:
            cp = subprocess.run(
                [sys.executable, "-m", "electrodrive.cli", "solve", "-h"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=float(self._HELP_TIMEOUT_S),
            )
            return cp.stdout or ""
        except Exception:
            return ""

    def _help_available(self) -> bool:
        return bool(self._solve_help_text().strip())

    def _flag_supported(self, flag: str) -> bool:
        txt = self._solve_help_text()
        if not txt:
            return False
        return flag in txt

    def _flag_takes_value(self, flag: str) -> bool:
        """Heuristic: True if help text suggests this flag expects an argument."""
        txt = self._solve_help_text()
        if not txt:
            return False

        # Look for patterns like: "--viz VIZ_DIR" or "--viz=<path>"
        pat = re.compile(re.escape(flag) + r"(?:\s+|=)(<[^>]+>|[A-Z][A-Z0-9_\-]*)")
        if pat.search(txt):
            return True

        for line in txt.splitlines():
            if flag not in line:
                continue
            if pat.search(line):
                return True
        return False

    def _problem_argv(self, spec_path: str) -> list[str]:
        # If help is unavailable (e.g., environment issues), default to the repo's
        # stable interface: --problem PATH.
        if not self._help_available():
            return ["--problem", spec_path]

        if self._flag_supported("--problem"):
            return ["--problem", spec_path]
        if self._flag_supported("--spec"):
            return ["--spec", spec_path]
        # Fallback: assume positional.
        return [spec_path]

    def _append_flag(self, cmd: List[str], flag: str, enabled: bool) -> None:
        if not enabled:
            return
        cmd.append(flag)

    def _append_flag_if_supported(self, cmd: List[str], flag: str, enabled: bool) -> None:
        if not enabled:
            return
        if self._flag_supported(flag):
            cmd.append(flag)

    def build_command(self, request: dict, out_dir: Path) -> list[str]:
        self.validate_request(request)

        spec_path = str(request.get("spec_path") or request.get("problem") or request.get("spec"))
        mode = str(request.get("mode", "auto") or "auto")

        cmd: List[str] = [
            sys.executable,
            "-m",
            "electrodrive.cli",
            "solve",
            *self._problem_argv(spec_path),
        ]

        # --mode is present in current repo; if help is available, respect it.
        if (not self._help_available()) or self._flag_supported("--mode"):
            cmd.extend(["--mode", mode])

        # --out is required in this repo's CLI.
        cmd.extend(["--out", str(out_dir)])

        cert = bool(request.get("cert", False))
        fast = bool(request.get("fast", False))
        cert_fast = bool(request.get("cert_fast", False) or request.get("cert-fast", False))

        viz_enable = bool(request.get("viz_enable", False))
        viz_dir = request.get("viz_dir") or request.get("viz-dir")
        viz_dir_s = str(viz_dir).strip() if isinstance(viz_dir, str) else ""

        # Optional flags: if help is unavailable, assume current repo interface.
        if self._help_available():
            self._append_flag_if_supported(cmd, "--cert", cert)
        else:
            self._append_flag(cmd, "--cert", cert)

        if cert_fast:
            if self._help_available():
                if self._flag_supported("--cert-fast"):
                    cmd.append("--cert-fast")
                else:
                    self._append_flag_if_supported(cmd, "--cert", True)
                    self._append_flag_if_supported(cmd, "--fast", True)
            else:
                # Conservative fallback when help isn't available:
                # prefer the stable pair over guessing --cert-fast exists.
                if cert:
                    cmd.append("--cert")
                cmd.append("--fast")
        else:
            if self._help_available():
                self._append_flag_if_supported(cmd, "--fast", fast)
            else:
                self._append_flag(cmd, "--fast", fast)

        # Visualization: only attach if help confirms --viz exists (avoids hard failures
        # on branches without viz support).
        viz_requested = bool(viz_enable or bool(viz_dir_s))
        if viz_requested and self._flag_supported("--viz"):
            if self._flag_takes_value("--viz"):
                viz_path = viz_dir_s or str((out_dir / "viz").resolve())
                cmd.extend(["--viz", viz_path])
            else:
                cmd.append("--viz")

        extra_args = request.get("extra_args") or []
        if isinstance(extra_args, list):
            cmd.extend([str(x) for x in extra_args if isinstance(x, str)])

        return cmd

    def build_env(self, request: dict, out_dir: Path, run_id: str) -> dict[str, str]:
        # The solve CLI already uses these env vars (EDE_RUN_DIR, EDE_RUN_ID) for downstream hooks.
        return {
            "EDE_RUN_DIR": str(out_dir),
            "EDE_RUN_ID": str(run_id),
            "PYTHONUNBUFFERED": "1",
        }

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "supports_controls": True,
            "description": "Run the core electrodrive solver via `python -m electrodrive.cli solve ...`",
            "request_template": {
                "spec_path": "specs/plane_point.json",
                "mode": "auto",
                "cert": True,
                "fast": False,
                "cert_fast": False,
                "viz_enable": False,
                "viz_dir": None,
                "extra_args": [],
            },
        }
