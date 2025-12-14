from __future__ import annotations

"""
Learn Train workflow launcher for ResearchED.

Design Doc mapping:
- FR-1: workflow discovery/launch -> build exact CLI argv
- §3.2 RunManager -> subprocess spawns electrodrive CLI

Notes:
- The learning stack is registered on the main electrodrive CLI via
  `electrodrive.learn.cli.register_learn_commands`.
- This workflow intentionally keeps request validation minimal and stable; users
  can pass advanced training flags via `extra_args`.
"""

import sys
from pathlib import Path
from typing import List


class LearnTrainWorkflow:
    name = "learn_train"
    supports_controls = False

    def validate_request(self, request: dict) -> None:
        if not isinstance(request, dict):
            raise ValueError("request must be a dict")
        cfg = request.get("config_path") or request.get("config")
        if isinstance(cfg, dict):
            # The underlying CLI expects a YAML file path, not inline config.
            raise ValueError("learn_train requires 'config_path' (or 'config') as a YAML file path string, not an inline dict")
        if not cfg or not isinstance(cfg, str):
            raise ValueError("learn_train requires 'config_path' (or 'config') as a string path")

        extra = request.get("extra_args", [])
        if extra is not None and not (isinstance(extra, list) and all(isinstance(x, str) for x in extra)):
            raise ValueError("'extra_args' must be a list[str] if provided")

    def build_command(self, request: dict, out_dir: Path) -> list[str]:
        self.validate_request(request)

        cfg_path = str(request.get("config_path") or request.get("config"))

        cmd: List[str] = [
            sys.executable,
            "-m",
            "electrodrive.cli",
            "train",
            "--config",
            cfg_path,
            "--out",
            str(out_dir),
        ]

        extra_args = request.get("extra_args") or []
        if isinstance(extra_args, list):
            cmd.extend([str(x) for x in extra_args if isinstance(x, str)])

        return cmd

    def build_env(self, request: dict, out_dir: Path, run_id: str) -> dict[str, str]:
        return {
            "EDE_RUN_DIR": str(out_dir),
            "EDE_RUN_ID": str(run_id),
            "PYTHONUNBUFFERED": "1",
        }

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "supports_controls": False,
            "description": "Learning training via `python -m electrodrive.cli train --config ... --out ...`",
            "request_template": {
                "config_path": "configs/train_example.yaml",
                "extra_args": [],
            },
        }
