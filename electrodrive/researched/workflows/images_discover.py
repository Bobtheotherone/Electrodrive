from __future__ import annotations

"""
Images Discover workflow launcher for ResearchED.

Design Doc mapping:
- FR-1: workflow discovery/launch -> build exact CLI argv
- §3.2 RunManager -> subprocess spawns electrodrive.tools.images_discover
- FR-3: run dir contract -> discovered_system.json + discovery_manifest.json

Notes:
- The underlying CLI accepts a large surface area. We validate a small, stable
  subset here and always allow power-user `extra_args`.
- We intentionally accept both snake_case and dash-case keys for compatibility
  with different frontends / JSON encoders.
"""

import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence


class ImagesDiscoverWorkflow:
    name = "images_discover"
    supports_controls = False

    _INT_KEYS = {
        "nmax",
        "n_points",
        "n-points",
        "adaptive_collocation_rounds",
        "adaptive-collocation-rounds",
        "restarts",
    }
    _FLOAT_KEYS = {
        "reg_l1",
        "reg-l1",
        "ratio_boundary",
        "ratio-boundary",
        "lambda_group",
        "lambda-group",
    }
    _BOOL_KEYS = {
        "operator_mode",
        "operator-mode",
        "aug_boundary",
        "aug-boundary",
        "subtract_physical",
        "subtract-physical",
        "intensive",
    }

    def _get(self, request: dict, *keys: str, default: Any = None) -> Any:
        """Return the first present key from request (accepting snake_case and dash-case)."""
        if not isinstance(request, dict):
            return default
        for k in keys:
            if k in request:
                return request.get(k)
        return default


    def _as_str_path(self, v: Any, *, field: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{field} must be a non-empty string path")
        return v.strip()

    def _as_basis_str(self, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        if isinstance(v, (list, tuple)):
            items: List[str] = []
            for x in v:
                if not isinstance(x, str):
                    raise ValueError("basis list must contain only strings")
                sx = x.strip()
                if sx:
                    items.append(sx)
            return ",".join(items) if items else None
        raise ValueError("basis must be a string or list[str]")

    def _as_int_opt(self, v: Any, *, field: str) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError(f"{field} must not be boolean")
        if isinstance(v, int):
            return int(v)
        if isinstance(v, float) and v.is_integer():
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                return int(s)
            except Exception as exc:
                raise ValueError(f"{field} must be int-like") from exc
        raise ValueError(f"{field} must be int-like")

    def _as_float_opt(self, v: Any, *, field: str) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError(f"{field} must not be boolean")
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                return float(s)
            except Exception as exc:
                raise ValueError(f"{field} must be float-like") from exc
        raise ValueError(f"{field} must be float-like")

    def validate_request(self, request: dict) -> None:
        if not isinstance(request, dict):
            raise ValueError("request must be a dict")

        spec = self._get(request, "spec_path", "spec")
        self._as_str_path(spec, field="spec_path")

        extra = request.get("extra_args", [])
        if extra is not None and not (isinstance(extra, list) and all(isinstance(x, str) for x in extra)):
            raise ValueError("'extra_args' must be a list[str] if provided")

        for k in self._BOOL_KEYS:
            if k in request and request.get(k) is not None and not isinstance(request.get(k), bool):
                raise ValueError(f"'{k}' must be boolean if provided")

        # Light numeric validation (reject booleans; allow stringified numbers).
        for k in self._INT_KEYS:
            if k in request:
                self._as_int_opt(request.get(k), field=k)
        for k in self._FLOAT_KEYS:
            if k in request:
                self._as_float_opt(request.get(k), field=k)

        # basis may be str or list[str]
        if "basis" in request:
            self._as_basis_str(request.get("basis"))

        # solver must be a string if present
        if "solver" in request and request.get("solver") is not None and not isinstance(request.get("solver"), str):
            raise ValueError("'solver' must be a string if provided")

    def _opt(self, cmd: List[str], flag: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, bool):
            raise ValueError(f"{flag} must not be boolean")
        cmd.extend([flag, str(value)])

    def build_command(self, request: dict, out_dir: Path) -> list[str]:
        self.validate_request(request)

        spec_path = str(self._get(request, "spec_path", "spec"))

        cmd: List[str] = [
            sys.executable,
            "-m",
            "electrodrive.tools.images_discover",
            "discover",
            "--spec",
            spec_path,
            "--out",
            str(out_dir),
        ]

        basis_val = self._as_basis_str(request.get("basis"))
        if basis_val is not None:
            self._opt(cmd, "--basis", basis_val)

        self._opt(cmd, "--nmax", self._as_int_opt(request.get("nmax"), field="nmax"))
        reg_l1_raw = request.get("reg_l1") if request.get("reg_l1") is not None else request.get("reg-l1")
        self._opt(cmd, "--reg-l1", self._as_float_opt(reg_l1_raw, field="reg_l1"))

        self._opt(cmd, "--n-points", self._as_int_opt(self._get(request, "n_points", "n-points"), field="n_points"))
        self._opt(cmd, "--ratio-boundary", self._as_float_opt(self._get(request, "ratio_boundary", "ratio-boundary"), field="ratio_boundary"))

        self._opt(cmd, "--solver", request.get("solver"))
        self._opt(
            cmd,
            "--adaptive-collocation-rounds",
            self._as_int_opt(self._get(request, "adaptive_collocation_rounds", "adaptive-collocation-rounds"), field="adaptive_collocation_rounds"),
        )
        self._opt(cmd, "--lambda-group", self._as_float_opt(self._get(request, "lambda_group", "lambda-group"), field="lambda_group"))
        self._opt(cmd, "--restarts", self._as_int_opt(request.get("restarts"), field="restarts"))

        self._opt(cmd, "--basis-generator", self._get(request, "basis_generator", "basis-generator"))
        self._opt(cmd, "--geo-encoder", self._get(request, "geo_encoder", "geo-encoder"))
        self._opt(cmd, "--basis-generator-mode", self._get(request, "basis_generator_mode", "basis-generator-mode"))
        self._opt(cmd, "--model-checkpoint", self._get(request, "model_checkpoint", "model-checkpoint"))

        if bool(self._get(request, "operator_mode", "operator-mode", default=False)):
            cmd.append("--operator-mode")
        if bool(self._get(request, "aug_boundary", "aug-boundary", default=False)):
            cmd.append("--aug-boundary")
        if bool(request.get("subtract_physical", False) or request.get("subtract-physical", False)):
            cmd.append("--subtract-physical")
        if bool(request.get("intensive", False)):
            cmd.append("--intensive")

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
            "description": "Sparse method-of-images discovery via `python -m electrodrive.tools.images_discover discover`",
            "request_template": {
                "spec_path": "specs/plane_point.json",
                "basis": "point",
                "nmax": 16,
                "reg_l1": 1e-3,
                "solver": None,
                "operator_mode": False,
                "adaptive_collocation_rounds": None,
                "aug_boundary": False,
                "subtract_physical": False,
                "lambda_group": 0.0,
                "restarts": None,
                "intensive": False,
                "basis_generator": "none",
                "geo_encoder": "egnn",
                "basis_generator_mode": "static_only",
                "model_checkpoint": None,
                "extra_args": [],
            },
        }
