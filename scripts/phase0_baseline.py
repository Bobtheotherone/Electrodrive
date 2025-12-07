#!/usr/bin/env python3
"""
Phase 0 baseline runner for the Electrodrive / EDE engine.

This script is intended to be run from the repo root (or via phase0_run.sh)
and will:

  1. Record environment + GPU info.
  2. Run core unit tests (CPU-only).
  3. Run the BEM probe on GPU.
  4. Run a basic EDE solve (plane_point analytic + BEM, auto mode).
  5. Run FMM sanity / stress tests + LaplaceFmm3D parallelism smoke (N=50, 500).

Artifacts:

  baseline/
    env.json / env.txt
    phase0_summary.json
    logs/
      env_check.log
      core_tests_cpu.log
      bem_probe.log
      ede_solve_plane_point.log
      fmm_pytests.log
      fmm_parallelism_N50.log
      fmm_parallelism_N500.log
    bem/
      bem_probe_summary.json (copied from runs/bem_probe)
      ...
    ede/
      metrics.json, manifest.json, evidence_log.jsonl, aggregate_verification_report.json (if present)
    fmm/
      fmm_pytests.jsonl (if JSONL logging is enabled)
      laplace_parallel_N50/events.jsonl
      laplace_parallel_N500/events.jsonl

The script is deliberately best-effort: it will attempt to run all stages
even if some fail, and will record success/failure in phase0_summary.json.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "baseline"
LOGS_DIR = BASELINE_DIR / "logs"
BEM_DIR = BASELINE_DIR / "bem"
EDE_DIR = BASELINE_DIR / "ede"
FMM_DIR = BASELINE_DIR / "fmm"


@dataclass
class StepResult:
    name: str
    ok: bool
    returncode: int
    log_path: Optional[str] = None
    extra: Optional[Dict[str, object]] = None

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "name": self.name,
            "ok": bool(self.ok),
            "returncode": int(self.returncode),
        }
        if self.log_path is not None:
            data["log_path"] = self.log_path
        if self.extra:
            data["extra"] = self.extra
        return data


def _print(msg: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[phase0 {timestamp}] {msg}")


def run_cmd(
    step_name: str,
    cmd: List[str],
    log_filename: str,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Path] = None,
) -> StepResult:
    """
    Run a subprocess, capture stdout+stderr to baseline/logs/log_filename,
    and return a StepResult describing success/failure.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / log_filename
    if cwd is None:
        cwd = REPO_ROOT

    _print(f"Running {step_name}: {' '.join(cmd)}")
    _print(f"  CWD = {cwd}")
    if env is not None:
        for key in ("CUDA_VISIBLE_DEVICES", "EDE_FMM_ENABLE_JSONL", "EDE_FMM_JSONL_PATH"):
            if key in env:
                _print(f"  env {key}={env[key]!r}")

    try:
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"# Step: {step_name}\n")
            f.write(f"# Command: {' '.join(cmd)}\n")
            f.write(f"# CWD: {cwd}\n")
            f.write(f"# Timestamp: {datetime.now().isoformat()}\n\n")
            cp = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            f.write(cp.stdout)

        ok = cp.returncode == 0
        rel_log = str(log_path.relative_to(REPO_ROOT))
        if ok:
            _print(f"{step_name} OK (rc={cp.returncode}) – log at {rel_log}")
        else:
            _print(f"{step_name} FAILED (rc={cp.returncode}) – log at {rel_log}")
        return StepResult(
            name=step_name,
            ok=ok,
            returncode=cp.returncode,
            log_path=rel_log,
        )
    except FileNotFoundError as exc:
        rel_log = str(log_path.relative_to(REPO_ROOT))
        _print(f"{step_name} FAILED – command not found: {exc}")
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"\n[phase0] ERROR: command not found: {exc}\n")
        except Exception:
            pass
        return StepResult(
            name=step_name,
            ok=False,
            returncode=-1,
            log_path=rel_log,
            extra={"error": f"command_not_found: {exc}"},
        )
    except Exception as exc:  # pragma: no cover - defensive
        rel_log = str(log_path.relative_to(REPO_ROOT))
        _print(f"{step_name} FAILED with unexpected exception: {exc}")
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"\n[phase0] ERROR: unexpected exception: {exc}\n")
        except Exception:
            pass
        return StepResult(
            name=step_name,
            ok=False,
            returncode=-2,
            log_path=rel_log,
            extra={"error": f"exception: {exc}"},
        )


def collect_env_info() -> Dict[str, object]:
    """
    Collect Python / platform / torch / CUDA info in a JSON-serializable dict.
    """
    info: Dict[str, object] = {}
    info["timestamp"] = datetime.now().isoformat()
    info["repo_root"] = str(REPO_ROOT)
    info["python_executable"] = sys.executable
    info["python_version"] = sys.version
    info["platform"] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    info["virtual_env"] = os.environ.get("VIRTUAL_ENV") or ""
    info["venv_dir_exists"] = (REPO_ROOT / ".venv").exists()

    # Git branch / commit (best-effort)
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        branch = None
    info["git_branch"] = branch

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        commit = None
    info["git_commit"] = commit

    if torch is None:
        info["torch"] = {"available": False, "import_error": True}
        info["torch_short_probe"] = None
        return info

    cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    device_name = None
    total_mem_gb = None
    try:
        if cuda_available:
            dev_idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(dev_idx)
            device_name = props.name
            total_mem_gb = float(props.total_memory) / (1024.0 ** 3)
    except Exception:
        cuda_available = False

    info["torch"] = {
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "total_memory_gb": total_mem_gb,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }

    probe = None
    if cuda_available:
        try:
            probe = {
                "torch_version": torch.__version__,
                "cuda_is_available": torch.cuda.is_available(),
                "device_name_0": torch.cuda.get_device_name(0),
            }
        except Exception:
            probe = None
    info["torch_short_probe"] = probe

    return info


def write_env_info() -> StepResult:
    """
    Write env.json and env.txt under baseline/ and return a StepResult.
    """
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    env_info = collect_env_info()
    env_json_path = BASELINE_DIR / "env.json"
    env_txt_path = BASELINE_DIR / "env.txt"

    with env_json_path.open("w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2, sort_keys=True)

    with env_txt_path.open("w", encoding="utf-8") as f:
        f.write(f"Phase 0 environment snapshot ({datetime.now().isoformat()})\n\n")
        f.write(json.dumps(env_info, indent=2, sort_keys=True))
        f.write("\n")

    _print("Environment snapshot written to baseline/env.json and baseline/env.txt")
    probe = env_info.get("torch_short_probe")
    torch_info = env_info.get("torch")
    if isinstance(probe, dict) and isinstance(torch_info, dict):
        _print(
            f"torch {probe.get('torch_version')} | "
            f"cuda_available={torch_info.get('cuda_available')} | "
            f"device0={probe.get('device_name_0')}"
        )
    else:
        _print("Torch / CUDA info unavailable or torch not installed.")

    if env_info.get("venv_dir_exists") and not env_info.get("virtual_env"):
        _print(
            "NOTE: .venv directory exists but VIRTUAL_ENV is not set. "
            "You may want to 'source .venv/bin/activate' before running heavy workloads."
        )

    return StepResult(
        name="env_check",
        ok=True,
        returncode=0,
        log_path=str(env_txt_path.relative_to(REPO_ROOT)),
    )


def run_core_tests_cpu() -> StepResult:
    """
    Run the core unit test suite on CPU by masking CUDA devices.

    This mirrors:
        pytest tests/test_bem* tests/test_fmm* tests/test_images_discover.py \
              tests/test_pinn_* tests/test_autotune_points_per_step.py

    but implemented as:
        pytest tests

    with CUDA disabled, so that all test modules under tests/ are exercised.
    """
    env = os.environ.copy()
    # Mask CUDA so tests run on CPU even if a GPU is present.
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = ["pytest", "tests"]
    return run_cmd("core_tests_cpu", cmd, log_filename="core_tests_cpu.log", env=env)


def run_bem_probe() -> StepResult:
    """
    Run the GPU BEM probe (_bem_probe.py) and copy its summary into baseline/bem/.
    """
    cmd = [sys.executable, "_bem_probe.py"]
    result = run_cmd("bem_probe", cmd, log_filename="bem_probe.log")

    src_dir = REPO_ROOT / "runs" / "bem_probe"
    BEM_DIR.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []

    if src_dir.is_dir():
        for name in (
            "bem_probe_summary.json",
            "metrics.json",
            "manifest.json",
            "evidence_log.jsonl",
        ):
            src = src_dir / name
            if src.is_file():
                dest = BEM_DIR / name
                try:
                    shutil.copy2(src, dest)
                    copied.append(name)
                except Exception as exc:  # pragma: no cover - best-effort
                    _print(f"Warning: failed to copy {src} -> {dest}: {exc}")
    else:
        _print(
            "No runs/bem_probe directory found after BEM probe; "
            "nothing copied to baseline/bem."
        )

    if copied:
        _print(f"BEM probe artifacts copied into baseline/bem/: {', '.join(copied)}")
    else:
        _print("No BEM probe artifacts were copied into baseline/bem/.")

    extra = {"copied_files": copied}
    if result.extra is None:
        result.extra = extra
    else:
        result.extra.update(extra)
    return result


def run_ede_solve_plane_point() -> StepResult:
    """
    Run:
        ede solve --problem specs/plane_point.json --mode auto --cert \
                  --out runs/solve_plane_point_auto

    falling back to ``python -m electrodrive.cli`` if the ``ede`` entrypoint
    is not on PATH. Copies metrics/manifest/evidence logs into baseline/ede/.
    """
    problem_rel = "specs/plane_point.json"
    out_rel = "runs/solve_plane_point_auto"

    problem_path = REPO_ROOT / problem_rel
    if not problem_path.is_file():
        _print(
            f"WARNING: {problem_rel} not found; skipping EDE solve plane_point step."
        )
        return StepResult(
            name="ede_solve_plane_point",
            ok=False,
            returncode=-1,
            log_path=None,
            extra={"error": f"{problem_rel} missing"},
        )

    if shutil.which("ede") is not None:
        cmd = [
            "ede",
            "solve",
            "--problem",
            problem_rel,
            "--mode",
            "auto",
            "--cert",
            "--out",
            out_rel,
        ]
    else:
        _print("ede entrypoint not found on PATH; falling back to python -m electrodrive.cli")
        cmd = [
            sys.executable,
            "-m",
            "electrodrive.cli",
            "solve",
            "--problem",
            problem_rel,
            "--mode",
            "auto",
            "--cert",
            "--out",
            out_rel,
        ]

    env = os.environ.copy()
    result = run_cmd(
        "ede_solve_plane_point",
        cmd,
        log_filename="ede_solve_plane_point.log",
        env=env,
    )

    EDE_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = REPO_ROOT / out_rel
    copied: List[str] = []

    if run_dir.is_dir():
        for name in (
            "metrics.json",
            "manifest.json",
            "aggregate_verification_report.json",
            "evidence_log.jsonl",
        ):
            src = run_dir / name
            if src.is_file():
                dest = EDE_DIR / name
                try:
                    shutil.copy2(src, dest)
                    copied.append(name)
                except Exception as exc:  # pragma: no cover - best-effort
                    _print(f"Warning: failed to copy {src} -> {dest}: {exc}")
    else:
        _print(
            f"No {out_rel} directory found after EDE solve; "
            "nothing copied to baseline/ede."
        )

    if copied:
        _print(f"EDE artifacts copied into baseline/ede/: {', '.join(copied)}")
    else:
        _print("No EDE artifacts were copied into baseline/ede/.")

    extra = {"copied_files": copied}
    if result.extra is None:
        result.extra = extra
    else:
        result.extra.update(extra)
    return result


def run_fmm_pytests() -> StepResult:
    """
    Run FMM sanity + stress tests (CPU+GPU):

        pytest electrodrive/fmm3d/tests/test_smoke_fmm.py \
               electrodrive/fmm3d/tests/test_accuracy.py \
               electrodrive/fmm3d/tests/test_stress.py

    with JSONL logging enabled via EDE_FMM_JSONL_PATH.
    """
    env = os.environ.copy()
    FMM_DIR.mkdir(parents=True, exist_ok=True)
    env["EDE_FMM_ENABLE_JSONL"] = "1"
    env["EDE_FMM_JSONL_PATH"] = str((FMM_DIR / "fmm_pytests.jsonl").resolve())

    cmd = [
        "pytest",
        "electrodrive/fmm3d/tests/test_smoke_fmm.py",
        "electrodrive/fmm3d/tests/test_accuracy.py",
        "electrodrive/fmm3d/tests/test_stress.py",
    ]
    return run_cmd("fmm_pytests", cmd, log_filename="fmm_pytests.log", env=env)


def run_fmm_parallelism_smoke(N: int, repeats: int, tile_size: int) -> StepResult:
    """
    Run the LaplaceFmm3D parallelism smoke test script for a given N.

    This mirrors the manual runs like:

        python -m electrodrive.fmm3d.tests.test_laplacefmm3d_parallelism \
            --N 50 --tile-size 65536 --repeats 5 --log-path baseline/fmm/laplace_parallel_N50
    """
    FMM_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = FMM_DIR / f"laplace_parallel_N{N}"
    cmd = [
        sys.executable,
        "-m",
        "electrodrive.fmm3d.tests.test_laplacefmm3d_parallelism",
        f"--N={N}",
        f"--tile-size={tile_size}",
        f"--repeats={repeats}",
        f"--log-path={log_dir}",
    ]

    step_name = f"fmm_parallelism_N{N}"
    log_filename = f"fmm_parallelism_N{N}.log"
    result = run_cmd(step_name, cmd, log_filename=log_filename, env=os.environ.copy())

    events = log_dir / "events.jsonl"
    extra: Dict[str, object] = {}
    if events.is_file():
        rel_events = str(events.relative_to(REPO_ROOT))
        _print(f"FMM parallelism N={N} events logged at {rel_events}")
        extra["events_jsonl"] = rel_events

    if extra:
        if result.extra is None:
            result.extra = extra
        else:
            result.extra.update(extra)

    return result


def main() -> int:
    _print(f"Phase 0 baseline runner starting from repo root: {REPO_ROOT}")
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    BEM_DIR.mkdir(parents=True, exist_ok=True)
    EDE_DIR.mkdir(parents=True, exist_ok=True)
    FMM_DIR.mkdir(parents=True, exist_ok=True)

    results: List[StepResult] = []

    # 0) Environment snapshot
    results.append(write_env_info())

    # 1) Core tests on CPU
    results.append(run_core_tests_cpu())

    # 2) BEM probe (GPU)
    results.append(run_bem_probe())

    # 3) EDE solve (analytic + BEM, auto mode)
    results.append(run_ede_solve_plane_point())

    # 4) FMM tests (CPU+GPU)
    results.append(run_fmm_pytests())
    results.append(run_fmm_parallelism_smoke(N=50, repeats=5, tile_size=65536))
    results.append(run_fmm_parallelism_smoke(N=500, repeats=5, tile_size=65536))

    summary = {
        "timestamp": datetime.now().isoformat(),
        "repo_root": str(REPO_ROOT),
        "results": [r.to_dict() for r in results],
    }
    summary_path = BASELINE_DIR / "phase0_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    _print(f"Wrote Phase 0 summary to {summary_path.relative_to(REPO_ROOT)}")

    ok_all = all(r.ok for r in results)
    if ok_all:
        _print("Phase 0 completed: all steps reported OK.")
        return 0

    _print(
        "Phase 0 completed: some steps FAILED. "
        "Inspect baseline/phase0_summary.json and baseline/logs/*.log for details."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
