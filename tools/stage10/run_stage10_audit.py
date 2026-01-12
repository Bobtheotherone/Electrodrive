from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required for run_stage10_audit.py") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _iter_run_dirs(runs_dir: Path) -> Iterable[Path]:
    if not runs_dir.exists():
        return []
    return [p for p in runs_dir.iterdir() if p.is_dir()]


def _wait_for_run_dir(
    *,
    runs_dir: Path,
    tag: str,
    start_time: float,
    existing: set[str],
    timeout_sec: float,
) -> Path:
    deadline = start_time + timeout_sec
    while time.time() < deadline:
        candidates = []
        for entry in _iter_run_dirs(runs_dir):
            if entry.name in existing:
                continue
            if not entry.name.endswith(f"_{tag}"):
                continue
            try:
                mtime = entry.stat().st_mtime
            except FileNotFoundError:
                continue
            if mtime >= start_time - 1.0:
                candidates.append((mtime, entry))
        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            return candidates[0][1]
        time.sleep(0.5)
    raise RuntimeError(f"run_dir not found within {timeout_sec:.0f}s for tag {tag}")


def _wait_for_file(path: Path, timeout_sec: float) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(0.5)
    return False


def _validate_preflight(preflight: Dict[str, Any], generations: int) -> None:
    counters = preflight.get("counters")
    extra = preflight.get("extra")
    if not isinstance(counters, dict):
        raise RuntimeError("preflight counters missing")
    if not isinstance(extra, dict):
        raise RuntimeError("preflight extra missing")
    per_gen = extra.get("per_gen")
    if not isinstance(per_gen, list) or len(per_gen) != generations:
        raise RuntimeError("preflight per_gen malformed")
    if any(not isinstance(entry, dict) for entry in per_gen):
        raise RuntimeError("preflight per_gen entries not dicts")
    for key in ("sampled_programs_total", "compiled_ok", "solved_ok", "fast_scored", "verified_written"):
        if counters.get(key, None) is None:
            raise RuntimeError(f"preflight counter {key} missing")


def _read_preflight(run_dir: Path, generations: int, timeout_sec: float) -> Dict[str, Any]:
    preflight_path = run_dir / "preflight.json"
    if not _wait_for_file(preflight_path, timeout_sec):
        raise RuntimeError(f"preflight.json missing after {timeout_sec:.0f}s")
    payload = _load_json(preflight_path)
    _validate_preflight(payload, generations)
    return payload


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_best_candidate(
    *,
    label: str,
    candidate: Dict[str, Any],
    repo_root: Path,
    out_dir: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    summary_path = candidate.get("summary_path")
    verify_path = candidate.get("verification_path")
    if not summary_path or not verify_path:
        return None, None
    summary_src = Path(summary_path)
    if not summary_src.is_absolute():
        summary_src = repo_root / summary_src
    cert_src = Path(verify_path) / "discovery_certificate.json"
    if not cert_src.is_absolute():
        cert_src = repo_root / cert_src
    summary_dst = out_dir / f"{label}_summary.json"
    cert_dst = out_dir / f"{label}_certificate.json"
    _copy_if_exists(summary_src, summary_dst)
    _copy_if_exists(cert_src, cert_dst)
    return summary_dst, cert_dst


def _triage_conclusion(triage: Dict[str, Any]) -> str:
    candidates = triage.get("candidates", [])
    if not candidates:
        return "Gate B triage: no candidates available."
    cand = candidates[0]
    modes = cand.get("modes", {})

    def _mode_line(name: str) -> Tuple[str, float, str]:
        metrics = modes.get(name, {})
        max_residual = float(metrics.get("max_residual", float("inf")))
        dominant = str(metrics.get("dominant_term", "unknown"))
        return name, max_residual, dominant

    lines: List[Tuple[str, float, str]] = [
        _mode_line("candidate_only"),
        _mode_line("candidate_plus_reference"),
        _mode_line("candidate_minus_reference"),
    ]
    best = min(lines, key=lambda item: item[1])
    return (
        "Gate B triage: "
        f"candidate_only max_residual={lines[0][1]:.3e} dominant={lines[0][2]}; "
        f"candidate_plus_reference max_residual={lines[1][1]:.3e} dominant={lines[1][2]}; "
        f"candidate_minus_reference max_residual={lines[2][1]:.3e} dominant={lines[2][2]}; "
        f"best_mode={best[0]}"
    )


def _write_report(
    *,
    out_path: Path,
    run_dir: Path,
    git_sha: str,
    preflight: Dict[str, Any],
    analysis: Dict[str, Any],
    triage: Dict[str, Any],
    best_partial_paths: Tuple[Optional[Path], Optional[Path]],
    best_abcde_paths: Tuple[Optional[Path], Optional[Path]],
) -> None:
    counters = preflight.get("counters", {})
    extra = preflight.get("extra", {})
    counts = analysis.get("counts", {})
    best_partial = analysis.get("best_partial") or {}

    lines = [
        "# Stage 10 audit report",
        "",
        f"run_dir: {run_dir}",
        f"run_id: {run_dir.name}",
        f"git_sha: {git_sha}",
        "",
        "## Preflight highlights",
        f"- fraction_dcim_candidates: {extra.get('fraction_dcim_candidates', 'n/a')}",
        f"- fraction_complex_candidates: {extra.get('fraction_complex_candidates', 'n/a')}",
        f"- verified_written: {counters.get('verified_written', 'n/a')}",
        f"- baseline_speed_backend_name: {extra.get('baseline_speed_backend_name', 'n/a')}",
        "",
        "## Gate counts",
        f"- A: {counts.get('A', 0)}",
        f"- AB: {counts.get('AB', 0)}",
        f"- ABC: {counts.get('ABC', 0)}",
        f"- ABCD: {counts.get('ABCD', 0)}",
        f"- ABCDE: {counts.get('ABCDE', 0)}",
        "",
        "## Best partial",
        f"- summary_path: {best_partial.get('summary_path', 'n/a')}",
        f"- gates_passed: {best_partial.get('gates_passed', [])}",
        f"- failed_gates: {best_partial.get('failed_gates', [])}",
        f"- fail_margins: {best_partial.get('fail_margins', {})}",
        "",
    ]

    abcde_count = int(counts.get("ABCDE", 0) or 0)
    if abcde_count > 0 and best_abcde_paths[0] and best_abcde_paths[1]:
        lines.extend(
            [
                "## A-E pass",
                "- YES",
                f"- summary_path: {best_abcde_paths[0]}",
                f"- discovery_certificate: {best_abcde_paths[1]}",
                "",
            ]
        )
    else:
        lines.extend(["## A-E pass", "- NO", ""])

    triage_note = _triage_conclusion(triage)
    lines.extend(["## Gate B triage", f"- {triage_note}", ""])

    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 10 audit discovery + bundle artifacts.")
    parser.add_argument(
        "--config",
        default="configs/stage10/discovery_stage10_audit_pilot.yaml",
        help="Stage 10 audit config path.",
    )
    parser.add_argument("--preflight-timeout", type=float, default=120.0)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    repo_root = _repo_root()
    cfg_path = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = _load_yaml(cfg_path)
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run", {}), dict) else {}
    tag = str(run_cfg.get("tag", "run")).strip() or "run"
    generations = int(run_cfg.get("generations", 1))

    runs_dir = repo_root / "runs"
    before = {p.name for p in _iter_run_dirs(runs_dir)}
    start_time = time.time()

    cmd = [sys.executable, "-m", "electrodrive.experiments.run_discovery", "--config", str(cfg_path)]
    proc = subprocess.Popen(cmd, cwd=repo_root)

    run_dir = _wait_for_run_dir(
        runs_dir=runs_dir,
        tag=tag,
        start_time=start_time,
        existing=before,
        timeout_sec=args.preflight_timeout,
    )

    run_log = run_dir / "run.log"
    if not _wait_for_file(run_log, args.preflight_timeout):
        proc.terminate()
        raise RuntimeError("run.log missing within timeout")

    try:
        preflight = _read_preflight(run_dir, generations, args.preflight_timeout)
    except Exception:
        proc.terminate()
        raise

    retcode = proc.wait()
    if retcode != 0:
        raise RuntimeError(f"run_discovery exited with code {retcode}")

    subprocess.run(
        [sys.executable, "tools/stage9/analyze_verifier_results.py", str(run_dir)],
        cwd=repo_root,
        check=True,
    )

    audit_root = repo_root / "stage10" / "audit" / run_dir.name
    audit_root.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            "tools/stage10/triage_gateB.py",
            str(run_dir),
            "--out-root",
            str(repo_root / "stage10" / "audit"),
            "--top-n",
            str(args.top_n),
        ],
        cwd=repo_root,
        check=True,
    )

    _copy_if_exists(run_dir / "config.yaml", audit_root / "config.yaml")
    _copy_if_exists(run_dir / "run.log", audit_root / "run.log")
    _copy_if_exists(run_dir / "preflight.json", audit_root / "preflight.json")

    analysis_dir = run_dir / "analysis"
    analysis_out = audit_root / "analysis"
    if analysis_dir.exists():
        analysis_out.mkdir(parents=True, exist_ok=True)
        for item in analysis_dir.iterdir():
            if item.is_file():
                _copy_if_exists(item, analysis_out / item.name)

    analysis_summary_path = analysis_dir / "analysis_summary.json"
    if not analysis_summary_path.exists():
        raise RuntimeError("analysis_summary.json missing; analyzer did not run?")
    analysis_summary = _load_json(analysis_summary_path)

    candidates_dir = audit_root / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    best_partial = analysis_summary.get("best_partial") or {}
    best_partial_paths = _copy_best_candidate(
        label="best_partial",
        candidate=best_partial,
        repo_root=repo_root,
        out_dir=candidates_dir,
    )
    best_candidates = analysis_summary.get("best_candidates") or []
    best_abcde_paths: Tuple[Optional[Path], Optional[Path]] = (None, None)
    if best_candidates:
        best_abcde_paths = _copy_best_candidate(
            label="best_abcde",
            candidate=best_candidates[0],
            repo_root=repo_root,
            out_dir=candidates_dir,
        )

    triage_path = audit_root / "triage" / "gateB_triage.json"
    triage_data = _load_json(triage_path) if triage_path.exists() else {}

    git_info = _load_json(run_dir / "git.json") if (run_dir / "git.json").exists() else {}
    git_sha = str(git_info.get("sha", "unknown"))

    report_path = audit_root / "stage10_report.md"
    _write_report(
        out_path=report_path,
        run_dir=run_dir,
        git_sha=git_sha,
        preflight=preflight,
        analysis=analysis_summary,
        triage=triage_data,
        best_partial_paths=best_partial_paths,
        best_abcde_paths=best_abcde_paths,
    )

    print(f"RUN_DIR={run_dir}")
    print(f"AUDIT_DIR={audit_root}")
    print(f"REPORT={report_path}")


if __name__ == "__main__":
    main()
