from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Tuple

import torch

from . import GateResult, _assert_cuda_inputs
from ..oracle_types import OracleQuery, OracleResult


def _prepare_points(points: torch.Tensor, n: int) -> torch.Tensor:
    if points.numel() == 0:
        return torch.randn(n, 3, device=points.device, dtype=points.dtype)
    if points.shape[0] >= n:
        return points[:n].contiguous()
    repeat = (n + points.shape[0] - 1) // points.shape[0]
    pts = points.repeat((repeat, 1))[:n]
    return pts.contiguous()


def _bench_eval(fn: Callable[[torch.Tensor], torch.Tensor], pts: torch.Tensor, *, repeat: int = 2) -> Tuple[float, float]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_times = []
    cuda_times = []
    for _ in range(max(1, repeat)):
        torch.cuda.synchronize()
        start_wall = time.perf_counter()
        start_event.record()
        out = fn(pts)
        if isinstance(out, tuple):
            out = out[0]
        if not torch.is_tensor(out):
            raise ValueError("Speed gate eval_fn must return tensor")
        if not out.is_cuda:
            raise ValueError("Speed gate eval_fn must return CUDA tensor")
        end_event.record()
        torch.cuda.synchronize()
        wall_times.append((time.perf_counter() - start_wall) * 1000.0)
        cuda_times.append(float(start_event.elapsed_time(end_event)))
    return float(sum(wall_times) / len(wall_times)), float(sum(cuda_times) / len(cuda_times))


def run_gate(
    query: OracleQuery,
    result: OracleResult,
    *,
    config: Optional[Dict[str, object]] = None,
) -> GateResult:
    _assert_cuda_inputs(query, result)
    cfg = dict(config or {})
    candidate_eval = cfg.get("candidate_eval", None)
    baseline_eval = cfg.get("baseline_eval", None)
    if not callable(candidate_eval):
        raise ValueError("Gate E requires 'candidate_eval' callable")
    if not callable(baseline_eval):
        raise ValueError("Gate E requires 'baseline_eval' callable")
    n_bench = int(cfg.get("n_bench", 4096))
    prior_pass = bool(cfg.get("prereq_pass", True))
    min_speedup = float(cfg.get("min_speedup", 1.1))

    device = query.points.device
    pts = _prepare_points(query.points, n_bench).to(device=device)

    cand_wall, cand_cuda = _bench_eval(candidate_eval, pts, repeat=3)
    base_wall, base_cuda = _bench_eval(baseline_eval, pts, repeat=3)

    speedup = base_cuda / max(cand_cuda, 1e-3)
    throughput = pts.shape[0] / max(cand_cuda, 1e-3) * 1000.0
    status = "pass" if prior_pass and speedup >= min_speedup else "borderline" if speedup >= min_speedup * 0.7 else "fail"

    evidence = {}
    if "artifact_dir" in cfg and cfg["artifact_dir"]:
        path = cfg["artifact_dir"] / "gateE_speed.json"  # type: ignore[operator]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "candidate_wall_ms": cand_wall,
                "candidate_cuda_ms": cand_cuda,
                "baseline_wall_ms": base_wall,
                "baseline_cuda_ms": base_cuda,
                "speedup": speedup,
                "throughput_points_per_s": throughput,
            }
            import json

            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            evidence["timings"] = str(path)
        except Exception:
            pass

    metrics = {
        "candidate_wall_ms": cand_wall,
        "candidate_cuda_ms": cand_cuda,
        "baseline_wall_ms": base_wall,
        "baseline_cuda_ms": base_cuda,
        "speedup": speedup,
        "throughput_points_per_s": throughput,
        "samples": float(pts.shape[0]),
    }
    thresholds = {"min_speedup": min_speedup}
    return GateResult(
        gate="E",
        status=status,
        metrics=metrics,
        thresholds=thresholds,
        evidence=evidence,
        oracle={"method": result.method, "fidelity": result.fidelity.value},
        notes=[],
        config=cfg,
    )
