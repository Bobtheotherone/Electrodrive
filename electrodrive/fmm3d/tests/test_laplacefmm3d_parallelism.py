# electrodrive/fmm3d/tests/test_laplacefmm3d_parallelism.py
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any

import torch
from torch import Tensor

from electrodrive.fmm3d.bem_fmm import make_laplace_fmm_backend
from electrodrive.utils.logging import (
    JsonlLogger,
    RuntimePerfFlags,
    log_runtime_environment,
    log_peak_vram,
)


def make_random_geometry(
    n_points: int,
    dtype: torch.dtype = torch.float64,
    seed: int = 1234,
) -> Dict[str, Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    src_centroids = torch.rand((n_points, 3), generator=g, dtype=dtype, device="cpu")
    areas = torch.full((n_points,), 1.0 / max(n_points, 1), dtype=dtype, device="cpu")
    return {"src_centroids": src_centroids, "areas": areas}


def make_random_sigma(
    n_points: int,
    dtype: torch.dtype = torch.float64,
    seed: int = 5678,
) -> Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    sigma = torch.randn((n_points,), generator=g, dtype=dtype, device="cpu")
    return sigma


def run_backend_benchmark(
    *,
    backend: str,
    src_centroids: Tensor,
    areas: Tensor,
    sigma: Tensor,
    tile_size: int,
    repeats: int,
    logger: Optional[JsonlLogger] = None,
) -> Dict[str, Any]:
    if backend == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but backend='gpu' was requested.")

    # IMPORTANT:
    # Do NOT pass the JsonlLogger into the FMM backend. If we do, all the deep
    # debug instrumentation (tensor stats, spectral logs, etc.) inside the FMM
    # stack gets enabled and adds a lot of overhead.
    #
    # We still use `logger` below only for high-level perf JSON.
    fmm = make_laplace_fmm_backend(
        src_centroids=src_centroids,
        areas=areas,
        backend=backend,
        logger=None,  # <-- internal FMM logging disabled
    )

    N = int(src_centroids.shape[0])
    dtype = src_centroids.dtype

    # Warm-up
    _ = fmm.matvec(
        sigma=sigma,
        src_centroids=src_centroids,
        areas=areas,
        tile_size=tile_size,
        self_integrals=None,
    )
    if backend == "gpu":
        torch.cuda.synchronize(fmm.fmm_device)

    timings = []
    V = None

    for i in range(repeats):
        t0 = time.perf_counter()
        V = fmm.matvec(
            sigma=sigma,
            src_centroids=src_centroids,
            areas=areas,
            tile_size=tile_size,
            self_integrals=None,
        )
        if backend == "gpu":
            torch.cuda.synchronize(fmm.fmm_device)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        timings.append(elapsed)

        # Progress print so you can see it's alive.
        print(f"[{backend}] matvec {i+1}/{repeats} took {elapsed:.4f} s", flush=True)

        if logger is not None:
            logger.smart_efficiency(
                backend=backend,
                iteration=i,
                elapsed_s=float(elapsed),
                N=int(N),
                tile_size=int(tile_size),
                dtype=str(dtype),
                matvec="laplace_fmm3d",
            )

    assert V is not None

    avg_time = sum(timings) / len(timings)
    min_time = min(timings)
    max_time = max(timings)
    avg_time_per_point = avg_time / max(N, 1)
    points_per_sec = N / avg_time if avg_time > 0.0 else float("nan")

    summary = {
        "backend": backend,
        "N": N,
        "dtype": str(dtype),
        "tile_size": tile_size,
        "repeats": repeats,
        "avg_time_s": avg_time,
        "min_time_s": min_time,
        "max_time_s": max_time,
        "avg_time_per_point_s": avg_time_per_point,
        "points_per_sec": points_per_sec,
    }

    if logger is not None:
        logger.smart_progress(
            backend=backend,
            N=int(N),
            tile_size=int(tile_size),
            repeats=int(repeats),
            avg_time_s=float(avg_time),
            min_time_s=float(min_time),
            max_time_s=float(max_time),
            avg_time_per_point_s=float(avg_time_per_point),
            points_per_sec=float(points_per_sec),
        )

    return {"fmm": fmm, "V": V, "summary": summary}


def run_parallelism_smoke_test_sequential(
    N: int,
    tile_size: int,
    repeats: int,
    log_dir: str,
) -> None:
    logger = JsonlLogger(log_dir)
    log_runtime_environment(logger, perf_flags=RuntimePerfFlags())

    logger.phase_start("laplace_fmm3d_smoke_seq", N=int(N), tile_size=int(tile_size), repeats=int(repeats))

    geom = make_random_geometry(N)
    src_centroids = geom["src_centroids"]
    areas = geom["areas"]
    sigma = make_random_sigma(N, dtype=src_centroids.dtype)

    # CPU then GPU
    cpu_result = run_backend_benchmark(
        backend="cpu",
        src_centroids=src_centroids,
        areas=areas,
        sigma=sigma,
        tile_size=tile_size,
        repeats=repeats,
        logger=logger,
    )
    V_cpu = cpu_result["V"]
    cpu_summary = cpu_result["summary"]

    if torch.cuda.is_available():
        gpu_result = run_backend_benchmark(
            backend="gpu",
            src_centroids=src_centroids,
            areas=areas,
            sigma=sigma,
            tile_size=tile_size,
            repeats=repeats,
            logger=logger,
        )
        V_gpu = gpu_result["V"]
        gpu_summary = gpu_result["summary"]

        diff = V_cpu - V_gpu
        num = torch.linalg.vector_norm(diff)
        denom = torch.linalg.vector_norm(V_cpu)
        rel_err = (num / denom).item() if denom.item() != 0.0 else float("nan")
        speedup = cpu_summary["avg_time_s"] / gpu_summary["avg_time_s"]

        logger.smart_accuracy(
            backend_cpu="cpu",
            backend_gpu="gpu",
            rel_l2_error=float(rel_err),
            N=int(N),
            cpu_avg_time_s=float(cpu_summary["avg_time_s"]),
            gpu_avg_time_s=float(gpu_summary["avg_time_s"]),
            speedup=float(speedup),
            mode="sequential",
        )
        log_peak_vram(logger, phase="laplace_fmm3d_smoke_seq")

        print("=== Sequential CPU→GPU test ===")
        print(f"N = {N}")
        print(f"CPU avg time: {cpu_summary['avg_time_s']:.4f} s")
        print(f"GPU avg time: {gpu_summary['avg_time_s']:.4f} s")
        print(f"Speedup CPU→GPU: {speedup:.2f}×")
        print(f"Relative L2 error (CPU vs GPU): {rel_err:.3e}")
    else:
        logger.info("GPU backend skipped (CUDA not available).")
        print("CUDA not available; only CPU backend was tested (sequential).")

    logger.phase_end("laplace_fmm3d_smoke_seq")
    print(f"Logs written under: {log_dir} (events.jsonl)")


def run_parallelism_smoke_test_concurrent(
    N: int,
    tile_size: int,
    repeats: int,
    log_dir: str,
) -> None:
    """
    Run CPU and GPU benchmarks concurrently in separate threads.
    This demonstrates real CPU/GPU overlap at the Python level.
    """
    logger = JsonlLogger(log_dir)
    log_runtime_environment(logger, perf_flags=RuntimePerfFlags())

    logger.phase_start("laplace_fmm3d_smoke_concurrent", N=int(N), tile_size=int(tile_size), repeats=int(repeats))

    geom = make_random_geometry(N)
    src_centroids = geom["src_centroids"]
    areas = geom["areas"]
    sigma = make_random_sigma(N, dtype=src_centroids.dtype)

    if not torch.cuda.is_available():
        logger.info("GPU backend skipped (CUDA not available).")
        print("CUDA not available; concurrent mode will just run CPU.")
        logger.phase_end("laplace_fmm3d_smoke_concurrent")
        return

    # Run CPU and GPU benchmarks in parallel threads.
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_cpu = ex.submit(
            run_backend_benchmark,
            backend="cpu",
            src_centroids=src_centroids,
            areas=areas,
            sigma=sigma,
            tile_size=tile_size,
            repeats=repeats,
            logger=logger,
        )
        fut_gpu = ex.submit(
            run_backend_benchmark,
            backend="gpu",
            src_centroids=src_centroids,
            areas=areas,
            sigma=sigma,
            tile_size=tile_size,
            repeats=repeats,
            logger=logger,
        )
        cpu_result = fut_cpu.result()
        gpu_result = fut_gpu.result()
    t1 = time.perf_counter()
    wall_time_concurrent = t1 - t0

    V_cpu = cpu_result["V"]
    cpu_summary = cpu_result["summary"]
    V_gpu = gpu_result["V"]
    gpu_summary = gpu_result["summary"]

    diff = V_cpu - V_gpu
    num = torch.linalg.vector_norm(diff)
    denom = torch.linalg.vector_norm(V_cpu)
    rel_err = (num / denom).item() if denom.item() != 0.0 else float("nan")
    speedup = cpu_summary["avg_time_s"] / gpu_summary["avg_time_s"]

    # Approximate "serial" total time as sum of per-backend averages * repeats
    approx_serial_time = (
        cpu_summary["avg_time_s"] * repeats + gpu_summary["avg_time_s"] * repeats
    )

    logger.smart_accuracy(
        backend_cpu="cpu",
        backend_gpu="gpu",
        rel_l2_error=float(rel_err),
        N=int(N),
        cpu_avg_time_s=float(cpu_summary["avg_time_s"]),
        gpu_avg_time_s=float(gpu_summary["avg_time_s"]),
        speedup=float(speedup),
        mode="concurrent",
        wall_time_concurrent=float(wall_time_concurrent),
        approx_serial_time=float(approx_serial_time),
    )
    log_peak_vram(logger, phase="laplace_fmm3d_smoke_concurrent")

    print("=== Concurrent CPU+GPU test ===")
    print(f"N = {N}")
    print(f"CPU avg time: {cpu_summary['avg_time_s']:.4f} s")
    print(f"GPU avg time: {gpu_summary['avg_time_s']:.4f} s")
    print(f"Approx serial total (CPU+GPU): {approx_serial_time:.4f} s")
    print(f"Concurrent wall time: {wall_time_concurrent:.4f} s")
    print(f"Relative L2 error (CPU vs GPU): {rel_err:.3e}")
    print(f"Logs written under: {log_dir} (events.jsonl)")

    logger.phase_end("laplace_fmm3d_smoke_concurrent")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke test for LaplaceFmm3D CPU vs GPU performance and correctness."
    )
    p.add_argument("--N", type=int, default=20000, help="Number of panels / points.")
    p.add_argument("--tile-size", type=int, default=1_000_000, help="P2P tile size (p2p_batch_size).")
    p.add_argument("--repeats", type=int, default=10, help="Number of timed matvecs per backend.")
    p.add_argument(
        "--log-path",
        type=str,
        default="fmm_parallelism_smoke",
        help="Directory for JsonlLogger (events.jsonl will be created inside).",
    )
    p.add_argument(
        "--concurrent",
        action="store_true",
        help="Run CPU and GPU benchmarks concurrently in separate threads.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.concurrent:
        run_parallelism_smoke_test_concurrent(
            N=args.N,
            tile_size=args.tile_size,
            repeats=args.repeats,
            log_dir=args.log_path,
        )
    else:
        run_parallelism_smoke_test_sequential(
            N=args.N,
            tile_size=args.tile_size,
            repeats=args.repeats,
            log_dir=args.log_path,
        )
