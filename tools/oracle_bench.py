from __future__ import annotations

import argparse
import time

import torch

from electrodrive.verify.oracle_backends import F0CoarseSpectralOracleBackend, F1SommerfeldOracleBackend
from electrodrive.verify.oracle_types import CachePolicy, OracleFidelity, OracleQuery, OracleQuantity


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarking (GPU-first)")


def _bench_backend(name: str, backend, query: OracleQuery) -> None:
    start = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    result = backend.evaluate(query)
    end_event.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - start) * 1e3
    cuda_ms = start_event.elapsed_time(end_event)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"{name}: wall={wall_ms:.2f} ms, cuda={cuda_ms:.2f} ms, peak_vram={peak_mb:.1f} MB, valid={bool(result.valid_mask.all())}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark F0 vs F1 Sommerfeld oracles (GPU-only).")
    parser.add_argument("--n", type=int, default=4096, help="Number of evaluation points")
    args = parser.parse_args()

    _require_cuda()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    spec = {
        "domain": "layered",
        "BCs": "dielectric_interfaces",
        "dielectrics": [
            {"name": "layer0", "epsilon": 2.0, "z_min": 0.0, "z_max": float("inf")},
            {"name": "layer1", "epsilon": 4.0, "z_min": -0.5, "z_max": 0.0},
            {"name": "layer2", "epsilon": 1.0, "z_min": -float("inf"), "z_max": -0.5},
        ],
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.6]}],
    }
    pts = torch.randn(args.n, 3, device=device, dtype=torch.float32)
    pts[:, 2] = torch.abs(pts[:, 2]) * 0.5
    pts = pts.contiguous()

    f0 = F0CoarseSpectralOracleBackend()
    f1 = F1SommerfeldOracleBackend()

    q_f0 = OracleQuery(
        spec=spec,
        points=pts,
        quantity=OracleQuantity.POTENTIAL,
        requested_fidelity=OracleFidelity.F0,
        cache_policy=CachePolicy.OFF,
    )
    _bench_backend("F0_coarse_spectral", f0, q_f0)

    q_f1 = OracleQuery(
        spec=spec,
        points=pts.to(dtype=torch.float64),
        quantity=OracleQuantity.POTENTIAL,
        requested_fidelity=OracleFidelity.F1,
        cache_policy=CachePolicy.OFF,
        budget={"sommerfeld": {"n_low": 96, "n_mid": 160, "n_high": 96, "k_max": 60.0}},
    )
    _bench_backend("F1_sommerfeld", f1, q_f1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
