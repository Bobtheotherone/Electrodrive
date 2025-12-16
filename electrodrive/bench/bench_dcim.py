from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from electrodrive.core.planar_stratified_reference import ThreeLayerConfig, potential_three_layer_region1
from electrodrive.images.basis_dcim import dcim_basis_from_block
from electrodrive.layers import DCIMCompilerConfig, SpectralKernelSpec, compile_dcim, layerstack_from_spec
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.config import K_E


def _spec(eps2: float = 4.0, h: float = 0.4) -> CanonicalSpec:
    return CanonicalSpec.from_json(
        {
            "domain": {"bbox": [[-1.0, -1.0, -2.0], [1.0, 1.0, 2.0]]},
            "conductors": [],
            "dielectrics": [
                {"name": "region1", "epsilon": 1.0, "z_min": 0.0, "z_max": math.inf},
                {"name": "slab", "epsilon": eps2, "z_min": -float(h), "z_max": 0.0},
                {"name": "region3", "epsilon": 1.0, "z_min": -math.inf, "z_max": -float(h)},
            ],
            "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.2]}],
            "BCs": "dielectric_interfaces",
            "symmetry": ["rot_z"],
            "queries": [],
        }
    )


def _compile_block(device: torch.device, cache: bool = True):
    stack = layerstack_from_spec(_spec())
    kernel = SpectralKernelSpec(source_region=0, obs_region=0, component="potential", bc_kind="dielectric_interfaces")
    cfg = DCIMCompilerConfig(
        k_min=0.05,
        k_mid=2.0,
        k_max=6.0,
        n_low=64,
        n_mid=64,
        n_high=0,
        log_low=False,
        log_high=False,
        vf_enabled=False,
        vf_for_images=False,
        exp_fit_enabled=True,
        exp_fit_requires_uniform_grid=True,
        exp_N=6,
        spectral_tol=0.3,
        spatial_tol=0.2,
        sample_points=[(0.3, 0.6), (0.5, 1.0), (0.2, 0.6)],
        cache_enabled=cache,
        cache_path=Path("runs/dcim_cache.jsonl"),
        device=device,
        dtype=torch.complex128,
        runtime_eval_mode="image_only",
        source_z=0.2,
        source_charge=1.0,
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    block = compile_dcim(stack, kernel, cfg)
    torch.cuda.synchronize()
    return block, time.perf_counter() - t0


def _eval_dcim(block, targets: torch.Tensor) -> tuple[torch.Tensor, float]:
    elems = [e for e in dcim_basis_from_block(block) if e.component == "real"]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    V = torch.zeros(targets.shape[0], device=targets.device, dtype=targets.dtype)
    for elem in elems:
        V = V + elem.potential(targets)
    torch.cuda.synchronize()
    return V, time.perf_counter() - t0


def _eval_oracle(targets: torch.Tensor) -> tuple[torch.Tensor, float]:
    cfg = ThreeLayerConfig(eps1=1.0, eps2=4.0, eps3=1.0, h=0.4, q=1.0, r0=(0.0, 0.0, 0.2), n_k=128, k_max=6.0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    V = potential_three_layer_region1(targets, cfg, device=targets.device, dtype=targets.dtype)
    torch.cuda.synchronize()
    return V, time.perf_counter() - t0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("bench_dcim requires CUDA.")

    print("Compiling DCIM block (cold)...")
    block_cold, t_cold = _compile_block(device, cache=True)
    print(f"Cold compile: {t_cold:.3f}s, stable={block_cold.certificate.stable}")

    print("Compiling DCIM block (cache hot)...")
    block_hot, t_hot = _compile_block(device, cache=True)
    print(f"Hot compile (cache hit expected): {t_hot:.3f}s, stable={block_hot.certificate.stable}")

    Ns = [128, 1024, 4096]
    for N in Ns:
        targets = torch.zeros((N, 3), device=device, dtype=torch.float64)
        targets[:, 0] = torch.linspace(0.05, 0.4, N, device=device)
        targets[:, 2] = torch.linspace(0.2, 1.0, N, device=device)

        V_dcim, t_dcim = _eval_dcim(block_hot, targets)
        V_oracle, t_oracle = _eval_oracle(targets)

        # Compare reflected part only.
        r_direct = torch.linalg.norm(
            targets - torch.tensor([0.0, 0.0, 0.2], device=device, dtype=torch.float64), dim=1
        ).clamp_min(1e-9)
        direct_term = K_E / (4.0 * math.pi * 1.0) * (1.0 / r_direct)
        V_ref_oracle = V_oracle - direct_term
        rel_err = (torch.linalg.norm(V_dcim - V_ref_oracle) / torch.linalg.norm(V_ref_oracle).clamp_min(1e-12)).item()
        speedup = t_oracle / max(t_dcim, 1e-6)

        print(f"N={N:5d} | dcim {t_dcim*1e3:.2f} ms | oracle {t_oracle*1e3:.2f} ms | speedup {speedup:.2f} | rel_err {rel_err:.3e}")


if __name__ == "__main__":
    main()
