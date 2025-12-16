from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch

from electrodrive.layers.dcim_types import ComplexImageTerm, DCIMBlock, DCIMCertificate
from electrodrive.layers.poles import PoleTerm, find_poles, PoleSearchConfig
from electrodrive.layers.rt_recursion import effective_reflection
from electrodrive.layers.spectral_kernels import SpectralKernelSpec
from electrodrive.layers.stack import LayerStack
from electrodrive.spectral.exp_fit import exp_fit
from electrodrive.spectral.vector_fitting import vector_fit


@dataclass
class DCIMCompilerConfig:
    k_min: float = 1e-3
    k_mid: float = 2.0
    k_max: float = 8.0
    n_low: int = 32
    n_mid: int = 64
    n_high: int = 32
    log_low: bool = True
    log_high: bool = True

    vf_enabled: bool = True
    vf_M: int = 6
    vf_max_iters: int = 6
    vf_tol: float = 1e-5
    vf_for_images: bool = False

    exp_fit_enabled: bool = True
    exp_N: int = 4
    exp_fit_requires_uniform_grid: bool = True
    runtime_eval_mode: str = "composite"  # "image_only" or "composite"

    poles_enabled: bool = False
    pole_cfg: PoleSearchConfig = field(default_factory=PoleSearchConfig)

    spectral_tol: float = 1e-3
    spatial_tol: float = 5e-2
    sample_points: Optional[Sequence[Tuple[float, float]]] = None  # (rho, z)
    source_z: float = 0.2
    source_charge: float = 1.0

    cache_enabled: bool = False
    cache_path: Optional[Path] = None

    device: str | torch.device = "cuda"
    dtype: torch.dtype = torch.complex128

    def cache_file(self) -> Optional[Path]:
        if not self.cache_enabled:
            return None
        path = self.cache_path or Path("runs/dcim_cache.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def _build_k_grid(cfg: DCIMCompilerConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    segments: List[torch.Tensor] = []
    if cfg.n_low > 0:
        if cfg.log_low:
            segments.append(torch.logspace(math.log10(cfg.k_min), math.log10(cfg.k_mid), cfg.n_low, device=device, dtype=torch.float64))
        else:
            segments.append(torch.linspace(cfg.k_min, cfg.k_mid, cfg.n_low, device=device, dtype=torch.float64))
    if cfg.n_mid > 0:
        segments.append(torch.linspace(cfg.k_mid, cfg.k_max, cfg.n_mid, device=device, dtype=torch.float64))
    if cfg.n_high > 0:
        if cfg.log_high:
            segments.append(torch.logspace(math.log10(cfg.k_mid), math.log10(cfg.k_max), cfg.n_high, device=device, dtype=torch.float64))
        else:
            segments.append(torch.linspace(cfg.k_mid, cfg.k_max, cfg.n_high, device=device, dtype=torch.float64))
    k = torch.unique(torch.cat(segments))
    return k.to(device=device, dtype=dtype)


def _eval_model(
    k: torch.Tensor,
    poles: torch.Tensor,
    residues: torch.Tensor,
    d: torch.Tensor,
    h: torch.Tensor,
    exp_terms: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    if poles.numel() == 0:
        F = d + h * k
    else:
        V = 1.0 / (k[:, None] - poles[None, :])
        F = torch.sum(residues[None, :] * V, dim=1) + d + h * k
    if exp_terms is not None and exp_terms[0].numel() > 0:
        A, B = exp_terms
        F = F + torch.sum(A[None, :] * torch.exp(-B[None, :] * k[:, None]), dim=1)
    return F


def _spectral_residual(F_ref: torch.Tensor, F_fit: torch.Tensor) -> Tuple[float, float]:
    diff = F_fit - F_ref
    l2 = (torch.linalg.norm(diff) / torch.linalg.norm(F_ref).clamp_min(1e-12)).item()
    linf = torch.max(torch.abs(diff)) / torch.max(torch.abs(F_ref)).clamp_min(1e-12)
    return float(l2), float(linf.item())


def _potential_from_reflection(
    k: torch.Tensor,
    R: torch.Tensor,
    eps_source: float,
    rho: torch.Tensor,
    z: torch.Tensor,
    z0: float,
    q: float,
) -> torch.Tensor:
    k_real = torch.real(k)
    kr = torch.outer(k_real, rho)
    j0 = torch.special.bessel_j0(kr)
    exp_dir = torch.exp(-torch.outer(k_real, torch.abs(z - z0)))
    exp_ref = torch.exp(-torch.outer(k_real, z + z0)) * R[:, None]
    integrand = k_real[:, None].to(R.dtype) * (exp_dir.to(R.dtype) + exp_ref) * j0.to(R.dtype)
    integral = torch.trapz(integrand, k_real.to(R.dtype), dim=0)
    direct = q / (4.0 * math.pi * eps_source) * (1.0 / torch.sqrt(rho * rho + (z - z0) ** 2).clamp_min(1e-9))
    reflected = (q / (2.0 * math.pi * 2.0 * eps_source)) * integral
    return direct + reflected


def _reflected_potential(
    k: torch.Tensor,
    R: torch.Tensor,
    eps_source: float,
    rho: torch.Tensor,
    z: torch.Tensor,
    z0: float,
    q: float,
) -> torch.Tensor:
    k_real = torch.real(k)
    kr = torch.outer(k_real, rho)
    j0 = torch.special.bessel_j0(kr)
    exp_ref = torch.exp(-torch.outer(k_real, z + z0)) * R[:, None]
    integrand = k_real[:, None].to(R.dtype) * exp_ref * j0.to(R.dtype)
    integral = torch.trapz(integrand, k_real.to(R.dtype), dim=0)
    return (q / (2.0 * math.pi * 2.0 * eps_source)) * integral


def _image_domain_potential(
    images: Sequence[ComplexImageTerm],
    rho: torch.Tensor,
    z: torch.Tensor,
    z0: float,
    eps1: float,
    q: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    assert rho.device.type == "cuda" and z.device.type == "cuda", "image-domain eval must run on CUDA."
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    rho = rho.to(device=device, dtype=real_dtype)
    z = z.to(device=device, dtype=real_dtype)
    V = torch.zeros_like(rho, device=device, dtype=dtype)
    if not images:
        return torch.real(V)
    for img in images:
        depth = torch.as_tensor(img.depth, device=device, dtype=dtype)
        weight = torch.as_tensor(img.weight, device=device, dtype=dtype)
        d = z.to(dtype) + torch.as_tensor(z0, device=device, dtype=dtype) + depth
        r_sq = rho * rho + d * d
        r = torch.sqrt(r_sq)
        contrib = weight * (q / (2.0 * math.pi * 2.0 * eps1)) * (d / (r * r * r))
        V = V + contrib
    return torch.real(V)


def _spatial_certificate(
    stack: LayerStack,
    k: torch.Tensor,
    R_ref: torch.Tensor,
    R_vf: torch.Tensor,
    images: Sequence[ComplexImageTerm],
    cfg: DCIMCompilerConfig,
) -> Tuple[float, float]:
    device = k.device
    dtype = k.dtype
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    eps1 = stack.layers[0].eps  # source region assumed top
    rho_vals: List[float] = [0.05, 0.2, 0.4]
    z_vals: List[float] = [0.05, 0.2, 0.4]
    if cfg.sample_points is not None:
        rho_vals = [p[0] for p in cfg.sample_points]
        z_vals = [p[1] for p in cfg.sample_points]
    rho = torch.tensor(rho_vals, device=device, dtype=real_dtype)
    z = torch.tensor(z_vals, device=device, dtype=real_dtype)

    V_ref = _reflected_potential(k, R_ref, float(eps1.real), rho, z, cfg.source_z, cfg.source_charge)
    try:
        V_img = _image_domain_potential(images, rho, z, cfg.source_z, float(eps1.real), cfg.source_charge, device, dtype)
    except Exception as exc:
        print(f"dcim_compile: image-domain eval failed: {exc}")
        return float("inf"), float("inf")
    V_pred = V_img
    if cfg.runtime_eval_mode != "image_only":
        V_vf_ref = _reflected_potential(k, R_vf, float(eps1.real), rho, z, cfg.source_z, cfg.source_charge)
        V_pred = V_vf_ref + V_img
    diff = V_pred - V_ref
    l2 = (torch.linalg.norm(diff) / torch.linalg.norm(V_ref).clamp_min(1e-12)).item()
    linf = torch.max(torch.abs(diff)) / torch.max(torch.abs(V_ref)).clamp_min(1e-12)
    return float(l2), float(linf.item())


def _cache_key(stack: LayerStack, kernel_spec: SpectralKernelSpec, cfg: DCIMCompilerConfig) -> str:
    payload = {
        "layers": [
            {"eps": (float(complex(l.eps).real), float(complex(l.eps).imag)), "z_min": l.z_min, "z_max": l.z_max, "name": l.name}
            for l in stack.layers
        ],
        "kernel": {
            "src": kernel_spec.source_region,
            "obs": kernel_spec.obs_region,
            "component": kernel_spec.component,
            "bc_kind": kernel_spec.bc_kind,
        },
        "cfg": {
            "k_min": cfg.k_min,
            "k_mid": cfg.k_mid,
            "k_max": cfg.k_max,
            "vf_M": cfg.vf_M,
            "exp_N": cfg.exp_N,
            "vf_enabled": cfg.vf_enabled,
            "runtime_eval_mode": cfg.runtime_eval_mode,
            "vf_for_images": cfg.vf_for_images,
        },
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _save_cache(block: DCIMBlock, path: Path, key: str) -> None:
    record = {"key": key, "block": block.to_json()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _load_cache(path: Path, key: str) -> Optional[DCIMBlock]:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except Exception:
            continue
        if record.get("key") == key:
            return DCIMBlock.from_json(record.get("block", {}))
    return None


def compile_dcim(
    stack: LayerStack,
    kernel_spec: SpectralKernelSpec,
    cfg: DCIMCompilerConfig,
) -> DCIMBlock:
    device = torch.device(cfg.device)
    dtype = cfg.dtype
    if cfg.runtime_eval_mode not in ("image_only", "composite"):
        raise ValueError(f"Unsupported runtime_eval_mode '{cfg.runtime_eval_mode}'.")
    if cfg.runtime_eval_mode == "image_only" and cfg.vf_enabled:
        raise ValueError("runtime_eval_mode='image_only' requires vf_enabled=False to avoid runtime spectral components.")
    if cfg.exp_fit_enabled and cfg.exp_fit_requires_uniform_grid:
        n_total = max(8, cfg.n_low + cfg.n_mid + cfg.n_high)
        real_dtype = torch.empty((), dtype=dtype).real.dtype
        k = torch.linspace(cfg.k_min, cfg.k_max, n_total, device=device, dtype=real_dtype).to(dtype=dtype)
    else:
        k = _build_k_grid(cfg, device, dtype)
    if k.device.type != "cuda":
        raise ValueError("compile_dcim requires CUDA k-grid (GPU-first rule).")

    cache_path = cfg.cache_file()
    key = _cache_key(stack, kernel_spec, cfg)
    if cache_path:
        cached = _load_cache(cache_path, key)
        if cached is not None:
            return cached

    F = effective_reflection(stack, k, source_region=kernel_spec.source_region, direction="down", device=device, dtype=dtype)

    pole_terms: List[PoleTerm] = []
    F_work = F.clone()
    if cfg.poles_enabled and cfg.pole_cfg.max_poles > 0:
        pole_terms = find_poles(stack, kernel_spec, cfg.pole_cfg, device=device)
        for p in pole_terms:
            F_work = F_work - (torch.as_tensor(p.residue, device=device, dtype=dtype) / (k - torch.as_tensor(p.pole, device=device, dtype=dtype)))

    vf_residues = torch.tensor([], device=device, dtype=dtype)
    vf_poles = torch.tensor([], device=device, dtype=dtype)
    vf_d = torch.tensor(0.0, device=device, dtype=dtype)
    vf_h = torch.tensor(0.0, device=device, dtype=dtype)
    F_vf = torch.zeros_like(F_work)
    if cfg.vf_enabled:
        vf_out = vector_fit(k, F_work, M=cfg.vf_M, max_iters=cfg.vf_max_iters, tol=cfg.vf_tol, device=device, dtype=dtype)
        vf_poles = vf_out["poles"]
        vf_residues = vf_out["residues"]
        vf_d = vf_out["d"]
        vf_h = vf_out["h"]
        F_vf = _eval_model(k, vf_poles, vf_residues, vf_d, vf_h)
        vf_residual_L2, _ = _spectral_residual(F_work, F_vf)
        if not math.isfinite(vf_residual_L2) or vf_residual_L2 > 10.0:
            vf_poles = torch.tensor([], device=device, dtype=dtype)
            vf_residues = torch.tensor([], device=device, dtype=dtype)
            vf_d = torch.tensor(0.0, device=device, dtype=dtype)
            vf_h = torch.tensor(0.0, device=device, dtype=dtype)
            F_vf = torch.zeros_like(F_work)

    images_list: List[ComplexImageTerm] = []
    exp_A = torch.tensor([], device=device, dtype=dtype)
    exp_B = torch.tensor([], device=device, dtype=dtype)
    if cfg.runtime_eval_mode == "image_only":
        F_target = F_work.clone()
        vf_poles = torch.tensor([], device=device, dtype=dtype)
        vf_residues = torch.tensor([], device=device, dtype=dtype)
        vf_d = torch.tensor(0.0, device=device, dtype=dtype)
        vf_h = torch.tensor(0.0, device=device, dtype=dtype)
        F_vf = torch.zeros_like(F_work)
        remainder = F_target
        if cfg.vf_for_images:
            vf_out = vector_fit(k, F_work, M=cfg.vf_M, max_iters=cfg.vf_max_iters, tol=cfg.vf_tol, device=device, dtype=dtype)
            vf_poles = vf_out["poles"]
            vf_residues = vf_out["residues"]
            vf_d = torch.tensor(0.0, device=device, dtype=dtype)
            vf_h = torch.tensor(0.0, device=device, dtype=dtype)
            F_vf = _eval_model(k, vf_poles, vf_residues, vf_d, vf_h)
            for i in range(vf_poles.numel()):
                images_list.append(
                    ComplexImageTerm(depth=complex(vf_poles[i].item()), weight=complex(vf_residues[i].item()), family="vf")
                )
            remainder = F_target - F_vf
        if cfg.exp_fit_enabled:
            exp_out = exp_fit(k, remainder, n_terms=cfg.exp_N, device=device, dtype=dtype, include_bias=True)
            exp_A = exp_out["A"]
            exp_B = exp_out["B"]
            k0 = torch.real(k[0])
            exp_A = exp_A * torch.exp(exp_B * k0)
            bias = exp_out.get("bias", torch.tensor(0.0, device=device, dtype=dtype))
            for i in range(exp_A.numel()):
                images_list.append(
                    ComplexImageTerm(depth=complex(exp_B[i].item()), weight=complex(exp_A[i].item()), family="exp_fit")
                )
            if torch.abs(bias) > 0:
                images_list.append(
                    ComplexImageTerm(depth=1e-6 + 0j, weight=complex(bias.item()), family="exp_fit_bias")
                )
        F_fit = _eval_model(k, vf_poles, vf_residues, vf_d, vf_h, (exp_A, exp_B) if cfg.exp_fit_enabled else None)
        if cfg.exp_fit_enabled:
            F_fit = F_fit + bias
        fit_L2, fit_Linf = _spectral_residual(F_target, F_fit)
        images = tuple(images_list)
        spatial_L2, spatial_Linf = _spatial_certificate(stack, k, F_target, torch.zeros_like(F_target), images, cfg)
        stable = (fit_L2 < cfg.spectral_tol) and (fit_Linf < 10 * cfg.spectral_tol) and (spatial_L2 < cfg.spatial_tol)
    else:
        if cfg.exp_fit_enabled:
            remainder = F_work - F_vf
            exp_out = exp_fit(k, remainder, n_terms=cfg.exp_N, device=device, dtype=dtype, include_bias=True)
            exp_A = exp_out["A"]
            exp_B = exp_out["B"]
            k0 = torch.real(k[0])
            exp_A = exp_A * torch.exp(exp_B * k0)
            bias = exp_out.get("bias", torch.tensor(0.0, device=device, dtype=dtype))
            for i in range(exp_A.numel()):
                images_list.append(
                    ComplexImageTerm(depth=complex(exp_B[i].item()), weight=complex(exp_A[i].item()), family="exp_fit")
                )
            if torch.abs(bias) > 0:
                images_list.append(
                    ComplexImageTerm(depth=1e-6 + 0j, weight=complex(bias.item()), family="exp_fit_bias")
                )
        F_fit = _eval_model(k, vf_poles, vf_residues, vf_d, vf_h, (exp_A, exp_B) if cfg.exp_fit_enabled else None)
        if cfg.exp_fit_enabled:
            F_fit = F_fit + bias
        fit_L2, fit_Linf = _spectral_residual(F, F_fit)
        images = tuple(images_list)
        spatial_L2, spatial_Linf = _spatial_certificate(stack, k, F, F_vf, images, cfg)
        stable = (fit_L2 < cfg.spectral_tol) and (fit_Linf < 10 * cfg.spectral_tol) and (spatial_L2 < cfg.spatial_tol)

    if cfg.runtime_eval_mode == "image_only":
        pole_terms_combined = tuple(pole_terms)
    else:
        pole_terms_combined = tuple(pole_terms) + tuple(
            PoleTerm(pole=complex(vf_poles[i].item()), residue=complex(vf_residues[i].item()), kind="vf")
            for i in range(vf_poles.numel())
        )

    certificate = DCIMCertificate(
        k_grid=tuple(float(x) for x in torch.real(k).detach().cpu().tolist()),
        fit_residual_L2=fit_L2,
        fit_residual_Linf=fit_Linf,
        spatial_check_rel_L2=spatial_L2,
        spatial_check_rel_Linf=spatial_Linf,
        stable=stable,
        meta={
            "vf": {
                "M": cfg.vf_M,
                "iters": cfg.vf_max_iters,
                "d": complex(vf_d.item()),
                "h": complex(vf_h.item()),
            },
            "exp_N": cfg.exp_N,
            "source_z": cfg.source_z,
            "source_charge": cfg.source_charge,
            "runtime_eval_mode": cfg.runtime_eval_mode,
        },
    )

    block = DCIMBlock(
        stack=stack,
        kernel=kernel_spec,
        poles=tuple(pole_terms_combined),
        images=images,
        certificate=certificate,
    )

    if cache_path:
        _save_cache(block, cache_path, key)
    return block
