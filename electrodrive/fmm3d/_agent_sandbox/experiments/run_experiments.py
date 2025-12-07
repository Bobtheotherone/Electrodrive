import argparse
import datetime
import json
import math
import os
import random
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

from electrodrive.utils.config import K_E
from electrodrive.fmm3d.bem_fmm import make_laplace_fmm_backend
from electrodrive.fmm3d.multipole_operators import MultipoleOperators, _rescale_locals_packed
from electrodrive.fmm3d.spherical_harmonics import (
    cartesian_to_spherical,
    spherical_harmonics_complex,
    pack_ylm,
    index_to_lm,
    num_harmonics,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
SANDBOX_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = SANDBOX_ROOT / "experiments" / "results"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_git_commit() -> str:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        commit = "UNKNOWN"
    return commit


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_l_vector(p: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    P2 = num_harmonics(p)
    l_vec = torch.zeros(P2, device=device, dtype=dtype)
    for idx in range(P2):
        l, _ = index_to_lm(idx)
        l_vec[idx] = l
    return l_vec


def direct_potential(x_src: torch.Tensor, q: torch.Tensor, x_tgt: torch.Tensor) -> torch.Tensor:
    diff = x_tgt[:, None, :] - x_src[None, :, :]
    r = torch.linalg.vector_norm(diff, dim=-1)
    eps = torch.finfo(r.dtype).eps
    r = torch.where(r < eps, torch.full_like(r, float("inf")), r)
    return torch.sum(q[None, :] / r, dim=1)


def p2local_spec(x_src: torch.Tensor, q: torch.Tensor, center: torch.Tensor, p: int) -> torch.Tensor:
    x_rel = x_src - center
    rho, theta, phi = cartesian_to_spherical(x_rel)
    Y = spherical_harmonics_complex(p, theta, phi)
    Y_packed = pack_ylm(Y, p)

    P2 = num_harmonics(p)
    L = torch.zeros(P2, dtype=torch.complex128, device=x_src.device)
    q_c = q.to(dtype=L.dtype)
    for idx in range(P2):
        l, _ = index_to_lm(idx)
        term = q_c * (rho ** (-(l + 1.0))) * torch.conj(Y_packed[:, idx])
        L[idx] = torch.sum(term)
    return L


def make_random_points(n: int, seed: int, mode: str = "uniform") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mirror of tests.test_stress.make_random_points.
    """
    torch.manual_seed(seed)
    if mode == "uniform":
        x = 2.0 * torch.rand(n, 3, dtype=torch.float64) - 1.0
    elif mode == "clusters":
        n1 = n // 2
        n2 = n - n1
        c1 = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
        c2 = torch.tensor([-0.5, 0.0, 0.0], dtype=torch.float64)
        x1 = c1 + 0.1 * torch.randn(n1, 3, dtype=torch.float64)
        x2 = c2 + 0.1 * torch.randn(n2, 3, dtype=torch.float64)
        x = torch.cat([x1, x2], dim=0)
    elif mode == "shell":
        r = 0.8
        u = torch.randn(n, 3, dtype=torch.float64)
        u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True)
        x = r * u + 0.05 * torch.randn(n, 3, dtype=torch.float64)
    else:
        raise ValueError(f"Unknown mode={mode!r}")

    q = torch.randn(n, dtype=torch.float64)
    return x, q


def direct_potential_physical(x: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Physical direct Laplace potential including Coulomb constant K_E.
    Mirrors tests.test_stress.direct_potential.
    """
    n = x.shape[0]
    dx = x.unsqueeze(1) - x.unsqueeze(0)
    r = torch.linalg.vector_norm(dx, dim=-1).clamp_min(eps)
    idx = torch.arange(n, device=x.device)
    r[idx, idx] = float("inf")
    kernel = K_E / r
    return (kernel * q.view(1, n)).sum(dim=1)


def rel_l2_err(phi: torch.Tensor, phi_ref: torch.Tensor) -> float:
    num = (phi - phi_ref).norm().item()
    den = phi_ref.norm().item()
    if den == 0.0:
        return 0.0 if num == 0.0 else math.inf
    return num / den


def target_error_stats(phi: torch.Tensor, phi_ref: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    rel_err = torch.abs(phi - phi_ref) / (torch.abs(phi_ref) + eps)
    rel_err_sorted, _ = torch.sort(rel_err)
    median = float(torch.quantile(rel_err_sorted, 0.5).item())
    p90 = float(torch.quantile(rel_err_sorted, 0.9).item())
    p99 = float(torch.quantile(rel_err_sorted, 0.99).item())
    rel_max = float(rel_err.max().item())
    max_abs = float(torch.max(torch.abs(phi - phi_ref)).item())
    return {
        "rel_err_median": median,
        "rel_err_p90": p90,
        "rel_err_p99": p99,
        "rel_err_max": rel_max,
        "max_abs_error": max_abs,
    }


def per_order_rel_error(estimate: torch.Tensor, reference: torch.Tensor, p: int) -> Dict[str, float]:
    device = reference.device
    dtype = reference.real.dtype if torch.is_complex(reference) else reference.dtype
    l_vec = get_l_vector(p, device=device, dtype=dtype)
    eps = torch.finfo(dtype).eps
    out: Dict[str, float] = {}
    for l in range(p + 1):
        mask = l_vec == l
        ref_slice = reference[mask]
        est_slice = estimate[mask]
        denom = torch.linalg.norm(ref_slice)
        num = torch.linalg.norm(est_slice - ref_slice)
        denom_val = float(denom)
        if denom_val < eps:
            out[str(l)] = math.nan
        else:
            out[str(l)] = float(num / denom)
    return out


def per_order_bias(estimate: torch.Tensor, reference: torch.Tensor, p: int) -> Dict[str, float]:
    device = reference.device
    dtype = reference.real.dtype if torch.is_complex(reference) else reference.dtype
    l_vec = get_l_vector(p, device=device, dtype=dtype)
    eps = torch.finfo(dtype).eps
    out: Dict[str, float] = {}
    for l in range(p + 1):
        mask = l_vec == l
        ref_slice = reference[mask]
        est_slice = estimate[mask]
        denom = torch.mean(torch.abs(ref_slice)) + eps
        ratio = est_slice / (ref_slice + eps)
        out[str(l)] = float(torch.mean(torch.real(ratio)).item())
    return out


def serialize_tensor_stats(arr: torch.Tensor) -> Dict[str, float]:
    arr_abs = torch.abs(arr)
    return {
        "max_mag": float(arr_abs.max().item()),
        "mean_mag": float(arr_abs.mean().item()),
    }


def log_jsonl(record: Dict, filename: str) -> None:
    ensure_results_dir()
    path = RESULTS_DIR / filename
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def make_base_record(experiment_name: str, fmm_stage: str) -> Dict:
    now = datetime.datetime.utcnow().isoformat() + "Z"
    return {
        "experiment_name": experiment_name,
        "run_id": str(uuid.uuid4()),
        "timestamp": now,
        "repo_root": str(REPO_ROOT),
        "fmm_stage": fmm_stage,
        "provenance": {
            "git_commit": get_git_commit(),
            "entrypoint": "electrodrive.fmm3d.multipole_operators.MultipoleOperators",
            "python": sys.version,
            "platform": sys.platform,
        },
    }


def run_local_rescaling_sweep(p: int, ratios: List[float], seed: int, n_realizations: int) -> None:
    device = torch.device("cpu")
    for idx_ratio, ratio in enumerate(ratios):
        for rep in range(n_realizations):
            seed_val = seed + 1000 * idx_ratio + rep
            set_seed(seed_val)

            P2 = num_harmonics(p)
            real = torch.randn(P2, device=device, dtype=torch.float64)
            imag = torch.randn(P2, device=device, dtype=torch.float64)
            L_from = real + 1j * imag

            l_vec = get_l_vector(p, device=device, dtype=torch.float64)
            expected = L_from * torch.pow(torch.as_tensor(ratio, dtype=torch.float64), -(l_vec + 1.0))
            got = _rescale_locals_packed(L_from, ratio, p)

            rel_l2 = torch.linalg.norm(got - expected) / (torch.linalg.norm(expected) + 1e-18)
            per_order_err = per_order_rel_error(got, expected, p)
            bias = per_order_bias(got, expected, p)

            record = make_base_record("local_rescaling_sweep", fmm_stage="LocalRescale")
            record.update(
                {
                    "geometry_mode": "random_coeffs",
                    "n_points": 0,
                    "parameters": {
                        "p": p,
                        "ratio": ratio,
                        "seed": seed_val,
                        "realizations": n_realizations,
                    },
                    "metrics": {
                        "rel_l2_error": float(rel_l2),
                        "max_abs_error": float(torch.max(torch.abs(got - expected)).item()),
                        "spectral": {
                            "per_order_rel_error": per_order_err,
                            "bias_factor_alpha": bias,
                            **serialize_tensor_stats(got),
                        },
                        "timing": {
                            "t_fmm": None,
                            "t_ref": None,
                        },
                    },
                    "notes": "Rescaling locals L(S_from)->L(S_to) via operator vs analytic ratio^{-(l+1)}",
                }
            )
            log_jsonl(record, "local_rescaling.jsonl")


def make_shell_sources(n_src: int, r_min: float, r_max: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    set_seed(seed)
    device = torch.device("cpu")
    dirs = torch.randn(n_src, 3, device=device, dtype=torch.float64)
    dirs = dirs / torch.linalg.vector_norm(dirs, dim=-1, keepdim=True)
    radii = torch.rand(n_src, 1, device=device, dtype=torch.float64) * (r_max - r_min) + r_min
    x_src = dirs * radii
    q = torch.randn(n_src, device=device, dtype=torch.float64)
    return x_src, q


def evaluate_l2l_case(
    *,
    p: int,
    n_src: int,
    scale: float,
    translation_vec: torch.Tensor,
    seed: int,
    experiment_name: str,
    record_filename: str,
    direction_index: Optional[int] = None,
) -> None:
    """
    Evaluate a single L2L translation case and append a JSONL record.
    """
    device = torch.device("cpu")
    set_seed(seed)
    center_parent = torch.zeros(3, dtype=torch.float64, device=device)
    center_child = translation_vec.to(device=device, dtype=torch.float64)

    x_src, q = make_shell_sources(n_src, r_min=1.2 * scale, r_max=1.8 * scale, seed=seed)

    L_parent = p2local_spec(x_src, q, center_parent, p)
    op = MultipoleOperators(p=p, dtype=torch.float64, device=device)
    t_dimless = (center_child - center_parent) / scale
    L_child_op = op.l2l(L_parent, t_dimless, scale)
    L_child_ref = p2local_spec(x_src, q, center_child, p)

    rel_l2 = torch.linalg.norm(L_child_op - L_child_ref) / (torch.linalg.norm(L_child_ref) + 1e-18)
    per_order_err = per_order_rel_error(L_child_op, L_child_ref, p)
    bias = per_order_bias(L_child_op, L_child_ref, p)

    # Field check at random targets near the child center
    tgt_dirs = torch.randn(16, 3, device=device, dtype=torch.float64)
    tgt_dirs = tgt_dirs / torch.linalg.vector_norm(tgt_dirs, dim=-1, keepdim=True)
    tgt_r = torch.rand(16, 1, device=device, dtype=torch.float64) * (0.6 * scale)
    x_tgt = center_child + tgt_dirs * tgt_r

    V_child_local = op.l2p(L_child_op, x_tgt, center_child, scale)
    V_direct = direct_potential(x_src, q, x_tgt)
    rel_l2_pot = torch.linalg.norm(V_child_local - V_direct) / (torch.linalg.norm(V_direct) + 1e-18)

    record = make_base_record(experiment_name, fmm_stage="L2L")
    record.update(
        {
            "geometry_mode": "external_shell",
            "n_points": int(n_src),
            "parameters": {
                "p": p,
                "seed": seed,
                "scale": scale,
                "translation": [float(x) for x in translation_vec.tolist()],
                "translation_norm_over_scale": float(torch.linalg.norm(translation_vec) / scale),
                "direction_index": direction_index,
            },
            "metrics": {
                "rel_l2_error": float(rel_l2),
                "max_abs_error": float(torch.max(torch.abs(L_child_op - L_child_ref)).item()),
                "spectral": {
                    "per_order_rel_error": per_order_err,
                    "bias_factor_alpha": bias,
                    **serialize_tensor_stats(L_child_op),
                },
                "potentials": {
                    "rel_l2_error": float(rel_l2_pot),
                    "max_abs_error": float(torch.max(torch.abs(V_child_local - V_direct)).item()),
                },
                "timing": {
                    "t_fmm": None,
                    "t_ref": None,
                },
            },
            "notes": f"L2L at |t|/scale={float(torch.linalg.norm(translation_vec) / scale):.3f} for p={p}",
        }
    )
    log_jsonl(record, record_filename)


def run_l2l_translation(
    p: int,
    n_src: int,
    center_parent: torch.Tensor,
    center_child: torch.Tensor,
    scale: float,
    seed: int,
) -> None:
    evaluate_l2l_case(
        p=p,
        n_src=n_src,
        scale=scale,
        translation_vec=center_child - center_parent,
        seed=seed,
        experiment_name="l2l_translation",
        record_filename="l2l_translation.jsonl",
        direction_index=None,
    )


def run_l2l_sweep(
    p_list: List[int],
    n_src: int,
    scale_list: List[float],
    translation_norm_list: List[float],
    n_directions: int,
    seed: int,
) -> None:
    device = torch.device("cpu")
    center_parent = torch.zeros(3, dtype=torch.float64, device=device)
    for ip, p in enumerate(p_list):
        for iscale, scale in enumerate(scale_list):
            for inorm, t_norm_over_scale in enumerate(translation_norm_list):
                for idir in range(n_directions):
                    seed_case = seed + ip * 10000 + iscale * 1000 + inorm * 100 + idir
                    set_seed(seed_case)
                    direction = torch.randn(3, device=device, dtype=torch.float64)
                    direction = direction / torch.linalg.vector_norm(direction)
                    translation_vec = direction * (t_norm_over_scale * scale)
                    evaluate_l2l_case(
                        p=p,
                        n_src=n_src,
                        scale=scale,
                        translation_vec=translation_vec,
                        seed=seed_case,
                        experiment_name="l2l_sweep",
                        record_filename="l2l_sweep.jsonl",
                        direction_index=idir,
                    )


def evaluate_m2l_case(
    *,
    p: int,
    n_src: int,
    scale: float,
    translation_vec: torch.Tensor,
    seed: int,
    experiment_name: str,
    record_filename: str,
    direction_index: Optional[int] = None,
) -> None:
    device = torch.device("cpu")
    set_seed(seed)
    center_src = torch.zeros(3, dtype=torch.float64, device=device)
    center_tgt = translation_vec.to(device=device, dtype=torch.float64)

    # Keep radii small relative to scale for convergence
    x_src, q = make_shell_sources(n_src, r_min=0.05 * scale, r_max=0.4 * scale, seed=seed)

    op = MultipoleOperators(p=p, dtype=torch.float64, device=device)
    M_src = op.p2m(x_src, q, center_src, scale)
    t_dimless = (center_tgt - center_src) / scale
    L_tgt_op = op.m2l(M_src, t_dimless, scale)
    L_tgt_ref = p2local_spec(x_src, q, center_tgt, p)

    rel_l2 = torch.linalg.norm(L_tgt_op - L_tgt_ref) / (torch.linalg.norm(L_tgt_ref) + 1e-18)
    per_order_err = per_order_rel_error(L_tgt_op, L_tgt_ref, p)
    bias = per_order_bias(L_tgt_op, L_tgt_ref, p)

    # Potentials near the target center
    tgt_dirs = torch.randn(32, 3, device=device, dtype=torch.float64)
    tgt_dirs = tgt_dirs / torch.linalg.vector_norm(tgt_dirs, dim=-1, keepdim=True)
    tgt_r = torch.rand(32, 1, device=device, dtype=torch.float64) * (0.3 * scale)
    x_tgt = center_tgt + tgt_dirs * tgt_r

    V_local = op.l2p(L_tgt_op, x_tgt, center_tgt, scale)
    V_direct = direct_potential(x_src, q, x_tgt)
    rel_l2_pot = torch.linalg.norm(V_local - V_direct) / (torch.linalg.norm(V_direct) + 1e-18)

    record = make_base_record(experiment_name, fmm_stage="M2L")
    record.update(
        {
            "geometry_mode": "shell_source",
            "n_points": int(n_src),
            "parameters": {
                "p": p,
                "seed": seed,
                "scale": scale,
                "translation": [float(x) for x in translation_vec.tolist()],
                "translation_norm_over_scale": float(torch.linalg.norm(translation_vec) / scale),
                "direction_index": direction_index,
            },
            "metrics": {
                "rel_l2_error": float(rel_l2),
                "max_abs_error": float(torch.max(torch.abs(L_tgt_op - L_tgt_ref)).item()),
                "spectral": {
                    "per_order_rel_error": per_order_err,
                    "bias_factor_alpha": bias,
                    **serialize_tensor_stats(L_tgt_op),
                },
                "potentials": {
                    "rel_l2_error": float(rel_l2_pot),
                    "max_abs_error": float(torch.max(torch.abs(V_local - V_direct)).item()),
                },
                "timing": {
                    "t_fmm": None,
                    "t_ref": None,
                },
            },
            "notes": f"M2L at |t|/scale={float(torch.linalg.norm(translation_vec) / scale):.3f} for p={p}",
        }
    )
    log_jsonl(record, record_filename)


def run_m2l_sweep(
    p_list: List[int],
    n_src: int,
    scale_list: List[float],
    translation_norm_list: List[float],
    n_directions: int,
    seed: int,
) -> None:
    device = torch.device("cpu")
    for ip, p in enumerate(p_list):
        for iscale, scale in enumerate(scale_list):
            for inorm, t_norm_over_scale in enumerate(translation_norm_list):
                for idir in range(n_directions):
                    seed_case = seed + ip * 10000 + iscale * 1000 + inorm * 100 + idir
                    set_seed(seed_case)
                    direction = torch.randn(3, device=device, dtype=torch.float64)
                    direction = direction / torch.linalg.vector_norm(direction)
                    translation_vec = direction * (t_norm_over_scale * scale)
                    evaluate_m2l_case(
                        p=p,
                        n_src=n_src,
                        scale=scale,
                        translation_vec=translation_vec,
                        seed=seed_case,
                        experiment_name="m2l_sweep",
                        record_filename="m2l_sweep.jsonl",
                        direction_index=idir,
                    )


def run_fmm_stress_grid(
    n_points_list: List[int],
    mode_list: List[str],
    expansion_orders: List[int],
    theta_list: List[float],
    max_leaf_size_list: List[int],
    seed: int,
) -> None:
    device = torch.device("cpu")
    for in_pts, n_points in enumerate(n_points_list):
        for mode in mode_list:
            for ip, p in enumerate(expansion_orders):
                for itheta, theta in enumerate(theta_list):
                    for ils, leaf_size in enumerate(max_leaf_size_list):
                        seed_case = seed + in_pts * 10000 + ip * 1000 + itheta * 100 + ils
                        set_seed(seed_case)
                        x, q = make_random_points(n_points, seed=seed_case, mode=mode)
                        x = x.to(device=device, dtype=torch.float64)
                        q = q.to(device=device, dtype=torch.float64)
                        areas = torch.ones_like(q)

                        backend = make_laplace_fmm_backend(
                            src_centroids=x,
                            areas=areas,
                            max_leaf_size=int(leaf_size),
                            theta=float(theta),
                            expansion_order=int(p),
                        )

                        phi_prod = backend.matvec(
                            sigma=q,
                            src_centroids=x,
                            areas=areas,
                            tile_size=1024,
                            self_integrals=None,
                        )
                        phi_ref = direct_potential_physical(x, q)

                        err_global = rel_l2_err(phi_prod, phi_ref)
                        stats = target_error_stats(phi_prod, phi_ref)

                        record = make_base_record("fmm_stress_grid", fmm_stage="FMM")
                        record.update(
                            {
                                "geometry_mode": mode,
                                "n_points": int(n_points),
                                "parameters": {
                                    "expansion_order": int(p),
                                    "theta": float(theta),
                                    "max_leaf_size": int(leaf_size),
                                    "seed": seed_case,
                                },
                                "metrics": {
                                    "global": {
                                        "rel_l2_error": float(err_global),
                                        "max_abs_error": float(torch.max(torch.abs(phi_prod - phi_ref)).item()),
                                    },
                                    "targets": stats,
                                    "timing": {
                                        "t_fmm": None,
                                        "t_ref": None,
                                    },
                                },
                                "notes": (
                                    f"FMM stress grid run: N={n_points}, mode={mode}, "
                                    f"p={p}, theta={theta}, leaf={leaf_size}"
                                ),
                            }
                        )
                        log_jsonl(record, "fmm_stress_grid.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sandbox FMM experiments.")
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    rescale = subparsers.add_parser("local_rescaling", help="Sweep local rescaling ratio^{-(l+1)} audits.")
    rescale.add_argument("--p", type=int, default=8)
    rescale.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.5, 2.0, 4.0])
    rescale.add_argument("--seed", type=int, default=12345)
    rescale.add_argument("--n-realizations", type=int, default=3)

    l2l = subparsers.add_parser("l2l", help="Check L2L translation vs explicit P2L at target.")
    l2l.add_argument("--p", type=int, default=8)
    l2l.add_argument("--n-src", type=int, default=64)
    l2l.add_argument("--scale", type=float, default=1.0)
    l2l.add_argument("--translation", type=float, nargs=3, default=[0.3, 0.2, -0.1])
    l2l.add_argument("--seed", type=int, default=20211)

    l2l_sweep = subparsers.add_parser("l2l_sweep", help="Sweep L2L translations over p, scale, and |t|/scale.")
    l2l_sweep.add_argument("--p-list", type=int, nargs="+", default=[4, 8])
    l2l_sweep.add_argument("--n-src", type=int, default=64)
    l2l_sweep.add_argument("--scale-list", type=float, nargs="+", default=[1.0])
    l2l_sweep.add_argument("--translation-norm-list", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    l2l_sweep.add_argument("--n-directions", type=int, default=3)
    l2l_sweep.add_argument("--seed", type=int, default=40000)

    m2l_sweep = subparsers.add_parser("m2l_sweep", help="Sweep M2L translations over p, scale, and |t|/scale.")
    m2l_sweep.add_argument("--p-list", type=int, nargs="+", default=[4, 6, 8])
    m2l_sweep.add_argument("--n-src", type=int, default=64)
    m2l_sweep.add_argument("--scale-list", type=float, nargs="+", default=[1.0])
    m2l_sweep.add_argument("--translation-norm-list", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    m2l_sweep.add_argument("--n-directions", type=int, default=3)
    m2l_sweep.add_argument("--seed", type=int, default=50000)

    fmm_grid = subparsers.add_parser("fmm_stress_grid", help="Global FMM stress grid mirroring test_stress.")
    fmm_grid.add_argument("--n-points-list", type=int, nargs="+", default=[512, 1024, 2048])
    fmm_grid.add_argument("--mode-list", type=str, nargs="+", default=["uniform", "clusters"])
    fmm_grid.add_argument("--expansion-orders", type=int, nargs="+", default=[4, 6, 8])
    fmm_grid.add_argument("--theta-list", type=float, nargs="+", default=[0.5])
    fmm_grid.add_argument("--max-leaf-size-list", type=int, nargs="+", default=[64])
    fmm_grid.add_argument("--seed", type=int, default=123)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experiment == "local_rescaling":
        run_local_rescaling_sweep(p=args.p, ratios=args.ratios, seed=args.seed, n_realizations=args.n_realizations)
    elif args.experiment == "l2l":
        center_parent = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        center_child = torch.tensor(args.translation, dtype=torch.float64)
        run_l2l_translation(
            p=args.p,
            n_src=args.n_src,
            center_parent=center_parent,
            center_child=center_child,
            scale=args.scale,
            seed=args.seed,
        )
    elif args.experiment == "l2l_sweep":
        run_l2l_sweep(
            p_list=args.p_list,
            n_src=args.n_src,
            scale_list=args.scale_list,
            translation_norm_list=args.translation_norm_list,
            n_directions=args.n_directions,
            seed=args.seed,
        )
    elif args.experiment == "m2l_sweep":
        run_m2l_sweep(
            p_list=args.p_list,
            n_src=args.n_src,
            scale_list=args.scale_list,
            translation_norm_list=args.translation_norm_list,
            n_directions=args.n_directions,
            seed=args.seed,
        )
    elif args.experiment == "fmm_stress_grid":
        run_fmm_stress_grid(
            n_points_list=args.n_points_list,
            mode_list=args.mode_list,
            expansion_orders=args.expansion_orders,
            theta_list=args.theta_list,
            max_leaf_size_list=args.max_leaf_size_list,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown experiment {args.experiment}")


if __name__ == "__main__":
    main()
