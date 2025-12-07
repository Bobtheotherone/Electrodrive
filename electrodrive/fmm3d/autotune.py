"""Auto-tuning and performance exploration for FMM backends.

This module provides a small, self-contained auto-tuner for the
:mod:`electrodrive.fmm3d` stack.  It is designed to be:

- **Safe by default**: if tuning is disabled or unsupported, callers
  still receive a validated :class:`FmmConfig`.
- **Cheap** for small problems: tuning runs on synthetic geometries
  with modest sizes.
- **Extensible**: the search strategy and performance model are
  deliberately simple and can be replaced by more sophisticated
  approaches (Bayesian optimisation, offline training, ...).

The primary entry point is :func:`autotune_for_problem`, which takes a
baseline :class:`FmmConfig` and a problem size (number of points /
panels) and returns a tuned configuration plus basic performance and
accuracy metrics.

Notes
-----
- The current implementation benchmarks **CPU-only** FMM backends,
  which matches the current Tier-3 reference implementation.  The API
  is written so that GPU backends can be wired in later without breaking
  callers.
- Accuracy is measured by comparing the FMM result against a direct
  O(N^2) kernel on a synthetic Laplace point-charge problem.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .config import FmmConfig
from .interaction_lists import InteractionLists, build_interaction_lists
from .kernels_cpu import (
    P2PResult,
    apply_p2p_cpu,
    p2m_cpu,
    m2m_cpu,
    m2l_cpu,
    l2l_cpu,
    l2p_cpu,
)
from .multipole_operators import (
    LocalCoefficients,
    MultipoleCoefficients,
    MultipoleOpStats,
)
from .tree import FmmTree, build_fmm_tree

# ---------------------------------------------------------------------------
# Dataclasses and public API
# ---------------------------------------------------------------------------


@dataclass
class TuningResult:
    """Result of an auto-tuning run.

    Parameters
    ----------
    config:
        The selected FMM configuration.
    metrics:
        Flat dictionary of scalar metrics (wall-time, GFLOPS estimates,
        error estimates, problem size, ...).
    """

    config: FmmConfig
    metrics: Dict[str, float]


# Small dataclass used only for serialisation of tuning records.
@dataclass
class _CachedTuningRecord:
    fingerprint: str
    problem_size_bucket: int
    config: Dict[str, Any]
    metrics: Dict[str, float]


AUTOTUNE_CACHE_ENV_VAR = "ELECTRODRIVE_FMM3D_AUTOTUNE_CACHE"
AUTOTUNE_CACHE_FILENAME = "fmm3d_autotune.json"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def autotune_for_problem(
    base_cfg: FmmConfig,
    problem_size: int,
    *,
    dry_run: bool = True,
    target_rel_error: float = 1e-2,
    max_trials: int = 24,
    device: Optional[torch.device] = None,
    use_cache: bool = True,
) -> TuningResult:
    """Autotune FMM parameters for a given problem size.

    The intent is that high-level code (e.g. BEM solvers) can call this
    once at the beginning of a run to obtain a configuration that is
    empirically good for the current machine and approximate problem
    size.

    Parameters
    ----------
    base_cfg:
        Baseline FMM configuration.  The tuner will explore a small
        neighbourhood around this configuration, respecting its kernel
        and precision settings.
    problem_size:
        Number of points / panels in the problem.
    dry_run:
        If True, skip all benchmarking and simply return ``base_cfg``
        wrapped in a :class:`TuningResult`.  This makes it cheap to keep
        calls to :func:`autotune_for_problem` in production code but
        disable tuning by default.
    target_rel_error:
        Desired relative L2 error for the synthetic FMM vs direct
        comparison.  Candidates whose estimated error exceeds this
        threshold are deprioritised.
    max_trials:
        Maximum number of candidate configurations to benchmark.  The
        search space is pruned to honour this limit.
    device:
        Logical device on which the *inputs* live.  Currently the FMM
        implementation is CPU-only; a CUDA device here simply indicates
        that the caller is otherwise GPU-heavy.  The tuner still builds
        trees and runs FMM on the CPU.
    use_cache:
        If True, attempt to reuse previously stored tuning results for
        the same hardware fingerprint and problem-size bucket.

    Returns
    -------
    TuningResult
        The best configuration found (or ``base_cfg`` if tuning was
        skipped or failed), together with metrics for the selected
        configuration.
    """
    if problem_size <= 0:
        raise ValueError("problem_size must be positive")

    # Always validate the baseline configuration early so errors are
    # raised deterministically, even in dry-run mode.
    base_cfg.validate()

    if dry_run:
        # Minimal metrics that are still somewhat informative.
        return TuningResult(
            config=base_cfg,
            metrics={
                "mode": 0.0,  # 0 == dry-run
                "problem_size": float(problem_size),
            },
        )

    # Resolve device and hardware fingerprint.
    if device is None:
        # Even if the caller prefers GPU for other work, the current
        # FMM backend is CPU-only.
        device = torch.device("cpu")

    fingerprint = _hardware_fingerprint(device=device)
    bucket = _bucket_problem_size(problem_size)

    # Try the cache first.
    if use_cache:
        cached = _load_best_from_cache(
            fingerprint=fingerprint,
            bucket=bucket,
            target_rel_error=target_rel_error,
        )
        if cached is not None:
            cfg = _cfg_from_dict(base_cfg, cached.config)
            cfg.validate()
            # Attach cache metadata to metrics.
            metrics = dict(cached.metrics)
            metrics.setdefault("problem_size", float(problem_size))
            metrics.setdefault("from_cache", 1.0)
            return TuningResult(config=cfg, metrics=metrics)

    # No suitable cached result → run an on-the-fly benchmark.
    tuner = _FmmAutotuner(
        base_cfg=base_cfg,
        problem_size=problem_size,
        device=device,
        target_rel_error=target_rel_error,
        max_trials=max_trials,
    )
    best_cfg, best_metrics = tuner.run()

    # Persist to cache (best effort only).
    if use_cache and best_cfg is not None and best_metrics:
        record = _CachedTuningRecord(
            fingerprint=fingerprint,
            problem_size_bucket=bucket,
            config=_cfg_to_dict(best_cfg),
            metrics=best_metrics,
        )
        _append_record_to_cache(record)

    if best_cfg is None:
        # Fall back to the baseline configuration; propagate the last
        # metrics we saw (if any) to aid debugging.
        return TuningResult(
            config=base_cfg,
            metrics=best_metrics or {
                "mode": 1.0,  # 1 == tuning_failed
                "problem_size": float(problem_size),
            },
        )

    return TuningResult(config=best_cfg, metrics=best_metrics)


# ---------------------------------------------------------------------------
# Core tuning logic
# ---------------------------------------------------------------------------


class _FmmAutotuner:
    """Internal helper implementing a simple grid search.

    The current strategy:

    - Samples a modest grid around the baseline expansion order,
      MAC theta, and leaf size.
    - For each candidate configuration:

        * builds a random point cloud in a unit box,
        * runs a single FMM matvec via the Tier-3 CPU backend,
        * compares against a direct O(N^2) kernel for accuracy,
        * estimates FLOP counts and GFLOPS,
        * keeps the candidate if it meets the target error.

    - Picks the candidate with the best wall time among those that meet
      the error target, breaking ties by smaller expansion order.
    """

    def __init__(
        self,
        base_cfg: FmmConfig,
        problem_size: int,
        device: torch.device,
        target_rel_error: float,
        max_trials: int,
    ) -> None:
        self._base_cfg = base_cfg
        self._problem_size = int(problem_size)
        self._device = device
        self._target_rel_error = float(target_rel_error)
        self._max_trials = int(max_trials)

        # Synthetic benchmark size for the direct O(N^2) solve.  We cap
        # this to keep tuning cheap; larger real problems still benefit
        # from tuned configs.
        self._benchmark_size = min(self._problem_size, 4096)

    # ------------------ high-level driver ------------------

    def run(self) -> Tuple[Optional[FmmConfig], Dict[str, float]]:
        candidates = list(self._generate_candidate_configs())
        if not candidates:
            return None, {}

        best_cfg: Optional[FmmConfig] = None
        best_metrics: Dict[str, float] = {}
        best_time: float = float("inf")

        for idx, cfg in enumerate(candidates):
            if idx >= self._max_trials:
                break

            metrics = self._benchmark_config(cfg)
            if not metrics:
                # Benchmark failed; skip.
                continue

            rel_l2 = metrics.get("rel_l2_error", float("inf"))
            if rel_l2 > self._target_rel_error:
                # Discard candidates that do not meet the accuracy target.
                continue

            wall_time = metrics.get("wall_time_s", float("inf"))
            if wall_time < best_time or (
                math.isclose(wall_time, best_time, rel_tol=1e-3)
                and cfg.expansion_order < (best_cfg.expansion_order if best_cfg else cfg.expansion_order + 1)
            ):
                best_time = wall_time
                best_cfg = cfg
                best_metrics = metrics

        return best_cfg, best_metrics

    # ------------------ candidate generation ------------------

    def _generate_candidate_configs(self) -> Iterable[FmmConfig]:
        """Yield a small set of candidate configurations.

        The search space is intentionally conservative.  It explores
        modest variations around the baseline expansion order, MAC theta
        and leaf size, while respecting global limits in
        :class:`FmmConfig`.
        """
        base = self._base_cfg

        # Expansion order candidates: {p-2, p, p+2, p+4}, clipped.
        p0 = base.expansion_order
        p_candidates: List[int] = sorted(
            {
                max(2, p0 - 2),
                p0,
                min(p0 + 2, 16),
                min(p0 + 4, 16),
            }
        )

        # MAC theta candidates: roughly {0.6*θ0, θ0, 1.25*θ0}, clipped.
        t0 = base.mac_theta
        t_candidates: List[float] = []
        for factor in (0.6, 1.0, 1.25):
            theta = float(max(0.15, min(0.85, factor * t0)))
            t_candidates.append(theta)
        t_candidates = sorted(set(t_candidates))

        # Leaf-size candidates: {leaf/2, leaf, 2*leaf}, clipped to [16, problem_size].
        l0 = base.leaf_size
        max_leaf = max(16, self._benchmark_size)
        l_candidates: List[int] = sorted(
            {
                max(16, min(l0 // 2, max_leaf)),
                max(16, min(l0, max_leaf)),
                max(16, min(l0 * 2, max_leaf)),
            }
        )

        for p in p_candidates:
            for theta in t_candidates:
                for leaf in l_candidates:
                    cfg = FmmConfig(
                        kernel=base.kernel,
                        expansion_order=p,
                        mac_theta=theta,
                        leaf_size=leaf,
                        backend=base.backend,
                        precision=base.precision,
                        use_fft_m2l=base.use_fft_m2l,
                        use_gpu=False,  # current backend is CPU-only
                        use_multi_gpu=base.use_multi_gpu,
                        use_mpi=base.use_mpi,
                        dtype=base.dtype,
                        p2p_batch_size=base.p2p_batch_size,
                    )
                    try:
                        cfg.validate()
                    except Exception:
                        # Skip invalid combinations (e.g. exceeding l_max limit).
                        continue
                    yield cfg

    # ------------------ single-config benchmark ------------------

    def _benchmark_config(self, cfg: FmmConfig) -> Dict[str, float]:
        """Benchmark a single candidate configuration.

        Returns a metrics dictionary, or an empty dict if benchmarking
        failed for any reason.
        """
        N = self._benchmark_size
        device = torch.device("cpu")  # FMM stack is CPU-only for now.
        dtype = cfg.dtype

        # Synthetic random point cloud in a unit box.
        g = torch.Generator(device="cpu").manual_seed(12345)
        points = torch.rand((N, 3), generator=g, dtype=dtype, device=device)
        charges = torch.randn((N,), generator=g, dtype=dtype, device=device)

        # Build tree + interaction lists once for this config.
        t0 = time.perf_counter()
        tree = build_fmm_tree(points, leaf_size=cfg.leaf_size)
        lists = build_interaction_lists(
            source_tree=tree,
            target_tree=tree,
            mac_theta=cfg.mac_theta,
        )
        t_tree = time.perf_counter()

        # FMM computation.
        stats = MultipoleOpStats()
        try:
            t_fmm_start = time.perf_counter()
            phi_fmm_tree, p2p_result = _run_fmm_pipeline(
                tree=tree,
                lists=lists,
                charges=charges,
                cfg=cfg,
                stats=stats,
            )
            t_fmm_end = time.perf_counter()
        except Exception:
            return {}

        # Direct reference solution for error estimation.
        try:
            t_ref_start = time.perf_counter()
            phi_ref = _direct_potential(points, points, charges, exclude_self=True)
            t_ref_end = time.perf_counter()
        except Exception:
            return {}

        # Reorder FMM result to original ordering; direct already uses original order.
        phi_fmm = phi_fmm_tree[tree.tree_to_original]

        # Error metrics.
        diff = phi_fmm - phi_ref
        rel_l2 = torch.linalg.norm(diff) / max(
            torch.linalg.norm(phi_ref),
            torch.tensor(torch.finfo(dtype).eps, dtype=dtype),
        )
        max_abs_err = torch.max(torch.abs(diff))
        max_abs_ref = torch.max(torch.abs(phi_ref))
        max_rel_err = (max_abs_err / max_abs_ref) if max_abs_ref > 0 else torch.tensor(
            0.0, dtype=dtype
        )

        # Performance metrics.
        wall_time_s = t_fmm_end - t_fmm_start
        build_time_s = t_tree - t0
        ref_time_s = t_ref_end - t_ref_start

        flops_estimate, gflops_estimate = _estimate_flops_and_gflops(
            tree=tree,
            cfg=cfg,
            stats=stats,
            p2p_result=p2p_result,
            wall_time_s=wall_time_s,
        )

        metrics: Dict[str, float] = {
            "problem_size": float(N),
            "expansion_order": float(cfg.expansion_order),
            "mac_theta": float(cfg.mac_theta),
            "leaf_size": float(cfg.leaf_size),
            "wall_time_s": float(wall_time_s),
            "build_time_s": float(build_time_s),
            "ref_time_s": float(ref_time_s),
            "rel_l2_error": float(rel_l2),
            "max_rel_error": float(max_rel_err),
            "max_abs_error": float(max_abs_err),
            "flops_estimate": float(flops_estimate),
            "gflops_estimate": float(gflops_estimate),
            "p2m_calls": float(stats.p2m_calls),
            "m2m_calls": float(stats.m2m_calls),
            "m2l_calls": float(stats.m2l_calls),
            "l2l_calls": float(stats.l2l_calls),
            "l2p_calls": float(stats.l2p_calls),
        }

        return metrics


# ---------------------------------------------------------------------------
# FMM pipeline + direct kernel for error estimation
# ---------------------------------------------------------------------------


def _run_fmm_pipeline(
    tree: FmmTree,
    lists: InteractionLists,
    charges: Tensor,
    cfg: FmmConfig,
    stats: Optional[MultipoleOpStats] = None,
) -> Tuple[Tensor, P2PResult]:
    """Run the full near-field + far-field FMM pipeline on a point cloud.

    This mirrors the structure used in the FMM sanity suite:

    - P2M → M2M build multipoles.
    - M2L → L2L → L2P produce far-field contributions.
    - P2P handles near-field corrections.

    Parameters
    ----------
    tree:
        FmmTree built over the input points.
    lists:
        Interaction lists (including U-list for P2P).
    charges:
        Charge vector in **tree order**.
    cfg:
        FMM configuration.
    stats:
        Optional :class:`MultipoleOpStats` that will be updated with
        call counts.

    Returns
    -------
    (phi_tree, p2p_result)
        - ``phi_tree``: potential in tree order.
        - ``p2p_result``: P2P diagnostics, including interaction count.
    """
    if stats is None:
        stats = MultipoleOpStats()

    # Far-field: P2M / M2M / M2L / L2L / L2P.
    multipoles: MultipoleCoefficients = p2m_cpu(
        tree=tree,
        charges=charges,
        cfg=cfg,
        stats=stats,
    )
    multipoles = m2m_cpu(
        tree=tree,
        multipoles=multipoles,
        cfg=cfg,
        stats=stats,
    )
    locals_: LocalCoefficients = m2l_cpu(
        source_tree=tree,
        target_tree=tree,
        multipoles=multipoles,
        cfg=cfg,
        stats=stats,
    )
    locals_ = l2l_cpu(
        tree=tree,
        locals_=locals_,
        cfg=cfg,
        stats=stats,
    )
    phi_far_tree: Tensor = l2p_cpu(
        tree=tree,
        locals_=locals_,
        cfg=cfg,
        stats=stats,
    )

    # Near-field: P2P correction using interaction lists.
    p2p_result = apply_p2p_cpu(
        source_tree=tree,
        target_tree=tree,
        charges_src=charges,
        lists=lists,
        cfg=cfg,
        logger=None,
        out=None,
    )
    phi_p2p_tree = p2p_result.potential

    phi_tree = phi_far_tree + phi_p2p_tree
    return phi_tree, p2p_result


def _direct_potential(
    points_src: Tensor,
    points_tgt: Tensor,
    charges_src: Tensor,
    *,
    exclude_self: bool = True,
) -> Tensor:
    """Naive O(N^2) Laplace potential for reference.

    This is intentionally simple and CPU-only.  It mirrors the direct
    reference used in :mod:`electrodrive.fmm3d.sanity_suite` but keeps
    the implementation local to avoid circular imports.
    """
    if (
        points_src.ndim != 2
        or points_tgt.ndim != 2
        or points_src.shape[1] != 3
        or points_tgt.shape[1] != 3
    ):
        raise ValueError("points_src and points_tgt must have shape (N, 3) and (M, 3)")
    if charges_src.shape[0] != points_src.shape[0]:
        raise ValueError("charges_src must have length N matching points_src")

    # Pure CPU implementation.
    pts_src = points_src.to("cpu")
    pts_tgt = points_tgt.to("cpu")
    q = charges_src.to("cpu")

    diff = pts_tgt[:, None, :] - pts_src[None, :, :]
    r2 = torch.sum(diff * diff, dim=-1)
    eps = torch.finfo(r2.dtype).eps
    r = torch.sqrt(torch.clamp(r2, min=eps))
    inv_r = torch.where(r > 0.0, 1.0 / r, torch.zeros_like(r))

    if (
        exclude_self
        and pts_src.data_ptr() == pts_tgt.data_ptr()
        and pts_src.shape[0] == pts_tgt.shape[0]
    ):
        n = pts_src.shape[0]
        diag = torch.eye(n, dtype=torch.bool, device=pts_src.device)
        inv_r = inv_r.masked_fill(diag, 0.0)

    # Coulomb constant is intentionally omitted here; we only care about
    # *relative* errors, so the global scaling factor cancels.
    contrib = q[None, :] * inv_r
    phi = contrib.sum(dim=1)
    return phi.to(points_tgt.device)


# ---------------------------------------------------------------------------
# FLOP estimates
# ---------------------------------------------------------------------------


def _estimate_flops_and_gflops(
    tree: FmmTree,
    cfg: FmmConfig,
    stats: MultipoleOpStats,
    p2p_result: P2PResult,
    wall_time_s: float,
) -> Tuple[float, float]:
    """Very rough FLOP and GFLOPS estimates for a single FMM run.

    The goal here is not cycle-accurate accounting, but rather a stable
    metric that allows for relative comparisons between configurations.

    Assumptions
    -----------
    - Each P2P interaction costs ~20 floating-point operations.
    - P2M and L2P are O(P^2) per node.
    - M2M and L2L are O(P^2) per translation.
    - M2L is O(P^4) with a small constant for the Laplace kernel.
    """
    P = (cfg.expansion_order + 1) ** 2
    l_max_m2l = 2 * cfg.expansion_order
    P_m2l = (l_max_m2l + 1) ** 2

    # P2P interactions.
    n_interactions = getattr(p2p_result, "n_interactions", 0)
    flops_p2p = 20.0 * float(n_interactions)

    # P2M / L2P / M2M / L2L / M2L.
    flops_p2m = 2.0 * P * float(tree.n_points)  # ~2 flops per term per particle
    flops_l2p = 2.0 * P * float(tree.n_points)

    flops_m2m = 4.0 * (P ** 2) * float(stats.m2m_calls)
    flops_l2l = 4.0 * (P ** 2) * float(stats.l2l_calls)
    flops_m2l = 4.0 * P * P_m2l * float(stats.m2l_calls)

    flops_total = flops_p2p + flops_p2m + flops_l2p + flops_m2m + flops_l2l + flops_m2l
    if wall_time_s <= 0.0:
        return flops_total, 0.0

    gflops = flops_total / (wall_time_s * 1e9)
    return flops_total, gflops


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------


def _bucket_problem_size(N: int) -> int:
    """Bucket problem sizes by rounding up to the next power of two."""
    if N <= 0:
        return 1
    return 1 << (int(math.ceil(math.log2(N))))


def _hardware_fingerprint(*, device: torch.device) -> str:
    """Return a simple, stable fingerprint for the current machine.

    The fingerprint intentionally avoids including ephemeral details
    such as driver versions; it is stable across minor software updates.
    """
    import platform

    parts: List[str] = [
        platform.node() or "unknown-host",
        platform.system() or "unknown-os",
        platform.machine() or "unknown-arch",
    ]

    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index or 0
        try:
            props = torch.cuda.get_device_properties(idx)
            parts.extend(
                [
                    "cuda",
                    props.name.replace(" ", "_"),
                    str(props.multi_processor_count),
                    str(props.total_memory),
                ]
            )
        except Exception:
            parts.append("cuda-unknown")
    else:
        parts.append("cpu")

    return "|".join(parts)


def _get_cache_path() -> Path:
    env = os.environ.get(AUTOTUNE_CACHE_ENV_VAR)
    if env:
        return Path(env)
    # Default to XDG cache directory if available, otherwise ~/.cache.
    base = os.environ.get("XDG_CACHE_HOME")
    if base is None:
        base = os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "electrodrive" / AUTOTUNE_CACHE_FILENAME


def _load_best_from_cache(
    *,
    fingerprint: str,
    bucket: int,
    target_rel_error: float,
) -> Optional[_CachedTuningRecord]:
    """Return the best cached record for a given hardware/size bucket."""
    path = _get_cache_path()
    if not path.is_file():
        return None

    try:
        with path.open("r", encoding="utf8") as f:
            raw = json.load(f)
    except Exception:
        return None

    records: List[Dict[str, Any]] = raw.get("records", [])
    best: Optional[_CachedTuningRecord] = None
    best_time: float = float("inf")

    for rec in records:
        if rec.get("fingerprint") != fingerprint:
            continue
        if int(rec.get("problem_size_bucket", -1)) != int(bucket):
            continue

        metrics = rec.get("metrics", {})
        rel_l2 = float(metrics.get("rel_l2_error", float("inf")))
        if rel_l2 > target_rel_error:
            continue

        wall = float(metrics.get("wall_time_s", float("inf")))
        if wall < best_time:
            best_time = wall
            best = _CachedTuningRecord(
                fingerprint=rec["fingerprint"],
                problem_size_bucket=int(rec["problem_size_bucket"]),
                config=dict(rec.get("config", {})),
                metrics={k: float(v) for k, v in metrics.items()},
            )

    return best


def _append_record_to_cache(record: _CachedTuningRecord) -> None:
    """Append a single tuning record to the cache file (best effort)."""
    path = _get_cache_path()
    try:
        if path.is_file():
            with path.open("r", encoding="utf8") as f:
                raw = json.load(f)
        else:
            raw = {"records": []}
    except Exception:
        raw = {"records": []}

    records: List[Dict[str, Any]] = raw.get("records", [])
    records.append(
        {
            "fingerprint": record.fingerprint,
            "problem_size_bucket": int(record.problem_size_bucket),
            "config": record.config,
            "metrics": record.metrics,
        }
    )
    raw["records"] = records

    # Ensure parent directory exists.
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf8") as f:
            json.dump(raw, f, indent=2, sort_keys=True)
    except Exception:
        # Cache failures should never be fatal.
        return


# ---------------------------------------------------------------------------
# Config (de-)serialisation helpers
# ---------------------------------------------------------------------------


def _cfg_to_dict(cfg: FmmConfig) -> Dict[str, Any]:
    """Serialise an :class:`FmmConfig` into a JSON-friendly dict.

    We keep only fields that are stable and cheap to reconstruct.
    """
    d: Dict[str, Any] = {
        "kernel": cfg.kernel,
        "expansion_order": int(cfg.expansion_order),
        "mac_theta": float(cfg.mac_theta),
        "leaf_size": int(cfg.leaf_size),
        "backend": cfg.backend,
        "precision": cfg.precision,
        "use_fft_m2l": bool(cfg.use_fft_m2l),
        "use_gpu": bool(cfg.use_gpu),
        "use_multi_gpu": bool(cfg.use_multi_gpu),
        "use_mpi": bool(cfg.use_mpi),
        "p2p_batch_size": int(cfg.p2p_batch_size)
        if cfg.p2p_batch_size is not None
        else None,
    }
    # Dtype is reconstructed from precision by FmmConfig.__post_init__ if
    # not explicitly provided, so we do not need to persist it.
    return d


def _cfg_from_dict(base: FmmConfig, data: Dict[str, Any]) -> FmmConfig:
    """Reconstruct an :class:`FmmConfig` from a dict.

    The baseline configuration is used to fill in any fields that are
    missing in the stored representation.
    """
    return FmmConfig(
        kernel=data.get("kernel", base.kernel),
        expansion_order=int(data.get("expansion_order", base.expansion_order)),
        mac_theta=float(data.get("mac_theta", base.mac_theta)),
        leaf_size=int(data.get("leaf_size", base.leaf_size)),
        backend=data.get("backend", base.backend),
        precision=data.get("precision", base.precision),
        use_fft_m2l=bool(data.get("use_fft_m2l", base.use_fft_m2l)),
        use_gpu=bool(data.get("use_gpu", base.use_gpu)),
        use_multi_gpu=bool(data.get("use_multi_gpu", base.use_multi_gpu)),
        use_mpi=bool(data.get("use_mpi", base.use_mpi)),
        dtype=base.dtype,
        p2p_batch_size=data.get("p2p_batch_size", base.p2p_batch_size),
    )
