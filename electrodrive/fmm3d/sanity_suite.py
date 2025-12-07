import argparse
import dataclasses
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

import torch
from torch import Tensor

from electrodrive.utils.config import K_E
from electrodrive.utils.logging import JsonlLogger  # noqa: F401 (for type hints)

from electrodrive.fmm3d.tree import FmmTree, build_fmm_tree
from electrodrive.fmm3d.interaction_lists import (
    InteractionLists,
    build_interaction_lists,
    verify_interaction_lists,
)
from electrodrive.fmm3d.kernels_cpu import (
    P2PResult,
    apply_p2p_cpu_tiled,
)
from electrodrive.fmm3d.multipole_operators import (
    FmmConfig,
    MultipoleOpStats,
    apply_fmm_laplace_potential,
)
from electrodrive.core.bem_kernel import (
    bem_matvec_gpu,
    DEFAULT_SINGLE_LAYER_KERNEL,
)
from electrodrive.fmm3d.bem_fmm import make_laplace_fmm_backend
from electrodrive.fmm3d.logging_utils import log_test_result_jsonl, want_jsonl


# ---------------------------------------------------------------------------
# Structured result type
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    """
    Simple container for a single numerical sanity test.

    Attributes
    ----------
    name:
        Logical test name, e.g. ``"p2p_vs_direct"``.
    ok:
        True if the test passed according to the requested tolerance.
    max_abs_err:
        Maximum absolute error between reference and test quantities.
    rel_l2_err:
        Relative L2 error (norm of difference divided by norm of reference).
    extra:
        Free-form metadata: seeds, timing, configuration, etc.
    """

    name: str
    ok: bool
    max_abs_err: float
    rel_l2_err: float
    extra: Dict[str, Any]

    def as_row(self) -> str:
        """Human-readable one-line summary."""
        status = "PASS" if self.ok else "FAIL"
        return (
            f"[{status}] {self.name:25s}  "
            f"max_abs_err={self.max_abs_err:.3e}  "
            f"rel_l2={self.rel_l2_err:.3e}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _direct_potential(x: Tensor, q: Tensor) -> Tensor:
    """
    Naive O(N²) Coulomb potential for sanity-checking FMM / P2P paths.

    Parameters
    ----------
    x : (N, 3)
        Point locations.
    q : (N,)
        Source charges.

    Returns
    -------
    phi : (N,)
        Potential ``phi_i = sum_{j != i} K_E * q_j / |x_i - x_j|``, with
        self-interactions (r = 0) contributing exactly zero.
    """
    N = x.shape[0]
    # Broadcasting:
    #   x[:, None, :] -> (N, 1, 3)
    #   x[None, :, :] -> (1, N, 3)
    diff = x[:, None, :] - x[None, :, :]  # (N, N, 3)
    r2 = torch.sum(diff * diff, dim=-1)   # (N, N)

    # Robust inversion: clamp small values to avoid division by zero.
    # We will explicitly zero the diagonal later.
    eps = 1e-12
    r = torch.sqrt(torch.clamp(r2, min=eps*eps))
    inv_r = 1.0 / r

    # Explicitly zero-out self-interactions using indices.
    # This is more robust than comparing floats (r2 == 0).
    if N > 0:
        idx = torch.arange(N, device=x.device)
        inv_r[idx, idx] = 0.0

    # charges: (N,) -> (1, N) for broadcasting
    phi = K_E * torch.sum(q[None, :] * inv_r, dim=1)
    return phi


def _find_fmm_config_class() -> type[FmmConfig]:
    """
    Locate a suitable FmmConfig class.

    This is defensive in case the project reorganises configuration modules.
    """
    # Prefer the canonical FmmConfig we imported above.
    return FmmConfig


def make_default_fmm_config(
    *,
    expansion_order: Optional[int] = None,
    mac_theta: Optional[float] = None,
    leaf_size: Optional[int] = None,
) -> FmmConfig:
    """
    Construct a default FmmConfig with optional overrides.

    Parameters
    ----------
    expansion_order:
        Optional override for multipole expansion order ``p``.
    mac_theta:
        Optional override for MAC parameter ``theta``.
    leaf_size:
        Optional override for maximum leaf size in the tree.
        This maps to ``FmmConfig.leaf_size``.
    """
    cfg_cls = _find_fmm_config_class()
    cfg = cfg_cls()  # type: ignore[call-arg]

    if expansion_order is not None:
        cfg.expansion_order = int(expansion_order)
    if mac_theta is not None:
        cfg.mac_theta = float(mac_theta)
    if leaf_size is not None:
        cfg.leaf_size = int(leaf_size)

    return cfg


# ---------------------------------------------------------------------------
# Generic runner with JSONL emission
# ---------------------------------------------------------------------------


def _run_test(
    suite_label: str,
    name: str,
    fn: Callable[..., TestResult],
    *args: Any,
    **kwargs: Any,
) -> TestResult:
    """
    Helper to run a test function and convert unexpected exceptions into
    a failing :class:`TestResult`, while also emitting a JSONL record
    (if enabled).

    Parameters
    ----------
    suite_label:
        Logical suite identifier, e.g. ``"sanity_suite"`` or
        ``"pytest_fmm_tests"``.
    name:
        Test name.
    fn:
        Callable implementing the numerical experiment, returning
        :class:`TestResult`.
    *args, **kwargs:
        Passed through to ``fn``.
    """
    t0 = time.perf_counter()
    try:
        res = fn(*args, **kwargs)
        if not isinstance(res, TestResult):  # defensive
            raise TypeError(
                f"Test function {fn.__name__} did not return a TestResult; "
                f"got {type(res)!r}"
            )
        dt = time.perf_counter() - t0
        res.extra.setdefault("wall_time_s", dt)
    except Exception as exc:  # pragma: no cover - defensive
        dt = time.perf_counter() - t0
        res = TestResult(
            name=name,
            ok=False,
            max_abs_err=float("inf"),
            rel_l2_err=float("inf"),
            extra={
                "error": repr(exc),
                "wall_time_s": dt,
            },
        )

    # Emit JSONL record for offline analysis, if enabled via env var.
    log_test_result_jsonl(suite_label, res)
    return res


# ---------------------------------------------------------------------------
# 1. Tree + interaction lists
# ---------------------------------------------------------------------------


def test_tree_and_interaction_lists(
    n_points: int,
    device: torch.device,
    dtype: torch.dtype,
) -> TestResult:
    """
    Build a random point cloud, construct the FMM tree and interaction lists,
    and run internal consistency checks on the lists.

    This does *not* involve any numerical kernel evaluations; it is purely
    geometric/structural.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    pts = torch.rand(n_points, 3, generator=g, device=device, dtype=dtype) - 0.5

    tree: FmmTree = build_fmm_tree(points=pts, leaf_size=64)
    lists: InteractionLists = build_interaction_lists(
        source_tree=tree,
        target_tree=tree,
        mac_theta=0.5,
    )

    # Verify structural invariants; this function raises on failure.
    verify_interaction_lists(tree, tree, lists)

    extra = {
        "n_points": int(n_points),
        "n_nodes": int(len(tree.nodes)),
        "n_p2p_pairs": int(lists.num_p2p_pairs),
        "n_m2l_pairs": int(lists.num_m2l_pairs),
        "seed": 0,
        "leaf_size": 64,
        "mac_theta": 0.5,
    }

    # If verify_interaction_lists did not raise, we treat this as a pass with
    # zero error. We keep the TestResult structure uniform.
    return TestResult(
        name="tree_and_interaction_lists",
        ok=True,
        max_abs_err=0.0,
        rel_l2_err=0.0,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# 2. P2P kernel vs direct
# ---------------------------------------------------------------------------


def test_p2p_against_direct(
    n_points: int,
    device: torch.device,
    dtype: torch.dtype,
    tol: float,
) -> TestResult:
    """
    Compare the tiled CPU P2P kernel against an explicit O(N²) computation.

    The random cloud is fixed via a deterministic seed.

    The tolerance ``tol`` is interpreted as a bound on the *relative* L2 error.
    """
    g = torch.Generator(device="cpu").manual_seed(1)
    x = torch.rand(n_points, 3, generator=g, device=device, dtype=dtype) - 0.5
    q = torch.randn(n_points, generator=g, device=device, dtype=dtype)

    # Reference (on requested device). Includes K_E.
    phi_ref = _direct_potential(x, q)

    # For the FMM tree + P2P kernel we currently run on CPU, which is also
    # what the tests use (device=_CPU_DEVICE). If in the future GPU support
    # is added here, we can move the tree to CUDA and keep the logic below.
    x_cpu = x.to("cpu")
    q_cpu = q.to("cpu")

    tree: FmmTree = build_fmm_tree(points=x_cpu, leaf_size=64)
    lists: InteractionLists = build_interaction_lists(
        source_tree=tree,
        target_tree=tree,
        mac_theta=1e-3,  # essentially force near-field for all pairs
    )

    charges_tree = tree.map_to_tree_order(q_cpu)

    p2p_res: P2PResult = apply_p2p_cpu_tiled(
        source_tree=tree,
        target_tree=tree,
        charges_src=charges_tree,
        lists=lists,
        tile_size_points=256,
        logger=None,
        out=None,
    )

    # The CPU P2P kernel returns pure 1/r. We must apply K_E to compare
    # against _direct_potential (which has K_E).
    phi_p2p_tree = p2p_res.potential * float(K_E)
    
    phi_p2p_cpu = tree.map_to_original_order(phi_p2p_tree)
    phi_p2p = phi_p2p_cpu.to(device=device)

    diff = phi_p2p - phi_ref
    max_abs_err = float(diff.abs().max().item())
    denom = float(phi_ref.norm().item())
    rel_l2 = 0.0 if denom == 0.0 else float(diff.norm().item() / denom)

    # Gate on relative error only; absolute error is not meaningful
    # in physical units with K_E.
    ok = rel_l2 <= tol

    extra = {
        "n_points": int(n_points),
        "mac_theta": 1e-3,
        "n_pairs": int(p2p_res.n_pairs),
        "n_interactions": int(p2p_res.n_interactions),
        "max_abs_err": max_abs_err,
        "rel_l2": rel_l2,
        "seed": 1,
        "leaf_size": 64,
        "tile_size_points": 256,
    }

    return TestResult(
        name="p2p_vs_direct",
        ok=ok,
        max_abs_err=max_abs_err,
        rel_l2_err=rel_l2,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# 3. Full FMM vs direct (currently via direct helper)
# ---------------------------------------------------------------------------


def test_full_fmm_against_direct(
    n_points: int,
    device: torch.device,
    dtype: torch.dtype,
    tol: float,
    *,
    expansion_order: Optional[int] = None,
    mac_theta: Optional[float] = None,
    leaf_size: Optional[int] = None,
    p2p_batch_size: Optional[int] = None,
) -> TestResult:
    """
    Full FMM test against explicit O(N²) Coulomb potential.

    At present this uses :func:`apply_fmm_laplace_potential` as a
    compatibility helper, which performs a vectorised direct Coulomb
    evaluation using tree geometry (no multipole compression yet), with
    self-interactions (r = 0) contributing zero to the potential.
    """
    g = torch.Generator(device="cpu").manual_seed(2)
    x = torch.rand(n_points, 3, generator=g, device=device, dtype=dtype) - 0.5
    q = torch.randn(n_points, generator=g, device=device, dtype=dtype)

    # Direct reference on the requested device.
    phi_ref = _direct_potential(x, q)

    cfg = make_default_fmm_config(
        expansion_order=expansion_order,
        mac_theta=mac_theta,
        leaf_size=leaf_size,
    )

    # Tree construction + helper currently run on CPU.
    x_cpu = x.to("cpu")
    q_cpu = q.to("cpu")

    tree: FmmTree = build_fmm_tree(
        points=x_cpu,
        leaf_size=cfg.leaf_size,
    )
    lists: InteractionLists = build_interaction_lists(
        source_tree=tree,
        target_tree=tree,
        mac_theta=cfg.mac_theta,
    )

    stats = MultipoleOpStats()

    # NOTE:
    # apply_fmm_laplace_potential expects tree geometry AND tree-ordered charges
    # because it performs a direct sum on tree.points.
    charges_tree = tree.map_to_tree_order(q_cpu)

    phi_tree = apply_fmm_laplace_potential(
        tree,
        charges_tree,  # Corrected: pass tree-ordered charges
        cfg,
        stats=stats,
    )

    # Map back to original order!
    phi_fmm_cpu = tree.map_to_original_order(phi_tree)
    phi_fmm = phi_fmm_cpu.to(device=device)

    diff = phi_fmm - phi_ref
    max_abs_err = float(diff.abs().max().item())
    denom = float(phi_ref.norm().item())
    rel_l2 = 0.0 if denom == 0.0 else float(diff.norm().item() / denom)
    ok = rel_l2 <= tol

    extra: Dict[str, Any] = {
        "n_points": int(n_points),
        "expansion_order": int(cfg.expansion_order),
        "mac_theta": float(cfg.mac_theta),
        "leaf_size": int(cfg.leaf_size),
        "n_p2p_pairs": int(lists.num_p2p_pairs),
        "n_m2l_pairs": int(lists.num_m2l_pairs),
        "rel_l2": rel_l2,
        "max_abs_err": max_abs_err,
        "seed": 2,
        "p2p_batch_size": int(p2p_batch_size or 32_768),
        "stats_keys": sorted(
            k for k, v in dataclasses.asdict(stats).items() if v
        ),
    }

    return TestResult(
        name="fmm_vs_direct",
        ok=ok,
        max_abs_err=max_abs_err,
        rel_l2_err=rel_l2,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# 4. BEM FMM vs BEM direct (LaplaceFmm3D backend)
# ---------------------------------------------------------------------------


def test_bem_fmm_against_bem(
    n_panels: int,
    device: torch.device,
    dtype: torch.dtype,
    tol: float,
) -> TestResult:
    """
    Compare the LaplaceFmm3D backend against the direct BEM matvec.

    The geometry is synthetic (random centroids + areas), but both reference
    and test paths use the *same* discretisation.
    """
    g = torch.Generator(device="cpu").manual_seed(3)
    centroids = torch.rand(
        n_panels, 3, generator=g, device=device, dtype=dtype
    ) - 0.5
    areas = torch.rand(
        n_panels, generator=g, device=device, dtype=dtype
    ) * 0.1
    sigma = torch.randn(
        n_panels, generator=g, device=device, dtype=dtype
    )

    # Direct BEM reference (torch-tiled backend).
    V_ref = bem_matvec_gpu(
        sigma=sigma,
        src_centroids=centroids,
        areas=areas,
        tile_size=256,
        self_integrals=None,
        logger=None,
        use_keops=False,
        kernel=DEFAULT_SINGLE_LAYER_KERNEL,
        backend="torch_tiled",
    )

    # FMM backend (LaplaceFmm3D).
    fmm = make_laplace_fmm_backend(
        src_centroids=centroids,
        areas=areas,
        max_leaf_size=64,
        theta=0.6,
        use_dipole=True,
        logger=None,
    )

    V_fmm = bem_matvec_gpu(
        sigma=sigma,
        src_centroids=centroids,
        areas=areas,
        tile_size=256,
        self_integrals=None,
        logger=None,
        use_keops=False,
        kernel=DEFAULT_SINGLE_LAYER_KERNEL,
        backend="external",
        matvec_impl=fmm.matvec,
    )

    diff = V_fmm - V_ref
    max_abs_err = float(diff.abs().max().item())
    denom = float(V_ref.norm().item())
    rel_l2 = 0.0 if denom == 0.0 else float(diff.norm().item() / denom)
    ok = rel_l2 <= tol

    extra = {
        "n_panels": int(n_panels),
        "theta": 0.6,
        "max_leaf_size": 64,
        "rel_l2": rel_l2,
        "max_abs_err": max_abs_err,
        "seed": 3,
        "tile_size": 256,
    }

    return TestResult(
        name="bem_fmm_vs_bem_direct",
        ok=ok,
        max_abs_err=max_abs_err,
        rel_l2_err=rel_l2,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Public run_* wrappers used by pytest
# ---------------------------------------------------------------------------


def run_tree_and_interaction_lists(
    *,
    n_points: int,
    device: torch.device,
    dtype: torch.dtype,
    suite_label: str = "pytest_fmm_tests",
) -> TestResult:
    """
    Public wrapper used by pytest to exercise the tree / list sanity check
    while emitting a JSONL record tagged with ``suite_label``.
    """
    return _run_test(
        suite_label,
        "tree_and_interaction_lists",
        test_tree_and_interaction_lists,
        n_points,
        device,
        dtype,
    )


def run_p2p_against_direct(
    *,
    n_points: int,
    device: torch.device,
    dtype: torch.dtype,
    tol: float,
    suite_label: str = "pytest_fmm_tests",
) -> TestResult:
    """
    Public wrapper used by pytest for the P2P vs direct comparison.
    """
    return _run_test(
        suite_label,
        "p2p_vs_direct",
        test_p2p_against_direct,
        n_points,
        device,
        dtype,
        tol,
    )


def run_full_fmm_against_direct(
    *,
    n_points: int,
    device: torch.device,
    dtype: torch.dtype,
    tol: float,
    expansion_order: Optional[int] = None,
    mac_theta: Optional[float] = None,
    leaf_size: Optional[int] = None,
    p2p_batch_size: Optional[int] = None,
    suite_label: str = "pytest_fmm_tests",
) -> TestResult:
    """
    Public wrapper used by pytest for the full FMM vs direct comparison.
    """
    return _run_test(
        suite_label,
        "fmm_vs_direct",
        test_full_fmm_against_direct,
        n_points,
        device,
        dtype,
        tol,
        expansion_order=expansion_order,
        mac_theta=mac_theta,
        leaf_size=leaf_size,
        p2p_batch_size=p2p_batch_size,
    )


def run_bem_fmm_against_bem(
    *,
    n_panels: int,
    device: torch.device,
    dtype: torch.dtype,
    tol: float,
    suite_label: str = "pytest_fmm_tests",
) -> TestResult:
    """
    Public wrapper used by pytest for the BEM–FMM vs BEM direct test.
    """
    return _run_test(
        suite_label,
        "bem_fmm_vs_bem_direct",
        test_bem_fmm_against_bem,
        n_panels,
        device,
        dtype,
        tol,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m electrodrive.fmm3d.sanity_suite",
        description="Run FMM sanity tests against direct evaluations.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for computations (e.g. 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="Floating-point dtype for tests.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=512,
        help="Number of points for point-cloud tests.",
    )
    parser.add_argument(
        "--tol-p2p",
        type=float,
        default=1e-10,
        help="Tolerance for P2P vs direct test (relative L2).",
    )
    parser.add_argument(
        "--tol-fmm",
        type=float,
        default=1e-2,
        help="Tolerance for FMM vs direct test (relative L2).",
    )
    parser.add_argument(
        "--tol-bem",
        type=float,
        default=1e-1,
        help="Tolerance for BEM–FMM vs BEM direct test (relative L2).",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help=(
            "Force structured JSONL test records by setting "
            "EDE_FMM_ENABLE_JSONL=1 inside this process. "
            "If EDE_FMM_JSONL_PATH is set, records will also be "
            "appended to that file."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    # If requested, force JSONL emission inside this process so that
    # log_test_result_jsonl will actually emit structured records.
    if getattr(args, "jsonl", False):
        os.environ["EDE_FMM_ENABLE_JSONL"] = "1"

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    # Human-facing summary on stdout.
    print(
        f"[sanity_suite] device={device}, dtype={dtype}, "
        f"n_points={args.n_points}"
    )

    # Explicitly report JSONL status + destination on stderr so your
    # PowerShell transcript makes the wiring obvious.
    jsonl_active = want_jsonl()
    jsonl_path = os.environ.get("EDE_FMM_JSONL_PATH", "").strip() or "<stdout>"
    print(
        f"[sanity_suite] jsonl_active={jsonl_active}, jsonl_path={jsonl_path}",
        file=sys.stderr,
    )

    suite_label = "sanity_suite"

    results: list[TestResult] = []

    # 1. Tree + interaction lists
    results.append(
        _run_test(
            suite_label,
            "tree_and_interaction_lists",
            test_tree_and_interaction_lists,
            args.n_points,
            device,
            dtype,
        )
    )

    # 2. P2P vs direct
    results.append(
        _run_test(
            suite_label,
            "p2p_vs_direct",
            test_p2p_against_direct,
            args.n_points,
            device,
            dtype,
            args.tol_p2p,
        )
    )

    # 3. Full FMM vs direct
    results.append(
        _run_test(
            suite_label,
            "fmm_vs_direct",
            test_full_fmm_against_direct,
            args.n_points,
            device,
            dtype,
            args.tol_fmm,
        )
    )

    # 4. BEM FMM vs BEM direct
    results.append(
        _run_test(
            suite_label,
            "bem_fmm_vs_bem_direct",
            test_bem_fmm_against_bem,
            512,
            device,
            dtype,
            args.tol_bem,
        )
    )

    print("\n=== FMM3D sanity summary ===")
    for res in results:
        print(res.as_row())
        if res.extra:
            print(f"    extra: {res.extra}")
        print()

    # Exit with non-zero status if any test failed.
    ok_all = all(r.ok for r in results)

    # Emit a final summary record so that log analyzers always see at least
    # one structured event for this suite when JSONL is enabled.
    if want_jsonl():
        summary = TestResult(
            name="sanity_suite_summary",
            ok=ok_all,
            max_abs_err=0.0,
            rel_l2_err=0.0,
            extra={
                "n_tests": len(results),
                "n_fail": sum(not r.ok for r in results),
            },
        )
        log_test_result_jsonl(suite_label, summary)

    if not ok_all:
        print(
            "At least one FMM-related test FAILED. See details above.",
            file=sys.stderr,
        )
        return 1
    return 0


__all__ = [
    "TestResult",
    "test_tree_and_interaction_lists",
    "test_p2p_against_direct",
    "test_full_fmm_against_direct",
    "test_bem_fmm_against_bem",
    "run_tree_and_interaction_lists",
    "run_p2p_against_direct",
    "run_full_fmm_against_direct",
    "run_bem_fmm_against_bem",
]


if __name__ == "__main__":
    raise SystemExit(main())