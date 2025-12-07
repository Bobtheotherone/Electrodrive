from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import torch
    from torch import Tensor
except Exception as e:  # pragma: no cover
    raise ImportError("Torch is required by bem_solver.py (GMRES).") from e

from electrodrive.utils.logging import JsonlLogger
    # telemetry
from electrodrive.utils.telemetry import TelemetryWriter

__all__ = ["gmres_restart", "setup_preconditioner", "SOLVER_TRACE_HEADERS"]

SOLVER_TRACE_HEADERS = [
    "ts",
    "pass_idx",
    "cycle",
    "iter",
    "resid_true_l2",
    "resid_precond_l2",
    "arnoldi_dim",
    "phase",
]


# ---------------------------------------------------------------------------
# Preconditioner helper
# ---------------------------------------------------------------------------


def setup_preconditioner(
    method: Optional[str],
    A_diag: Optional[Tensor] = None,
    areas: Optional[Tensor] = None,
    logger: Optional[JsonlLogger] = None,
) -> Optional[Callable[[Tensor], Tensor]]:
    """
    Returns a left-preconditioner apply function M(x) ≈ P^{-1} x, or None.

    Supported methods
    -----------------
    - None / "none":   no preconditioning.
    - "diag"/"jacobi": use A_diag as diagonal (Jacobi) preconditioner.
    - "mass":          use panel areas as a simple mass-like preconditioner.

    Notes
    -----
    - If "diag"/"jacobi" is requested but A_diag is None, we fall back to
      "mass" if areas are provided, otherwise no preconditioner.
    - In all cases we clamp denominators and mask non-finite entries to avoid
      division by 0 and NaN propagation.
    """
    if method is None or method == "none":
        return None

    if method in ("diag", "jacobi"):
        if A_diag is None:
            if logger:
                logger.warning(
                    "Diagonal preconditioner requested but A_diag is None; "
                    "falling back to 'mass' if areas provided."
                )
            # Fallback: if areas supplied, use mass preconditioner
            if areas is None:
                return None
            method = "mass"
        else:
            # Mask non-finite and too-small entries; they act as identity.
            finite = torch.isfinite(A_diag)
            safe_diag = A_diag.clone()
            bad = (~finite) | (safe_diag.abs() < 1e-18)
            if bad.any() and logger:
                logger.warning(
                    "Diagonal preconditioner: masking %d non-finite/near-zero entries.",
                    int(bad.sum().item()),
                )
            safe_diag[bad] = 1.0
            P_inv = 1.0 / safe_diag

            def apply(x: Tensor) -> Tensor:
                return P_inv * x

            return apply

    if method == "mass":
        if areas is None:
            if logger:
                logger.warning("Mass preconditioner requested but areas is None.")
            return None
        finite = torch.isfinite(areas)
        safe_areas = areas.clone()
        bad = (~finite) | (safe_areas.abs() < 1e-18)
        if bad.any() and logger:
            logger.warning(
                "Mass preconditioner: masking %d non-finite/near-zero entries.",
                int(bad.sum().item()),
            )
        if not torch.any(~bad):
            if logger:
                logger.warning("Mass preconditioner: no usable finite entries.")
            return None
        safe_areas[bad] = 1.0
        P_inv = 1.0 / safe_areas

        def apply(x: Tensor) -> Tensor:
            return P_inv * x

        return apply

    if logger:
        logger.warning("Unknown preconditioner '%s'; using none.", str(method))
    return None


# ---------------------------------------------------------------------------
# Restarted GMRES
# ---------------------------------------------------------------------------


@torch.no_grad()
def gmres_restart(
    A: Callable[[Tensor], Tensor],
    b: Tensor,
    M: Optional[Callable[[Tensor], Tensor]] = None,
    restart: int = 80,
    tol: float = 1e-8,
    maxiter: int = 500,
    logger: Optional[JsonlLogger] = None,
    x0: Optional[Tensor] = None,
    telemetry: Optional[TelemetryWriter] = None,
    pass_index: int = 0,
    live_control_poll_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    log_every: int = 1,
    *,
    # Extra options (backward compatible)
    callback: Optional[Callable[[int, float, Dict[str, Any]], None]] = None,
    callback_context: Optional[Dict[str, Any]] = None,
    precond: Optional[Union[str, Callable[[Tensor], Tensor]]] = None,
    areas: Optional[Tensor] = None,
    A_diag: Optional[Tensor] = None,
    # New options
    validate_operators: bool = False,
    _retry_without_precond: bool = False,
) -> Tuple[Tensor, Dict[str, Any]]:
    """
    Restarted GMRES with optional (left) preconditioner M and CSV telemetry.

    Parameters
    ----------
    A : callable
        Matrix-vector apply function v -> A v (already on correct device).
    b : Tensor
        Right-hand-side vector.
    M : callable or None
        Left-preconditioner apply function z -> M z. If None but `precond` is
        provided, we construct M via `setup_preconditioner`.
    restart : int
        Maximum Krylov subspace dimension before restart.
    tol : float
        Relative tolerance on the TRUE residual.
    maxiter : int
        Maximum number of Arnoldi/Givens iterations (across all cycles).
    logger : JsonlLogger or None
        Logger for JSONL telemetry.
    x0 : Tensor or None
        Initial guess. Defaults to zeros_like(b).
    telemetry : TelemetryWriter or None
        Optional CSV telemetry writer (uses SOLVER_TRACE_HEADERS).
    pass_index : int
        Integer index to tag this call (e.g., BEM refine pass).
    live_control_poll_fn : callable or None
        Function returning a dict with optional keys {"terminate", "pause"}.
    log_every : int
        Log/telemetry interval in iterations.

    callback : callable or None
        If provided, called as callback(iter_idx, resid_norm, ctx) every
        iteration or when log_every divides the iteration index.
    callback_context : dict or None
        Extra context dictionary passed into callback.
    precond : str or callable or None
        If str in {"jacobi","diag","mass"}, we call setup_preconditioner().
        If callable, used directly as M.
    areas : Tensor or None
        Panel areas (for "mass" preconditioner).
    validate_operators : bool
        If True, run cheap sanity checks on A and M (shape/finite) before
        starting iterations. This is useful in tests / health checks.
    _retry_without_precond : bool
        Internal flag used to avoid infinite recursion when we retry once
        without a preconditioner after a preconditioned run fails.

    Returns
    -------
    x : Tensor
        Approximate solution.
    info : dict
        Dictionary with fields:
          - iters:   total Arnoldi iterations
          - resid:   final (true) residual norm
          - success: bool
          - code:    integer code:
              0  = converged
              1  = maxiter reached (no convergence)
              2  = terminate/stagnation/abort
              3  = non-finite initial true residual (input/operator issue)
              4  = internal Arnoldi indexing error
              5  = non-finite inner residual
              6  = non-finite post-update residual
              7  = non-finite initial preconditioned residual (precond issue)
          - resid_true0: initial true residual norm
          - tol_abs: absolute tolerance used
          - used_preconditioner: bool
    """
    if restart <= 0:
        raise ValueError("restart must be positive")
    if maxiter <= 0:
        raise ValueError("maxiter must be positive")

    orig_dtype = b.dtype
    orig_device = b.device
    N = int(b.numel())
    use_fp64_krylov = (
        N <= 64
        and hasattr(torch, "float64")
        and orig_dtype in (torch.float16, torch.bfloat16, torch.float32)
    )
    work_dtype = torch.float64 if use_fp64_krylov else orig_dtype

    # Wrap A so that it always consumes and returns work_dtype on the original device.
    def A_apply(v: Tensor) -> Tensor:
        v_in = v.to(dtype=orig_dtype, device=orig_device)
        out = A(v_in)
        return out.to(dtype=work_dtype, device=orig_device)

    # Build/choose preconditioner
    if M is None and precond is not None:
        if isinstance(precond, str):
            M_local = setup_preconditioner(precond, A_diag=A_diag, areas=areas, logger=logger)
        elif callable(precond):
            M_local = precond
        else:
            M_local = None
    else:
        M_local = M

    # Wrap M similarly (we'll probe it once below).
    def M_apply(v: Tensor) -> Tensor:
        if M_local is None:
            return v
        v_in = v.to(dtype=orig_dtype, device=orig_device)
        out = M_local(v_in)
        return out.to(dtype=work_dtype, device=orig_device)

    # Optional operator validation (A & M).
    if validate_operators:
        # Simple probe with unit-norm random vector
        probe = torch.randn_like(b, dtype=work_dtype, device=orig_device)
        norm_probe = float(torch.linalg.norm(probe))
        if norm_probe > 0:
            probe = probe / norm_probe

        # Validate A
        try:
            Ap = A_apply(probe)
        except Exception as exc:  # pragma: no cover - defensive
            if logger:
                logger.error("GMRES operator A probe failed.", error=str(exc))
            return b.to(dtype=orig_dtype), dict(
                iters=0,
                resid=float("nan"),
                success=False,
                code=4,
                resid_true0=float("nan"),
                tol_abs=float("nan"),
                used_preconditioner=False,
            )
        if Ap.shape != b.shape or not torch.isfinite(Ap).all():
            if logger:
                logger.error(
                    "GMRES operator A produced invalid output on probe.",
                    shape=str(Ap.shape),
                    finite=bool(torch.isfinite(Ap).all()),
                )
            return b.to(dtype=orig_dtype), dict(
                iters=0,
                resid=float("nan"),
                success=False,
                code=4,
                resid_true0=float("nan"),
                tol_abs=float("nan"),
                used_preconditioner=False,
            )

    # One-time preconditioner probe: disable M if it misbehaves.
    if M_local is not None:
        probe = torch.ones_like(b, dtype=work_dtype, device=orig_device)
        try:
            Mp = M_apply(probe)
        except Exception as exc:  # pragma: no cover - defensive
            if logger:
                logger.error("Preconditioner probe failed; disabling.", error=str(exc))
            M_local = None
        else:
            if Mp.shape != b.shape or not torch.isfinite(Mp).all():
                if logger:
                    logger.error(
                        "Preconditioner produced invalid output on probe; disabling.",
                        shape=str(Mp.shape),
                        finite=bool(torch.isfinite(Mp).all()),
                    )
                M_local = None

    # Re-wrap M_apply to reflect possibly-disabled M_local
    def M_apply(v: Tensor) -> Tensor:
        if M_local is None:
            return v
        v_in = v.to(dtype=orig_dtype, device=orig_device)
        out = M_local(v_in)
        return out.to(dtype=work_dtype, device=orig_device)

    device = orig_device
    x = (
        torch.zeros_like(b, dtype=work_dtype)
        if x0 is None
        else x0.to(device=device, dtype=work_dtype).clone()
    )
    b_work = b.to(device=device, dtype=work_dtype)

    # Initial residuals
    r_true = b_work - A_apply(x)
    r_true_norm = float(torch.linalg.norm(r_true))

    if not math.isfinite(r_true_norm):
        if logger:
            logger.error(
                "GMRES initial true residual is non-finite.",
                resid_true=r_true_norm,
            )
        return x.to(dtype=orig_dtype), dict(
            iters=0,
            resid=r_true_norm,
            success=False,
            code=3,
            resid_true0=r_true_norm,
            tol_abs=float("nan"),
            used_preconditioner=bool(M_local is not None),
        )

    r = M_apply(r_true)
    beta = torch.linalg.norm(r)
    beta_val = float(beta)

    if not math.isfinite(beta_val):
        if logger:
            logger.error(
                "GMRES initial preconditioned residual is non-finite.",
                resid_precond=beta_val,
                resid_true=r_true_norm,
            )
        # If we had a preconditioner and haven't retried yet, try once more without it.
        if M_local is not None and not _retry_without_precond:
            if logger:
                logger.warning(
                    "Retrying GMRES without preconditioner due to "
                    "non-finite initial preconditioned residual."
                )
            return gmres_restart(
                A=A,
                b=b,
                M=None,
                restart=restart,
                tol=tol,
                maxiter=maxiter,
                logger=logger,
                x0=x0,
                telemetry=telemetry,
                pass_index=pass_index,
                live_control_poll_fn=live_control_poll_fn,
                log_every=log_every,
                callback=callback,
                callback_context=callback_context,
                precond=None,
                areas=None,
                validate_operators=False,
                _retry_without_precond=True,
            )
        return x.to(dtype=orig_dtype), dict(
            iters=0,
            resid=r_true_norm,
            success=False,
            code=7,
            resid_true0=r_true_norm,
            tol_abs=float("nan"),
            used_preconditioner=bool(M_local is not None),
        )

    r_norm = beta_val
    init_true = r_true_norm if r_true_norm > 1e-16 else 1.0
    # Absolute tolerance. The extra clamp at 1e-12 keeps us realistic for
    # float32 while still honoring strict tolerances for float64 tests.
    tol_abs = max(tol * init_true, tol * 1e-12, 1e-12)

    if logger:
        logger.info(
            "GMRES started.",
            N=int(b_work.numel()),
            tol=float(tol),
            tol_abs=float(tol_abs),
            restart=int(restart),
            maxiter=int(maxiter),
            preconditioned=bool(M_local is not None),
            pass_index=int(pass_index),
        )

    if telemetry:
        telemetry.append_row(
            dict(
                ts=time.time(),
                pass_idx=int(pass_index),
                cycle=0,
                iter=0,
                resid_true_l2=r_true_norm,
                resid_precond_l2=r_norm,
                arnoldi_dim=0,
                phase="start",
            )
        )

    # Already solved?
    if beta_val == 0.0:
        # If the preconditioned residual is exactly zero, treat this as
        # converged only if the true residual is also small.
        if r_true_norm <= tol_abs:
            return x.to(dtype=orig_dtype), dict(
                iters=0,
                resid=0.0,
                success=True,
                code=0,
                resid_true0=r_true_norm,
                tol_abs=float(tol_abs),
                used_preconditioner=bool(M_local is not None),
                converged_on="true",
            )
        else:
            if logger:
                logger.warning(
                    "GMRES beta=0.0 but true residual above tolerance.",
                    resid_true=r_true_norm,
                    tol_abs=float(tol_abs),
                )
            return x.to(dtype=orig_dtype), dict(
                iters=0,
                resid=r_true_norm,
                success=False,
                code=6,
                resid_true0=r_true_norm,
                tol_abs=float(tol_abs),
                used_preconditioner=bool(M_local is not None),
            )

    total_iters = 0
    max_cycles = max(1, (maxiter + restart - 1) // restart)
    last_poll = time.time()

    # Threshold for automatic fallback: only small systems will be retried with M=None
    fallback_N_threshold = 512

    for cycle in range(max_cycles):
        # Arnoldi basis and upper-Hessenberg
        V: List[Tensor] = [r / beta]  # v_0
        H = torch.zeros((restart + 1, restart), device=device, dtype=work_dtype)
        g = torch.zeros(restart + 1, device=device, dtype=work_dtype)
        g[0] = beta
        givens: List[Tuple[float, float]] = []
        j = 0

        while j < restart and total_iters < maxiter:
            # Live control poll (terminate/pause)
            if live_control_poll_fn is not None and (time.time() - last_poll) >= 5.0:
                try:
                    state = live_control_poll_fn() or {}
                except Exception:
                    state = {}
                last_poll = time.time()
                if state.get("terminate"):
                    if logger:
                        logger.warning(
                            "GMRES termination requested by live control.",
                            cycle=int(cycle),
                            iters=int(total_iters),
                        )
                    return x.to(dtype=orig_dtype), dict(
                        iters=int(total_iters),
                        resid=float("nan"),
                        success=False,
                        code=2,
                        resid_true0=r_true_norm,
                        tol_abs=float(tol_abs),
                        used_preconditioner=bool(M_local is not None),
                    )
                if state.get("pause"):
                    time.sleep(0.2)

            # Safety: we must always have len(V) >= j+1 here
            if j >= len(V):
                if logger:
                    logger.error(
                        "GMRES internal error: Arnoldi index j out of range.",
                        j=int(j),
                        len_V=int(len(V)),
                        cycle=int(cycle),
                        iters=int(total_iters),
                    )
                return x.to(dtype=orig_dtype), dict(
                    iters=int(total_iters),
                    resid=float("nan"),
                    success=False,
                    code=4,
                    resid_true0=r_true_norm,
                    tol_abs=float(tol_abs),
                    used_preconditioner=bool(M_local is not None),
                )

            # Arnoldi step
            vj = V[j]
            w = A_apply(vj)
            w = M_apply(w)
            total_iters += 1

            # Arnoldi orthogonalization
            for i in range(j + 1):
                H[i, j] = torch.dot(V[i], w)
                w = w - H[i, j] * V[i]
            H[j + 1, j] = torch.linalg.norm(w)
            h_next = float(H[j + 1, j])

            # Handle breakdown / non-finite column
            if (not math.isfinite(h_next)) or (h_next <= 0.0):
                if logger:
                    logger.error(
                        "GMRES Arnoldi breakdown or non-finite column.",
                        cycle=int(cycle),
                        iters=int(total_iters),
                        j=int(j),
                        h_next=h_next,
                        preconditioned=bool(M_local is not None),
                    )
                # If we have a preconditioner, try once without it for small systems.
                if (
                    M_local is not None
                    and not _retry_without_precond
                    and N <= fallback_N_threshold
                ):
                    if logger:
                        logger.warning(
                            "Retrying GMRES without preconditioner after Arnoldi breakdown."
                        )
                    return gmres_restart(
                        A=A,
                        b=b,
                        M=None,
                        restart=restart,
                        tol=tol,
                        maxiter=maxiter,
                        logger=logger,
                        x0=x0,
                        telemetry=telemetry,
                        pass_index=pass_index,
                        live_control_poll_fn=live_control_poll_fn,
                        log_every=log_every,
                        callback=callback,
                        callback_context=callback_context,
                        precond=None,
                        areas=None,
                        validate_operators=False,
                        _retry_without_precond=True,
                    )
                # Stop inner iterations for this cycle. We'll use m_use=j below.
                break

            # Extend basis
            V.append(w / H[j + 1, j])

            # Apply existing Givens rotations
            for i_g, (cs_i, sn_i) in enumerate(givens):
                temp = cs_i * H[i_g, j] + sn_i * H[i_g + 1, j]
                H[i_g + 1, j] = -sn_i * H[i_g, j] + cs_i * H[i_g + 1, j]
                H[i_g, j] = temp

            # New Givens for this column
            denom = torch.sqrt(H[j, j] ** 2 + H[j + 1, j] ** 2)
            denom_val = float(denom)

            if denom_val < 1e-14 or (not math.isfinite(denom_val)):
                cs_j, sn_j = 1.0, 0.0
            else:
                cs_j = float(H[j, j] / denom)
                sn_j = float(H[j + 1, j] / denom)

            givens.append((cs_j, sn_j))

            # Apply to H and g
            H[j, j] = cs_j * H[j, j] + sn_j * H[j + 1, j]
            H[j + 1, j] = 0.0
            g_j1 = -sn_j * g[j]
            g_j = cs_j * g[j]
            g[j] = g_j
            g[j + 1] = g_j1

            r_norm = abs(float(g[j + 1]))

            # Detect non-finite residuals early
            if not math.isfinite(r_norm):
                if logger:
                    logger.error(
                        "GMRES produced non-finite residual.",
                        cycle=int(cycle),
                        iters=int(total_iters),
                        arnoldi_dim=int(j + 1),
                        resid=r_norm,
                        preconditioned=bool(M_local is not None),
                    )
                # Retry once without preconditioner for small systems.
                if (
                    M_local is not None
                    and not _retry_without_precond
                    and N <= fallback_N_threshold
                ):
                    if logger:
                        logger.warning(
                            "Retrying GMRES without preconditioner after non-finite residual."
                        )
                    return gmres_restart(
                        A=A,
                        b=b,
                        M=None,
                        restart=restart,
                        tol=tol,
                        maxiter=maxiter,
                        logger=logger,
                        x0=x0,
                        telemetry=telemetry,
                        pass_index=pass_index,
                        live_control_poll_fn=live_control_poll_fn,
                        log_every=log_every,
                        callback=callback,
                        callback_context=callback_context,
                        precond=None,
                        areas=None,
                        validate_operators=False,
                        _retry_without_precond=True,
                    )
                return x.to(dtype=orig_dtype), dict(
                    iters=int(total_iters),
                    resid=r_norm,
                    success=False,
                    code=5,
                    resid_true0=r_true_norm,
                    tol_abs=float(tol_abs),
                    used_preconditioner=bool(M_local is not None),
                )

            # Progress logging / telemetry / callback
            if logger and (total_iters % max(1, log_every) == 0):
                logger.debug(
                    "GMRES progress.",
                    cycle=int(cycle),
                    iters=int(total_iters),
                    resid_precond=float(r_norm),
                    arnoldi_dim=int(j + 1),
                )
            if telemetry and (total_iters % max(1, log_every) == 0):
                telemetry.append_row(
                    dict(
                        ts=time.time(),
                        pass_idx=int(pass_index),
                        cycle=int(cycle),
                        iter=int(total_iters),
                        resid_true_l2=float("nan"),
                        resid_precond_l2=r_norm,
                        arnoldi_dim=int(j + 1),
                        phase="inner",
                    )
                )
            if callback is not None:
                try:
                    ctx = dict(callback_context or {})
                    ctx.setdefault("pass_index", int(pass_index))
                    ctx.setdefault("arnoldi_dim", int(j + 1))
                    callback(int(total_iters), float(r_norm), ctx)
                except Exception:
                    # Never let a callback failure kill the solver
                    pass

            # We no longer treat small preconditioned residual as convergence;
            # that is tracked and logged but not used as a stopping criterion here.
            if r_norm <= tol_abs:
                # *Do not* break on preconditioned residual alone; let the cycle
                # continue so that the true residual test below can decide.
                j += 1
                continue

            j += 1  # next Arnoldi step

        # Dimension actually used in this cycle
        m_use = j
        if m_use == 0:
            # No Arnoldi steps completed -> stagnation / breakdown
            if logger:
                logger.warning(
                    "GMRES stagnation: no Arnoldi steps.",
                    cycle=int(cycle),
                    iters=int(total_iters),
                    preconditioned=bool(M_local is not None),
                )
            # Retry once without preconditioner for small systems.
            if (
                M_local is not None
                and not _retry_without_precond
                and N <= fallback_N_threshold
            ):
                if logger:
                    logger.warning(
                        "Retrying GMRES without preconditioner after stagnation."
                    )
                return gmres_restart(
                    A=A,
                    b=b,
                    M=None,
                    restart=restart,
                    tol=tol,
                    maxiter=maxiter,
                    logger=logger,
                    x0=x0,
                    telemetry=telemetry,
                    pass_index=pass_index,
                    live_control_poll_fn=live_control_poll_fn,
                    log_every=log_every,
                    callback=callback,
                    callback_context=callback_context,
                    precond=None,
                    areas=None,
                    validate_operators=False,
                    _retry_without_precond=True,
                )
            return x.to(dtype=orig_dtype), dict(
                iters=int(total_iters),
                resid=float(r_norm),
                success=False,
                code=2,
                resid_true0=r_true_norm,
                tol_abs=float(tol_abs),
                used_preconditioner=bool(M_local is not None),
            )

        # Back-solve R y = g_top for y (upper triangular R = H[:m_use, :m_use])
        R = H[:m_use, :m_use]
        g_top = g[:m_use]
        # Triangular solve; fallback to lstsq for robustness
        try:
            y = torch.linalg.solve_triangular(R, g_top, upper=True)
        except Exception:
            y = torch.linalg.lstsq(R, g_top).solution

        V_m = torch.stack(V[:m_use], dim=1)  # (n, m_use)
        dx = V_m @ y
        x = x + dx

        # Recompute true + preconditioned residuals after cycle update
        r_true = b_work - A_apply(x)
        r_true_norm = float(torch.linalg.norm(r_true))
        if not math.isfinite(r_true_norm):
            if logger:
                logger.error(
                    "GMRES post-update true residual is non-finite.",
                    resid_true=r_true_norm,
                    iters=int(total_iters),
                    preconditioned=bool(M_local is not None),
                )
            # Retry once without preconditioner for small systems.
            if (
                M_local is not None
                and not _retry_without_precond
                and N <= fallback_N_threshold
            ):
                if logger:
                    logger.warning(
                        "Retrying GMRES without preconditioner after "
                        "non-finite post-update residual."
                    )
                return gmres_restart(
                    A=A,
                    b=b,
                    M=None,
                    restart=restart,
                    tol=tol,
                    maxiter=maxiter,
                    logger=logger,
                    x0=x0,
                    telemetry=telemetry,
                    pass_index=pass_index,
                    live_control_poll_fn=live_control_poll_fn,
                    log_every=log_every,
                    callback=callback,
                    callback_context=callback_context,
                    precond=None,
                    areas=None,
                    validate_operators=False,
                    _retry_without_precond=True,
                )
            return x.to(dtype=orig_dtype), dict(
                iters=int(total_iters),
                resid=r_true_norm,
                success=False,
                code=6,
                resid_true0=r_true_norm,
                tol_abs=float(tol_abs),
                used_preconditioner=bool(M_local is not None),
            )

        r = M_apply(r_true)
        beta = torch.linalg.norm(r)
        r_norm = float(beta)

        if logger:
            logger.info(
                "GMRES cycle done.",
                cycle=int(cycle),
                iters=int(total_iters),
                resid_precond=r_norm,
                resid_true=r_true_norm,
                arnoldi_dim=int(m_use),
            )
        if telemetry:
            telemetry.append_row(
                dict(
                    ts=time.time(),
                    pass_idx=int(pass_index),
                    cycle=int(cycle),
                    iter=int(total_iters),
                    resid_true_l2=r_true_norm,
                    resid_precond_l2=r_norm,
                    arnoldi_dim=int(m_use),
                    phase="cycle_end",
                )
            )

        # Convergence test: require TRUE residual to be small.
        if r_true_norm <= tol_abs:
            return x.to(dtype=orig_dtype), dict(
                iters=int(total_iters),
                resid=float(r_true_norm),
                success=True,
                code=0,
                resid_true0=r_true_norm,
                tol_abs=float(tol_abs),
                used_preconditioner=bool(M_local is not None),
                converged_on="true",
            )

        if total_iters >= maxiter:
            break

    # Did not converge within maxiter
    if logger:
        logger.warning(
            "GMRES failed to converge within max iterations.",
            iters=int(total_iters),
            resid=float(r_norm),
            resid_true=float(r_true_norm),
            maxiter=int(maxiter),
            preconditioned=bool(M_local is not None),
        )

    # For small systems where preconditioner might hurt, retry once without it.
    if (
        M_local is not None
        and not _retry_without_precond
        and N <= fallback_N_threshold
    ):
        if logger:
            logger.warning(
                "Retrying GMRES without preconditioner after non-convergence.",
            )
        return gmres_restart(
            A=A,
            b=b,
            M=None,
            restart=restart,
            tol=tol,
            maxiter=maxiter,
            logger=logger,
            x0=x0,
            telemetry=telemetry,
            pass_index=pass_index,
            live_control_poll_fn=live_control_poll_fn,
            log_every=log_every,
            callback=callback,
            callback_context=callback_context,
            precond=None,
            areas=None,
            validate_operators=False,
            _retry_without_precond=True,
        )

    return x.to(dtype=orig_dtype), dict(
        iters=int(total_iters),
        resid=float(r_norm),
        success=False,
        code=1,
        resid_true0=r_true_norm,
        tol_abs=float(tol_abs),
        used_preconditioner=bool(M_local is not None),
    )
