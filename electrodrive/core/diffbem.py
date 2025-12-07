# -*- coding: utf-8 -*-
"""
Differentiable BEM solver using xitorch (opt-in) with verbose logging.

This module wraps a differentiable matrix-free matvec in a xitorch LinearOperator
and solves A x = b with autograd support.

It logs:
- tensor shapes/dtypes/devices and requires_grad flags
- chosen solver method and options
- operator parameter names exposed to xitorch
- residuals and norm stats
- fallback decisions (e.g., when a method isn't available)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional
import os

import torch

# Optional xitorch import
try:  # pragma: no cover
    from xitorch import LinearOperator as _XiLinearOperator  # type: ignore
    from xitorch.linalg import solve as x_solve  # type: ignore

    _HAVE_XITORCH = True
except Exception:  # pragma: no cover
    _XiLinearOperator = object  # type: ignore
    x_solve = None  # type: ignore
    _HAVE_XITORCH = False


def _kvfmt(**kw) -> Dict[str, str]:
    """
    Safe tensor/string formatter for logging.
    """
    out: Dict[str, str] = {}
    for k, v in kw.items():
        if isinstance(v, torch.Tensor):
            out[k] = (
                f"Tensor(shape={tuple(v.shape)}, "
                f"dtype={v.dtype}, device={v.device}, "
                f"grad={v.requires_grad})"
            )
        else:
            out[k] = str(v)
    return out


class _BEMLinearOperator(_XiLinearOperator):  # type: ignore[misc]
    """
    xitorch LinearOperator wrapper around a differentiable matrix-free matvec.
    """

    def __init__(
        self,
        *,
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        matvec: Callable[[torch.Tensor], torch.Tensor],
        params: Optional[Dict[str, torch.Tensor]] = None,
        logger: Optional[Any] = None,
    ) -> None:
        assert isinstance(size, int) and size > 0
        self._size = int(size)
        self._dtype = dtype
        self._device = device
        self._matvec = matvec
        self._logger = logger
        self._paramnames: list[str] = []

        # Register tensor params so xitorch can track them for autograd.
        if params:
            for name, tensor in params.items():
                if isinstance(tensor, torch.Tensor):
                    setattr(self, name, tensor)
                    self._paramnames.append(name)

        super().__init__(
            shape=(self._size, self._size),
            dtype=self._dtype,
            device=self._device,
        )

        if self._logger is not None:
            try:
                self._logger.info(
                    "XiLinearOperator constructed.",
                    size=self._size,
                    dtype=str(self._dtype),
                    device=str(self._device),
                    params=(
                        ",".join(self._paramnames)
                        if self._paramnames
                        else "(none)"
                    ),
                )
            except Exception:
                pass

    def _getparamnames(self, prefix: str = "") -> list[str]:  # type: ignore[override]
        """
        Expose tensor parameter names to xitorch for autograd.
        """
        return [prefix + n for n in self._paramnames]

    def _mv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Matrix-vector or matrix-matrix product implementation.

        Supports:
        - x: [n] or [n, k] (columns as vectors)
        - x with batch dimensions, where one of the last two dims is n.
        """
        n = self._size

        def apply_cols(X: torch.Tensor) -> torch.Tensor:
            if X.shape[0] != n:
                raise RuntimeError(
                    f"_BEMLinearOperator: expected leading dim {n}, "
                    f"got {tuple(X.shape)}"
                )
            cols = [self._matvec(X[:, j]) for j in range(X.shape[1])]
            return torch.stack(cols, dim=1)

        # Vector case: [n]
        if x.dim() == 1:
            if x.shape[0] != n:
                raise RuntimeError(
                    f"_BEMLinearOperator: expected (n,) with n={n}, "
                    f"got {tuple(x.shape)}"
                )
            return self._matvec(x)

        # Matrix case: [n, k] or [k, n]
        if x.dim() == 2:
            r, c = x.shape
            if r == n:
                return apply_cols(x)
            if c == n:
                y = apply_cols(x.t().contiguous())
                return y.t().contiguous()
            raise RuntimeError(
                f"_BEMLinearOperator: expected (n,k) or (k,n) with n={n}, "
                f"got {tuple(x.shape)}"
            )

        # Higher-rank: handle when last dim is n
        if x.shape[-1] == n:
            X = x.reshape(-1, n)
            Y = torch.stack(
                [self._matvec(v) for v in X],
                dim=0,
            )
            return Y.reshape(*x.shape[:-1], n)

        # Or when second-to-last dim is n
        if x.shape[-2] == n:
            # Flatten batches; keep (n, k)
            if x.shape[:-2]:
                B = int(torch.tensor(x.shape[:-2]).prod().item())
            else:
                B = 1
            k = x.shape[-1]
            X = x.reshape(B, n, k)
            Y = torch.stack(
                [apply_cols(X[b]) for b in range(B)],
                dim=0,
            )
            return Y.reshape(*x.shape)

        raise RuntimeError(
            f"_BEMLinearOperator: unsupported input shape "
            f"{tuple(x.shape)} for n={n}"
        )


def _as_2d_column(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure x is [n,1] for use as a single RHS column.
    """
    if x.dim() == 1:
        return x.unsqueeze(-1)
    if x.dim() == 2:
        if x.shape[1] == 1:
            return x
        if x.shape[0] == 1:
            return x.t().contiguous()
    raise RuntimeError(
        f"Expected vector or single-column matrix, got shape {tuple(x.shape)}"
    )


def _effective_method(cfg: Any) -> str:
    """
    Choose xitorch solver method based on cfg, defaulting conservatively.

    Many xitorch builds support 'bicgstab' and 'cg'; 'gmres' may not be present.
    """
    preferred = getattr(cfg, "xitorch_method", None)
    if isinstance(preferred, str) and preferred.strip():
        return preferred.strip().lower()
    return "bicgstab"


def solve_diffbem(
    spec: Optional[Any],
    cfg: Any,
    logger: Optional[Any],
    *,
    C: torch.Tensor,
    N: torch.Tensor,
    A: torch.Tensor,
    rhs: torch.Tensor,
    matvec: Callable[[torch.Tensor], torch.Tensor],
    x0: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Autograd-capable linear solve for the BEM system using xitorch.

    Parameters:
        spec: Optional problem description (not used in core solve).
        cfg:  BEMConfig-like object with gmres_maxiter/gmres_tol/xitorch_method.
        logger: Optional JsonlLogger-like for structured logging.
        C:    [N,3] centroids tensor (can carry gradients if desired).
        N:    [N,3] normals tensor (registered as parameter; typically const).
        A:    [N]   areas tensor (registered as parameter; typically const).
        rhs:  [N]   right-hand side vector; gradients can flow w.r.t. rhs.
        matvec: callable implementing y = A(sigma) in a differentiable way.
        x0:   optional initial guess [N] or [N,1].

    Returns:
        {
            "sigma": Tensor [N],
            "stats": {
                "solver": str,
                "resid": float,
                "iters": None,
                "success": True,
            },
        }

    Notes:
        - This function assumes that `matvec` and provided tensors are set up
          so that xitorch's autograd can track any desired parameters.
        - It does NOT itself try to differentiate w.r.t. the mesh generation
          or CanonicalSpec; callers must pass tensors directly.
    """
    if not _HAVE_XITORCH:
        raise RuntimeError(
            "differentiable=True requested but xitorch is not installed. "
            "Install xitorch or run with differentiable=False."
        )

    # Extra verbosity controlled by env var.
    trace_env = os.getenv("EDE_TRACE", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    device = C.device
    dtype = C.dtype
    n = C.shape[0]

    if logger is not None:
        try:
            logger.info(
                "DiffBEM entry.",
                **_kvfmt(C=C, N=N, A=A, rhs=rhs),
                n=n,
                trace_env=trace_env,
            )
        except Exception:
            pass

    # Wrap matvec + parameters in a LinearOperator so xitorch can differentiate.
    op = _BEMLinearOperator(
        size=n,
        dtype=dtype,
        device=device,
        matvec=matvec,
        params={"C": C, "N": N, "A": A},
        logger=logger,
    )

    B = _as_2d_column(rhs.to(device=device, dtype=dtype))
    if x0 is None:
        x0_opt: Optional[torch.Tensor] = None
    else:
        x0_opt = _as_2d_column(
            x0.to(device=device, dtype=dtype)
        )

    # Solver options
    method = _effective_method(cfg)
    solve_opts: Dict[str, Any] = {
        "method": method,
        "max_niter": int(
            getattr(cfg, "gmres_maxiter", 200)
        ),
        "tol": float(
            getattr(cfg, "gmres_tol", 1e-6)
        ),
        "verbose": bool(trace_env),
    }
    if x0_opt is not None:
        # xitorch expects x0 with same shape semantics as RHS
        solve_opts["x0"] = x0_opt  # type: ignore[assignment]

    if logger is not None:
        try:
            logger.info(
                "DiffBEM about to solve.",
                method=str(method),
                max_niter=int(solve_opts["max_niter"]),
                tol=float(solve_opts["tol"]),
                x0_present=bool(x0_opt is not None),
            )
        except Exception:
            pass

    # Try requested method, then fall back to bicgstab -> cg.
    tried: list[str] = []
    last_exc: Optional[Exception] = None

    def _attempt(m: str) -> Optional[torch.Tensor]:
        nonlocal last_exc
        tried.append(m)
        try:
            if logger is not None:
                logger.info(
                    "xitorch.solve attempt.",
                    method=m,
                )
            return x_solve(  # type: ignore[arg-type]
                op,
                B,
                method=m,
                max_niter=solve_opts["max_niter"],
                tol=solve_opts["tol"],
                verbose=solve_opts["verbose"],
            )
        except Exception as e:  # pragma: no cover - environment/solver dependent
            last_exc = e
            if logger is not None:
                logger.warning(
                    "xitorch.solve failed; will consider fallback.",
                    method=m,
                    error=str(e),
                )
            return None

    sigma_2d: Optional[torch.Tensor] = _attempt(method)
    if sigma_2d is None and method.lower() != "bicgstab":
        sigma_2d = _attempt("bicgstab")
    if sigma_2d is None and method.lower() != "cg":
        sigma_2d = _attempt("cg")

    if sigma_2d is None:
        if logger is not None:
            try:
                logger.error(
                    "All xitorch methods failed.",
                    tried=",".join(tried),
                    error=str(last_exc)
                    if last_exc
                    else "unknown",
                    exc_info=True,
                )
            except Exception:
                pass
        raise last_exc if last_exc else RuntimeError(
            "xitorch.solve failed for all attempted methods"
        )

    sigma = sigma_2d.squeeze(-1)

    # Compute best-effort residual in no-grad mode for logging.
    with torch.no_grad():
        try:
            r = matvec(sigma) - rhs.to(
                device=device,
                dtype=dtype,
            )
            resid = float(
                (r.norm() / (rhs.norm() + 1e-12)).item()
            )
            rnorm = float(r.norm().item())
            bnorm = float(rhs.norm().item())
            xnorm = float(sigma.norm().item())
        except Exception:
            resid = float("nan")
            rnorm = float("nan")
            bnorm = float("nan")
            xnorm = float("nan")

    if logger is not None:
        try:
            logger.info(
                "DiffBEM solve complete.",
                resid=resid,
                rnorm=rnorm,
                bnorm=bnorm,
                xnorm=xnorm,
                method_used=tried[-1] if tried else method,
            )
        except Exception:
            pass

    stats: Dict[str, Any] = {
        "solver": f"xitorch_{tried[-1] if tried else method}",
        "resid": resid,
        "iters": None,
        "success": True,
    }
    return {"sigma": sigma, "stats": stats}