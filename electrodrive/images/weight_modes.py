from __future__ import annotations

"""
Weight-mode utilities for axis sweeps:
- assemble weight matrices across z-grids,
- compute SVDs,
- fit lightweight symbolic laws (poly / rational),
- export bundles for downstream controllers and vaulting.

The fitting paths avoid heavy dependencies; if PySR / AI Feynman / SINDy
are available they can be added later, but this module is intentionally
numpy-only to keep smoke runs portable.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class WeightModeBundle:
    """Container for weight matrices and their SVD."""

    weights: np.ndarray  # shape (K, M) padded with zeros
    z_grid: np.ndarray   # shape (M,)
    U: np.ndarray        # shape (K, K') or (K, M)
    S: np.ndarray        # shape (min(K, M),)
    VT: np.ndarray       # shape (min(K, M), M)
    mode_curves: np.ndarray  # shape (r, M) where r = len(S)
    sigma_norm: np.ndarray
    effective_rank: Dict[str, int]
    recon_error: Dict[str, float]


@dataclass
class SymbolicFit:
    """Lightweight record of a symbolic fit for one mode."""

    mode: int
    method: str  # "poly" or "rational"
    expression: str
    coefficients: Dict[str, Any]
    rmse: float
    mae: float
    max_abs: float
    rel_rmse: float
    backend: str = "numpy"


def _to_numpy(weights: Sequence[torch.Tensor | np.ndarray | Sequence[float]]) -> List[np.ndarray]:
    arrs: List[np.ndarray] = []
    for w in weights:
        if isinstance(w, torch.Tensor):
            arr = w.detach().cpu().numpy()
        else:
            arr = np.asarray(w, dtype=float)
        arrs.append(arr.reshape(-1))
    return arrs


def assemble_weight_matrix(
    weights: Sequence[torch.Tensor | np.ndarray | Sequence[float]],
    z_grid: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad variable-length weight vectors into a dense matrix W[K, M]."""
    arrs = _to_numpy(weights)
    K = max((a.shape[0] for a in arrs), default=0)
    M = len(z_grid)
    W = np.zeros((K, M), dtype=float)
    for j, arr in enumerate(arrs):
        if j >= M:
            break
        k = min(K, arr.shape[0])
        W[:k, j] = arr[:k]
    return W, np.asarray(z_grid, dtype=float).reshape(-1)


def compute_weight_modes(
    weights: Sequence[torch.Tensor | np.ndarray | Sequence[float]],
    z_grid: Sequence[float],
    max_rank: int = 3,
) -> WeightModeBundle:
    """Build weight matrix and its truncated SVD."""
    W, z_arr = assemble_weight_matrix(weights, z_grid)
    if W.size == 0 or z_arr.size == 0:
        return WeightModeBundle(
            weights=W,
            z_grid=z_arr,
            U=np.zeros((0, 0)),
            S=np.zeros((0,)),
            VT=np.zeros((0, 0)),
            mode_curves=np.zeros((0, 0)),
            sigma_norm=np.zeros((0,)),
            effective_rank={},
            recon_error={},
        )

    U, S, VT = np.linalg.svd(W, full_matrices=False)
    r = min(max_rank, S.shape[0])
    mode_curves = (S[:r, None] * VT[:r, :])
    sigma_norm = S / (S[0] if S.size > 0 else 1.0)
    eff_rank = {
        "eps_1e-1": int((S > 0.1 * S[0]).sum()) if S.size > 0 else 0,
        "eps_1e-2": int((S > 0.01 * S[0]).sum()) if S.size > 0 else 0,
    }

    recon_error: Dict[str, float] = {}
    norm_W = float(np.linalg.norm(W)) + 1e-9
    for rr in range(1, r + 1):
        W_rr = U[:, :rr] @ np.diag(S[:rr]) @ VT[:rr, :]
        recon_error[f"rank{rr}_rel_fro"] = float(np.linalg.norm(W - W_rr) / norm_W)

    return WeightModeBundle(
        weights=W,
        z_grid=z_arr,
        U=U,
        S=S,
        VT=VT,
        mode_curves=mode_curves,
        sigma_norm=sigma_norm,
        effective_rank=eff_rank,
        recon_error=recon_error,
    )


def _poly_fit(z: np.ndarray, y: np.ndarray, deg: int) -> Optional[SymbolicFit]:
    try:
        coeffs = np.polyfit(z, y, deg)
    except Exception:
        return None
    pred = np.polyval(coeffs, z)
    err = y - pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mmax = float(np.max(np.abs(err)))
    scale = float(np.max(np.abs(y)) + 1e-9)
    rel_rmse = rmse / scale

    terms = []
    for i, c in enumerate(coeffs):
        power = deg - i
        term = f"{c:.6g}"
        if power == 1:
            term += " * z"
        elif power > 1:
            term += f" * z^{power}"
        terms.append(term)
    expr = " + ".join(terms) if terms else "0"

    return SymbolicFit(
        mode=-1,
        method="poly",
        expression=expr,
        coefficients={"poly_coeffs": coeffs.tolist()},
        rmse=rmse,
        mae=mae,
        max_abs=mmax,
        rel_rmse=rel_rmse,
    )


def _rational_fit(z: np.ndarray, y: np.ndarray, deg_num: int, deg_den: int) -> Optional[SymbolicFit]:
    if deg_num < 0 or deg_den < 0:
        return None
    z = z.reshape(-1)
    y = y.reshape(-1)
    M = z.shape[0]
    cols: List[np.ndarray] = []
    # numerator terms a0 + a1 z + ...
    for p in range(deg_num + 1):
        cols.append(z ** p)
    # denominator terms (excluding leading 1) multiply y
    for q in range(1, deg_den + 1):
        cols.append(-y * (z ** q))
    if not cols:
        return None
    A = np.stack(cols, axis=1)
    b = y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    a = sol[: deg_num + 1]
    b_coeffs = sol[deg_num + 1 :]

    num = sum(a[p] * (z ** p) for p in range(deg_num + 1))
    denom = 1.0 + sum(b_coeffs[q - 1] * (z ** q) for q in range(1, deg_den + 1))
    if np.any(np.abs(denom) < 1e-8):
        return None
    pred = num / denom
    err = y - pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mmax = float(np.max(np.abs(err)))
    scale = float(np.max(np.abs(y)) + 1e-9)
    rel_rmse = rmse / scale

    num_terms = " + ".join(f"{a[p]:.6g} * z^{p}" if p > 0 else f"{a[p]:.6g}" for p in range(deg_num + 1))
    den_terms = " + ".join(f"{b_coeffs[q - 1]:.6g} * z^{q}" for q in range(1, deg_den + 1))
    expr = f"({num_terms}) / (1 + {den_terms})" if den_terms else num_terms

    return SymbolicFit(
        mode=-1,
        method="rational",
        expression=expr,
        coefficients={"numerator": a.tolist(), "denominator": [1.0] + b_coeffs.tolist()},
        rmse=rmse,
        mae=mae,
        max_abs=mmax,
        rel_rmse=rel_rmse,
    )


def fit_symbolic_modes(
    z_grid: Sequence[float],
    mode_curves: np.ndarray,
    max_rank: int = 3,
    max_poly_degree: int = 4,
    rational_candidates: Sequence[Tuple[int, int]] = ((1, 1), (2, 1)),
) -> List[SymbolicFit]:
    """Fit simple laws to mode curves; chooses the best RMSE per mode."""
    z_arr = np.asarray(z_grid, dtype=float)
    fits: List[SymbolicFit] = []
    if mode_curves.size == 0:
        return fits
    r = min(max_rank, mode_curves.shape[0])
    for i in range(r):
        y = mode_curves[i]
        candidates: List[SymbolicFit] = []
        for deg in range(1, max_poly_degree + 1):
            poly_fit = _poly_fit(z_arr, y, deg)
            if poly_fit is not None:
                poly_fit.mode = i
                candidates.append(poly_fit)
        for deg_num, deg_den in rational_candidates:
            rat = _rational_fit(z_arr, y, deg_num, deg_den)
            if rat is not None:
                rat.mode = i
                candidates.append(rat)
        if not candidates:
            continue
        best = min(candidates, key=lambda f: f.rmse)
        fits.append(best)
    return fits


def export_weight_mode_bundle(
    out_dir: Path,
    bundle: WeightModeBundle,
    fits: Sequence[SymbolicFit],
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist bundle + fits to disk with the required filenames."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "weights_vs_axis.npy"
    svd_path = out_dir / "svd_modes.npy"
    fits_path = out_dir / "symbolic_fits.json"
    metrics_path = out_dir / "metrics.json"

    np.save(weights_path, bundle.weights)
    np.save(
        svd_path,
        {
            "U": bundle.U,
            "S": bundle.S,
            "VT": bundle.VT,
            "z_grid": bundle.z_grid,
            "mode_curves": bundle.mode_curves,
            "sigma_norm": bundle.sigma_norm,
            "effective_rank": bundle.effective_rank,
        },
        allow_pickle=True,
    )

    fits_serialized = [
        {
            "mode": f.mode,
            "method": f.method,
            "expression": f.expression,
            "coefficients": f.coefficients,
            "rmse": f.rmse,
            "mae": f.mae,
            "max_abs": f.max_abs,
            "rel_rmse": f.rel_rmse,
            "backend": f.backend,
        }
        for f in fits
    ]
    with fits_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "z_grid": bundle.z_grid.tolist(),
                "fits": fits_serialized,
                "sigma_norm": bundle.sigma_norm.tolist(),
            },
            fp,
            indent=2,
        )

    metrics: Dict[str, Any] = {
        "effective_rank": bundle.effective_rank,
        "reconstruction": bundle.recon_error,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    return {
        "weights": str(weights_path),
        "svd": str(svd_path),
        "fits": str(fits_path),
        "metrics": str(metrics_path),
    }


def _evaluate_fit(fit: SymbolicFit | Dict[str, Any], z: float) -> Optional[float]:
    if isinstance(fit, SymbolicFit):
        fdict = {
            "method": fit.method,
            "coefficients": fit.coefficients,
        }
    else:
        fdict = fit
    method = fdict.get("method", "")
    coeffs = fdict.get("coefficients", {})
    zz = float(z)
    if method == "poly":
        c = np.asarray(coeffs.get("poly_coeffs", []), dtype=float)
        if c.size == 0:
            return None
        return float(np.polyval(c, zz))
    if method == "rational":
        num = np.asarray(coeffs.get("numerator", []), dtype=float)
        den = np.asarray(coeffs.get("denominator", []), dtype=float)
        if num.size == 0 or den.size == 0:
            return None
        num_val = float(sum(num[p] * (zz ** p) for p in range(num.size)))
        den_val = float(sum(den[q] * (zz ** q) for q in range(den.size)))
        if abs(den_val) < 1e-8:
            return None
        return num_val / den_val
    return None


def load_svd_bundle(path: Path | str) -> Optional[Dict[str, Any]]:
    try:
        data = np.load(path, allow_pickle=True).item()
        return {
            "U": np.asarray(data.get("U", [])),
            "S": np.asarray(data.get("S", [])),
            "VT": np.asarray(data.get("VT", [])),
            "z_grid": np.asarray(data.get("z_grid", [])),
            "mode_curves": np.asarray(data.get("mode_curves", [])),
            "sigma_norm": np.asarray(data.get("sigma_norm", [])),
            "effective_rank": data.get("effective_rank", {}),
        }
    except Exception:
        return None


def load_symbolic_fits(path: Path | str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    try:
        data = json.loads(Path(path).read_text())
        return np.asarray(data.get("z_grid", []), dtype=float), list(data.get("fits", []))
    except Exception:
        return np.asarray([]), []


def predict_weights_from_modes(
    z0: float,
    svd_bundle: Dict[str, Any],
    fits: Sequence[Dict[str, Any] | SymbolicFit],
    max_rank: int = 3,
) -> Optional[np.ndarray]:
    U = np.asarray(svd_bundle.get("U", []))
    S = np.asarray(svd_bundle.get("S", []))
    if U.size == 0 or S.size == 0:
        return None
    r = min(max_rank, min(U.shape[1], len(S)))
    if r == 0:
        return None
    mode_vals = np.zeros((r,), dtype=float)
    for fit in fits:
        idx = int(fit.mode) if isinstance(fit, SymbolicFit) else int(fit.get("mode", 0))  # type: ignore[call-arg]
        if idx >= r or idx < 0:
            continue
        val = _evaluate_fit(fit, z0)
        if val is not None and math.isfinite(val):
            mode_vals[idx] = val
    if not np.any(mode_vals):
        return None
    w_pred = np.zeros((U.shape[0],), dtype=float)
    for i in range(r):
        w_pred += U[:, i] * mode_vals[i]
    return w_pred


def spectral_gap_ok(S: Sequence[float], rank: int, thresh: float = 0.1) -> bool:
    arr = np.asarray(S, dtype=float).reshape(-1)
    if arr.size <= rank:
        return False
    if arr[0] <= 0.0:
        return False
    return float(arr[rank] / arr[0]) < float(thresh)


def fit_quality_ok(fits: Sequence[Dict[str, Any] | SymbolicFit], rel_rmse_tol: float = 0.2) -> bool:
    if not fits:
        return False
    for fit in fits:
        rel_rmse = fit.rel_rmse if isinstance(fit, SymbolicFit) else fit.get("rel_rmse", 1.0)  # type: ignore[call-arg]
        if not math.isfinite(rel_rmse):
            return False
        if rel_rmse > rel_rmse_tol:
            return False
    return True


def render_summary(
    label: str,
    geometry: str,
    basis: Sequence[str],
    z_grid: Sequence[float],
    bundle: WeightModeBundle,
    fits: Sequence[SymbolicFit],
    research_wishlist: Sequence[str],
) -> str:
    lines = [
        f"{label}",
        "",
        f"Geometry: {geometry}",
        f"Basis: {', '.join(basis)}",
        f"Axis sweep z-grid: {', '.join(f'{z:.3g}' for z in z_grid)}",
        "",
        "Spectral summary:",
        f"- sigma_norm: {', '.join(f'{s:.3g}' for s in bundle.sigma_norm[:5])}",
        f"- effective_rank (1e-1 / 1e-2): {bundle.effective_rank.get('eps_1e-1', 0)} / {bundle.effective_rank.get('eps_1e-2', 0)}",
    ]
    if bundle.recon_error:
        recon_parts = [f"{k}:{v:.3g}" for k, v in bundle.recon_error.items()]
        lines.append(f"- reconstruction rel_fro: {', '.join(recon_parts)}")

    if fits:
        lines.append("")
        lines.append("Symbolic mode laws:")
        for fit in fits:
            lines.append(
                f"- mode {fit.mode} [{fit.method}] rmse={fit.rmse:.3g} rel_rmse={fit.rel_rmse:.3g}: {fit.expression}"
            )

    if research_wishlist:
        lines.append("")
        lines.append("research_wishlist:")
        for item in research_wishlist:
            lines.append(f"- {item}")

    return "\n".join(lines) + "\n"
