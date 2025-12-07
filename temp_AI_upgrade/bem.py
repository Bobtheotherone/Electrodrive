from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import numpy as np

from electrodrive.core.bem_mesh import TriMesh, generate_mesh
from electrodrive.core.bem_kernel import (
    _bem_E_field_targets_core_torch,  # differentiable E-field targets
    _bem_matvec_core_torch,  # differentiable matvec core
    _bem_potential_targets_core_torch,  # differentiable potential targets
    bem_E_field_targets,  # non-diff E-field at targets
    bem_matvec_gpu,  # non-diff matvec (no_grad, KeOps-capable)
    bem_potential_targets,  # non-diff potential at targets
)
from electrodrive.core.bem_quadrature import (
    self_integral_correction,
    near_singular_quadrature,
)

# Optional CUDA near-field extension (panel-panel / target-panel).
try:  # pragma: no cover - optional dependency
    from electrodrive.core import bem_near_cuda  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    bem_near_cuda = None  # type: ignore[assignment]

from electrodrive.core.bem_solver import gmres_restart
from electrodrive.utils.config import BEMConfig, K_E
from electrodrive.utils.logging import JsonlLogger
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.debug import bem_intercept

__all__ = ["bem_solve", "BEMSolution"]


# ---------------------------------------------------------------------------
# Lazy diffbem import
# ---------------------------------------------------------------------------


def _get_diffbem_module():
    """
    Lazy import for the differentiable BEM solver.

    This avoids introducing a hard xitorch dependency for users who only
    rely on the standard non-differentiable BEM path.
    """
    try:  # pragma: no cover - import path only
        from electrodrive.core import diffbem as _diffbem  # type: ignore
        return _diffbem
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Small helpers / control loop primitives
# ---------------------------------------------------------------------------


class _ControlSignal(Exception):
    """
    Sentinel exception used to abort GMRES early in response to control.json.

    This is intentionally local to bem.py and never exposed as part of the
    public API. Callers should catch it inside bem_solve and translate to a
    structured error.
    """


@dataclass
class _ControlState:
    pause: bool = False
    terminate: bool = False
    write_every: Optional[int] = None
    snapshot_marks: List[str] = field(default_factory=list)
    last_poll_ts: float = 0.0


def _safe_import_exists(mod_name: str) -> bool:
    try:
        __import__(mod_name)
        return True
    except Exception:
        return False


def _detect_backends() -> Dict[str, bool]:
    return {
        "torch": "torch" in sys.modules or _safe_import_exists("torch"),
        "keops": _safe_import_exists("pykeops") or _safe_import_exists("keopscore"),
        "cupy": _safe_import_exists("cupy"),
    }


def _has_bem_near_cuda() -> bool:
    """
    Return True if the optional bem_near_cuda extension is available.

    This helper is defensive and never raises; if the extension or its
    capability flag is missing, it simply returns False.
    """
    global bem_near_cuda
    if bem_near_cuda is None:
        return False
    try:
        # Prefer the explicit availability checker if present.
        fn = getattr(bem_near_cuda, "is_bem_near_cuda_available", None)
        if callable(fn):
            return bool(fn())
        # Backwards-compatible fallbacks.
        legacy_fn = getattr(bem_near_cuda, "has_bem_near_cuda", None)
        if callable(legacy_fn):
            return bool(legacy_fn())
        flag = getattr(bem_near_cuda, "HAS_BEM_NEAR_CUDA", None)
        if flag is not None:
            return bool(flag)
        # If the module imported successfully and exposes neither helper,
        # assume it is usable.
        return True
    except Exception:
        return False


def _get_run_dir_from_env_or_cfg(cfg: BEMConfig) -> Optional[Path]:
    """
    Determine run directory for control/manifest/metrics integration.

    Preference:
    - cfg.run_dir if present and non-empty.
    - env EDE_RUN_DIR if set.
    """
    run_dir_val = getattr(cfg, "run_dir", None)
    if isinstance(run_dir_val, (str, os.PathLike)) and str(run_dir_val).strip():
        p = Path(run_dir_val).expanduser()
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            return None

    env_val = os.getenv("EDE_RUN_DIR", "").strip()
    if env_val:
        p = Path(env_val).expanduser()
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            return None
    return None


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """
    Atomically write JSON to path (tmp + replace + fsync).

    Cross-platform safe; never raises to caller (best-effort).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp, path)
    except Exception:
        # Best-effort; do not crash solver on logging failures.
        pass


def _poll_control_file(
    run_dir: Optional[Path],
    state: _ControlState,
    logger: Optional[JsonlLogger],
    *,
    max_hz: float = 4.0,
) -> None:
    """
    Poll control.json (if present) at most max_hz for pause/terminate/write_every.

    This function:
    - Updates state.pause / state.terminate / state.write_every.
    - Appends any snapshot marks to state.snapshot_marks.
    - Implements a bounded wait loop for pause, with small sleeps (no busy spin).
    """
    if run_dir is None:
        return

    now = time.time()
    min_dt = 1.0 / max_hz
    if now - state.last_poll_ts < min_dt:
        return
    state.last_poll_ts = now

    path = run_dir / "control.json"
    if not path.is_file():
        return

    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return

    if not isinstance(obj, dict):
        return

    pause = bool(obj.get("pause", False))
    terminate = bool(obj.get("terminate", False))
    write_every = obj.get("write_every", None)

    if isinstance(write_every, int) and write_every > 0:
        state.write_every = write_every
    elif write_every is None:
        # leave unchanged
        pass

    snapshot = obj.get("snapshot", None)
    if isinstance(snapshot, str) and snapshot:
        state.snapshot_marks.append(snapshot)

    if pause and not state.pause and logger:
        logger.info("Control: entering pause state.")
    if not pause and state.pause and logger:
        logger.info("Control: resuming from pause state.")
    state.pause = pause

    if terminate and not state.terminate and logger:
        logger.warning("Control: terminate requested; will abort solve.")
    state.terminate = terminate

    # Handle blocking pause (bounded; no busy loop).
    if state.pause and not state.terminate:
        if logger:
            logger.info("Control: paused. Waiting until pause=false or terminate=true.")
        max_wait = 3600.0  # 1 hour safety cap
        t0 = time.time()
        while True:
            if state.terminate:
                break
            if time.time() - t0 > max_wait:
                if logger:
                    logger.warning(
                        "Control: pause exceeded max_wait; auto-resuming.",
                        max_wait=max_wait,
                    )
                state.pause = False
                break
            time.sleep(0.25)
            _poll_control_file(run_dir, state, logger, max_hz=max_hz)
            if not state.pause:
                break


def _init_device(cfg: BEMConfig) -> torch.device:
    if getattr(cfg, "use_gpu", False) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _free_space_at_points(
    spec: CanonicalSpec,
    P: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Free-space potential from explicit point charges at points P[N,3].

    V_free(x) = sum_k K_E * q_k / |x - r_k|
    """
    if P.numel() == 0:
        return torch.zeros(0, device=device, dtype=dtype)

    V = torch.zeros(P.shape[0], device=device, dtype=dtype)
    for ch in spec.charges:
        if ch.get("type") != "point":
            continue
        try:
            q = torch.as_tensor(float(ch["q"]), device=device, dtype=dtype)
            pos = torch.as_tensor(
                [float(x) for x in ch["pos"]],
                device=device,
                dtype=dtype,
            )
        except Exception:
            continue
        r = torch.linalg.norm(P - pos[None, :], dim=1).clamp_min(1e-12)
        V = V + (K_E * q) / r
    return V


def _free_space_potential_on_centroids(
    spec: CanonicalSpec,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """
    Convenience wrapper: free-space potential evaluated at panel centroids.

    This uses the same point-charge model as `_free_space_at_points`.
    """
    return _free_space_at_points(
        spec=spec,
        P=centroids,
        dtype=centroids.dtype,
        device=centroids.device,
    )


def _free_space_E_field_at_points(
    spec: CanonicalSpec,
    P: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Free-space E-field from explicit point charges at points P[N,3].

    E_free(x) = sum_k K_E * q_k (x - r_k) / |x - r_k|^3
    """
    if P.numel() == 0:
        return torch.zeros(0, 3, device=device, dtype=dtype)

    E = torch.zeros(P.shape[0], 3, device=device, dtype=dtype)
    for ch in spec.charges:
        if ch.get("type") != "point":
            continue
        try:
            q = torch.as_tensor(float(ch["q"]), device=device, dtype=dtype)
            pos = torch.as_tensor(
                [float(x) for x in ch["pos"]],
                device=device,
                dtype=dtype,
            )
        except Exception:
            continue
        R = P - pos[None, :]
        r = torch.linalg.norm(R, dim=1, keepdim=True).clamp_min(1e-12)
        E = E + (K_E * q) * R / (r**3)
    return E


def _bc_vector(
    spec: CanonicalSpec,
    mesh: TriMesh,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Construct per-panel Dirichlet BC values (V on conductor surfaces).

    Each panel inherits the potential of its owning conductor. Panels whose
    conductor has no explicit 'potential' field default to 0.0 (grounded).
    """
    # Default: all panels grounded
    bc = torch.zeros(mesh.n_panels, device=device, dtype=dtype)

    # Map conductor ID -> potential
    id_to_V: Dict[int, float] = {}
    for i, c in enumerate(getattr(spec, "conductors", []) or []):
        try:
            V = float(c.get("potential", 0.0))
        except Exception:
            V = 0.0
        # Some specs may carry an explicit "id"; otherwise fall back to index.
        cid = int(c.get("id", i))
        id_to_V[cid] = V

    # mesh.conductor_ids is length n_panels; panels with unknown IDs get 0.0
    for i, cid in enumerate(mesh.conductor_ids):
        bc[i] = id_to_V.get(int(cid), 0.0)

    return bc


def _offset_from_boundary(
    spec: Optional[CanonicalSpec],
    P: torch.Tensor,
    areas: torch.Tensor,
    centroids: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    sign: float = 1.0,
) -> torch.Tensor:
    """
    Slightly nudge points that lie exactly on idealized boundaries (plane/sphere)
    to avoid numerical coincidences when evaluating induced fields.

    Only applied to evaluation points; does not affect the solved system.
    """
    if P.numel() == 0:
        return P

    P2 = P.clone()

    if areas.numel() > 0:
        try:
            req = float(
                torch.median(
                    torch.sqrt(areas.clamp_min(1e-30) / math.pi)
                ).item()
            )
        except Exception:
            req = 1e-3
    else:
            req = 1e-3

    # Extremely small nudge to avoid exact coincident evaluations without
    # materially moving points away from the enforced boundary.
    delta = torch.as_tensor(sign * 1e-9 * req, device=device, dtype=dtype)

    if spec is None:
        return P2

    for c in spec.conductors:
        t = c.get("type")
        if t == "plane":
            z0 = float(c.get("z", 0.0))
            mask = torch.abs(P2[:, 2] - z0) < 10.0 * torch.finfo(dtype).eps
            if mask.any():
                P2[mask, 2] = z0 + delta
        elif t == "sphere":
            try:
                center = torch.as_tensor(
                    [float(x) for x in c.get("center", [0.0, 0.0, 0.0])],
                    device=device,
                    dtype=dtype,
                )
                a = float(c.get("radius", 1.0))
            except Exception:
                continue
            r = torch.linalg.norm(P2 - center[None, :], dim=1)
            mask = (
                torch.abs(r - a)
                < 10.0 * torch.finfo(dtype).eps * max(1.0, a)
            )
            if torch.any(mask):
                v = P2[mask] - center[None, :]
                v = v / torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(1e-24)
                P2[mask] = center[None, :] + (a + float(delta)) * v

    return P2


def _build_near_pairs_for_panels(
    centroids_np: np.ndarray,
    areas_np: np.ndarray,
    distance_factor: float,
    n_panels: Optional[int] = None,
) -> np.ndarray:
    """
    Construct a list of (i, j) indices for panel–panel interactions that are
    considered "near" based on centroid distance and panel radii.

    Panels i and j are marked near if

        |C_i - C_j| <= distance_factor * (R_i + R_j),

    where R_k = sqrt(A_k / pi) is the equal-area disk radius associated with
    panel k. Self-interactions (i == j) are excluded; they are handled by
    `self_integral_correction`.

    Returns
    -------
    near_pairs : ndarray, shape (P, 2)
        Possibly empty array of (i, j) index pairs.
    """
    # Normalise inputs
    C_full = np.asarray(centroids_np, dtype=float)
    A_full = np.asarray(areas_np, dtype=float).reshape(-1)

    N_C = int(C_full.shape[0])
    N_A = int(A_full.shape[0])

    if n_panels is not None:
        N = max(0, min(int(n_panels), N_C, N_A))
    else:
        N = max(0, min(N_C, N_A))

    if N <= 1:
        return np.zeros((0, 2), dtype=np.int64)

    # Truncate to the common prefix to avoid any shape mismatch surprises.
    C = C_full[:N]
    A = A_full[:N]

    # Equal-area disk radii R = sqrt(A/pi), clipping to avoid negatives
    A_clipped = np.clip(A, 0.0, np.inf)
    radii = np.sqrt(A_clipped / math.pi)

    df = float(distance_factor)

    # Collect near pairs incrementally to avoid dense N×N allocations.
    pairs_i: List[np.ndarray] = []
    pairs_j: List[np.ndarray] = []

    for i in range(N):
        # Distances from panel i to all panels.
        diff = C[i] - C  # (N, 3)
        dists = np.linalg.norm(diff, axis=1)  # (N,)

        # Pair-dependent near threshold: df * (R_i + R_j).
        thresh_i = df * (radii[i] + radii)  # (N,)

        # Robustness: ignore any non-finite distances/thresholds.
        mask = (
            np.isfinite(dists)
            & np.isfinite(thresh_i)
            & (dists > 0.0)
            & (dists <= thresh_i)
        )

        js = np.nonzero(mask)[0]
        if js.size == 0:
            continue

        pairs_i.append(np.full(js.shape, i, dtype=np.int64))
        pairs_j.append(js.astype(np.int64))

    if not pairs_i:
        return np.zeros((0, 2), dtype=np.int64)

    i_idx = np.concatenate(pairs_i, axis=0)
    j_idx = np.concatenate(pairs_j, axis=0)
    return np.stack([i_idx, j_idx], axis=1)


def _precompute_near_correction_weights(
    centroids: np.ndarray,
    areas: np.ndarray,
    panel_vertices: np.ndarray,
    near_pairs: np.ndarray,
    quad_order: int,
) -> np.ndarray:
    """
    Precompute geometry-only near-field correction weights for panel pairs.

    For each (i, j) in near_pairs, this computes the scalar delta

        delta_ij = I_near(C_i, panel_j) - K_E * A_j / |C_i - C_j|,

    where I_near is evaluated using near_singular_quadrature.  These
    deltas depend only on geometry and quadrature order, not on sigma,
    so they can be reused across GMRES iterations.
    """
    if near_pairs.size == 0:
        return np.zeros((0,), dtype=float)

    C_np = np.asarray(centroids, dtype=float)
    A_np = np.asarray(areas, dtype=float).reshape(-1)
    verts_np = np.asarray(panel_vertices, dtype=float)

    N_panels = min(int(C_np.shape[0]), int(A_np.shape[0]), int(verts_np.shape[0]))
    if N_panels <= 0:
        return np.zeros((near_pairs.shape[0],), dtype=float)

    weights = np.zeros((near_pairs.shape[0],), dtype=float)

    for idx_pair, (idx_i_raw, idx_j_raw) in enumerate(near_pairs):
        idx_i = int(idx_i_raw)
        idx_j = int(idx_j_raw)

        if not (0 <= idx_i < N_panels) or not (0 <= idx_j < N_panels):
            continue

        Aj = float(A_np[idx_j])
        if Aj <= 0.0:
            continue

        target = C_np[idx_i]
        verts_j = verts_np[idx_j]

        pts, w = near_singular_quadrature(
            target=target,
            panel_vertices=verts_j,
            method="telles",
            order=quad_order,
        )
        if w.size == 0:
            continue

        r = np.linalg.norm(pts - target[None, :], axis=1)
        r = np.maximum(r, 1e-12)
        kernel_vals = K_E / r
        I_near = float(np.sum(kernel_vals * w))

        r_far = np.linalg.norm(target - C_np[idx_j])
        if r_far < 1e-12:
            continue
        I_far = float(K_E * Aj / r_far)

        weights[idx_pair] = I_near - I_far

    return weights


# ---------------------------------------------------------------------------
# Torch-native near-field correction for panel-panel matvec
# ---------------------------------------------------------------------------


def _apply_near_quadrature_matvec_torch(
    V_far: torch.Tensor,
    sigma: torch.Tensor,
    *,
    near_pairs: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Fast, Torch-native near-field correction for the centroid-lumped matvec.

    This assumes that the geometry-only near weights
        delta_ij = I_near(C_i, panel_j) - K_E * A_j / |C_i - C_j|
    have already been precomputed and stored in `weights`.

    Parameters
    ----------
    V_far : (N,) tensor
        Output of the far-field matvec (e.g. bem_matvec_gpu).
    sigma : (N,) tensor
        Surface charge density vector.
    near_pairs : (P,2) tensor (long)
        [i, j] indices of near interactions.
    weights : (P,) tensor
        Precomputed delta_ij weights.

    Returns
    -------
    V_corr : (N,) tensor
        V_far plus near-field corrections.
    """
    if near_pairs is None or weights is None:
        return V_far
    if near_pairs.numel() == 0 or weights.numel() == 0 or sigma.numel() == 0:
        return V_far

    device = V_far.device
    dtype = V_far.dtype

    near_pairs = near_pairs.to(device=device, dtype=torch.long)
    weights = weights.to(device=device, dtype=dtype)

    i_idx = near_pairs[:, 0]
    j_idx = near_pairs[:, 1]

    N_v = V_far.shape[0]
    N_s = sigma.shape[0]

    valid = (i_idx >= 0) & (i_idx < N_v) & (j_idx >= 0) & (j_idx < N_s)
    if not torch.any(valid):
        return V_far

    i_idx = i_idx[valid]
    j_idx = j_idx[valid]
    w = weights[valid]

    contrib = sigma.index_select(0, j_idx) * w

    V_out = V_far.clone()
    V_out.index_add_(0, i_idx, contrib)
    return V_out


def _apply_near_quadrature_matvec(
    V_far: torch.Tensor,
    sigma: torch.Tensor,
    *,
    centroids: torch.Tensor,
    areas: torch.Tensor,
    panel_vertices: np.ndarray,
    near_pairs: np.ndarray,
    quad_order: int,
    near_pair_weights: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Apply a near-field quadrature correction to a matrix-vector product V_far.

    This function assumes:
      - centroids, areas, sigma, and V_far are Tensors (CPU or GPU).
      - panel_vertices and near_pairs are numpy arrays built from TriMesh.

    The correction replaces the point-lumped approximation

        K_E * A_j / |C_i - C_j|

    with a quadrature-based estimate of the integral

        ∫_panel_j K_E / |C_i - y| dS_y

    for all (i, j) listed in near_pairs.
    """
    if near_pairs.size == 0:
        return V_far

    device = V_far.device
    dtype = V_far.dtype

    V_np = V_far.detach().cpu().numpy()
    sigma_np = sigma.detach().cpu().numpy()
    C_np = centroids.detach().cpu().numpy()
    A_np = areas.detach().cpu().numpy()

    # Guard against any shape mismatches: work with the common prefix length.
    N_V = int(V_np.shape[0])
    N_C = int(C_np.shape[0])
    N_A = int(A_np.shape[0])
    N_panels = min(N_V, N_C, N_A, int(panel_vertices.shape[0]))

    if N_panels <= 0:
        return V_far

    weights_np: Optional[np.ndarray] = None
    if near_pair_weights is not None:
        weights_np = np.asarray(near_pair_weights, dtype=float).reshape(-1)
        if weights_np.shape[0] != near_pairs.shape[0]:
            weights_np = None

    from electrodrive.utils.config import K_E  # local import to avoid cycles

    for pair_idx, (idx_i_raw, idx_j_raw) in enumerate(near_pairs):
        idx_i = int(idx_i_raw)
        idx_j = int(idx_j_raw)

        # Robust index bounds check: skip any out-of-range pairs.
        if not (0 <= idx_i < N_panels):
            continue
        if not (0 <= idx_j < N_panels):
            continue

        Aj = float(A_np[idx_j])
        sig_j = float(sigma_np[idx_j])
        if Aj <= 0.0 or sig_j == 0.0:
            continue

        target = C_np[idx_i]
        verts_j = panel_vertices[idx_j]

        delta = None
        if weights_np is not None and pair_idx < weights_np.shape[0]:
            delta = float(weights_np[pair_idx])
            if not math.isfinite(delta):
                delta = None

        if delta is None:
            pts, w = near_singular_quadrature(
                target=target,
                panel_vertices=verts_j,
                method="telles",
                order=quad_order,
            )
            if w.size == 0:
                continue

            r = np.linalg.norm(pts - target[None, :], axis=1)
            r = np.maximum(r, 1e-12)
            kernel_vals = K_E / r
            I_near = float(np.sum(kernel_vals * w))

            r_far = np.linalg.norm(target - C_np[idx_j])
            if r_far < 1e-12:
                # Diagonal terms rely on self_integral_correction instead.
                continue
            I_far = float(K_E * Aj / r_far)
            delta = I_near - I_far

        if delta == 0.0:
            continue

        V_np[idx_i] += sig_j * delta

    return torch.as_tensor(V_np, device=device, dtype=dtype)


def _apply_near_quadrature_matvec_cuda(
    V_far: torch.Tensor,
    sigma: torch.Tensor,
    *,
    centroids: torch.Tensor,
    areas: torch.Tensor,
    panel_vertices: torch.Tensor,
    near_pairs: torch.Tensor,
    quad_order: int,
    panel_vertices_np: Optional[np.ndarray] = None,
    near_pairs_np: Optional[np.ndarray] = None,
    near_pair_weights_np: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    CUDA-accelerated near-field quadrature correction to a matrix-vector
    product V_far.

    This wrapper expects all tensor arguments to reside on the same CUDA
    device. It delegates the heavy lifting to the optional
    `electrodrive.core.bem_near_cuda` extension when available and
    falls back to the CPU implementation `_apply_near_quadrature_matvec`
    when needed.

    Parameters are analogous to `_apply_near_quadrature_matvec`, except
    that `panel_vertices` and `near_pairs` are Tensors.
    """
    # Nothing to do if there are no near interactions.
    if near_pairs is None or near_pairs.numel() == 0:
        return V_far

    device = V_far.device
    if device.type != "cuda":
        # Not on CUDA; fall back to the CPU helper if we have NumPy arrays.
        if panel_vertices_np is not None and near_pairs_np is not None:
            return _apply_near_quadrature_matvec(
                V_far,
                sigma,
                centroids=centroids,
                areas=areas,
                panel_vertices=panel_vertices_np,
                near_pairs=near_pairs_np,
                quad_order=quad_order,
                near_pair_weights=near_pair_weights_np,
            )
        return V_far

    if not _has_bem_near_cuda():
        if panel_vertices_np is not None and near_pairs_np is not None:
            return _apply_near_quadrature_matvec(
                V_far,
                sigma,
                centroids=centroids,
                areas=areas,
                panel_vertices=panel_vertices_np,
                near_pairs=near_pairs_np,
                quad_order=quad_order,
                near_pair_weights=near_pair_weights_np,
            )
        return V_far

    try:
        # Delegate to the high-level CUDA wrapper, which normalises inputs
        # and calls the compiled extension.
        return bem_near_cuda.apply_near_quadrature_matvec_cuda(  # type: ignore[attr-defined]
            V_far,
            sigma,
            centroids=centroids,
            areas=areas,
            panel_vertices=panel_vertices,
            near_pairs=near_pairs,
            quad_order=quad_order,
            K_E=None,
        )
    except Exception:
        # Graceful fallback to CPU helper if we can.
        if panel_vertices_np is not None and near_pairs_np is not None:
            return _apply_near_quadrature_matvec(
                V_far,
                sigma,
                centroids=centroids,
                areas=areas,
                panel_vertices=panel_vertices_np,
                near_pairs=near_pairs_np,
                quad_order=quad_order,
            )
        return V_far


def _apply_near_quadrature_potentials(
    V_far: torch.Tensor,
    targets: torch.Tensor,
    centroids: torch.Tensor,
    areas: torch.Tensor,
    sigma: torch.Tensor,
    panel_vertices: torch.Tensor,
    *,
    distance_factor: float,
    quad_order: int,
    cache: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Near-field quadrature correction for potentials at arbitrary targets.

    Parameters
    ----------
    V_far : (M,) tensor
        Potential computed via centroid-lumped kernel.
    targets : (M,3) tensor
        Evaluation points.
    centroids : (N,3) tensor
    areas : (N,) tensor
    sigma : (N,) tensor
    panel_vertices : (N,3,3) tensor

    Returns
    -------
    V_corr : (M,) tensor
        Corrected potential (same shape as V_far).
    """
    if targets.numel() == 0 or sigma.numel() == 0:
        return V_far

    device = V_far.device
    dtype = V_far.dtype

    V_np = V_far.detach().cpu().numpy()
    T_np = targets.detach().cpu().numpy()

    if cache is not None:
        C_np = cache["C_np"]
        A_np = cache["A_np"]
        sigma_np = cache["S_np"]
        panel_vertices_np = cache["PV_np"]
        radii = cache["radii"]
    else:
        C_np = centroids.detach().cpu().numpy()
        A_np = areas.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()
        panel_vertices_np = panel_vertices.detach().cpu().numpy()
        areas_clipped = np.clip(A_np, 0.0, np.inf)
        radii = np.sqrt(areas_clipped / math.pi)

    from electrodrive.utils.config import K_E  # local import to avoid cycles

    M = int(T_np.shape[0])
    N = int(C_np.shape[0])
    if N == 0:
        return V_far

    for i in range(M):
        x = T_np[i]
        dists = np.linalg.norm(C_np - x[None, :], axis=1)
        # Panels whose centroid is within distance_factor * R_j are treated
        # with the refined rule.
        thresh = distance_factor * radii
        near_idx = np.nonzero(dists <= thresh)[0]
        if near_idx.size == 0:
            continue

        for j in near_idx:
            Aj = float(A_np[j])
            sig_j = float(sigma_np[j])
            if Aj <= 0.0 or sig_j == 0.0:
                continue

            verts_j = panel_vertices_np[j]
            pts, w = near_singular_quadrature(
                target=x, panel_vertices=verts_j, method="telles", order=quad_order
            )
            if w.size == 0:
                continue

            r = np.linalg.norm(pts - x[None, :], axis=1)
            r = np.maximum(r, 1e-12)
            kernel_vals = K_E / r
            I_near = float(np.sum(kernel_vals * w))

            r_far = dists[j]
            if r_far < 1e-12:
                # Replace the singular far-field self term already present in
                # V_np (from bem_potential_targets) with the properly
                # integrated on-surface value.
                r_safe = max(r_far, 1e-12)
                I_far = float(K_E * Aj / r_safe)
                V_np[i] -= sig_j * I_far
                V_np[i] += sig_j * I_near
            else:
                I_far = float(K_E * Aj / r_far)
                V_np[i] += sig_j * (I_near - I_far)

    return torch.as_tensor(V_np, device=device, dtype=dtype)


def _autotune_tile_size(
    N: int,
    dtype: torch.dtype,
    device: torch.device,
    logger: JsonlLogger,
    target_vram_fraction: float = 0.8,
    max_vram_gb: float = 24.0,
    *,
    min_tile: int = 512,
    max_tile: Optional[int] = None,
    target_peak_gb: Optional[float] = None,
    tile_mem_divisor: float = 3.0,
) -> int:
    """
    Choose a safe tile size for kernels under a VRAM budget.

    The goal here is robustness first:
    - stay well inside the available VRAM envelope
    - respect dtype differences (fp64 is more expensive than fp32)
    - avoid overly large tiles on CPU-only runs
    """
    # CPU / no-CUDA path: pick a conservative power-of-two tile size.
    if device.type != "cuda" or N <= 0 or not torch.cuda.is_available():
        if N <= 0:
            return min_tile
        # cap at 2048 by default on CPU to avoid blowing caches
        T = min(N, max(min_tile, 2048))
        # round down to power of two
        T = 2 ** int(math.log2(max(1, T)))
        return int(T)

    try:
        total_vram = torch.cuda.get_device_properties(device).total_memory
    except Exception:
        logger.warning(
            "Could not query total VRAM. Falling back to configured cap.",
            max_vram_gb=max_vram_gb,
        )
        total_vram = max_vram_gb * (1024**3)

    available = total_vram * float(target_vram_fraction)
    bytes_per = torch.finfo(dtype).bits // 8

    # Vector/state memory we expect to keep around during the solve
    vector_mem = N * 10 * bytes_per

    divisor = float(tile_mem_divisor) if tile_mem_divisor and tile_mem_divisor > 0 else 3.0
    memory_for_tiling = max(0.0, available - vector_mem) / divisor

    # Rough model: kernel needs ~4 * bytes_per * T^2
    T = int(math.sqrt(max(1.0, memory_for_tiling / (4 * bytes_per))))
    T = min(N, max(min_tile, T))

    if T > 0:
        # snap to power of two for better kernel behavior
        T = 2 ** int(math.log2(T))
    else:
        T = min_tile

    # Optional: if caller provides a stricter peak budget, respect it by
    # shrinking the tile size (never enlarging it).
    try:
        total_gb = total_vram / (1024**3)
        if target_peak_gb and target_peak_gb > 0:
            budget_gb = min(target_peak_gb, total_gb * float(target_vram_fraction))
            mem_bytes = budget_gb * (1024**3)
            mem_for_tiles = max(0.0, mem_bytes - vector_mem) / 3.0
            T2 = int(math.sqrt(max(1.0, mem_for_tiles / (4 * bytes_per))))
            T2 = min(N, max(min_tile, T2))
            if T2 > 0:
                T2 = 2 ** int(math.log2(T2))
            else:
                T2 = min_tile
            # Never exceed the original T when honoring a stricter peak budget.
            T = min(T, T2)
    except Exception:
        pass

    # Apply caller-provided max_tile clamp if present.
    if max_tile is not None:
        T = min(T, max_tile)

    # Dtype-aware hard caps as an additional safety net.
    try:
        if dtype == torch.float64:
            # fp64 is expensive; keep tiles moderate
            T = min(T, 4096)
        else:
            # fp32 can tolerate somewhat larger tiles
            T = min(T, 8192)
    except Exception:
        # If anything goes wrong, just leave T as-is.
        pass

    logger.info(
        "VRAM autotune result.",
        tile_size=int(T),
        N_dof=int(N),
        dtype=str(dtype),
        divisor=float(divisor),
        total_vram_gb=f"{total_vram / (1024 ** 3):.2f}",
    )

    return int(T)


def _jacobi_fallback_solve(
    A: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    diag: torch.Tensor,
    logger: JsonlLogger,
    *,
    maxiter: int = 1024,
    tol: float = 1e-6,
    omega: float = 0.8,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Very simple damped Jacobi fallback solver for Ax = b.

    Returns (x, stats) with similar 'info' shape to gmres_restart.
    """
    x = torch.zeros_like(b)
    Dinv = 1.0 / diag.clamp_min(1e-18)

    with torch.no_grad():
        r = b - A(x)
        r_norm0 = float(torch.linalg.norm(r))

        if not math.isfinite(r_norm0):
            logger.error(
                "Jacobi fallback: initial residual is non-finite.",
                r0=r_norm0,
            )
            return x, {
                "iters": 0,
                "resid": r_norm0,
                "success": False,
                "solver": "jacobi_fallback",
            }

        target = tol * max(r_norm0, 1.0)
        logger.warning(
            "Jacobi fallback: starting iterative solve.",
            r0=r_norm0,
            tol=tol,
            target=target,
            maxiter=maxiter,
        )

        it = 0
        for k in range(maxiter):
            it = k + 1
            x = x + omega * Dinv * r
            r = b - A(x)
            r_norm = float(torch.linalg.norm(r))

            if not math.isfinite(r_norm):
                logger.error(
                    "Jacobi fallback: residual became non-finite.",
                    iter=it,
                    resid=r_norm,
                )
                return x, {
                    "iters": it,
                    "resid": r_norm,
                    "success": False,
                    "solver": "jacobi_fallback",
                }

            if k == 0 or (k + 1) % 25 == 0:
                logger.info(
                    "Jacobi fallback iter.",
                    iter=it,
                    resid=r_norm,
                )

            if r_norm <= target:
                logger.info(
                    "Jacobi fallback converged.",
                    iters=it,
                    resid=r_norm,
                    target=target,
                )
                return x, {
                    "iters": it,
                    "resid": r_norm,
                    "success": True,
                    "solver": "jacobi_fallback",
                }

        logger.warning(
            "Jacobi fallback reached maxiter without convergence.",
            iters=it,
            resid=r_norm,
            target=target,
        )
        return x, {
            "iters": it,
            "resid": r_norm,
            "success": False,
            "solver": "jacobi_fallback",
        }


def compute_bem_capacitive_energy(
    V_total: torch.Tensor,
    sigma: torch.Tensor,
    areas: torch.Tensor,
) -> float:
    """
    Capacitive energy from boundary data (Route A variant):

        U = 1/2 * ∫ (V * sigma) dS  ≈ 0.5 * sum_i V_i * sigma_i * A_i

    For numerical robustness, accumulation is always carried out in float64,
    even if the main solve ran in float32.
    """
    try:
        V64 = V_total.to(torch.float64)
        s64 = sigma.to(torch.float64)
        a64 = areas.to(torch.float64)
        energy = 0.5 * torch.sum(V64 * s64 * a64)
        return float(energy.item())
    except Exception:
        return float("nan")


def _sample_gpu_peak_mb() -> Dict[str, float]:
    """
    Return current peak GPU memory usage in MB.
    """
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0}
    try:
        dev = torch.cuda.current_device()
        torch.cuda.synchronize(dev)
        max_alloc = float(torch.cuda.max_memory_allocated(dev))
        max_res = float(torch.cuda.max_memory_reserved(dev))
        return {
            "allocated": max_alloc / (1024.0 * 1024.0),
            "reserved": max_res / (1024.0 * 1024.0),
        }
    except Exception:
        return {"allocated": 0.0, "reserved": 0.0}


def _write_manifest(
    run_dir: Optional[Path],
    *,
    run_id: str,
    device: torch.device,
    dtype: torch.dtype,
    requested_mode: str,
    selected_mode: str,
    planner_rationale: str = "",
    fallback_reason: Optional[str] = None,
) -> None:
    """
    Best-effort manifest.json writer for BEM runs.
    """
    if run_dir is None:
        return

    backends = _detect_backends()
    gpu_peak = _sample_gpu_peak_mb()

    tf32_enabled = False
    try:
        if hasattr(torch, "get_float32_matmul_precision"):
            tf32_mode = str(torch.get_float32_matmul_precision())
            tf32_enabled = tf32_mode.lower() in ("high", "highest", "medium")
    except Exception:
        tf32_enabled = False

    device_name = "cpu"
    gpu_available = bool(device.type == "cuda" and torch.cuda.is_available())
    if gpu_available:
        try:
            props = torch.cuda.get_device_properties(device)
            device_name = props.name
        except Exception:
            device_name = "unknown"

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "git_sha": os.getenv("EDE_GIT_SHA", ""),
        "versions": {
            "python": sys.version,
            "torch": getattr(torch, "__version__", "unavailable"),
        },
        "planner": {
            "requested_mode": requested_mode,
            "selected_mode": selected_mode,
            "rationale": planner_rationale,
        },
        "device": {
            "gpu_available": gpu_available,
            "device_name": device_name,
            "dtype": str(dtype),
            "tf32": tf32_enabled,
            "gpu_mem_peak_mb": gpu_peak,
        },
        "backend": {
            "available": backends,
            "selected": "torch",
            "fallback_reason": fallback_reason,
        },
    }

    _atomic_write_json(run_dir / "manifest.json", manifest)


# ---------------------------------------------------------------------------
# BEMSolution: evaluation wrapper
# ---------------------------------------------------------------------------


class BEMSolution:
    """
    Simple evaluator wrapping the solved boundary data.

    Provides:
    - eval(p: (x,y,z)) -> float potential
    - eval_V_E_batched(P: [N,3]) -> (V[N], E[N,3])

    When constructed with differentiable=True, eval_V_E_batched uses
    differentiable kernel cores so that gradients can flow back to sigma.
    """

    def __init__(
        self,
        spec: CanonicalSpec,
        centroids: torch.Tensor,
        areas: torch.Tensor,
        sigma: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        tile_size: int,
        normals: Optional[torch.Tensor] = None,
        differentiable: bool = False,
        panel_vertices: Optional[torch.Tensor] = None,
        near_quadrature: bool = False,
        near_quad_order: int = 2,
        near_quad_dist_factor: float = 1.5,
    ):
        self._spec = spec
        self._C = centroids
        self._A = areas
        self._N = normals
        self._S = sigma
        self._device = device
        self._dtype = dtype
        self._tile = tile_size
        self._differentiable = bool(differentiable)
        self._panel_vertices = panel_vertices
        # Near-field quadrature is implemented for the non-differentiable
        # evaluation path. The correction kernels themselves run on CPU but
        # accept inputs from either CPU or GPU and return results on the
        # original device.
        self._near_quad_enabled = bool(
            near_quadrature
            and (panel_vertices is not None)
            and not self._differentiable
        )
        self._near_quad_order = int(near_quad_order)
        self._near_quad_dist_factor = float(near_quad_dist_factor)
        self.meta: Dict[str, Any] = {}

        # Optional CPU-side cache for near-field potential correction to avoid
        # repeated Tensor→NumPy conversions on every eval call.
        self._near_eval_cache: Optional[Dict[str, Any]] = None
        if self._near_quad_enabled and panel_vertices is not None:
            try:
                C_np = self._C.detach().cpu().numpy()
                A_np = self._A.detach().cpu().numpy()
                S_np = self._S.detach().cpu().numpy()
                PV_np = panel_vertices.detach().cpu().numpy()
                A_clipped = np.clip(A_np, 0.0, np.inf)
                radii_np = np.sqrt(A_clipped / math.pi)
                self._near_eval_cache = {
                    "C_np": C_np,
                    "A_np": A_np,
                    "S_np": S_np,
                    "PV_np": PV_np,
                    "radii": radii_np,
                }
            except Exception:
                self._near_eval_cache = None

    def _eval_at_points(
        self,
        P: torch.Tensor,
        compute_V: bool,
        compute_E: bool,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        P = P.to(device=self._device, dtype=self._dtype)

        P_shift = _offset_from_boundary(
            self._spec,
            P,
            self._A,
            self._C,
            self._dtype,
            self._device,
        )

        V_total: Optional[torch.Tensor] = None
        E_total: Optional[torch.Tensor] = None

        if compute_V:
            V_free = _free_space_at_points(
                self._spec, P_shift, self._dtype, self._device
            )
            if self._differentiable:
                V_ind = _bem_potential_targets_core_torch(
                    targets=P_shift,
                    src_centroids=self._C,
                    areas=self._A,
                    sigma=self._S,
                    tile_size=self._tile,
                )
            else:
                V_ind = bem_potential_targets(
                    targets=P_shift,
                    src_centroids=self._C,
                    areas=self._A,
                    sigma=self._S,
                    tile_size=self._tile,
                )
                if (
                    self._near_quad_enabled
                    and self._panel_vertices is not None
                    and self._panel_vertices.numel() > 0
                ):
                    V_ind = _apply_near_quadrature_potentials(
                        V_ind,
                        targets=P_shift,
                        centroids=self._C,
                        areas=self._A,
                        sigma=self._S,
                        panel_vertices=self._panel_vertices,
                        distance_factor=self._near_quad_dist_factor,
                        quad_order=self._near_quad_order,
                        cache=self._near_eval_cache,
                    )
            V_total = V_free + V_ind

        if compute_E:
            E_free = _free_space_E_field_at_points(
                self._spec, P_shift, self._dtype, self._device
            )
            if self._differentiable:
                E_ind = _bem_E_field_targets_core_torch(
                    targets=P_shift,
                    src_centroids=self._C,
                    areas=self._A,
                    sigma=self._S,
                    tile_size=self._tile,
                )
            else:
                E_ind = bem_E_field_targets(
                    targets=P_shift,
                    src_centroids=self._C,
                    areas=self._A,
                    sigma=self._S,
                    tile_size=self._tile,
                )
            E_total = E_free + E_ind

        return V_total, E_total

    def eval(self, p: Tuple[float, float, float]) -> float:
        P = torch.tensor(
            [[float(p[0]), float(p[1]), float(p[2])]],
            device=self._device,
            dtype=self._dtype,
        )
        V, _ = self._eval_at_points(P, compute_V=True, compute_E=False)
        return float(V[0].item() if V is not None and V.numel() > 0 else 0.0)

    def eval_V_E_batched(
        self,
        P: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        V, E = self._eval_at_points(P, compute_V=True, compute_E=True)
        if V is None:
            V = torch.empty(0, device=self._device, dtype=self._dtype)
        if E is None:
            E = torch.empty(0, 3, device=self._device, dtype=self._dtype)
        return V, E


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def bem_solve(
    spec: CanonicalSpec,
    cfg: BEMConfig,
    logger: JsonlLogger,
    differentiable: bool = False,
) -> Dict[str, Any]:
    """
    Single-layer BEM for Dirichlet problems with explicit point charges.

    Returns either a rich dict with a BEMSolution, or {"error": "..."}.
    """
    device = _init_device(cfg)
    dtype: torch.dtype = torch.float64 if getattr(cfg, "fp64", False) else torch.float32

    run_dir = _get_run_dir_from_env_or_cfg(cfg)
    run_id = os.getenv("EDE_RUN_ID", "") or f"bem-{int(time.time())}"

    logger.info(
        "BEM solver start.",
        device=str(device),
        fp64=bool(getattr(cfg, "fp64", False)),
        max_refine_passes=getattr(cfg, "max_refine_passes", 3),
        differentiable=bool(differentiable),
        run_id=run_id,
    )

    intercept_ctx = bem_intercept.maybe_start_intercept(
        spec, test_name="bem_solve", bem_cfg=cfg
    )
    stop_reason = "unset"

    control = _ControlState()

    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass

    current_h = float(getattr(cfg, "initial_h", 0.3))
    refine_factor = float(getattr(cfg, "refine_factor", 0.5))
    target_bc = float(getattr(cfg, "target_bc_inf_norm", 1e-7))

    # Hoist config scalars out of the refinement loop to avoid repeated getattr
    max_refine_passes = int(getattr(cfg, "max_refine_passes", 3))
    gmres_restart_val = int(getattr(cfg, "gmres_restart", 50))
    gmres_tol_val = float(getattr(cfg, "gmres_tol", 1e-8))
    gmres_maxiter_val = int(getattr(cfg, "gmres_maxiter", 500))
    gmres_log_every_val = int(getattr(cfg, "gmres_log_every", 25))
    use_precond_cfg = bool(getattr(cfg, "use_precond", True))
    allow_jacobi_cfg = bool(getattr(cfg, "allow_jacobi_fallback", True))
    jacobi_maxiter_cfg = int(getattr(cfg, "jacobi_maxiter", 256))
    jacobi_tol_cfg = float(getattr(cfg, "jacobi_tol", 1e-5))
    jacobi_omega_cfg = float(getattr(cfg, "jacobi_omega", 0.8))

    history: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_bc = float("inf")
    plateau = 0
    # Plateau parameters are now configurable via cfg:
    max_plateau = int(getattr(cfg, "plateau_max", 1))
    plateau_rel_improvement = float(getattr(cfg, "plateau_rel_improvement", 0.05))

    def _error_result(msg: str, **extra: Any) -> Dict[str, Any]:
        """
        Centralized structured error helper.

        Ensures that any error surfaced to callers also carries whatever
        refinement history and best-pass diagnostics are available, and
        that a manifest is written for post-mortem analysis.
        """
        payload: Dict[str, Any] = {"error": msg}
        if history:
            payload["refinement_history"] = history
        if best is not None:
            payload["best_bc_resid_linf"] = best.get("bc_resid_linf")
            payload["best_dof"] = best.get("dof")
        payload.update(extra)
        try:
            _write_manifest(
                run_dir,
                run_id=run_id,
                device=device,
                dtype=dtype,
                requested_mode="bem",
                selected_mode="bem",
                planner_rationale="bem_solve_direct",
                fallback_reason=msg,
            )
        except Exception:
            # Logging/manifest failures must not mask the core error.
            pass
        try:
            bem_intercept.finalize(intercept_ctx)
        except Exception:
            pass
        return payload

    def _maybe_poll_control() -> None:
        _poll_control_file(run_dir, control, logger)
        if control.terminate:
            raise _ControlSignal("Terminate requested via control.json")

    try:
        for rp in range(max_refine_passes):
            _maybe_poll_control()

            logger.info(
                f"Refine pass {rp + 1}: target h={current_h:.4f}",
                refine_pass=rp + 1,
                target_h=current_h,
            )

            try:
                mesh: TriMesh = generate_mesh(
                    spec,
                    target_h=current_h,
                    logger=logger,
                )
            except Exception as e:
                logger.error("Mesh generation failed.", error=str(e))
                break

            N = mesh.n_panels
            logger.info("Mesh built.", dof=N)

            if N == 0:
                logger.error("Generated mesh has zero panels; aborting.")
                break

            if int(getattr(cfg, "tile_size", 0)) > 0:
                tile_size = int(getattr(cfg, "tile_size", 0))
            else:
                target_peak_gb = getattr(cfg, "target_peak_gb", None)
                tile_mem_divisor = getattr(cfg, "tile_mem_divisor", 3.0)
                tile_size = _autotune_tile_size(
                    N,
                    dtype,
                    device,
                    logger,
                    target_vram_fraction=getattr(cfg, "target_vram_fraction", 0.8),
                    max_vram_gb=getattr(cfg, "vram_cap_gb", 24.0),
                    min_tile=getattr(cfg, "min_tile", 512),
                    max_tile=getattr(cfg, "max_tile", None),
                    target_peak_gb=target_peak_gb,
                    tile_mem_divisor=tile_mem_divisor,
                )

            C = torch.as_tensor(mesh.centroids, device=device, dtype=dtype)
            A = torch.as_tensor(mesh.areas, device=device, dtype=dtype)
            Nrm = torch.as_tensor(mesh.normals, device=device, dtype=dtype)

            bc = _bc_vector(spec, mesh, device, dtype)
            V_free = _free_space_potential_on_centroids(spec, C)

            # Basic sanity checks on BC and free-space potential.
            if not torch.isfinite(bc).all():
                n_bad = int((~torch.isfinite(bc)).sum().item())
                logger.error(
                    "BEM BC vector has non-finite entries.",
                    n_bad=n_bad,
                    max_abs=float(
                        torch.nan_to_num(
                            torch.abs(bc), nan=0.0, posinf=0.0, neginf=0.0
                        ).max().item()
                    ),
                )
                return _error_result("BEM BC vector contains NaN/Inf.")

            if not torch.isfinite(V_free).all():
                n_bad = int((~torch.isfinite(V_free)).sum().item())
                logger.error(
                    "Free-space potential on centroids has non-finite entries.",
                    n_bad=n_bad,
                    max_abs=float(
                        torch.nan_to_num(
                            torch.abs(V_free), nan=0.0, posinf=0.0, neginf=0.0
                        ).max().item()
                    ),
                )
                return _error_result(
                    "Free-space potential on centroids contains NaN/Inf."
                )

            # Self-panel diagonal:
            # - self_integrals: raw self integral I_self(A) = ∫_panel G dS
            # - self_corr:     per-area diagonal K_ii = I_self / A
            self_integrals = torch.empty(N, device=device, dtype=dtype)
            for i in range(N):
                self_integrals[i] = self_integral_correction(A[i])

            A_safe = A.clamp_min(torch.finfo(dtype).tiny)
            self_corr = self_integrals / A_safe

            # Near-field quadrature setup (panel geometry for matvec).
            use_near_quad_matvec = bool(
                getattr(cfg, "use_near_quadrature_matvec", False)
            )
            near_quad_order = int(
                getattr(cfg, "near_quadrature_order", 2)
            )
            near_quad_dist_factor = float(
                getattr(cfg, "near_quadrature_distance_factor", 1.5)
            )

            panel_vertices_np: np.ndarray | None = None
            near_pairs_np: np.ndarray = np.zeros((0, 2), dtype=np.int64)
            near_pair_weights_np: Optional[np.ndarray] = None
            # Optional CUDA-side / Torch-side backing arrays for near-field data.
            panel_vertices_cuda: Optional[torch.Tensor] = None
            near_pairs_cuda: Optional[torch.Tensor] = None
            near_pair_weights_torch: Optional[torch.Tensor] = None
            near_pairs_torch: Optional[torch.Tensor] = None
            use_near_quad_matvec_cuda = False

            if use_near_quad_matvec and N > 1:
                # Sanity: TriMesh invariants should guarantee these shapes.
                assert mesh.centroids.shape[0] >= N and mesh.areas.shape[0] >= N, (
                    "TriMesh inconsistent: centroids/areas length mismatch "
                    f"(C={mesh.centroids.shape[0]}, "
                    f"A={mesh.areas.shape[0]}, N={N})"
                )
                try:
                    # Ensure triangles are an integer array and in-bounds.
                    tris = np.asarray(mesh.triangles, dtype=np.int64)
                    verts = np.asarray(mesh.vertices, dtype=float)
                    panel_vertices_np = verts[tris]

                    # Build robust near pairs using the explicit panel count N.
                    near_pairs_np = _build_near_pairs_for_panels(
                        mesh.centroids,
                        mesh.areas,
                        distance_factor=near_quad_dist_factor,
                        n_panels=N,
                    )

                    if near_pairs_np.size == 0:
                        use_near_quad_matvec = False
                    else:
                        logger.info(
                            "Precomputing near-field matvec geometry corrections.",
                            dof=int(N),
                            n_pairs=int(near_pairs_np.shape[0]),
                            quad_order=int(near_quad_order),
                        )
                        near_pair_weights_np = _precompute_near_correction_weights(
                            mesh.centroids,
                            mesh.areas,
                            panel_vertices_np,
                            near_pairs_np,
                            quad_order=near_quad_order,
                        )

                        # Torch mirrors for fast matvec correction (CPU or GPU).
                        try:
                            near_pairs_torch = torch.as_tensor(
                                near_pairs_np, device=device, dtype=torch.long
                            )
                            near_pair_weights_torch = torch.as_tensor(
                                near_pair_weights_np, device=device, dtype=dtype
                            )
                        except Exception as exc:
                            logger.warning(
                                "Failed to build Torch near-field buffers; "
                                "falling back to NumPy near-field path.",
                                error=str(exc),
                            )
                            near_pairs_torch = None
                            near_pair_weights_torch = None

                        if device.type == "cuda":
                            # Try to prepare CUDA buffers and use bem_near_cuda.
                            if _has_bem_near_cuda():
                                try:
                                    panel_vertices_cuda = torch.as_tensor(
                                        panel_vertices_np,
                                        device=device,
                                        dtype=dtype,
                                    )
                                    near_pairs_cuda = near_pairs_torch
                                    use_near_quad_matvec_cuda = True
                                    logger.info(
                                        "Near-field quadrature for matvec "
                                        "enabled on CUDA via bem_near_cuda.",
                                        dof=int(N),
                                        n_pairs=int(near_pairs_np.shape[0]),
                                    )
                                except Exception as exc:
                                    panel_vertices_cuda = None
                                    near_pairs_cuda = None
                                    use_near_quad_matvec_cuda = False
                                    logger.warning(
                                        "Failed to move near-field quadrature "
                                        "data to CUDA; falling back to Torch/CPU "
                                        "near corrections.",
                                        error=str(exc),
                                    )
                            if not use_near_quad_matvec_cuda:
                                logger.info(
                                    "Near-field quadrature for matvec enabled: "
                                    "far-field matvec on device, Torch near-field "
                                    "correction (no C++ extension).",
                                    device=str(device),
                                    dof=int(N),
                                    n_pairs=int(near_pairs_np.shape[0]),
                                )
                        else:
                            logger.info(
                                "Near-field quadrature for matvec enabled on CPU.",
                                dof=int(N),
                                n_pairs=int(near_pairs_np.shape[0]),
                            )
                except Exception as exc:
                    logger.warning(
                        "Failed to set up near-field quadrature; "
                        "continuing without it.",
                        error=str(exc),
                        exc_type=type(exc).__name__,
                        traceback=traceback.format_exc(),
                    )
                    panel_vertices_np = None
                    near_pairs_np = np.zeros((0, 2), dtype=np.int64)
                    near_pair_weights_np = None
                    panel_vertices_cuda = None
                    near_pairs_cuda = None
                    near_pairs_torch = None
                    near_pair_weights_torch = None
                    use_near_quad_matvec = False
                    use_near_quad_matvec_cuda = False
            elif use_near_quad_matvec:
                # N <= 1, nothing to correct.
                use_near_quad_matvec = False
                use_near_quad_matvec_cuda = False

            def _matvec_sigma_nondiff_base(sig: torch.Tensor) -> torch.Tensor:
                # Far field via the standard tiled kernel.
                V_far = bem_matvec_gpu(
                    sigma=sig,
                    src_centroids=C,
                    areas=A,
                    tile_size=tile_size,
                    self_integrals=self_corr,
                    use_near_quad=False,
                )
                # Optional near-field correction using triangle quadrature.
                if (
                    use_near_quad_matvec
                    and near_pairs_np.size > 0
                ):
                    if (
                        use_near_quad_matvec_cuda
                        and panel_vertices_cuda is not None
                        and near_pairs_cuda is not None
                    ):
                        V_far = _apply_near_quadrature_matvec_cuda(
                            V_far,
                            sig,
                            centroids=C,
                            areas=A,
                            panel_vertices=panel_vertices_cuda,
                            near_pairs=near_pairs_cuda,
                            quad_order=near_quad_order,
                            panel_vertices_np=panel_vertices_np,
                            near_pairs_np=near_pairs_np,
                            near_pair_weights_np=near_pair_weights_np,
                        )
                    elif (
                        near_pairs_torch is not None
                        and near_pair_weights_torch is not None
                    ):
                        V_far = _apply_near_quadrature_matvec_torch(
                            V_far,
                            sig,
                            near_pairs=near_pairs_torch,
                            weights=near_pair_weights_torch,
                        )
                    elif panel_vertices_np is not None:
                        V_far = _apply_near_quadrature_matvec(
                            V_far,
                            sig,
                            centroids=C,
                            areas=A,
                            panel_vertices=panel_vertices_np,
                            near_pairs=near_pairs_np,
                            quad_order=near_quad_order,
                            near_pair_weights=near_pair_weights_np,
                        )
                return V_far

            def _matvec_sigma_diff_base(sig: torch.Tensor) -> torch.Tensor:
                return _bem_matvec_core_torch(
                    centroids=C,
                    areas=A,
                    sigma=sig,
                    self_integrals=self_corr,
                    tile_size=tile_size,
                )

            alpha = float(getattr(cfg, "near_alpha", 0.0))
            if alpha > 0.0:
                k = int(getattr(cfg, "near_k", 8))
                k = max(1, min(k, max(1, N - 1)))
                with torch.no_grad():
                    dists = torch.cdist(C, C, p=2)
                    idx = torch.arange(N, device=device)
                    dists[idx, idx] = float("inf")
                    near_idx = torch.topk(dists, k=k, largest=False, dim=1).indices
            else:
                near_idx = None

            def _matvec_with_near(
                base_mv: Callable[[torch.Tensor], torch.Tensor],
                sig: torch.Tensor,
            ) -> torch.Tensor:
                v = base_mv(sig)
                if near_idx is None or alpha <= 0.0 or sig.numel() == 0:
                    return v
                idx_l = near_idx
                idx_clamped = idx_l.clamp_min(0).clamp_max(sig.shape[0] - 1)
                neigh = v.index_select(0, idx_clamped.view(-1)).view(idx_clamped.shape)
                neigh_mean = neigh.mean(dim=1)
                return (1.0 - alpha) * v + alpha * neigh_mean

            def matvec_sigma_nondiff(sig: torch.Tensor) -> torch.Tensor:
                return _matvec_with_near(_matvec_sigma_nondiff_base, sig)

            def matvec_sigma_diff(sig: torch.Tensor) -> torch.Tensor:
                return _matvec_with_near(_matvec_sigma_diff_base, sig)

            # 7) RHS and initial guess
            b = bc - V_free
            # Use the raw self-integral I_self(A) as an approximation to the
            # matrix diagonal; self_corr contains the per-area K_ii entries.
            diag = self_integrals.clamp_min(1e-18)
            x0 = b / diag

            # Sanity checks to catch NaN/Inf before GMRES
            b_finite = torch.isfinite(b)
            diag_finite = torch.isfinite(diag)
            x0_finite = torch.isfinite(x0)

            if not b_finite.all():
                n_bad = int((~b_finite).sum().item())
                logger.error(
                    "BEM RHS has non-finite entries.",
                    n_bad=n_bad,
                    max_abs=float(
                        torch.nan_to_num(
                            torch.abs(b), nan=0.0, posinf=0.0, neginf=0.0
                        ).max().item()
                    ),
                )
                return _error_result("BEM RHS contains NaN/Inf.")

            if not diag_finite.all():
                n_bad = int((~diag_finite).sum().item())
                logger.error(
                    "BEM diagonal has non-finite entries.",
                    n_bad=n_bad,
                )
                return _error_result(
                    "BEM diagonal (self-integrals) contains NaN/Inf."
                )

            if not x0_finite.all():
                n_bad = int((~x0_finite).sum().item())
                logger.warning(
                    "Initial guess x0 has non-finite entries; resetting to zeros.",
                    n_bad=n_bad,
                )
                x0 = torch.zeros_like(b)

            # 7b) GMRES callback with control + telemetry
            def _gmres_callback(
                iter_idx: int,
                resid_norm: float,
                ctx: Dict[str, Any],
            ) -> None:
                try:
                    _poll_control_file(run_dir, control, logger)
                    if control.terminate:
                        raise _ControlSignal(
                            "Terminate requested via control.json during GMRES."
                        )

                    tile_sz = int(ctx.get("tile_size", tile_size))
                    pass_idx = int(ctx.get("pass_index", rp + 1))

                    extra = {
                        k: v
                        for k, v in (ctx or {}).items()
                        if k not in ("iter", "resid", "tile_size", "pass_index")
                    }

                    logger.debug(
                        "GMRES iter.",
                        iter=int(iter_idx),
                        resid=float(resid_norm),
                        tile_size=tile_sz,
                        dof=int(N),
                        pass_index=pass_idx,
                        write_every=control.write_every,
                        **extra,
                    )
                except _ControlSignal:
                    raise
                except Exception:
                    pass

            # 8) Linear solve
            if not differentiable:
                use_precond = use_precond_cfg
                precond_label: Optional[str] = "jacobi" if use_precond else None

                try:
                    sigma, info = gmres_restart(
                        matvec_sigma_nondiff,
                        b,
                        restart=gmres_restart_val,
                        tol=gmres_tol_val,
                        maxiter=gmres_maxiter_val,
                        precond=precond_label,
                        areas=A,
                        A_diag=diag,
                        logger=logger,
                        x0=x0,
                        callback=_gmres_callback,
                        callback_context={
                            "tile_size": int(tile_size),
                            "pass_index": int(rp + 1),
                        },
                        log_every=gmres_log_every_val,
                    )
                except _ControlSignal as cs:
                    logger.warning(
                        "GMRES aborted via control signal.",
                        reason=str(cs),
                    )
                    return _error_result(
                        f"BEM solve aborted via control: {cs}"
                    )
                except Exception as e:
                    logger.error("GMRES failed.", error=str(e))
                    return _error_result(f"GMRES failed: {e}")

                # Inspect GMRES outcome; optionally retry without preconditioner,
                # then fall back to Jacobi if still unsuccessful.
                resid_val = info.get("resid", float("inf"))
                try:
                    resid_float = float(resid_val)
                except Exception:
                    resid_float = float("inf")
                success_flag = bool(info.get("success", True))
                code_val = int(info.get("code", 0) or 0)

                # If GMRES with preconditioner fails or produces a non-finite
                # residual, retry once without any preconditioner before
                # falling back to Jacobi.
                if precond_label is not None and (
                    (not success_flag) or (not math.isfinite(resid_float))
                ):
                    logger.warning(
                        "GMRES reported failure or non-finite residual with "
                        "preconditioner; retrying without preconditioner.",
                        gmres_resid=resid_float,
                        gmres_code=code_val,
                    )
                    try:
                        sigma2, info2 = gmres_restart(
                            matvec_sigma_nondiff,
                            b,
                            restart=gmres_restart_val,
                            tol=gmres_tol_val,
                            maxiter=gmres_maxiter_val,
                            precond=None,
                            areas=A,
                            A_diag=diag,
                            logger=logger,
                            x0=x0,
                            callback=_gmres_callback,
                            callback_context={
                                "tile_size": int(tile_size),
                                "pass_index": int(rp + 1),
                            },
                            log_every=gmres_log_every_val,
                        )
                        sigma, info = sigma2, info2
                        resid_val = info.get("resid", float("inf"))
                        try:
                            resid_float = float(resid_val)
                        except Exception:
                            resid_float = float("inf")
                        success_flag = bool(info.get("success", True))
                        code_val = int(info.get("code", 0) or 0)
                    except _ControlSignal as cs:
                        logger.warning(
                            "GMRES aborted via control signal on retry.",
                            reason=str(cs),
                        )
                        return _error_result(
                            f"BEM solve aborted via control: {cs}"
                        )
                    except Exception as e:
                        logger.error(
                            "GMRES retry without preconditioner failed.",
                            error=str(e),
                        )
                        return _error_result(
                            f"GMRES failed on retry without preconditioner: {e}"
                        )

                if (not success_flag) or (not math.isfinite(resid_float)):
                    use_jacobi = allow_jacobi_cfg
                    if use_jacobi:
                        logger.warning(
                            "GMRES reported failure; attempting Jacobi fallback.",
                            gmres_resid=resid_float,
                            gmres_code=code_val,
                        )
                        sigma_j, info_j = _jacobi_fallback_solve(
                            A=matvec_sigma_nondiff,
                            b=b,
                            diag=diag,
                            logger=logger,
                            maxiter=jacobi_maxiter_cfg,
                            tol=jacobi_tol_cfg,
                            omega=jacobi_omega_cfg,
                        )
                        resid_j = info_j.get("resid", float("inf"))
                        try:
                            resid_j_float = float(resid_j)
                        except Exception:
                            resid_j_float = float("inf")

                        if bool(info_j.get("success", False)) and math.isfinite(
                            resid_j_float
                        ):
                            sigma = sigma_j
                            info = info_j
                            logger.info(
                                "Jacobi fallback succeeded.",
                                resid=resid_j_float,
                                iters=info_j.get("iters"),
                            )
                        else:
                            logger.error(
                                "Jacobi fallback failed after GMRES failure.",
                                gmres_resid=resid_float,
                                gmres_code=code_val,
                                jacobi_resid=resid_j_float,
                                jacobi_iters=info_j.get("iters"),
                            )
                            return _error_result(
                                (
                                    "BEM linear solve failed: "
                                    f"GMRES(code={code_val}, resid={resid_float}) "
                                    f"and Jacobi(resid={resid_j_float})"
                                ),
                                gmres_resid=resid_float,
                                gmres_code=code_val,
                                jacobi_resid=resid_j_float,
                                jacobi_iters=info_j.get("iters"),
                            )
                    else:
                        logger.error(
                            "BEM linear solve failed; GMRES reported failure and "
                            "Jacobi fallback is disabled.",
                            gmres_resid=resid_float,
                            gmres_code=code_val,
                        )
                        return _error_result(
                            (
                                "BEM linear solve failed: "
                                f"GMRES(code={code_val}, resid={resid_float})"
                            ),
                            gmres_resid=resid_float,
                            gmres_code=code_val,
                        )
            else:
                diffbem = _get_diffbem_module()
                if diffbem is None:
                    msg = (
                        "differentiable=True requested for bem_solve, "
                        "but electrodrive.core.diffbem or xitorch is unavailable."
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)

                diff_out = diffbem.solve_diffbem(
                    spec,
                    cfg,
                    logger,
                    C=C,
                    N=Nrm,
                    A=A,
                    rhs=b,
                    matvec=matvec_sigma_diff,
                    x0=x0,
                )
                sigma = diff_out["sigma"]
                info = dict(diff_out.get("stats", {}))
                info.setdefault("solver", "xitorch_gmres")

                resid_val = info.get("resid", None)
                if resid_val is not None:
                    try:
                        resid_float = float(resid_val)
                    except Exception:
                        resid_float = float("inf")
                else:
                    resid_float = 0.0
                success_flag = bool(info.get("success", True))
                if (not success_flag) or (not math.isfinite(resid_float)):
                    logger.error(
                        "Differentiable BEM linear solve reported failure.",
                        resid=resid_float,
                        info=info,
                    )
                    return _error_result(
                        (
                            "Differentiable BEM linear solve failed: "
                            f"resid={resid_float}"
                        ),
                        resid=resid_float,
                        info=info,
                    )

            logger.info(
                "BEM linear solve done.",
                iters=info.get("iters"),
                resid=info.get("resid"),
                success=info.get("success", True),
                differentiable=bool(differentiable),
            )

            # 9) Post-process: induced potential at centroids and BC residual
            sigma_finite = torch.isfinite(sigma)
            if not sigma_finite.all():
                n_bad = int((~sigma_finite).sum().item())
                logger.error(
                    "Surface charge density contains non-finite entries.",
                    n_bad=n_bad,
                )
                return _error_result(
                    "BEM sigma contains NaN/Inf after linear solve."
                )

            if not differentiable:
                V_ind = matvec_sigma_nondiff(sigma)
            else:
                V_ind = matvec_sigma_diff(sigma)

            V_tot = torch.nan_to_num(
                V_free + V_ind,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            bc_resid_linf = float(torch.max(torch.abs(V_tot - bc)).item())

            sigma_sane = torch.nan_to_num(
                sigma,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            energy_A = compute_bem_capacitive_energy(V_tot, sigma_sane, A)

            logger.info(
                "Pass results.",
                bc_residual_linf=bc_resid_linf,
                energy_A=f"{energy_A:.6e}",
            )

            pass_payload: Dict[str, Any] = {
                "pass": rp + 1,
                "h": current_h,
                "dof": N,
                "bc_resid_linf": bc_resid_linf,
                "energy_A": energy_A,
                "gmres_info": info,
                "tile_size": tile_size,
                "artifacts": (
                    spec,
                    C,
                    A,
                    Nrm,
                    sigma,
                    V_tot,
                    mesh,
                ),
            }
            history.append(pass_payload)
            try:
                bem_intercept.record_bem_pass(
                    intercept_ctx,
                    {
                        "pass": rp + 1,
                        "h": current_h,
                        "n_panels": N,
                        "tile_size": tile_size,
                        "bc_resid_linf": bc_resid_linf,
                        "gmres_iters": info.get("iters"),
                        "gmres_resid_true": info.get("resid"),
                        "gmres_tol_abs": info.get("tol_abs"),
                        "gmres_code": info.get("code"),
                        "gmres_used_precond": info.get("used_preconditioner", False),
                        "device": str(device),
                        "dtype": str(dtype),
                        "plateau": plateau,
                        "near_quad_eval": bool(getattr(cfg, "use_near_quadrature", False)),
                    },
                )
            except Exception:
                pass

            if bc_resid_linf < best_bc:
                improvement = (
                    best_bc - bc_resid_linf
                    if math.isfinite(best_bc)
                    else float("inf")
                )
                best = pass_payload
                best_bc = bc_resid_linf
                plateau = (
                    0
                    if improvement
                    > max(
                        1e-16,
                        plateau_rel_improvement * max(best_bc, 1e-16),
                    )
                    else plateau + 1
                )
            else:
                plateau += 1

            stop_by_bc = bc_resid_linf <= target_bc
            have_min_panels = N >= int(getattr(cfg, "min_panels", 0))
            have_min_passes = (rp + 1) >= int(getattr(cfg, "min_refine_passes", 1))

            if stop_by_bc and have_min_panels and have_min_passes:
                logger.info(
                    "Target BC residual reached and min_panels/min_passes satisfied.",
                    min_panels=int(getattr(cfg, "min_panels", 0)),
                    passes=rp + 1,
                    dof=N,
                )
                stop_reason = "target_bc"
                break

            if plateau >= max_plateau and have_min_panels and have_min_passes:
                logger.info(
                    "Plateau detected; stopping refinement to avoid long runs.",
                    plateau=plateau,
                )
                stop_reason = "plateau"
                break

            current_h *= refine_factor

    except _ControlSignal as cs_outer:
        logger.warning(
            "BEM solve aborted via control signal.",
            reason=str(cs_outer),
        )
        if not history:
            return _error_result(f"BEM solve aborted via control: {cs_outer}")

    if stop_reason == "unset":
        stop_reason = "max_passes_reached"

    # Fallback if no best pass was selected
    if best is None:
        if not history:
            logger.error("BEM solve produced no passes.")
            return _error_result("BEM solve failed (no passes)")

        # Prefer passes whose linear solve actually succeeded and has finite residual.
        def _pass_ok(p: Dict[str, Any]) -> bool:
            info_p = p.get("gmres_info") or {}
            success_p = bool(info_p.get("success", True))
            resid_p = info_p.get("resid", None)
            if resid_p is None:
                return success_p
            try:
                resid_f = float(resid_p)
            except Exception:
                resid_f = float("inf")
            return success_p and math.isfinite(resid_f)

        good_passes = [p for p in history if _pass_ok(p)]
        if not good_passes:
            last_info = history[-1].get("gmres_info", {})
            logger.error(
                "BEM solve produced passes but none had a successful linear solve.",
                last_gmres_info=last_info,
            )
            return _error_result(
                "BEM solve failed (no successful linear solves)",
                gmres_last=last_info,
            )

        best = min(good_passes, key=lambda p: p.get("bc_resid_linf", float("inf")))
        logger.warning(
            "Using best successful pass for diagnostics.",
            dof=best["dof"],
            bc_resid_linf=best.get("bc_resid_linf"),
        )

    (
        spec_b,
        C_b,
        A_b,
        Nrm_b,
        sigma_b,
        Vtot_b,
        mesh_b,
    ) = best["artifacts"]
    N_b = best["dof"]
    tile_b = best["tile_size"]

    # Panel vertices for near-field evaluation quadrature on the final mesh.
    panel_vertices_b: Optional[torch.Tensor] = None
    use_near_quad_eval = bool(getattr(cfg, "use_near_quadrature", False))
    if use_near_quad_eval and N_b > 0:
        try:
            verts_b = torch.as_tensor(
                mesh_b.vertices, device=C_b.device, dtype=C_b.dtype
            )
            tris_b = torch.as_tensor(
                mesh_b.triangles, device=C_b.device, dtype=torch.long
            )
            panel_vertices_b = verts_b[tris_b]
        except Exception as exc:
            use_near_quad_eval = False
            panel_vertices_b = None
            logger.warning(
                "Failed to build panel vertices for near-field evaluation; "
                "continuing without near quadrature at targets.",
                error=str(exc),
            )

    solution = BEMSolution(
        spec_b,
        C_b,
        A_b,
        sigma_b,
        C_b.device,
        C_b.dtype,
        tile_b,
        normals=Nrm_b,
        differentiable=bool(differentiable),
        panel_vertices=panel_vertices_b,
        near_quadrature=use_near_quad_eval,
        near_quad_order=int(getattr(cfg, "near_quadrature_order", 2)),
        near_quad_dist_factor=float(
            getattr(cfg, "near_quadrature_distance_factor", 1.5)
        ),
    )
    solution.meta["energy_A"] = best["energy_A"]
    solution.meta["bem_vram_config"] = {
        "target_peak_gb": float(getattr(cfg, "target_peak_gb", 0.0) or 0.0),
        "tile_mem_divisor": float(getattr(cfg, "tile_mem_divisor", 3.0) or 3.0),
        "fp64": bool(getattr(cfg, "fp64", False)),
    }

    if N_b <= 0:
        sample_points: List[List[float]] = []
        boundary_samples: List[float] = []
    else:
        n_samples = min(1024, max(1, N_b))
        if n_samples == 1:
            idx = torch.tensor([0], device=C_b.device, dtype=torch.long)
        else:
            step = max(1, N_b // n_samples)
            idx = torch.arange(0, N_b, step, device=C_b.device, dtype=torch.long)[
                :n_samples
            ]

        sample_points = C_b[idx].detach().cpu().numpy().tolist()
        boundary_samples = Vtot_b[idx].detach().cpu().numpy().tolist()

    patch_L: Optional[float] = None
    try:
        # Use the best-pass spec explicitly; for current workflows this is
        # identical to the input spec, but it makes the intent clearer.
        if any(c.get("type") == "plane" for c in spec_b.conductors):
            total_area = float(A_b.sum().item()) if N_b > 0 else 0.0
            if total_area > 0.0:
                patch_L = math.sqrt(total_area)
                logger.info(
                    "Plane patch extent recorded.",
                    total_area=f"{total_area:.6f}",
                    patch_L=f"{patch_L:.6f}",
                )
            else:
                logger.warning("Plane patch extent unavailable (zero area).")
    except Exception as exc:
        logger.warning("Failed computing patch extent.", error=str(exc))
        patch_L = None

    gpu_peak_mb = _sample_gpu_peak_mb()

    mesh_stats: Dict[str, Any] = {
        "n_panels": N_b,
        "total_area": float(A_b.sum().item()) if N_b > 0 else 0.0,
        "bc_residual_linf": float(best["bc_resid_linf"]),
        "h_final": float(best["h"]),
        "tile_size_final": int(tile_b),
        "gpu_mem_peak_mb": gpu_peak_mb,
    }
    if patch_L is not None:
        mesh_stats["patch_L"] = patch_L
        try:
            if intercept_ctx is not None:
                for p in intercept_ctx.payload.get("refinement_passes", []):
                    if int(p.get("pass", -1)) == int(best.get("pass", -1)):
                        p["patch_L"] = patch_L
        except Exception:
            pass

    gmres_stats = dict(best.get("gmres_info", {}))
    gmres_stats.setdefault("gpu_mem_peak_mb", gpu_peak_mb)

    out: Dict[str, Any] = {
        "boundary_samples": boundary_samples,
        "sample_points": sample_points,
        "solution": solution,
        "surface_charge_density": sigma_b.detach().cpu().numpy(),
        "mesh_stats": mesh_stats,
        "gmres_stats": gmres_stats,
        "refinement_history": history,
    }

    try:
        _write_manifest(
            run_dir,
            run_id=run_id,
            device=device,
            dtype=dtype,
            requested_mode="bem",
            selected_mode="bem",
            planner_rationale="bem_solve_direct",
        )
    except Exception:
        pass

    try:
        bem_intercept.set_stop_reason(intercept_ctx, stop_reason)
        bem_intercept.finalize(intercept_ctx)
    except Exception:
        pass

    return out
