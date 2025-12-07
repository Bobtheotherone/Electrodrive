from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import hashlib
import os

import numpy as np
import torch

from electrodrive.images.basis import (
    ImageBasisElement,
    generate_candidate_basis,
    build_dictionary,
)
# NOTE: images currently depends on learn.collocation for collocation sampling.
# In a future refactor, this should move to a shared core.collocation module.
from electrodrive.learn.collocation import make_collocation_batch_for_spec
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.logging import JsonlLogger


class ImageSystem:
    """A concrete image system: basis elements + weights."""

    def __init__(self, elements: List[ImageBasisElement], weights: torch.Tensor):
        self.elements = elements
        self.weights = weights
        if weights.numel() > 0:
            self.device = weights.device
            self.dtype = weights.dtype
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32

    def potential(self, targets: torch.Tensor) -> torch.Tensor:
        """Evaluate the image-system potential at a batch of points."""
        V = torch.zeros(
            targets.shape[0],
            device=targets.device,
            dtype=targets.dtype,
        )
        for elem, w in zip(self.elements, self.weights):
            V = V + w.to(targets.dtype) * elem.potential(targets)
        return V


def _make_collocation_rng() -> np.random.Generator:
    """Deterministic RNG for image-discovery collocation sampling.

    Uses a fixed base seed and optionally folds in EDE_RUN_ID to keep
    runs reproducible but still vary across run IDs.
    """
    base_seed = 12345
    run_id = os.getenv("EDE_RUN_ID", "")
    if run_id:
        h = hashlib.sha1(run_id.encode("utf-8")).digest()
        run_hash = int.from_bytes(h[:8], "little") & 0xFFFFFFFF
        seed = (base_seed ^ run_hash) & 0xFFFFFFFF
    else:
        seed = base_seed
    return np.random.default_rng(seed)


def get_collocation_data(
    spec: CanonicalSpec,
    logger: JsonlLogger,
    device: torch.device,
    dtype: torch.dtype,
    return_is_boundary: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build collocation points + targets for a given spec.

    This delegates to :func:`make_collocation_batch_for_spec` from the
    learning stack so that analytic shortcuts and BEM fallbacks are
    shared between training and the sparse image discovery path.

    Returns
    -------
    colloc_pts : torch.Tensor
        [N, 3] collocation points.
    target : torch.Tensor
        [N] oracle potential values.
    is_boundary : torch.Tensor (optional)
        [N] boolean mask of boundary points when requested.
    """
    device = torch.device(device)

    # Allow light-weight overrides for experimentation via environment.
    try:
        n_points_env = int(os.getenv("EDE_IMAGES_N_POINTS", "0"))
        n_points = n_points_env if n_points_env > 0 else 512
    except Exception:
        n_points = 512

    try:
        ratio_env = float(os.getenv("EDE_IMAGES_RATIO_BOUNDARY", "nan"))
        ratio_boundary = ratio_env if 0.0 < ratio_env < 1.0 else 0.5
    except Exception:
        ratio_boundary = 0.5

    rng = _make_collocation_rng()

    try:
        batch = make_collocation_batch_for_spec(
            spec=spec,
            n_points=n_points,
            ratio_boundary=ratio_boundary,
            supervision_mode="auto",
            device=device,
            dtype=dtype,
            rng=rng,
        )
    except Exception as e:  # defensive path
        logger.error(
            "Collocation batch construction failed.",
            error=str(e),
        )
        return (
            torch.empty(0, 3, device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=dtype),
        )

    X = batch.get("X")
    V = batch.get("V_gt")

    if X is None or V is None or X.numel() == 0 or V.numel() == 0:
        logger.error(
            "Collocation helper returned an empty batch.",
            n_points_requested=int(n_points),
        )
        return (
            torch.empty(0, 3, device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=dtype),
        )

    mask_finite = batch.get("mask_finite")
    if mask_finite is not None and mask_finite.shape == (X.shape[0],):
        mask = mask_finite.to(device=device, dtype=torch.bool) & torch.isfinite(V)
    else:
        mask = torch.isfinite(V)

    if not mask.any():
        logger.error(
            "Collocation batch has no finite targets.",
            n_points_total=int(X.shape[0]),
        )
        return (
            torch.empty(0, 3, device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=dtype),
        )

    X_f = X[mask].to(device=device, dtype=dtype)
    V_f = V[mask].to(device=device, dtype=dtype)
    is_boundary_out: torch.Tensor | None = None

    N = int(X_f.shape[0])

    # Estimate boundary fraction if available.
    n_boundary = 0
    is_boundary = batch.get("is_boundary")
    if return_is_boundary and is_boundary is not None and is_boundary.shape == (X.shape[0],):
        is_boundary = is_boundary.to(device=device)
        n_boundary = int(is_boundary[mask].sum().item())
        is_boundary_out = is_boundary[mask]

    frac_boundary = float(n_boundary) / float(N) if N > 0 else 0.0

    V_min = float(V_f.min().item()) if N > 0 else float("nan")
    V_max = float(V_f.max().item()) if N > 0 else float("nan")

    logger.info(
        "Collocation data prepared.",
        n_points=N,
        n_boundary=n_boundary,
        frac_boundary=frac_boundary,
        V_min=V_min,
        V_max=V_max,
    )

    # Backward-compatible return: if no boundary mask requested, return two tensors.
    if is_boundary_out is None:
        return X_f, V_f
    return X_f, V_f, is_boundary_out


def assemble_basis_matrix(
    basis_set: List[ImageBasisElement],
    points: torch.Tensor,
) -> torch.Tensor:
    """Assemble A[N,K] with columns A[:, k] = basis_set[k].potential(points).

    This delegates to :func:`build_dictionary` in electrodrive.images.basis
    so that basis evaluation stays centralized.
    """
    return build_dictionary(
        basis_set,
        points,
        device=points.device,
        dtype=points.dtype,
    )


def solve_l1_ista(
    A: torch.Tensor,
    g: torch.Tensor,
    reg_l1: float,
    logger: JsonlLogger,
    max_iter: int = 1000,
    tol: float = 1e-6,
    per_elem_reg: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[int]]:
    """Solve the L1-regularised least-squares problem via ISTA.

    Minimises 0.5 * ||A w - g||_2^2 + reg_l1 * ||w||_1.

    The implementation is intentionally simple and robust rather than
    aggressively tuned; it is shared between all geometries and does not
    hard-code any problem-specific structure.
    """
    if A.numel() == 0 or g.numel() == 0:
        # Nothing to do; construct a correctly-shaped zero vector.
        K = A.shape[1] if A.ndim == 2 else 0
        return torch.zeros(K, device=A.device, dtype=A.dtype), []

    N_k = A.shape[1]

    # Estimate the Lipschitz constant L of A^T A using either SVD or a
    # short power iteration, falling back cleanly on failure.
    try:
        _, S, _ = torch.linalg.svd(A, full_matrices=False)
        L = float(S[0] ** 2) if S.numel() > 0 else 1.0
    except Exception:  # rarely hit in practice
        x = torch.randn(N_k, device=A.device, dtype=A.dtype)
        for _ in range(20):
            x = A.T @ (A @ x)
            n = torch.linalg.norm(x)
            if float(n) < 1e-9:
                break
            x = x / n
        L = float(torch.linalg.norm(A.T @ (A @ x))) or 1.0

    if L <= 0.0:
        logger.warning("ISTA: non-positive Lipschitz estimate, aborting.")
        return torch.zeros(N_k, device=A.device, dtype=A.dtype), []

    alpha = 1.0 / L
    thr = reg_l1 * alpha
    thr_vec: Optional[torch.Tensor] = None
    if per_elem_reg is not None:
        thr_vec = per_elem_reg.to(device=A.device, dtype=A.dtype) * alpha
    w = torch.zeros(N_k, device=A.device, dtype=A.dtype)

    last_rel_change: float = float("inf")

    for it in range(max_iter):
        w_prev = w.clone()
        r = A @ w - g
        grad = A.T @ r
        w = w - alpha * grad
        # Soft-thresholding (L1 proximal step).
        if thr_vec is not None:
            w = torch.sign(w) * torch.clamp(torch.abs(w) - thr_vec, min=0.0)
        else:
            w = torch.sign(w) * torch.clamp(torch.abs(w) - thr, min=0.0)
        num = float(torch.linalg.norm(w - w_prev))
        den = float(torch.linalg.norm(w) + 1e-9)
        if den > 0.0:
            last_rel_change = num / den
        if den > 0.0 and last_rel_change < tol:
            logger.info(
                "ISTA converged.",
                iters=int(it + 1),
                rel_change=float(last_rel_change),
            )
            break
    else:
        logger.warning(
            "ISTA did not converge.",
            max_iter=int(max_iter),
            final_rel_change=float(last_rel_change),
        )

    # Determine support using a relative threshold so that rescaling A or
    # g does not spuriously drive all coefficients below a fixed cutoff.
    w_abs = torch.abs(w)
    if w_abs.numel() == 0:
        support: List[int] = []
    else:
        max_abs = float(w_abs.max().item())
        if max_abs == 0.0:
            support = []
        else:
            rel_tol = 1e-6
            abs_tol = 1e-12
            thr_support = max(abs_tol, rel_tol * max_abs)
            support = torch.where(w_abs > thr_support)[0].tolist()

    return w, support


def optimize_parameters_lbfgs(
    system: ImageSystem,
    points: torch.Tensor,
    g: torch.Tensor,
    logger: JsonlLogger,
) -> ImageSystem:
    """
    Optional second-stage refinement using L-BFGS.

    We treat the image weights and any float Tensor-valued parameters in
    each ImageBasisElement (for example, point-charge positions) as
    optimization variables. The objective is the mean-squared error
    between the image-system potential and the oracle targets ``g`` on
    the supplied collocation points.

    This routine is deliberately defensive: it never raises to callers
    and falls back to the incoming system on failure.
    """
    # Trivial early-exit guards.
    if system.weights.numel() == 0:
        logger.info("L-BFGS skipped: empty image system.")
        return system
    if points.numel() == 0 or g.numel() == 0:
        logger.info("L-BFGS skipped: no collocation data passed in.")
        return system

    device = system.weights.device
    dtype = system.weights.dtype
    points = points.to(device=device, dtype=dtype)
    g = g.to(device=device, dtype=dtype)

    # Clone weights so we do not backprop through the ISTA step.
    w = system.weights.detach().clone().to(device=device, dtype=dtype)
    w.requires_grad_(True)
    system.weights = w

    params: List[torch.Tensor] = [w]

    # Promote any floating-point Tensor parameters inside basis elements.
    for elem in system.elements:
        new_params: Dict[str, torch.Tensor] = {}
        for name, value in elem.params.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                p = value.detach().clone().to(device=device, dtype=dtype)
                p.requires_grad_(True)
                new_params[name] = p
                params.append(p)
            else:
                new_params[name] = value
        elem.params = new_params

    # If only the weights are trainable, a closed-form least-squares
    # update is cheaper and more stable than running L-BFGS.
    if len(params) == 1:
        logger.info(
            "L-BFGS skipped: only weights are trainable; using LS re-fit."
        )
        try:
            with torch.no_grad():
                A = assemble_basis_matrix(system.elements, points)
                if A.numel() == 0:
                    return system
                # Normal equations with tiny Tikhonov regularisation.
                reg = 1e-8
                ATA = A.T @ A + reg * torch.eye(
                    A.shape[1], device=device, dtype=dtype
                )
                ATg = A.T @ g
                w_ls = torch.linalg.solve(ATA, ATg)
                return ImageSystem(system.elements, w_ls)
        except Exception as exc:  # defensive
            logger.warning(
                "Least-squares refinement failed; keeping original system.",
                error=str(exc),
            )
            return system

    # L-BFGS over weights + element parameters.
    try:
        max_iter_env = os.getenv("EDE_IMAGES_LBFGS_MAX_ITER", "")
        try:
            max_iter = int(max_iter_env) if max_iter_env else 50
        except Exception:
            max_iter = 50

        optimizer = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=max_iter,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            V_pred = system.potential(points)
            loss = torch.mean((V_pred - g) ** 2)
            if not torch.isfinite(loss):
                # Abort cleanly if numerics blow up.
                return torch.tensor(float("inf"), device=device, dtype=dtype)
            loss.backward()
            return loss

        final_loss = optimizer.step(closure)
        try:
            loss_val = float(final_loss.detach().cpu())
        except Exception:
            loss_val = float("nan")
        logger.info(
            "L-BFGS refinement complete.",
            loss=loss_val,
            n_images=len(system.elements),
        )
    except Exception as exc:  # defensive
        logger.warning(
            "L-BFGS refinement failed; returning original system.",
            error=str(exc),
        )
        return system

    with torch.no_grad():
        new_weights = w.detach()
    return ImageSystem(system.elements, new_weights)


def discover_images(
    spec: CanonicalSpec,
    basis_types: List[str],
    n_max: int,
    reg_l1: float,
    restarts: int,
    logger: JsonlLogger,
    per_type_reg: Optional[Dict[str, float]] = None,
    boundary_weight: Optional[float] = None,
    two_stage: bool = False,
    nonlocal_types: Optional[List[str]] = None,
) -> ImageSystem:
    """Top-level entry point for sparse image discovery."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    logger.info(
        "Sparse image discovery started.",
        basis_types=basis_types,
        n_max=int(n_max),
        reg_l1=float(reg_l1),
        device=str(device),
    )

    # 1) Propose candidate basis elements for this spec.
    candidates = generate_candidate_basis(
        spec,
        basis_types=basis_types,
        n_candidates=max(1, n_max * 4),
        device=device,
        dtype=dtype,
    )
    if not candidates:
        logger.warning(
            "No candidate basis elements generated for this configuration."
        )
        return ImageSystem([], torch.zeros(0, device=device, dtype=dtype))

    logger.info(
        "Generated candidate basis elements.",
        n_candidates=len(candidates),
    )

    # 2) Build collocation set from the oracle.
    colloc_out = get_collocation_data(
        spec,
        logger,
        device=device,
        dtype=dtype,
        return_is_boundary=boundary_weight is not None,
    )
    if isinstance(colloc_out, tuple) and len(colloc_out) == 3:
        colloc_pts, target, is_boundary = colloc_out  # type: ignore[misc]
    else:
        colloc_pts, target = colloc_out  # type: ignore[misc]
        is_boundary = None
    if colloc_pts.shape[0] == 0:
        logger.error(
            "Collocation data generation failed; returning empty image system."
        )
        return ImageSystem([], torch.zeros(0, device=device, dtype=dtype))

    # 3) Assemble dictionary and optional boundary weighting.
    A = assemble_basis_matrix(candidates, colloc_pts)

    row_weights = None
    if boundary_weight is not None and is_boundary is not None and is_boundary.shape == (colloc_pts.shape[0],):
        alpha = float(max(0.0, min(1.0, boundary_weight)))
        beta = 1.0 - alpha
        is_boundary = is_boundary.to(device=device)
        row_weights = torch.where(
            is_boundary,
            torch.full_like(is_boundary, alpha, dtype=dtype),
            torch.full_like(is_boundary, beta, dtype=dtype),
        )
        rw_sqrt = torch.sqrt(row_weights).view(-1, 1)
        A = A * rw_sqrt
        target = target * rw_sqrt.view(-1)

    per_elem_reg_vec: Optional[torch.Tensor] = None
    if per_type_reg:
        reg_list = []
        for elem in candidates:
            reg_list.append(float(per_type_reg.get(elem.type, reg_l1)))
        per_elem_reg_vec = torch.tensor(reg_list, device=device, dtype=dtype)

    def _run_solver(A_in: torch.Tensor, g_in: torch.Tensor, elems: List[ImageBasisElement], reg_vec: Optional[torch.Tensor]) -> Tuple[torch.Tensor, List[int]]:
        return solve_l1_ista(
            A_in,
            g_in,
            reg_l1,
            logger,
            per_elem_reg=reg_vec,
        )

    # Helper: assemble system from selected indices and weights.
    def _build_system(idx_list: List[int], weights_vec: torch.Tensor) -> ImageSystem:
        selected = [candidates[i] for i in idx_list]
        w_sel = weights_vec[: len(selected)]
        return ImageSystem(selected, w_sel)

    # Two-stage path: solve non-local then point cleanup on residual.
    if two_stage:
        nonlocal_types = nonlocal_types or [
            "ring",
            "ring_gauss",
            "poloidal_ring",
            "ring_ladder_inner",
            "ring_ladder_outer",
            "toroidal_mode_cluster",
        ]
        nonlocal_idx = [i for i, c in enumerate(candidates) if c.type in nonlocal_types]
        point_idx = [i for i, c in enumerate(candidates) if c.type == "point"]

        if not nonlocal_idx:
            two_stage = False  # fall back
        else:
            A_non = A[:, nonlocal_idx]
            reg_non = per_elem_reg_vec[nonlocal_idx] if per_elem_reg_vec is not None else None
            w_non, supp_non = _run_solver(A_non, target, [candidates[i] for i in nonlocal_idx], reg_non)

            target_res = target
            if supp_non:
                w_non_sel = w_non[supp_non]
                A_non_sel = A_non[:, supp_non]
                target_res = target - A_non_sel @ w_non_sel

            w_point = torch.zeros(len(point_idx), device=device, dtype=dtype)
            supp_point: List[int] = []
            if point_idx:
                A_point = A[:, point_idx]
                reg_point = per_elem_reg_vec[point_idx] if per_elem_reg_vec is not None else None
                w_point, supp_point = _run_solver(
                    A_point,
                    target_res,
                    [candidates[i] for i in point_idx],
                    reg_point,
                )

            combined_idx: List[int] = []
            combined_w: List[torch.Tensor] = []
            for j in supp_non:
                combined_idx.append(nonlocal_idx[j])
                combined_w.append(w_non[j])
            for j in supp_point:
                combined_idx.append(point_idx[j])
                combined_w.append(w_point[j])

            if not combined_idx:
                # Fall back to dense LS below.
                weights = torch.zeros(len(candidates), device=device, dtype=dtype)
                support_idx = []
            else:
                w_concat = torch.stack(combined_w) if combined_w else torch.zeros(0, device=device, dtype=dtype)
                # Enforce n_max globally.
                if len(combined_idx) > n_max:
                    abs_w = torch.abs(w_concat)
                    topk = torch.topk(abs_w, k=n_max, largest=True)
                    combined_idx = [combined_idx[i] for i in topk.indices.tolist()]
                    w_concat = w_concat[topk.indices]
                # LS refit on combined support.
                try:
                    A_sel = A[:, combined_idx][:, : len(combined_idx)]
                    reg_ls = 1e-8
                    ATA = A_sel.T @ A_sel + reg_ls * torch.eye(A_sel.shape[1], device=A_sel.device, dtype=A_sel.dtype)
                    ATg = A_sel.T @ target
                    w_ls = torch.linalg.solve(ATA, ATg)
                    w_concat = w_ls[: len(combined_idx)]
                except Exception as exc:
                    logger.warning("Two-stage LS refinement failed; keeping sparse weights.", error=str(exc))
                weights = torch.zeros(len(candidates), device=device, dtype=dtype)
                for idx_val, w_val in zip(combined_idx, w_concat):
                    weights[idx_val] = w_val
                support_idx = combined_idx
    # Standard single-stage path.
    if not two_stage:
        weights, support_idx = _run_solver(A, target, candidates, per_elem_reg_vec)

    # 4) If ISTA finds no clearly non-zero coefficients, fall back to a
    #    dense least-squares fit over all candidates instead of giving up.
    if not support_idx:
        logger.warning(
            "Sparse solver selected no non-zero image weights; "
            "falling back to dense least-squares fit."
        )
        try:
            if A.numel() == 0 or A.shape[1] == 0:
                return ImageSystem([], torch.zeros(0, device=device, dtype=dtype))

            # Normal equations with tiny Tikhonov regularisation.
            reg_ls = 1e-8
            ATA = A.T @ A + reg_ls * torch.eye(A.shape[1], device=device, dtype=dtype)
            ATg = A.T @ target
            w_ls = torch.linalg.solve(ATA, ATg)

            # Enforce n_max by keeping the largest-|w| coefficients.
            k = min(n_max, w_ls.numel())
            if k <= 0:
                return ImageSystem([], torch.zeros(0, device=device, dtype=dtype))

            topk = torch.topk(w_ls.abs(), k=k, largest=True)
            idx = topk.indices
            selected = [candidates[int(i)] for i in idx]
            w_sel = w_ls[idx]

            system = ImageSystem(selected, w_sel)

            if restarts > 0:
                system = optimize_parameters_lbfgs(
                    system,
                    colloc_pts,
                    target,
                    logger,
                )

            logger.info(
                "Image discovery complete (LS fallback).",
                n_images=len(system.elements),
            )
            return system
        except Exception as exc:  # defensive
            logger.warning(
                "Dense least-squares fallback failed; returning empty image system.",
                error=str(exc),
            )
            return ImageSystem([], torch.zeros(0, device=device, dtype=dtype))

    # 5) Sparse-support path: keep up to n_max ISTA-selected candidates.
    selected = [candidates[i] for i in support_idx][:n_max]
    w_sel = weights[support_idx][: len(selected)]

    # Optional LS refit on the selected support to improve fit quality.
    try:
        A_sel = A[:, support_idx][:, : len(selected)]
        if A_sel.numel() > 0 and A_sel.shape[1] > 0:
            reg_ls = 1e-8
            ATA = A_sel.T @ A_sel + reg_ls * torch.eye(
                A_sel.shape[1], device=A_sel.device, dtype=A_sel.dtype
            )
            ATg = A_sel.T @ target
            w_ls = torch.linalg.solve(ATA, ATg)
            w_sel = w_ls[: len(selected)]
    except Exception as exc:
        logger.warning("Least-squares refinement on support failed.", error=str(exc))
    system = ImageSystem(selected, w_sel)

    if restarts > 0:
        system = optimize_parameters_lbfgs(
            system,
            colloc_pts,
            target,
            logger,
        )

    logger.info(
        "Image discovery complete.",
        n_images=len(system.elements),
    )
    return system
