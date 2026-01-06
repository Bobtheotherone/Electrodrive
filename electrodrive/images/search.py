from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import math
from dataclasses import dataclass
from pathlib import Path

import hashlib
import os

import numpy as np
import torch

from electrodrive.images.basis import (
    ImageBasisElement,
    generate_candidate_basis,
    build_dictionary,
    BasisGenerator,
    ChargeNodeInfo,
    CondNodeInfo,
    compute_group_ids,
    annotate_group_info,
    BASIS_FAMILY_ENUM,
    BASIS_FAMILY_NAMES,
)
from electrodrive.images.operator import BasisOperator
# NOTE: images currently depends on learn.collocation for collocation sampling.
# In a future refactor, this should move to a shared core.collocation module.
from electrodrive.learn.collocation import (
    make_collocation_batch_for_spec,
    _infer_geom_type_from_spec,
    compute_layered_reference_potential,
)
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.logging import JsonlLogger
from electrodrive.utils.config import K_E


# ---------------------------------------------------------------------------
# Internal helpers: device / dtype selection and linear-algebra utilities
# ---------------------------------------------------------------------------


@dataclass
class AugLagrangeConfig:
    """Hyperparameters for augmented-Lagrangian BC enforcement."""

    rho0: float = 10.0
    rho_growth: float = 10.0
    rho_max: float = 1e6
    max_outer: int = 3
    base_tol: float = 1e-6


def _normalize_aug_lagrange_cfg(cfg: Optional[Any]) -> Optional[AugLagrangeConfig]:
    """Convert a user-provided AL config into a validated dataclass."""
    if cfg is None:
        return None
    if isinstance(cfg, AugLagrangeConfig):
        return cfg
    if isinstance(cfg, dict):
        try:
            return AugLagrangeConfig(
                rho0=float(cfg.get("rho0", 10.0)),
                rho_growth=float(cfg.get("rho_growth", 10.0)),
                rho_max=float(cfg.get("rho_max", 1e6)),
                max_outer=int(cfg.get("max_outer", 3)),
                base_tol=float(cfg.get("base_tol", 1e-6)),
            )
        except Exception:
            return None
    return None


def _get_default_device() -> torch.device:
    """Select the compute device, allowing a lightweight ENV override.

    EDE_DEVICE can be "cuda", "cuda:1", "cpu", etc. If invalid, we fall
    back to "cuda" when available, else CPU.
    """
    dev_str = os.getenv("EDE_DEVICE", "").strip().lower()
    if dev_str:
        try:
            return torch.device(dev_str)
        except Exception:
            # Fall through to autodetect if the override is invalid.
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_default_dtype() -> torch.dtype:
    """Select the compute dtype for the image search pipeline.

    Default is float32. An optional EDE_IMAGES_DTYPE override can be
    "float64"/"double", "float16"/"fp16", or "bfloat16"/"bf16".
    """
    dt = os.getenv("EDE_IMAGES_DTYPE", "").strip().lower()
    if dt in {"float64", "double"}:
        return torch.float64
    if dt in {"float16", "fp16"}:
        return torch.float16
    if dt in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float32


def _intensive_enabled(override: Optional[bool] = None) -> bool:
    """Return True if intensive mode is requested via override or env."""
    if override is not None:
        return bool(override)
    val = os.getenv("EDE_IMAGES_INTENSIVE", "0").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _estimate_lipschitz(
    A: torch.Tensor,
    logger: JsonlLogger,
    max_power_iters: int = 50,
) -> float:
    """Estimate the Lipschitz constant L of A^T A in a GPU-friendly way.

    Small problems (min(N, K) <= 256) use an exact SVD in float32 for a
    tight bound. Larger problems use power iteration on A^T A, which is
    much cheaper than a full SVD but still robust enough for ISTA.
    """
    if A.ndim != 2:
        return 1.0

    N, K = A.shape
    # Exact path for small systems: helps reduce ISTA iterations.
    if min(N, K) <= 256:
        try:
            A32 = A.detach().to(dtype=torch.float32)
            _, s, _ = torch.linalg.svd(A32, full_matrices=False)
            if s.numel() > 0:
                return float((s[0] ** 2).item())
        except Exception as exc:
            logger.warning(
                "ISTA: SVD Lipschitz estimate failed; falling back to power iteration.",
                error=str(exc),
            )

    # Power iteration on A^T A: cheap and GPU-friendly.
    with torch.no_grad():
        x = torch.randn(K, device=A.device, dtype=A.dtype)
        n0 = torch.linalg.norm(x)
        if float(n0) > 0.0:
            x = x / n0

        for _ in range(max_power_iters):
            y = A.T @ (A @ x)  # (A^T A) x
            n = torch.linalg.norm(y)
            if float(n) < 1e-7:
                break
            x = (y / n).detach()
        Ax = A @ x
        L = float(torch.dot(Ax, Ax).item())  # ||A x||^2

    return L if L > 0.0 else 1.0


def _estimate_lipschitz_operator(
    A_op: BasisOperator,
    logger: JsonlLogger,
    max_power_iters: int = 50,
) -> float:
    """Power-iteration Lipschitz estimate for operator-form dictionaries."""
    return _estimate_lipschitz_from_ops(
        lambda x: A_op.matvec(x),
        lambda r: A_op.rmatvec(r),
        A_op.shape,
        logger,
        max_power_iters=max_power_iters,
        device=A_op.device,
        dtype=A_op.dtype,
    )


def _estimate_lipschitz_from_ops(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rmatvec: Callable[[torch.Tensor], torch.Tensor],
    shape: Tuple[int, int],
    logger: JsonlLogger,
    max_power_iters: int = 50,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> float:
    """Power-iteration Lipschitz estimate for arbitrary matvec/rmatvec pairs."""
    N, K = shape
    if N == 0 or K == 0:
        return 1.0

    device = device if device is not None else torch.device("cpu")
    dtype = dtype if dtype is not None else torch.float32

    with torch.no_grad():
        x = torch.randn(K, device=device, dtype=dtype)
        n0 = torch.linalg.norm(x)
        if float(n0) > 0.0:
            x = x / n0

        for _ in range(max_power_iters):
            y = matvec(x)
            z = rmatvec(y)
            n = torch.linalg.norm(z)
            if float(n) < 1e-7:
                break
            x = (z / n).detach()

        Ax = matvec(x)
        L = float(torch.dot(Ax, Ax).item())

    if L <= 0.0:
        logger.warning(
            "ISTA: operator Lipschitz estimate non-positive; falling back to 1.0.",
        )
        return 1.0
    return L


def _solve_normal_eq_psd(
    ATA: torch.Tensor,
    ATg: torch.Tensor,
) -> torch.Tensor:
    """Solve (ATA) w = ATg for SPD-ish normal equations on GPU.

    We try a Cholesky factorisation (fast, numerically nice) and fall back
    to a generic solver on failure.
    """
    try:
        L = torch.linalg.cholesky(ATA)
        w = torch.cholesky_solve(ATg.unsqueeze(-1), L).squeeze(-1)
    except Exception:
        try:
            w = torch.linalg.solve(ATA, ATg)
        except Exception:
            eps = 1e-6 * torch.max(torch.abs(torch.diag(ATA))).clamp_min(1.0)
            ATA = ATA + eps * torch.eye(ATA.shape[0], device=ATA.device, dtype=ATA.dtype)
            w = torch.linalg.solve(ATA, ATg)
    return w


def _group_prox(
    w: torch.Tensor,
    group_ids: Optional[torch.Tensor],
    lambda_group: float | torch.Tensor,
) -> torch.Tensor:
    """Group-lasso proximal operator applied after elementwise shrinkage."""
    if group_ids is None:
        return w
    if not torch.is_tensor(lambda_group) and lambda_group <= 0.0:
        return w
    if group_ids.shape[0] != w.shape[0]:
        raise ValueError(
            f"group_ids length {group_ids.shape[0]} does not match weights {w.shape[0]}"
        )
    w_out = w.clone()
    group_ids = group_ids.to(device=w.device)
    if torch.is_tensor(lambda_group):
        lam_vec = lambda_group.to(device=w.device, dtype=w.dtype).view(-1)
        if lam_vec.numel() == 1:
            lam_vec = lam_vec.expand_as(w)
        if lam_vec.numel() != w.shape[0]:
            raise ValueError(
                f"lambda_group tensor has shape {tuple(lam_vec.shape)}, expected ({w.shape[0]},)"
            )
    else:
        lam_vec = None
    unique_groups = torch.unique(group_ids)
    for g_val in unique_groups:
        mask = group_ids == g_val
        if not bool(mask.any()):
            continue
        w_g = w_out[mask]
        norm_g = torch.linalg.norm(w_g)
        lam = float(lam_vec[mask].mean().item()) if lam_vec is not None else float(lambda_group)
        if float(norm_g) <= lam:
            w_out[mask] = 0.0
        else:
            shrink = (norm_g - lam) / norm_g
            w_out[mask] = shrink * w_g
    return w_out


class ImageSystem:
    """A concrete image system: basis elements + weights."""

    def __init__(self, elements: List[ImageBasisElement], weights: torch.Tensor, metadata: Optional[Dict[str, Any]] = None):
        self.elements = elements
        self.weights = weights
        self.metadata: Dict[str, Any] = metadata or {}
        self._v2 = None
        self._v2_cache_key = None
        if weights.numel() > 0:
            self.device = weights.device
            self.dtype = weights.dtype
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32

    def potential(self, targets: torch.Tensor) -> torch.Tensor:
        """Evaluate the image-system potential at a batch of points.

        Internally we align to the system's device/dtype to avoid
        device-mismatch surprises, then convert back to the original
        layout on return.
        """
        flag = os.getenv("EDE_IMAGE_SYSTEM_V2", "").strip().lower()
        if flag in {"1", "true", "yes", "on"}:
            return self._potential_v2(targets)
        orig_device = targets.device
        orig_dtype = targets.dtype

        # Move to the system's device/dtype for basis evaluation.
        targets_sys = targets.to(device=self.device, dtype=self.dtype)

        V_sys = torch.zeros(
            targets_sys.shape[0],
            device=self.device,
            dtype=self.dtype,
        )
        # Avoid per-element dtype casts and extra allocations.
        for elem, w in zip(self.elements, self.weights):
            V_sys.add_(w * elem.potential(targets_sys))

        # Preserve the caller's expectations about device/dtype.
        return V_sys.to(device=orig_device, dtype=orig_dtype)

    def _potential_v2(self, targets: torch.Tensor) -> torch.Tensor:
        key = (
            id(self.weights),
            tuple(self.weights.shape),
            self.weights.device,
            self.weights.dtype,
            len(self.elements),
        )
        if self._v2 is None or self._v2_cache_key != key:
            from electrodrive.images.image_system_v2 import ImageSystemV2

            self._v2 = ImageSystemV2(self.elements, self.weights, metadata=self.metadata)
            self._v2_cache_key = key
        return self._v2.potential(targets)


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


def _make_torch_rng(device: torch.device) -> torch.Generator:
    """Create a deterministic torch RNG tied to the run ID without polluting global state."""
    # Reuse the run-id seeded numpy RNG to derive a stable torch seed.
    seed = int(_make_collocation_rng().integers(0, 2**31 - 1, dtype=np.int64))
    try:
        gen = torch.Generator(device=device)
    except Exception:
        gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def _build_boundary_row_weights(
    boundary_mask: Optional[torch.Tensor],
    boundary_weight: Optional[float],
    boundary_mode: str,
    boundary_penalty_default: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[str], Optional[float], int, int]:
    """Compute per-row weights for legacy boundary handling (non-AL).

    Returns
    -------
    row_weights : torch.Tensor or None
        Per-row weights (length N) or None when unweighted.
    row_weights_sqrt : torch.Tensor or None
        Square roots for easy scaling of targets/matrix rows.
    mode_label : str or None
        Human-readable label of the weighting mode.
    boundary_metric : float or None
        Magnitude of the boundary emphasis for logging.
    n_boundary : int
        Count of boundary rows.
    n_interior : int
        Count of interior rows.
    """
    if boundary_mask is None:
        return None, None, None, None, 0, 0

    boundary_mask = boundary_mask.to(device=device)
    n_boundary = int(boundary_mask.sum().item())
    n_total = int(boundary_mask.shape[0])
    n_interior = n_total - n_boundary

    if n_boundary == 0 or n_total == 0:
        return None, None, None, None, n_boundary, n_interior

    row_weights: Optional[torch.Tensor] = None
    mode_label: Optional[str] = None
    boundary_metric: Optional[float] = None

    if boundary_weight is None:
        if boundary_mode in {"penalty", "hard"} or boundary_penalty_default > 0.0:
            penalty_val = float(boundary_penalty_default) if boundary_penalty_default > 0.0 else 1.0
            penalty = max(penalty_val, 1.0)
            row_weights = torch.where(
                boundary_mask,
                torch.full_like(boundary_mask, penalty, dtype=dtype),
                torch.ones_like(boundary_mask, dtype=dtype),
            )
            mode_label = "penalty"
            boundary_metric = penalty
    else:
        try:
            bw = float(boundary_weight)
        except Exception:
            bw = 0.0
        if 0.0 < bw < 1.0:
            alpha = float(max(0.0, min(1.0, bw)))
            beta = 1.0 - alpha
            row_weights = torch.where(
                boundary_mask,
                torch.full_like(boundary_mask, alpha, dtype=dtype),
                torch.full_like(boundary_mask, beta, dtype=dtype),
            )
            mode_label = "mix"
            boundary_metric = alpha
        elif bw >= 1.0:
            row_weights = torch.where(
                boundary_mask,
                torch.full_like(boundary_mask, bw, dtype=dtype),
                torch.ones_like(boundary_mask, dtype=dtype),
            )
            mode_label = "ratio"
            boundary_metric = bw

    if row_weights is None:
        return None, None, None, None, n_boundary, n_interior

    return row_weights, row_weights.sqrt(), mode_label, boundary_metric, n_boundary, n_interior


def _maybe_shuffle_candidates(
    candidates: List[ImageBasisElement],
    logger: JsonlLogger,
) -> List[ImageBasisElement]:
    """Deterministically shuffle candidates to reduce index bias.

    Shuffling is controlled by the environment knob
    ``EDE_IMAGES_SHUFFLE_CANDIDATES`` (defaults to on) and reuses the
    same run-ID seeded RNG as collocation so ordering remains
    reproducible per run.
    """
    if len(candidates) <= 1:
        return candidates

    flag = os.getenv("EDE_IMAGES_SHUFFLE_CANDIDATES", "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return candidates

    rng = _make_collocation_rng()
    perm = rng.permutation(len(candidates))
    shuffled = [candidates[int(i)] for i in perm]
    logger.info(
        "Candidate basis order shuffled.",
        n_candidates=len(shuffled),
        shuffle_seed=os.getenv("EDE_RUN_ID", ""),
    )
    return shuffled


def _normalize_generator_mode(mode: Optional[str]) -> str:
    allowed = {
        "static_only",
        "static_plus_learned",
        "learned_only",
        "diffusion",
        "hybrid_diffusion",
        "gfn",
        "gfn_flow",
        "gfdsl",
    }
    if not mode:
        return "static_only"
    m = mode.strip().lower()
    if m == "gflownet":
        m = "gfn"
    if m in {"gflownet_flow", "gfnflow"}:
        m = "gfn_flow"
    if m in {"gfdsl_programs", "gfdsl_json"}:
        m = "gfdsl"
    if m not in allowed:
        return "static_only"
    return m


def _extract_slab_bounds(spec: CanonicalSpec) -> Optional[Tuple[float, float]]:
    if getattr(spec, "BCs", "") != "dielectric_interfaces":
        return None
    layers = getattr(spec, "dielectrics", None) or []
    if len(layers) != 3:
        return None
    try:
        triples: List[Tuple[float, float, float, Dict[str, Any]]] = []
        for layer in layers:
            z_min = float(layer["z_min"])
            z_max = float(layer["z_max"])
            thickness = z_max - z_min
            triples.append((z_min, z_max, thickness, layer))
    except Exception:
        return None
    if len(triples) != 3:
        return None
    triples.sort(key=lambda t: (t[0] + t[1]) * 0.5)
    bottom, middle, top = triples
    if middle[2] <= 0:
        return None
    z_bottom_slab = middle[0]
    z_top_slab = middle[1]
    return z_bottom_slab, z_top_slab


def _dedup_candidates_by_position(candidates: List[ImageBasisElement], tol: float = 1e-6) -> List[ImageBasisElement]:
    seen: List[Tuple[float, float, float]] = []
    deduped: List[ImageBasisElement] = []
    for elem in candidates:
        pos = getattr(elem, "params", {}).get("position", None)
        if isinstance(pos, torch.Tensor):
            coords = pos.detach().cpu().view(-1)
            key = (float(coords[0]), float(coords[1]), float(coords[2]))
        else:
            deduped.append(elem)
            continue
        keep = True
        for px, py, pz in seen:
            if abs(px - key[0]) < tol and abs(py - key[1]) < tol and abs(pz - key[2]) < tol:
                keep = False
                break
        if keep:
            seen.append(key)
            deduped.append(elem)
    return deduped


def _domain_extent_from_spec(spec: CanonicalSpec) -> float:
    z_vals: List[float] = []
    for ch in getattr(spec, "charges", []) or []:
        try:
            z_vals.append(abs(float(ch.get("pos", [0.0, 0.0, 0.0])[2])))
        except Exception:
            continue
    for layer in getattr(spec, "dielectrics", []) or []:
        for key in ("z_min", "z_max"):
            try:
                z_vals.append(abs(float(layer.get(key, 0.0))))
            except Exception:
                continue
    domain = getattr(spec, "domain", {})
    if isinstance(domain, dict):
        bbox = domain.get("bbox", None)
        if bbox and len(bbox) == 2 and len(bbox[0]) >= 3 and len(bbox[1]) >= 3:
            try:
                z_vals.append(abs(float(bbox[0][2])))
                z_vals.append(abs(float(bbox[1][2])))
            except Exception:
                pass
    max_z = max(z_vals) if z_vals else 1.0
    return max(1.0, max_z)


def _numeric_diagnostics(system: ImageSystem, colloc_pts: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
    """Compute residual and weight diagnostics for a solved system."""
    if colloc_pts.numel() == 0 or target.numel() == 0:
        return {"numeric_status": "ok", "rel_resid": float("nan"), "max_abs_weight": float("nan"), "min_nonzero_weight": float("nan")}
    with torch.no_grad():
        preds = system.potential(colloc_pts)
        diff = preds - target
        denom = torch.linalg.norm(target).clamp_min(1e-12)
        rel_resid = float((torch.linalg.norm(diff) / denom).item())

        weights = system.weights.detach().cpu()
        max_abs_weight = float(weights.abs().max().item()) if weights.numel() > 0 else 0.0
        nonzero = weights[weights != 0]
        min_nonzero_weight = float(nonzero.abs().min().item()) if nonzero.numel() > 0 else float("nan")
        status = "ok"
        if not torch.isfinite(weights).all():
            status = "nonfinite"
        elif rel_resid > 100.0:
            status = "catastrophic"
        return {
            "numeric_status": status,
            "rel_resid": rel_resid,
            "max_abs_weight": max_abs_weight,
            "min_nonzero_weight": min_nonzero_weight,
        }


def _family_name(elem: ImageBasisElement) -> str:
    info = getattr(elem, "_group_info", None)
    if isinstance(info, dict) and "family_name" in info:
        try:
            return str(info["family_name"])
        except Exception:
            pass
    return elem.type


def _build_lambda_group_vector(
    candidates: Sequence[ImageBasisElement],
    device: torch.device,
    dtype: torch.dtype,
    lambda_default: float,
) -> torch.Tensor:
    """Per-element group regularisation from family-specific env knobs."""
    fam_regs: Dict[str, float] = {}
    for fam in BASIS_FAMILY_ENUM.keys():
        val = os.getenv(f"EDE_IMAGES_GROUP_REG_{fam}", None)
        if val is None:
            continue
        try:
            fam_regs[fam] = float(val)
        except Exception:
            continue
    lam_vec = torch.full((len(candidates),), float(lambda_default), device=device, dtype=dtype)
    for idx, elem in enumerate(candidates):
        fam = _family_name(elem)
        if fam in fam_regs:
            lam_vec[idx] = fam_regs[fam]
    return lam_vec


def _build_learned_candidates(
    spec: CanonicalSpec,
    basis_generator: BasisGenerator,
    geo_encoder: Optional[Any],
    n_candidates: int,
    device: torch.device,
    dtype: torch.dtype,
    logger: JsonlLogger,
) -> List[ImageBasisElement]:
    """Invoke a learned BasisGenerator with a lightweight GeoEncoder."""
    if n_candidates <= 0:
        return []

    try:
        from electrodrive.images.geo_encoder import GeoEncoder  # type: ignore[assignment]
    except Exception:
        GeoEncoder = None  # type: ignore[assignment]
    try:
        from electrodrive.images.learned_generator import SimpleGeoEncoder  # type: ignore[assignment]
    except Exception:
        SimpleGeoEncoder = None  # type: ignore[assignment]

    encoder = geo_encoder
    if encoder is None:
        choice = os.getenv("EDE_IMAGES_GEO_ENCODER", "egnn").strip().lower()
        if choice in {"egnn", "geo", "graph"} and GeoEncoder is not None:
            encoder = GeoEncoder()
        elif choice in {"simple", "mlp"} and SimpleGeoEncoder is not None:
            encoder = SimpleGeoEncoder()
        elif GeoEncoder is not None:
            encoder = GeoEncoder()
        elif SimpleGeoEncoder is not None:
            encoder = SimpleGeoEncoder()
        else:
            logger.warning(
                "BasisGenerator requested but no GeoEncoder available; skipping learned candidates."
            )
            return []

    # Ensure modules live on the correct device/dtype.
    try:
        encoder = encoder.to(device=device)  # type: ignore[assignment]
    except Exception:
        pass
    try:
        basis_generator = basis_generator.to(device=device)  # type: ignore[assignment]
    except Exception:
        pass

    encoder.eval()
    basis_generator.eval()

    with torch.no_grad():
        try:
            z_global, charge_nodes, conductor_nodes = encoder.encode(
                spec,
                device=device,
                dtype=dtype,
            )
            slab_bounds = _extract_slab_bounds(spec)
            if slab_bounds and hasattr(basis_generator, "set_slab_bounds"):
                try:
                    basis_generator.set_slab_bounds(slab_bounds)  # type: ignore[call-arg]
                except Exception:
                    pass
            learned = basis_generator(
                z_global=z_global,
                charge_nodes=charge_nodes,
                conductor_nodes=conductor_nodes,
                n_candidates=n_candidates,
            )
        except Exception as exc:  # defensive
            logger.warning(
                "BasisGenerator invocation failed; continuing without learned candidates.",
                error=str(exc),
            )
            return []

    # Filter out any malformed entries to keep downstream code robust.
    safe: List[ImageBasisElement] = []
    motif_counter = 1
    for elem in learned:
        try:
            pos = elem.params.get("position", None)
            if pos is None or not torch.isfinite(pos).all():
                continue
            conductor_hint = 0
            try:
                conductor_hint = int(torch.as_tensor(elem.params.get("conductor_id", 0)).item())  # type: ignore[arg-type]
            except Exception:
                conductor_hint = 0
            annotate_group_info(
                elem,
                conductor_id=conductor_hint,
                family_name=elem.type,
                motif_index=motif_counter,
            )
            motif_counter += 1
            safe.append(elem)
        except Exception:
            continue

    if safe:
        logger.info(
            "Learned candidates generated.",
            n_candidates=len(safe),
            mode="learned",
        )
    return safe[:n_candidates]


def _build_gfn_candidates(
    spec: CanonicalSpec,
    gfn_generator: Any,
    geo_encoder: Optional[Any],
    n_candidates: int,
    device: torch.device,
    dtype: torch.dtype,
    logger: JsonlLogger,
    seed: Optional[int],
    mode: str = "gfn",
) -> List[ImageBasisElement]:
    """Invoke a GFlowNetProgramGenerator to produce candidates."""
    if n_candidates <= 0:
        return []

    try:
        from electrodrive.images.geo_encoder import GeoEncoder  # type: ignore[assignment]
    except Exception:
        GeoEncoder = None  # type: ignore[assignment]
    try:
        from electrodrive.images.learned_generator import SimpleGeoEncoder  # type: ignore[assignment]
    except Exception:
        SimpleGeoEncoder = None  # type: ignore[assignment]

    encoder = geo_encoder
    if encoder is None:
        choice = os.getenv("EDE_IMAGES_GEO_ENCODER", "egnn").strip().lower()
        if choice in {"egnn", "geo", "graph"} and GeoEncoder is not None:
            encoder = GeoEncoder()
        elif choice in {"simple", "mlp"} and SimpleGeoEncoder is not None:
            encoder = SimpleGeoEncoder()
        elif GeoEncoder is not None:
            encoder = GeoEncoder()
        elif SimpleGeoEncoder is not None:
            encoder = SimpleGeoEncoder()
        else:
            raise ValueError("GFlowNet generator requested but no GeoEncoder available.")

    try:
        encoder = encoder.to(device=device)  # type: ignore[assignment]
    except Exception:
        pass

    encoder.eval()
    try:
        gfn_generator.set_spec(spec)
    except Exception:
        pass

    with torch.no_grad():
        z_global, charge_nodes, conductor_nodes = encoder.encode(
            spec,
            device=device,
            dtype=dtype,
        )
        candidates = gfn_generator.generate(
            spec=spec,
            spec_embedding=z_global,
            n_candidates=n_candidates,
            seed=seed,
        )

    safe: List[ImageBasisElement] = []
    for elem in candidates:
        try:
            pos = elem.params.get("position", None)
            if pos is None or not torch.isfinite(pos).all():
                continue
            info = getattr(elem, "_group_info", None)
            if not isinstance(info, dict) or not info:
                annotate_group_info(
                    elem,
                    conductor_id=0,
                    family_name=elem.type,
                    motif_index=0,
                )
            safe.append(elem)
        except Exception:
            continue

    if safe:
        logger.info(
            "GFlowNet candidates generated.",
            n_candidates=len(safe),
            mode=mode,
        )
    return safe[:n_candidates]


def _resolve_collocation_params(
    n_points_override: Optional[int],
    ratio_override: Optional[float],
) -> Tuple[int, float]:
    """Resolve collocation sampling parameters with env-compatible defaults."""
    if n_points_override is not None and n_points_override > 0:
        n_points = int(n_points_override)
    else:
        try:
            n_points_env = int(os.getenv("EDE_IMAGES_N_POINTS", "0"))
            n_points = n_points_env if n_points_env > 0 else 512
        except Exception:
            n_points = 512

    if ratio_override is not None and 0.0 < ratio_override < 1.0:
        ratio_boundary = float(ratio_override)
    else:
        try:
            ratio_env = float(os.getenv("EDE_IMAGES_RATIO_BOUNDARY", "nan"))
            ratio_boundary = ratio_env if 0.0 < ratio_env < 1.0 else 0.5
        except Exception:
            ratio_boundary = 0.5

    return n_points, ratio_boundary


def _prepare_collocation_batch(
    batch: Dict[str, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    logger: JsonlLogger,
    return_is_boundary: bool,
    pass_label: str,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
    """Shared post-processing for make_collocation_batch_for_spec outputs."""
    X = batch.get("X")
    V = batch.get("V_gt")

    if X is None or V is None or X.numel() == 0 or V.numel() == 0:
        logger.error(
            "Collocation helper returned an empty batch.",
            n_points_requested=int(batch.get("n_points", 0)) if isinstance(batch.get("n_points", None), (int, float)) else 0,  # type: ignore[arg-type]
            pass_label=pass_label,
        )
        empty_pts = torch.empty(0, 3, device=device, dtype=dtype)
        empty_v = torch.empty(0, device=device, dtype=dtype)
        return empty_pts, empty_v, None, {
            "n_points": 0,
            "n_boundary": 0,
            "frac_boundary": 0.0,
            "V_min": float("nan"),
            "V_max": float("nan"),
        }

    mask_finite = batch.get("mask_finite")
    if mask_finite is not None and mask_finite.shape == (X.shape[0],):
        mask = mask_finite.to(device=device, dtype=torch.bool) & torch.isfinite(V)
    else:
        mask = torch.isfinite(V)

    if not mask.any():
        logger.error(
            "Collocation batch has no finite targets.",
            n_points_total=int(X.shape[0]),
            pass_label=pass_label,
        )
        empty_pts = torch.empty(0, 3, device=device, dtype=dtype)
        empty_v = torch.empty(0, device=device, dtype=dtype)
        return empty_pts, empty_v, None, {
            "n_points": 0,
            "n_boundary": 0,
            "frac_boundary": 0.0,
            "V_min": float("nan"),
            "V_max": float("nan"),
        }

    X_f = X[mask].to(device=device, dtype=dtype).contiguous()
    V_f = V[mask].to(device=device, dtype=dtype).contiguous()

    is_boundary_out: Optional[torch.Tensor] = None
    n_boundary = 0
    is_boundary = batch.get("is_boundary")
    if return_is_boundary and is_boundary is not None and is_boundary.shape == (X.shape[0],):
        is_boundary = is_boundary.to(device=device)
        n_boundary = int(is_boundary[mask].sum().item())
        is_boundary_out = is_boundary[mask]
    elif return_is_boundary and is_boundary is not None and is_boundary.shape != (X.shape[0],):
        logger.warning(
            "Boundary mask present but has unexpected shape; dropping.",
            expected_shape=(int(X.shape[0]),),
            actual_shape=tuple(is_boundary.shape),
            pass_label=pass_label,
        )

    N = int(X_f.shape[0])
    frac_boundary = float(n_boundary) / float(N) if N > 0 else 0.0
    V_min = float(V_f.min().item()) if N > 0 else float("nan")
    V_max = float(V_f.max().item()) if N > 0 else float("nan")

    stats = {
        "n_points": N,
        "n_boundary": n_boundary,
        "frac_boundary": frac_boundary,
        "V_min": V_min,
        "V_max": V_max,
    }

    return X_f, V_f, is_boundary_out, stats


def get_collocation_data(
    spec: CanonicalSpec,
    logger: JsonlLogger,
    device: torch.device,
    dtype: torch.dtype,
    return_is_boundary: bool = False,
    rng: np.random.Generator | None = None,
    n_points_override: Optional[int] = None,
    ratio_override: Optional[float] = None,
    subtract_physical_potential: bool = False,
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

    Notes
    -----
    Optional ``rng``, ``n_points_override``, and ``ratio_override`` parameters
    allow callers to reuse a deterministic generator and explicitly control
    sample sizes when running adaptive refinement passes.
    """
    device = torch.device(device)
    rng = rng if rng is not None else _make_collocation_rng()
    try:
        geom = _infer_geom_type_from_spec(spec)
    except Exception:
        geom = "unknown"

    n_points, ratio_boundary = _resolve_collocation_params(
        n_points_override, ratio_override
    )

    try:
        # Discovery uses collocation purely as an oracle; no gradients needed.
        with torch.no_grad():
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

    X_f, V_f, is_boundary_out, stats = _prepare_collocation_batch(
        batch,
        device=device,
        dtype=dtype,
        logger=logger,
        return_is_boundary=return_is_boundary,
        pass_label="initial",
    )

    if subtract_physical_potential and X_f.numel() > 0:
        try:
            if geom == "layered_planar":
                V_ref = compute_layered_reference_potential(spec, X_f, device=device, dtype=dtype)
                V_f = V_f - V_ref
            else:
                charges = getattr(spec, "charges", []) or []
                q_list = []
                pos_list = []
                for c in charges:
                    if c.get("type") != "point":
                        continue
                    q_list.append(float(c.get("q", 0.0)))
                    pos_list.append([float(v) for v in c.get("pos", [0.0, 0.0, 0.0])])
                if q_list:
                    Q = torch.tensor(q_list, device=device, dtype=dtype).view(1, -1)
                    P = torch.tensor(pos_list, device=device, dtype=dtype).view(len(q_list), 3)
                    diff = X_f[:, None, :] - P[None, :, :]
                    R = torch.linalg.norm(diff, dim=2).clamp_min(1e-9)
                    V_phys = (K_E * Q / R).sum(dim=1)
                    V_f = V_f - V_phys
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to subtract physical potential; leaving targets unchanged.",
                error=str(exc),
            )

    logger.info(
        "Collocation data prepared.",
        **stats,
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


def assemble_basis(
    basis_elems: Sequence[ImageBasisElement],
    points: torch.Tensor,
    operator_mode: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | BasisOperator:
    """Return a dense dictionary or BasisOperator depending on operator_mode.

    This is the polymorphic assembly entry point used by discover_images.
    New code should depend on this helper, while ``assemble_basis_matrix``
    remains as a thin compatibility wrapper for legacy callers.
    """
    device = torch.device(device)
    dtype = dtype
    operator_mode = bool(operator_mode)
    if operator_mode:
        pts = points.to(device=device, dtype=dtype).contiguous()
        return BasisOperator(list(basis_elems), points=pts, device=device, dtype=dtype)
    return assemble_basis_matrix(list(basis_elems), points.to(device=device, dtype=dtype))


def _matvec(A: torch.Tensor | BasisOperator, w: torch.Tensor, X: Optional[torch.Tensor]) -> torch.Tensor:
    """Dispatch matvec for dense or operator dictionaries."""
    if isinstance(A, BasisOperator):
        pts = X if X is not None else getattr(A, "points", None)
        return A.matvec(w, pts)
    return A.matmul(w)


def _rmatvec(A: torch.Tensor | BasisOperator, r: torch.Tensor, X: Optional[torch.Tensor]) -> torch.Tensor:
    """Dispatch rmatvec for dense or operator dictionaries."""
    if isinstance(A, BasisOperator):
        pts = X if X is not None else getattr(A, "points", None)
        return A.rmatvec(r, pts)
    return A.transpose(0, 1).matmul(r)


def solve_l1_ista(
    A: torch.Tensor | BasisOperator,
    g: torch.Tensor,
    reg_l1: float,
    logger: JsonlLogger,
    max_iter: int = 1000,
    tol: float = 1e-6,
    per_elem_reg: Optional[torch.Tensor] = None,
    collocation: Optional[torch.Tensor] = None,
    group_ids: Optional[torch.Tensor] = None,
    lambda_group: float | torch.Tensor = 0.0,
    weight_prior: Optional[torch.Tensor] = None,
    lambda_weight_prior: float | torch.Tensor = 0.0,
    *,
    is_boundary: Optional[torch.Tensor] = None,
    aug_lagrange_cfg: Optional[AugLagrangeConfig] = None,
    boundary_weight: Optional[float] = None,
    normalize_columns: bool = True,
    ls_refit: bool = False,
    support_k: Optional[int] = None,
    boundary_mode: str = "penalty",
    boundary_penalty_default: float = 0.0,
    stats: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, List[int]]:
    """Solve the L1-regularised least-squares problem via ISTA.

    Minimises 0.5 * ||A w - g||_2^2 + reg_l1 * ||w||_1 with optional
    boundary weighting, augmented-Lagrangian penalties, group sparsity,
    and a final least-squares refit on the selected support.
    """
    stats_out = stats if stats is not None else None
    is_operator = isinstance(A, BasisOperator)
    group_ids_tensor: Optional[torch.Tensor] = None

    if is_operator:
        try:
            N_shape, N_k = A.shape  # type: ignore[assignment]
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("Operator A must expose .shape = (N, K)") from exc
        device = getattr(A, "device", torch.device("cpu"))
        dtype = getattr(A, "dtype", torch.float32)
        X = collocation if collocation is not None else getattr(A, "points", None)
        if X is None:
            raise ValueError("Operator-mode ISTA requires collocation points.")
        X = X.to(device=device, dtype=dtype).contiguous()
        N = X.shape[0]
        try:
            group_ids_tensor = group_ids if group_ids is not None else getattr(A, "groups", None)
        except Exception:
            group_ids_tensor = group_ids
        if N_shape not in (-1, N) and N_shape != N:
            logger.warning(
                "Operator shape does not match collocation size; proceeding with provided targets.",
                operator_shape=str((N_shape, N_k)),
                n_targets=int(N),
            )
    else:
        if not isinstance(A, torch.Tensor):
            raise TypeError("solve_l1_ista expects a torch.Tensor or BasisOperator")
        if A.ndim != 2:
            raise ValueError(f"ISTA expects a 2D matrix A, got shape {tuple(A.shape)}")
        N, N_k = A.shape
        device = A.device
        dtype = A.dtype
        X = None
        group_ids_tensor = group_ids

    if N == 0 or N_k == 0 or g.numel() == 0:
        return torch.zeros(N_k, device=device, dtype=dtype), []

    g = g.to(device=device, dtype=dtype).view(-1)
    if g.shape[0] != N:
        raise ValueError(f"g has shape {tuple(g.shape)}, expected ({N},)")

    # Optional environment overrides to keep legacy behaviour.
    try:
        max_iter_env = int(os.getenv("EDE_IMAGES_ISTA_MAX_ITER", "0"))
        if max_iter_env > 0:
            max_iter = max_iter_env
    except Exception:
        pass
    if _intensive_enabled():
        max_iter = max(max_iter, 200)

    try:
        tol_env = float(os.getenv("EDE_IMAGES_ISTA_TOL", "nan"))
        if tol_env > 0.0:
            tol = tol_env
    except Exception:
        pass

    boundary_mask = None
    if is_boundary is not None:
        boundary_mask = torch.as_tensor(is_boundary, device=device, dtype=torch.bool).view(-1)
        if boundary_mask.shape[0] != N:
            raise ValueError(f"is_boundary has shape {tuple(boundary_mask.shape)}, expected ({N},)")

    # Column norms and scaled matvec/rmatvec.
    if is_operator:
        A_op: BasisOperator = A  # type: ignore[assignment]
        col_norms = getattr(A_op, "col_norms", None)
        if normalize_columns:
            if col_norms is None:
                try:
                    col_norms = A_op.estimate_col_norms(X)  # type: ignore[arg-type]
                except Exception as exc:
                    logger.warning(
                        "ISTA: column-norm estimation failed; falling back to ones.",
                        error=str(exc),
                    )
                    col_norms = torch.ones(N_k, device=device, dtype=dtype)
            col_norms = col_norms.to(device=device, dtype=dtype).view(-1).clamp_min(1e-6)
            A_op.col_norms = col_norms
        else:
            col_norms = torch.ones(N_k, device=device, dtype=dtype)
            A_op._inv_col_norms = torch.ones_like(col_norms)

        inv_norms = A_op.inv_col_norms()

        def _matvec_scaled(w: torch.Tensor) -> torch.Tensor:
            return _matvec(A_op, w * inv_norms, X)  # type: ignore[arg-type]

        def _rmatvec_scaled(r_vec: torch.Tensor) -> torch.Tensor:
            return inv_norms * _rmatvec(A_op, r_vec, X)
    else:
        A_tensor = A.to(device=device, dtype=dtype)
        if normalize_columns:
            col_norms = torch.linalg.norm(A_tensor, dim=0).clamp_min(1e-6)
        else:
            col_norms = torch.ones(N_k, device=device, dtype=dtype)

        def _matvec_scaled(w: torch.Tensor) -> torch.Tensor:
            return A_tensor @ (w * (1.0 / col_norms))

        def _rmatvec_scaled(r_vec: torch.Tensor) -> torch.Tensor:
            return (A_tensor.T @ r_vec) * (1.0 / col_norms)

    inv_norms = 1.0 / col_norms

    # Thresholds in the column-normalised coordinates.
    base_reg = (
        per_elem_reg.to(device=device, dtype=dtype).view(-1)
        if per_elem_reg is not None
        else torch.full((N_k,), reg_l1, device=device, dtype=dtype)
    )
    if base_reg.shape[0] != N_k:
        raise ValueError(f"per_elem_reg has shape {tuple(base_reg.shape)}, expected ({N_k},)")

    if group_ids_tensor is not None:
        group_ids_tensor = torch.as_tensor(group_ids_tensor, device=device, dtype=torch.long).view(-1)
        if group_ids_tensor.shape[0] != N_k:
            raise ValueError(f"group_ids has shape {tuple(group_ids_tensor.shape)}, expected ({N_k},)")

    weight_prior_tensor: Optional[torch.Tensor] = None
    lambda_prior_eff: Optional[torch.Tensor | float] = None
    if weight_prior is not None:
        w_prior = torch.as_tensor(weight_prior, device=device, dtype=dtype).view(-1)
        if w_prior.numel() == 1:
            w_prior = w_prior.expand(N_k)
        if w_prior.numel() < N_k:
            w_prior = torch.cat(
                [w_prior, torch.zeros(N_k - w_prior.numel(), device=device, dtype=dtype)],
                dim=0,
            )
        elif w_prior.numel() > N_k:
            w_prior = w_prior[:N_k]
        weight_prior_tensor = w_prior

        if torch.is_tensor(lambda_weight_prior):
            lam_prior = lambda_weight_prior.to(device=device, dtype=dtype).view(-1)
            if lam_prior.numel() == 1:
                lam_prior = lam_prior.expand(N_k)
            if lam_prior.numel() < N_k:
                lam_prior = torch.cat(
                    [lam_prior, torch.zeros(N_k - lam_prior.numel(), device=device, dtype=dtype)],
                    dim=0,
                )
            elif lam_prior.numel() > N_k:
                lam_prior = lam_prior[:N_k]
            if float(lam_prior.max().item()) > 0.0:
                lambda_prior_eff = lam_prior
        else:
            lam_scalar = float(lambda_weight_prior)
            if lam_scalar > 0.0:
                lambda_prior_eff = lam_scalar

    def _make_ops(row_weights_sqrt: Optional[torch.Tensor]) -> Tuple[
        Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], float
    ]:
        def mv(w: torch.Tensor) -> torch.Tensor:
            y = _matvec_scaled(w)
            if row_weights_sqrt is not None:
                y = row_weights_sqrt * y
            return y

        def rtv(r_vec: torch.Tensor) -> torch.Tensor:
            r_in = r_vec
            if row_weights_sqrt is not None:
                r_in = row_weights_sqrt * r_vec
            return _rmatvec_scaled(r_in)

        L_est = _estimate_lipschitz_from_ops(
            mv,
            rtv,
            (N, N_k),
            logger,
            max_power_iters=min(max_iter, 50),
            device=device,
            dtype=dtype,
        )
        return mv, rtv, L_est

    def _ista_loop(
        g_vec: torch.Tensor,
        row_weights_sqrt: Optional[torch.Tensor],
        *,
        max_iter_inner: int,
        tol_inner: float,
    ) -> Tuple[torch.Tensor, List[int]]:
        matvec_fn, rmatvec_fn, L_val = _make_ops(row_weights_sqrt)
        if L_val <= 0.0:
            logger.warning("ISTA: non-positive Lipschitz estimate, aborting.")
            return torch.zeros(N_k, device=device, dtype=dtype), []

        alpha = 1.0 / L_val
        thr_vec = base_reg * inv_norms * alpha

        lambda_group_eff: float | torch.Tensor
        if torch.is_tensor(lambda_group):
            lg_vec = lambda_group.to(device=device, dtype=dtype).view(-1)
            if lg_vec.numel() == 1:
                lg_vec = lg_vec.expand(N_k)
            if lg_vec.numel() not in (0, N_k):
                raise ValueError(f"lambda_group tensor has shape {tuple(lg_vec.shape)}, expected ({N_k},)")
            lambda_group_eff = lg_vec * alpha
        else:
            lambda_group_eff = float(lambda_group) * alpha

        g_eff = g_vec if row_weights_sqrt is None else row_weights_sqrt * g_vec

        w = torch.zeros(N_k, device=device, dtype=dtype)
        w_prev = torch.empty_like(w)
        diff = torch.empty_like(w)
        last_rel_change: float = float("inf")
        iters_done = 0
        converged_flag = False

        with torch.no_grad():
            for it in range(max_iter_inner):
                iters_done = int(it + 1)
                w_prev.copy_(w)

                r_vec = matvec_fn(w) - g_eff
                grad = rmatvec_fn(r_vec)
                if weight_prior_tensor is not None and lambda_prior_eff is not None:
                    w_phys = w * inv_norms
                    grad_prior = w_phys - weight_prior_tensor
                    if torch.is_tensor(lambda_prior_eff):
                        grad_prior = grad_prior * lambda_prior_eff
                    else:
                        grad_prior = grad_prior * float(lambda_prior_eff)
                    grad = grad + grad_prior * inv_norms

                w.add_(grad, alpha=-alpha)

                # Elementwise L1 prox
                abs_w = torch.abs(w)
                shrunk = torch.clamp(abs_w - thr_vec, min=0.0)
                w.copy_(torch.sign(w) * shrunk)

                if group_ids_tensor is not None:
                    if torch.is_tensor(lambda_group_eff):
                        if lambda_group_eff.numel() > 0:
                            w.copy_(_group_prox(w, group_ids_tensor, lambda_group_eff))
                    elif lambda_group_eff > 0.0:
                        w.copy_(_group_prox(w, group_ids_tensor, lambda_group_eff))

                diff.copy_(w).add_(w_prev, alpha=-1.0)
                num = float(torch.linalg.norm(diff))
                den = float(torch.linalg.norm(w) + 1e-9)
                if den > 0.0:
                    last_rel_change = num / den
                    if last_rel_change < tol_inner:
                        converged_flag = True
                        logger.info(
                            "ISTA converged.",
                            iters=int(it + 1),
                            rel_change=float(last_rel_change),
                        )
                        break
            else:
                logger.warning(
                    "ISTA did not converge.",
                    max_iter=int(max_iter_inner),
                    final_rel_change=float(last_rel_change),
                )
        if stats_out is not None:
            stats_out["iters"] = int(iters_done)
            stats_out["converged"] = bool(converged_flag)
            stats_out["rel_change"] = float(last_rel_change)

        w_phys = w * inv_norms
        w_abs = torch.abs(w_phys)
        idx_sorted = torch.argsort(w_abs, descending=True)
        support = [int(i) for i in idx_sorted.tolist() if float(w_abs[int(i)].item()) > 0.0]
        if support_k is not None and support_k > 0:
            support = support[: support_k]
        return w_phys, support

    def _ls_refit(
        w_phys: torch.Tensor,
        support: List[int],
        g_vec: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int]]:
        if not ls_refit or not support:
            return w_phys, support

        support_unique = list(dict.fromkeys(support))
        if isinstance(A, BasisOperator):
            elems = getattr(A, "elements", None)
            dense: Optional[torch.Tensor] = None
            if elems is not None and X is not None:
                try:
                    dense = assemble_basis_matrix([elems[i] for i in support_unique], X)
                except Exception:
                    dense = None
            if dense is None:
                try:
                    op_sub = A.subset(support_unique)
                    dense = op_sub.to_dense(targets=X)
                except Exception:
                    dense = None
            if dense is None or dense.numel() == 0:
                return w_phys, support_unique
            A_dense = dense
        else:
            A_dense = A.to(device=device, dtype=dtype)[:, support_unique]

        reg_ls = 1e-8
        ATA = A_dense.T @ A_dense + reg_ls * torch.eye(A_dense.shape[1], device=device, dtype=dtype)
        ATg = A_dense.T @ g_vec
        w_sub = _solve_normal_eq_psd(ATA, ATg)
        w_refit = torch.zeros_like(w_phys)
        for idx_val, w_val in zip(support_unique, w_sub):
            w_refit[idx_val] = w_val
        return w_refit, support_unique

    # Legacy boundary weighting when AL is disabled.
    row_weights = None
    row_weights_sqrt = None
    if aug_lagrange_cfg is None and boundary_mask is not None:
        row_weights, row_weights_sqrt, _, _, _, _ = _build_boundary_row_weights(
            boundary_mask,
            boundary_weight,
            boundary_mode,
            boundary_penalty_default,
            device=device,
            dtype=dtype,
        )

    aug_cfg = _normalize_aug_lagrange_cfg(aug_lagrange_cfg)
    if aug_cfg is None or boundary_mask is None or not bool(boundary_mask.any()):
        w_raw, support = _ista_loop(
            g,
            row_weights_sqrt=row_weights_sqrt,
            max_iter_inner=max_iter,
            tol_inner=tol,
        )
        w_final, support_final = _ls_refit(w_raw, support, g if row_weights_sqrt is None else g * row_weights_sqrt)
        return w_final, support_final

    # Augmented Lagrangian outer loop.
    mu = torch.zeros(int(boundary_mask.sum().item()), device=device, dtype=dtype)
    rho = max(float(aug_cfg.rho0), 1.0)
    rho_growth = max(float(aug_cfg.rho_growth), 1.0)
    rho_max = max(float(aug_cfg.rho_max), rho)
    max_outer = max(int(aug_cfg.max_outer), 1)

    boundary_norms: List[float] = []
    support_out: List[int] = []
    w_out = torch.zeros(N_k, device=device, dtype=dtype)

    for outer in range(max_outer):
        weights_vec = torch.ones(N, device=device, dtype=dtype)
        weights_vec = torch.where(boundary_mask, torch.full_like(weights_vec, rho, dtype=dtype), weights_vec)
        weights_sqrt = weights_vec.sqrt()

        target_aug = g.clone()
        target_aug[boundary_mask] = -mu / (2.0 * rho)

        tol_inner = float(aug_cfg.base_tol) / (1.0 + math.log10(max(rho, 1.0)))
        w_iter, support_iter = _ista_loop(
            target_aug,
            row_weights_sqrt=weights_sqrt,
            max_iter_inner=max_iter,
            tol_inner=tol_inner,
        )

        pred_full = _matvec(A, w_iter, X if is_operator else None)
        y_b = pred_full[boundary_mask] - g[boundary_mask]
        bc_norm = float(torch.linalg.norm(y_b).item())
        boundary_norms.append(bc_norm)

        mu = mu + rho * y_b
        w_out = w_iter
        support_out = support_iter

        if len(boundary_norms) >= 2:
            prev = boundary_norms[-2]
            if prev > 0.0 and bc_norm >= 0.9 * prev:
                logger.warning(
                    "AL loop: boundary norm stagnated; freezing rho growth.",
                    prev_bc=float(prev),
                    curr_bc=float(bc_norm),
                    rho=float(rho),
                )
                rho_growth = 1.0
        rho = min(rho * rho_growth, rho_max)

    w_final, support_final = _ls_refit(w_out, support_out, g)
    logger.info(
        "AL diagnostics.",
        n_outer=int(len(boundary_norms)),
        bc_norm_before=float(boundary_norms[0]) if boundary_norms else float("nan"),
        bc_norm_after=float(boundary_norms[-1]) if boundary_norms else float("nan"),
        rho_final=float(rho),
    )
    if stats_out is not None:
        stats_out["aug_rho_final"] = float(rho)
        stats_out["aug_bc_norm_before"] = float(boundary_norms[0]) if boundary_norms else float("nan")
        stats_out["aug_bc_norm_after"] = float(boundary_norms[-1]) if boundary_norms else float("nan")
    return w_final, support_final


def solve_l1_augmented_lagrangian(
    g: torch.Tensor,
    boundary_mask: torch.Tensor,
    *,
    make_weighted_dict: Callable[[torch.Tensor], torch.Tensor | BasisOperator],
    predict_unweighted: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    collocation: torch.Tensor,
    reg_l1: float,
    logger: JsonlLogger,
    cfg: AugLagrangeConfig,
    per_elem_reg: Optional[torch.Tensor] = None,
    group_ids: Optional[torch.Tensor] = None,
    lambda_group: float | torch.Tensor = 0.0,
    solver_mode: str = "ista",
    lista_model: Optional[Any] = None,
    weight_prior: Optional[torch.Tensor] = None,
    lambda_weight_prior: float | torch.Tensor = 0.0,
    stats: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor | BasisOperator, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Outer augmented-Lagrangian loop enforcing boundary residuals.

    Parameters
    ----------
    g : torch.Tensor
        Unweighted collocation targets (length N).
    boundary_mask : torch.Tensor
        Boolean mask of length N indicating boundary rows.
    make_weighted_dict : Callable
        Function building a weighted dictionary from a per-row weight vector.
    predict_unweighted : Callable
        Function returning A w on the (optional) subset mask without row
        weighting; used to measure boundary residuals.
    collocation : torch.Tensor
        Collocation points, forwarded to ISTA in operator mode.

    Returns
    -------
    w : torch.Tensor
        Final weights after the AL outer loop.
    A_last : torch.Tensor | BasisOperator
        The weighted dictionary used in the last inner solve.
    g_last : torch.Tensor
        Weighted target vector used in the last inner solve.
    row_weights_last : torch.Tensor
        Final per-row weights (length N) used in the last inner solve.
    diagnostics : Dict[str, Any]
        Boundary norm and rho schedule for logging.
    """
    device = g.device
    dtype = g.dtype
    stats_out = stats if stats is not None else None
    boundary_mask = boundary_mask.to(device=device, dtype=torch.bool).view(-1)
    g = g.to(device=device, dtype=dtype).view(-1)

    n_boundary = int(boundary_mask.sum().item())
    if n_boundary == 0 or g.numel() == 0:
        logger.warning("Augmented Lagrangian requested but boundary mask is empty; falling back to vanilla ISTA.")
        A_last = make_weighted_dict(torch.ones_like(g))
        w, _ = solve_l1_ista(
            A_last,
            g,
            reg_l1,
            logger,
            per_elem_reg=per_elem_reg,
            collocation=collocation,
            group_ids=group_ids,
            lambda_group=lambda_group,
            weight_prior=weight_prior,
            lambda_weight_prior=lambda_weight_prior,
            stats=stats_out,
        )
        if stats_out is not None:
            stats_out["aug_rho_final"] = float(cfg.rho0)
        return (
            w,
            A_last,
            g,
            torch.ones_like(g),
            {"n_outer": 1, "bc_norms": [], "rho_final": float(cfg.rho0)},
        )

    mu = torch.zeros(n_boundary, device=device, dtype=dtype)
    rho = max(float(cfg.rho0), 1.0)
    rho_growth = max(float(cfg.rho_growth), 1.0)
    rho_max = max(float(cfg.rho_max), rho)
    max_outer = max(int(cfg.max_outer), 1)

    boundary_norms: List[float] = []
    A_last: torch.Tensor | BasisOperator = make_weighted_dict(torch.ones_like(g))
    g_last: torch.Tensor = g
    row_weights_last: torch.Tensor = torch.ones_like(g)

    stop_growth = False

    for outer in range(max_outer):
        # Build weighted targets and operator for this rho.
        weights_vec = torch.ones_like(g, device=device, dtype=dtype)
        weights_vec = torch.where(boundary_mask, torch.full_like(weights_vec, rho, dtype=dtype), weights_vec)

        target_aug = g.clone()
        target_aug[boundary_mask] = -mu / (2.0 * rho)

        weights_sqrt = weights_vec.sqrt()
        g_weighted = target_aug * weights_sqrt

        A_weighted = make_weighted_dict(weights_vec)
        tol_inner = float(cfg.base_tol) / (1.0 + math.log10(max(rho, 1.0)))

        if solver_mode == "lista" and lista_model is not None:
            try:
                # Prefer newer LISTA checkpoints that accept reg_l1; fall back for older ones.
                try:
                    w = lista_model(
                        A_weighted,
                        collocation,
                        g_weighted,
                        reg_l1=reg_l1,
                        group_ids=group_ids,
                        lambda_group=lambda_group,
                    )
                except TypeError:
                    w = lista_model(
                        A_weighted,
                        collocation,
                        g_weighted,
                        group_ids=group_ids,
                        lambda_group=lambda_group,
                    )
            except Exception as exc:
                logger.warning(
                    "AL loop: LISTA solve failed; falling back to ISTA.",
                    error=str(exc),
                )
                w, _ = solve_l1_ista(
                    A_weighted,
                    g_weighted,
                    reg_l1,
                    logger,
                    tol=tol_inner,
                    per_elem_reg=per_elem_reg,
                    collocation=collocation,
                    group_ids=group_ids,
                    lambda_group=lambda_group,
                    weight_prior=weight_prior,
                    lambda_weight_prior=lambda_weight_prior,
                )
        else:
            w, _ = solve_l1_ista(
                A_weighted,
                g_weighted,
                reg_l1,
                logger,
                tol=tol_inner,
                per_elem_reg=per_elem_reg,
                collocation=collocation,
                group_ids=group_ids,
                lambda_group=lambda_group,
                weight_prior=weight_prior,
                lambda_weight_prior=lambda_weight_prior,
            )

        # Boundary residual A_b w - g_b (g_b assumed zero but keep subtraction for safety).
        pred_b = predict_unweighted(w, boundary_mask)
        g_b = g[boundary_mask]
        y_b = pred_b - g_b
        bc_norm = float(torch.linalg.norm(y_b).item())
        boundary_norms.append(bc_norm)

        A_last = A_weighted
        g_last = g_weighted
        row_weights_last = weights_vec

        mu = mu + rho * y_b

        if len(boundary_norms) >= 2:
            prev = boundary_norms[-2]
            if prev > 0.0 and bc_norm >= 0.9 * prev and not stop_growth:
                logger.warning(
                    "AL loop: boundary norm did not decrease by 10%; freezing rho growth.",
                    prev_bc=float(prev),
                    curr_bc=float(bc_norm),
                    rho=float(rho),
                )
                stop_growth = True

        if not stop_growth:
            rho = min(rho * rho_growth, rho_max)

    diagnostics = {
        "n_outer": int(len(boundary_norms)),
        "bc_norm_before": float(boundary_norms[0]) if boundary_norms else float("nan"),
        "bc_norm_after": float(boundary_norms[-1]) if boundary_norms else float("nan"),
        "rho_final": float(rho),
        "rho_frozen": bool(stop_growth),
    }
    if stats_out is not None:
        stats_out["aug_rho_final"] = diagnostics["rho_final"]
        stats_out["aug_bc_norm_before"] = diagnostics["bc_norm_before"]
        stats_out["aug_bc_norm_after"] = diagnostics["bc_norm_after"]
    if boundary_norms and boundary_norms[-1] >= boundary_norms[0] * 0.99:
        logger.warning(
            "AL loop: boundary norm did not improve over baseline.",
            bc_start=float(boundary_norms[0]),
            bc_end=float(boundary_norms[-1]),
        )

    logger.info(
        "AL diagnostics.",
        n_outer=int(len(boundary_norms)),
        bc_norm_before=diagnostics["bc_norm_before"],
        bc_norm_after=diagnostics["bc_norm_after"],
        rho_final=diagnostics["rho_final"],
        rho_frozen=diagnostics["rho_frozen"],
    )

    return w, A_last, g_last, row_weights_last, diagnostics


def solve_sparse(
    A: torch.Tensor | BasisOperator,
    X: torch.Tensor,
    V_gt: torch.Tensor,
    is_boundary: Optional[torch.Tensor],
    logger: JsonlLogger,
    *,
    reg_l1: float,
    solver: str = "ista",
    lista_model: Optional[Any] = None,
    aug_lagrange_cfg: Optional[Any] = None,
    boundary_weight: Optional[float] = None,
    boundary_mode: str = "penalty",
    boundary_penalty_default: float = 0.0,
    per_elem_reg: Optional[torch.Tensor] = None,
    group_ids: Optional[torch.Tensor] = None,
    lambda_group: float | torch.Tensor = 0.0,
    weight_prior: Optional[torch.Tensor] = None,
    lambda_weight_prior: float | torch.Tensor = 0.0,
    normalize_columns: bool = True,
    ls_refit: bool = True,
    support_k: Optional[int] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    lista_refine: bool = True,
    return_stats: bool = False,
    constraints: Optional[list] = None,
    dtype_policy: Optional[Any] = None,
    warm_start: Optional[torch.Tensor] = None,
    admm_cfg: Optional[Any] = None,
    constraint_mode: str = "none",
) -> Tuple[torch.Tensor, List[int]] | Tuple[torch.Tensor, List[int], Dict[str, Any]]:
    """Unified sparse solver wrapper handling ISTA, LISTA, and AL."""
    stats: Dict[str, Any] = {"solver": (solver or "ista").strip().lower()}
    stats.setdefault("iters", 0)
    stats.setdefault("converged", False)
    stats.setdefault("rel_change", float("nan"))
    solver_mode = (solver or "ista").strip().lower()
    if solver_mode not in {"ista", "lista", "implicit_lasso", "implicit_grouplasso", "admm_constrained"}:
        solver_mode = "ista"

    if isinstance(A, BasisOperator) and group_ids is None:
        try:
            group_ids = getattr(A, "groups", None)
        except Exception:
            group_ids = None

    aug_cfg = _normalize_aug_lagrange_cfg(aug_lagrange_cfg)
    if aug_cfg is not None and solver_mode == "lista":
        logger.info("AL requested; forcing ISTA for stability with AL loop.")
        solver_mode = "ista"

    def _ista_call(max_iter_inner: int, tol_inner: float, do_refit: bool) -> Tuple[torch.Tensor, List[int]]:
        return solve_l1_ista(
            A,
            V_gt,
            reg_l1,
            logger,
            max_iter=max_iter_inner,
            tol=tol_inner,
            per_elem_reg=per_elem_reg,
            collocation=X,
            group_ids=group_ids,
            lambda_group=lambda_group,
            weight_prior=weight_prior,
            lambda_weight_prior=lambda_weight_prior,
            is_boundary=is_boundary,
            aug_lagrange_cfg=aug_cfg,
            boundary_weight=boundary_weight,
            normalize_columns=normalize_columns,
            ls_refit=do_refit,
            support_k=support_k,
            boundary_mode=boundary_mode,
            boundary_penalty_default=boundary_penalty_default,
            stats=stats,
        )

    w_out: torch.Tensor
    support_out: List[int]

    if solver_mode in {"implicit_lasso", "implicit_grouplasso"}:
        from electrodrive.images.optim import (
            SparseSolveRequest,
            implicit_grouplasso_solve,
            implicit_lasso_solve,
        )

        req = SparseSolveRequest(
            A=A,
            X=X if isinstance(A, BasisOperator) else None,
            g=V_gt,
            is_boundary=is_boundary,
            lambda_l1=float(reg_l1),
            lambda_group=lambda_group if lambda_group is not None else 0.0,
            group_ids=group_ids,
            weight_prior=weight_prior,
            lambda_weight_prior=lambda_weight_prior,
            normalize_columns=normalize_columns,
            col_norms=None,
            constraints=constraints or [],
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            return_stats=return_stats,
            dtype_policy=dtype_policy,
        )
        if solver_mode == "implicit_grouplasso":
            result = implicit_grouplasso_solve(req)
        else:
            result = implicit_lasso_solve(req)
        w_out = result.w
        support_out = [int(i) for i in result.support.to(device="cpu", dtype=torch.long).tolist()]
        stats.update(result.stats)
        stats["solver"] = solver_mode
    elif solver_mode == "admm_constrained":
        from electrodrive.images.optim import ADMMConfig, SparseSolveRequest, admm_constrained_solve

        constraint_mode_norm = (constraint_mode or "none").strip().lower()
        constraint_list = [] if constraint_mode_norm in {"none", "off"} else (constraints or [])
        if isinstance(admm_cfg, ADMMConfig):
            cfg = admm_cfg
        elif isinstance(admm_cfg, dict):
            cfg = ADMMConfig(**admm_cfg)
        elif admm_cfg is None:
            cfg = ADMMConfig()
        else:
            cfg = admm_cfg
        req = SparseSolveRequest(
            A=A,
            X=X if isinstance(A, BasisOperator) else None,
            g=V_gt,
            is_boundary=is_boundary,
            lambda_l1=float(reg_l1),
            lambda_group=lambda_group if lambda_group is not None else 0.0,
            group_ids=group_ids,
            weight_prior=weight_prior,
            lambda_weight_prior=lambda_weight_prior,
            normalize_columns=normalize_columns,
            col_norms=None,
            constraints=constraint_list,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            return_stats=return_stats,
            dtype_policy=dtype_policy,
        )
        result = admm_constrained_solve(req, cfg if isinstance(cfg, ADMMConfig) else None)
        w_out = result.w
        support_out = [int(i) for i in result.support.to(device="cpu", dtype=torch.long).tolist()]
        stats.update(result.stats)
        stats["solver"] = solver_mode
    elif solver_mode == "lista" and lista_model is not None:
        try:
            lista_model = lista_model.to(device=V_gt.device, dtype=V_gt.dtype)  # type: ignore[assignment]
        except Exception:
            pass
        try:
            lista_model.eval()  # type: ignore[call-arg]
        except Exception:
            pass
        try:
            # Prefer LISTA checkpoints that accept reg_l1; fall back gracefully if not.
            if hasattr(lista_model, "solve"):
                w_lista = lista_model.solve(
                    A,
                    V_gt,
                    X=X,
                    group_ids=group_ids,
                    lambda_group=lambda_group,
                    reg_l1=reg_l1,
                )
            else:
                try:
                    w_lista = lista_model(
                        A,
                        X,
                        V_gt,
                        reg_l1=reg_l1,
                        group_ids=group_ids,
                        lambda_group=lambda_group,
                    )
                except TypeError:
                    w_lista = lista_model(
                        A,
                        X,
                        V_gt,
                        group_ids=group_ids,
                        lambda_group=lambda_group,
                    )
            # Regardless of LISTA quality, finish with a full ISTA solve to ensure parity.
            stats["solver"] = "lista+ista_full"
            w_out, support_out = _ista_call(max_iter_inner=max_iter, tol_inner=tol, do_refit=ls_refit)
        except Exception as exc:
            logger.warning(
                "LISTA solver failed; falling back to ISTA.",
                error=str(exc),
            )
            w_out, support_out = _ista_call(max_iter_inner=max_iter, tol_inner=tol, do_refit=ls_refit)
    else:
        # Default ISTA path (with optional AL handled inside).
        w_out, support_out = _ista_call(max_iter_inner=max_iter, tol_inner=tol, do_refit=ls_refit)

    try:
        preds = _matvec(A, w_out, X if isinstance(A, BasisOperator) else None)
        resid = preds - V_gt
        bc_norm = float(
            torch.linalg.norm(resid[is_boundary])
            if is_boundary is not None and bool(is_boundary.any())
            else torch.linalg.norm(resid)
        )
        stats["bc_norm"] = bc_norm
        if is_boundary is not None and bool(is_boundary.any()):
            interior_mask = ~is_boundary
            stats["int_norm"] = (
                float(torch.linalg.norm(resid[interior_mask]))
                if bool(interior_mask.any())
                else float("nan")
            )
            stats["frac_boundary"] = float(is_boundary.float().mean().item())
        else:
            stats["int_norm"] = float(torch.linalg.norm(resid))
            stats["frac_boundary"] = 0.0
    except Exception:
        stats["bc_norm"] = float("nan")
        stats["int_norm"] = float("nan")

    try:
        stats["col_norm_min"] = float(col_norms.min().item())
        stats["col_norm_max"] = float(col_norms.max().item())
        stats["col_norm_median"] = float(col_norms.median().item())
        stats["normalized_columns"] = bool(normalize_columns)
    except Exception:
        pass

    try:
        if torch.is_tensor(lambda_group):
            stats["lambda_group"] = float(lambda_group.max().item())
        else:
            stats["lambda_group"] = float(lambda_group)
    except Exception:
        stats["lambda_group"] = float("nan")

    if return_stats:
        return w_out, support_out, stats
    return w_out, support_out


def optimize_parameters_lbfgs(
    system: ImageSystem,
    points: torch.Tensor,
    g: torch.Tensor,
    logger: JsonlLogger,
    *,
    domain_extent: Optional[float] = None,
    max_iter_override: Optional[int] = None,
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
                w_ls = _solve_normal_eq_psd(ATA, ATg)
                return ImageSystem(system.elements, w_ls)
        except Exception as exc:  # defensive
            logger.warning(
                "Least-squares refinement failed; keeping original system.",
                error=str(exc),
            )
            return system

    # Snapshot pre-refinement state for potential rollback.
    pre_serialized = [elem.serialize() for elem in system.elements]
    pre_weights = system.weights.detach().clone()

    def _rel_resid(sys: ImageSystem) -> float:
        preds = sys.potential(points)
        diff = preds - g
        denom = torch.linalg.norm(g).clamp_min(1e-12)
        return float((torch.linalg.norm(diff) / denom).item())

    rel_resid_before = _rel_resid(system)

    # L-BFGS over weights + element parameters.
    try:
        max_iter_env = os.getenv("EDE_IMAGES_LBFGS_MAX_ITER", "")
        try:
            max_iter = int(max_iter_env) if max_iter_env else 50
        except Exception:
            max_iter = 50
        if max_iter_override is not None:
            try:
                max_iter = int(max_iter_override)
            except Exception:
                pass

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
        refined = ImageSystem(system.elements, new_weights)

    # Guardrails: revert on divergence or unphysical positions.
    try:
        rel_resid_after = _rel_resid(refined)
    except Exception:
        rel_resid_after = float("inf")

    out_of_bounds = False
    if domain_extent is not None and math.isfinite(domain_extent):
        thresh = float(domain_extent) * 3.0
        for elem in refined.elements:
            pos = getattr(elem, "params", {}).get("position", None)
            if isinstance(pos, torch.Tensor):
                z_val = float(pos.view(-1)[2].item())
                if abs(z_val) > thresh:
                    out_of_bounds = True
                    break

    if (rel_resid_after > rel_resid_before * 10.0) or out_of_bounds:
        logger.warning(
            "LBFGS refinement diverged; reverting to pre-LBFGS.",
            rel_resid_before=rel_resid_before,
            rel_resid_after=rel_resid_after,
            out_of_bounds=bool(out_of_bounds),
        )
        restored_elems = [
            ImageBasisElement.deserialize(e, device=device, dtype=dtype) for e in pre_serialized
        ]
        return ImageSystem(restored_elems, pre_weights.to(device=device, dtype=dtype))

    return refined


def cheap_discover_images_for_collocation(
    spec: CanonicalSpec,
    basis_types: List[str],
    logger: JsonlLogger,
    n_max: int = 8,
    reg_l1: float = 1e-4,
    subtract_physical_potential: bool = False,
) -> ImageSystem:
    """Lightweight ISTA-based solve to seed adaptive collocation."""
    device = _get_default_device()
    dtype = _get_default_dtype()
    cand_rng = _make_torch_rng(device)

    candidates = generate_candidate_basis(
        spec,
        basis_types=basis_types,
        n_candidates=max(1, 4 * n_max),
        device=device,
        dtype=dtype,
        rng=cand_rng,
    )
    if not candidates:
        logger.warning("Cheap solve skipped: no candidates generated.")
        return ImageSystem([], torch.zeros(0, device=device, dtype=dtype))

    rng = _make_collocation_rng()
    with torch.no_grad():
        batch = make_collocation_batch_for_spec(
            spec=spec,
            n_points=256,
            ratio_boundary=0.7,
            supervision_mode="auto",
            device=device,
            dtype=dtype,
            rng=rng,
        )
    colloc_pts, target_vec, _, stats = _prepare_collocation_batch(
        batch,
        device=device,
        dtype=dtype,
        logger=logger,
        return_is_boundary=False,
        pass_label="cheap_collocation",
    )
    if subtract_physical_potential and colloc_pts.numel() > 0:
        try:
            charges = getattr(spec, "charges", []) or []
            q_list = []
            pos_list = []
            for c in charges:
                if c.get("type") != "point":
                    continue
                q_list.append(float(c.get("q", 0.0)))
                pos_list.append([float(v) for v in c.get("pos", [0.0, 0.0, 0.0])])
            if q_list:
                Q = torch.tensor(q_list, device=device, dtype=dtype).view(1, -1)
                P = torch.tensor(pos_list, device=device, dtype=dtype).view(len(q_list), 3)
                diff = colloc_pts[:, None, :] - P[None, :, :]
                R = torch.linalg.norm(diff, dim=2).clamp_min(1e-9)
                V_phys = (K_E * Q / R).sum(dim=1)
                target_vec = target_vec - V_phys
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Cheap collocation: failed to subtract physical potential; leaving targets unchanged.",
                error=str(exc),
            )
    if colloc_pts.numel() == 0 or target_vec.numel() == 0:
        logger.warning("Cheap solve skipped: empty collocation batch.")
        return ImageSystem([], torch.zeros(0, device=device, dtype=dtype))

    A = assemble_basis_matrix(candidates, colloc_pts)
    weights, _ = solve_l1_ista(
        A,
        target_vec,
        reg_l1=reg_l1,
        logger=logger,
        max_iter=200,
        tol=1e-4,
    )
    logger.info(
        "Cheap provisional image solve complete.",
        n_candidates=len(candidates),
        n_collocation=int(stats.get("n_points", 0)),
    )
    return ImageSystem(candidates, weights)


def build_adaptive_collocation(
    spec: CanonicalSpec,
    logger: JsonlLogger,
    device: torch.device,
    dtype: torch.dtype,
    n_points: int,
    ratio_boundary: float,
    n_rounds: int,
    initial_system: Optional[ImageSystem],
    subtract_physical_potential: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Residual-driven collocation builder for adaptive discovery runs.

    In round 1 we keep the full batch; subsequent rounds keep the
    highest-residual points against a provisional image system. Oracle
    calls are bounded to ``n_rounds`` batches of ``n_points`` each.
    """
    device = torch.device(device)
    dtype = dtype
    n_rounds = max(1, int(n_rounds))
    ratio_boundary = float(max(0.0, min(1.0, ratio_boundary)))

    rng = _make_collocation_rng()

    X_all: List[torch.Tensor] = []
    V_all: List[torch.Tensor] = []
    is_b_all: List[torch.Tensor] = []

    system = initial_system

    for r in range(n_rounds):
        pass_label = f"adaptive_round_{r + 1}"
        with torch.no_grad():
            batch = make_collocation_batch_for_spec(
                spec=spec,
                n_points=n_points,
                ratio_boundary=ratio_boundary,
                supervision_mode="auto",
                device=device,
                dtype=dtype,
                rng=rng,
            )
        X_r, V_r, is_b_r, stats = _prepare_collocation_batch(
            batch,
            device=device,
            dtype=dtype,
            logger=logger,
            return_is_boundary=True,
            pass_label=pass_label,
        )
        if X_r.numel() == 0:
            logger.warning(
                "Adaptive collocation round produced no points; stopping early.",
                pass_label=pass_label,
            )
            break

        if subtract_physical_potential:
            try:
                charges = getattr(spec, "charges", []) or []
                q_list = []
                pos_list = []
                for c in charges:
                    if c.get("type") != "point":
                        continue
                    q_list.append(float(c.get("q", 0.0)))
                    pos_list.append([float(v) for v in c.get("pos", [0.0, 0.0, 0.0])])
                if q_list:
                    Q = torch.tensor(q_list, device=device, dtype=dtype).view(1, -1)
                    P = torch.tensor(pos_list, device=device, dtype=dtype).view(len(q_list), 3)
                    diff = X_r[:, None, :] - P[None, :, :]
                    R = torch.linalg.norm(diff, dim=2).clamp_min(1e-9)
                    V_phys = (K_E * Q / R).sum(dim=1)
                    V_r = V_r - V_phys
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Adaptive collocation: failed to subtract physical potential; leaving targets unchanged.",
                    error=str(exc),
                    pass_label=pass_label,
                )

        if is_b_r is None:
            is_b_r = torch.zeros(X_r.shape[0], device=device, dtype=torch.bool)

        logger.info(
            "Adaptive collocation round prepared.",
            pass_label=pass_label,
            n_points=int(stats["n_points"]),
            n_boundary=int(stats["n_boundary"]),
            frac_boundary=stats["frac_boundary"],
        )

        if r == 0 or system is None:
            X_all.append(X_r)
            V_all.append(V_r)
            is_b_all.append(is_b_r)

            if system is None:
                try:
                    system = cheap_discover_images_for_collocation(
                        spec,
                        ["axis_point"],
                        logger,
                        subtract_physical_potential=subtract_physical_potential,
                    )
                except Exception as exc:
                    logger.warning(
                        "Cheap provisional solve failed; continuing without it.",
                        error=str(exc),
                        pass_label=pass_label,
                    )
            continue

        with torch.no_grad():
            V_img_r = system.potential(X_r)
            resid = (V_img_r - V_r).abs()

        k = min(int(n_points), int(resid.shape[0]))
        if k <= 0:
            logger.warning(
                "Adaptive round retained zero points after residual scoring.",
                pass_label=pass_label,
            )
            continue

        top_idx = torch.topk(resid, k=k, largest=True).indices
        X_all.append(X_r[top_idx])
        V_all.append(V_r[top_idx])
        is_b_all.append(is_b_r[top_idx])

        logger.info(
            "Adaptive collocation residual selection.",
            pass_label=pass_label,
            n_selected=int(top_idx.shape[0]),
            k_requested=int(k),
        )

    if not X_all:
        return (
            torch.empty(0, 3, device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=torch.bool),
        )

    X = torch.cat(X_all, dim=0)
    V = torch.cat(V_all, dim=0)
    is_boundary = torch.cat(is_b_all, dim=0) if is_b_all else torch.zeros(
        X.shape[0], device=device, dtype=torch.bool
    )

    logger.info(
        "Adaptive collocation assembled.",
        n_rounds=len(X_all),
        n_points=int(X.shape[0]),
        n_boundary=int(is_boundary.sum().item()),
    )

    return X, V, is_boundary


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
    solver: str = "ista",
    solver_explicit: bool = False,
    operator_mode: Optional[bool] = None,
    lista_model: Optional[Any] = None,
    aug_lagrange: Optional[Any] = None,
    adaptive_collocation_rounds: int = 1,
    n_points_override: Optional[int] = None,
    ratio_boundary_override: Optional[float] = None,
    lambda_group: float = 0.0,
    basis_generator: Optional[BasisGenerator] = None,
    basis_generator_mode: str = "static_only",
    geo_encoder: Optional[Any] = None,
    weight_prior: Optional[torch.Tensor | Sequence[float]] = None,
    lambda_weight_prior: float | torch.Tensor = 0.0,
    weight_prior_label: Optional[str] = None,
    model_checkpoint: Optional[str] = None,
    gfn_checkpoint: Optional[str] = None,
    gfn_seed: Optional[int] = None,
    flow_checkpoint: Optional[str] = None,
    flow_steps: Optional[int] = None,
    flow_solver: Optional[str] = None,
    flow_temp: Optional[float] = None,
    flow_dtype: Optional[str] = None,
    flow_seed: Optional[int] = None,
    allow_random_flow: bool = False,
    subtract_physical_potential: bool = False,
    intensive: Optional[bool] = None,
    constraint_specs: Optional[list[Any]] = None,
    admm_cfg: Optional[Any] = None,
    constraint_mode: str = "none",
    gfdsl_program_dir: Optional[str] = None,
) -> ImageSystem:
    """Top-level entry point for sparse image discovery."""

    device = _get_default_device()
    dtype = _get_default_dtype()
    if intensive is True:
        os.environ["EDE_IMAGES_INTENSIVE"] = "1"
    intensive_mode = _intensive_enabled(intensive)

    solver_mode_raw = (solver or "ista").strip().lower()
    solver_mode = solver_mode_raw if solver_mode_raw else "ista"
    solver_user_explicit = bool(solver_explicit)
    if not solver_user_explicit and solver_mode in {"", "ista", "auto"} and lista_model is not None and aug_lagrange is None:
        solver_mode = "lista"

    aug_cfg = _normalize_aug_lagrange_cfg(aug_lagrange)
    if aug_lagrange is not None and aug_cfg is None:
        logger.warning("Invalid augmented Lagrangian config provided; disabling AL.")

    boundary_mode_env = os.getenv("EDE_IMAGES_BOUNDARY_MODE", "mix").strip().lower()
    boundary_mode = boundary_mode_env if boundary_mode_env in {"mix", "ratio", "penalty", "hard"} else "mix"
    try:
        boundary_penalty_default = float(os.getenv("EDE_IMAGES_BOUNDARY_PENALTY", "0.0"))
    except Exception:
        boundary_penalty_default = 0.0
    if aug_cfg is not None:
        boundary_penalty_default = 0.0

    operator_flag_env = os.getenv("EDE_IMAGES_USE_OPERATOR", "").strip().lower()
    use_operator = True if operator_mode is None else bool(operator_mode)
    operator_env_applied = operator_mode is None
    if operator_env_applied:
        if operator_flag_env in {"1", "true", "yes", "on"}:
            use_operator = True
        elif operator_flag_env in {"0", "false", "no", "off"}:
            use_operator = False
    try:
        operator_min_cols = int(os.getenv("EDE_IMAGES_OPERATOR_MIN_COLS", "0"))
    except Exception:
        operator_min_cols = 0

    use_lista = solver_mode == "lista"
    if use_lista and lista_model is None:
        logger.warning("LISTA solver requested but no model provided; falling back to ISTA.")
        solver_mode = "ista"
        use_lista = False
    if use_lista:
        try:
            lista_model = lista_model.to(device=device, dtype=dtype)  # type: ignore[assignment]
        except Exception:
            pass
        try:
            lista_model.eval()  # type: ignore[call-arg]
        except Exception:
            pass

    logger.info(
        "Sparse image discovery started.",
        basis_types=basis_types,
        n_max=int(n_max),
        reg_l1=float(reg_l1),
        device=str(device),
        dtype=str(dtype),
        solver=solver_mode,
        operator_mode=bool(use_operator),
        aug_lagrange=bool(aug_cfg is not None),
        model_checkpoint=model_checkpoint,
        gfn_checkpoint=gfn_checkpoint,
        gfn_seed=gfn_seed,
        flow_checkpoint=flow_checkpoint,
        flow_steps=flow_steps,
        flow_solver=flow_solver,
        flow_temp=flow_temp,
        flow_dtype=flow_dtype,
        flow_seed=flow_seed,
        allow_random_flow=bool(allow_random_flow),
        intensive=bool(intensive_mode),
    )

    mode_env = os.getenv("EDE_IMAGES_BASIS_GENERATOR_MODE", "")
    mode_raw = mode_env if mode_env else basis_generator_mode
    mode = _normalize_generator_mode(mode_raw)
    cand_multiplier = 16 if intensive_mode else 4
    learned_multiplier = max(2, cand_multiplier // 2)
    cand_rng = _make_torch_rng(device)
    if gfn_seed is None:
        seed_env = os.getenv("EDE_IMAGES_GFN_SEED", "").strip()
        if seed_env:
            try:
                gfn_seed = int(seed_env)
            except Exception:
                gfn_seed = None

    gfdsl_candidates: List[ImageBasisElement] = []
    if mode == "gfdsl":
        gfdsl_dir = gfdsl_program_dir or os.getenv("EDE_GFDSl_PROGRAM_DIR", "").strip()
        if not gfdsl_dir:
            raise ValueError("gfdsl mode requires gfdsl_program_dir or EDE_GFDSl_PROGRAM_DIR.")
        limit_env = os.getenv("EDE_GFDSl_MAX_PROGRAMS", "").strip()
        limit = None
        if limit_env:
            try:
                limit = int(limit_env)
            except Exception:
                limit = None
        from electrodrive.gfdsl.program_loader import load_gfdsl_programs

        gfdsl_candidates = load_gfdsl_programs(
            Path(gfdsl_dir),
            spec=spec,
            device=device,
            dtype=dtype,
            eval_backend="operator",
            logger=logger,
            limit=limit,
        )
        logger.info(
            "Using GFDSL program generator.",
            mode=mode,
            program_dir=str(gfdsl_dir),
            n_candidates=len(gfdsl_candidates),
        )

    if mode in {"diffusion", "hybrid_diffusion"} and basis_generator is None:
        try:
            from electrodrive.images.diffusion_generator import DiffusionBasisGenerator, DiffusionGeneratorConfig

            basis_generator = DiffusionBasisGenerator(
                DiffusionGeneratorConfig(
                    k_max=max(4, n_max * learned_multiplier),
                    n_steps=64 if intensive_mode else 32,
                    hidden_dim=256 if intensive_mode else 128,
                    n_layers=6 if intensive_mode else 4,
                    n_heads=8 if intensive_mode else 4,
                )
            )
            logger.info("Using diffusion BasisGenerator.", mode=mode)
        except Exception as exc:
            logger.warning(
                "Failed to construct DiffusionBasisGenerator; reverting to static-only.",
                error=str(exc),
            )
            mode = "static_only"
    if mode == "gfn":
        try:
            from electrodrive.gfn.integration import GFlowNetProgramGenerator
        except Exception as exc:
            raise ValueError(f"Failed to import GFlowNetProgramGenerator: {exc}") from exc
        if basis_generator is not None and not isinstance(basis_generator, GFlowNetProgramGenerator):
            raise ValueError("GFlowNet mode requires a GFlowNetProgramGenerator instance.")
        if basis_generator is None:
            if not gfn_checkpoint:
                raise ValueError("GFlowNet generator requires a checkpoint; random weights are not allowed.")
            basis_generator = GFlowNetProgramGenerator(
                checkpoint_path=gfn_checkpoint,
                device=device,
                dtype=dtype,
            )
        logger.info("Using GFlowNet BasisGenerator.", mode=mode)
    if mode == "gfn_flow":
        if device.type == "cpu" and torch.cuda.is_available():
            device = torch.device("cuda")
        try:
            from electrodrive.flows.types import FlowConfig
            from electrodrive.gfn.integration import HybridGFlowFlowGenerator
        except Exception as exc:
            raise ValueError(f"Failed to import HybridGFlowFlowGenerator: {exc}") from exc
        if basis_generator is not None and not isinstance(basis_generator, HybridGFlowFlowGenerator):
            raise ValueError("gfn_flow mode requires a HybridGFlowFlowGenerator instance.")
        if basis_generator is None:
            if not gfn_checkpoint:
                raise ValueError("gfn_flow requires a GFlowNet checkpoint; random weights are not allowed.")
            if not flow_checkpoint and not allow_random_flow:
                raise ValueError("gfn_flow requires a flow checkpoint unless allow_random_flow is enabled.")
            base_cfg = FlowConfig()
            flow_cfg = FlowConfig(
                latent_dim=base_cfg.latent_dim,
                model_dim=base_cfg.model_dim,
                max_tokens=base_cfg.max_tokens,
                max_ast_len=base_cfg.max_ast_len,
                n_steps=int(flow_steps) if flow_steps is not None else base_cfg.n_steps,
                solver=str(flow_solver or base_cfg.solver),
                temperature=float(flow_temp) if flow_temp is not None else base_cfg.temperature,
                dtype=str(flow_dtype or base_cfg.dtype),
                seed=flow_seed,
            )
            basis_generator = HybridGFlowFlowGenerator(
                checkpoint_path=gfn_checkpoint,
                flow_checkpoint_path=flow_checkpoint,
                flow_config=flow_cfg,
                allow_random_flow=allow_random_flow,
                device=device,
                dtype=dtype,
            )
        logger.info("Using Hybrid GFlowFlow BasisGenerator.", mode=mode)

    static_candidates: List[ImageBasisElement] = []
    if mode in {"static_only", "static_plus_learned", "hybrid_diffusion"}:
        static_candidates = generate_candidate_basis(
            spec,
            basis_types=basis_types,
            n_candidates=max(1, n_max * cand_multiplier),
            device=device,
            dtype=dtype,
            rng=cand_rng,
        )

    learned_candidates: List[ImageBasisElement] = []
    gfn_candidates: List[ImageBasisElement] = []
    if mode in {"gfn", "gfn_flow"} and basis_generator is not None:
        gfn_candidates = _build_gfn_candidates(
            spec=spec,
            gfn_generator=basis_generator,
            geo_encoder=geo_encoder,
            n_candidates=max(1, n_max * learned_multiplier),
            device=device,
            dtype=dtype,
            logger=logger,
            seed=gfn_seed,
            mode=mode,
        )
    elif basis_generator is not None and mode in {"static_plus_learned", "learned_only", "diffusion", "hybrid_diffusion"}:
        learned_candidates = _build_learned_candidates(
            spec=spec,
            basis_generator=basis_generator,
            geo_encoder=geo_encoder,
            n_candidates=max(1, n_max * learned_multiplier),
            device=device,
            dtype=dtype,
            logger=logger,
        )

    if mode == "gfdsl":
        candidates = gfdsl_candidates
    elif mode in {"gfn", "gfn_flow"}:
        candidates = gfn_candidates
    elif mode in {"learned_only", "diffusion"}:
        candidates = learned_candidates
    elif mode in {"static_plus_learned", "hybrid_diffusion"}:
        combined = static_candidates + learned_candidates
        candidates = _dedup_candidates_by_position(combined, tol=1e-6) if mode == "hybrid_diffusion" else combined
    else:
        candidates = static_candidates

    candidates = _maybe_shuffle_candidates(candidates, logger)
    if operator_env_applied and operator_min_cols > 0 and len(candidates) >= operator_min_cols:
        use_operator = True

    if not candidates:
        logger.warning("No candidate basis elements generated for this configuration.")
        return ImageSystem([], torch.zeros(0, device=device, dtype=dtype))

    colloc_n_points_base, colloc_ratio_base = _resolve_collocation_params(
        n_points_override, ratio_boundary_override
    )
    colloc_n_points_eff = colloc_n_points_base
    candidates_work: List[ImageBasisElement] = candidates
    try:
        adaptive_rounds_env = int(os.getenv("EDE_IMAGES_ADAPTIVE_ROUNDS", "0"))
    except Exception:
        adaptive_rounds_env = 0
    try:
        adaptive_passes_env = int(os.getenv("EDE_IMAGES_ADAPTIVE_PASSES", "0"))
    except Exception:
        adaptive_passes_env = 0
    adaptive_rounds = adaptive_collocation_rounds
    if adaptive_rounds_env > 0:
        adaptive_rounds = adaptive_rounds_env
    elif adaptive_passes_env > 0:
        adaptive_rounds = 1 + adaptive_passes_env
    adaptive_rounds = max(1, int(adaptive_rounds))

    need_boundary_mask = (
        boundary_weight is not None
        or boundary_mode in {"mix", "ratio", "penalty", "hard"}
        or aug_cfg is not None
    )

    attempt = 0
    while True:
        try:
            weight_prior_vec: Optional[torch.Tensor] = None
            lambda_weight_prior_vec: Optional[torch.Tensor | float] = None
            if weight_prior is not None:
                w_prior = torch.as_tensor(weight_prior, device=device, dtype=dtype).view(-1)
                if w_prior.numel() == 1:
                    w_prior = w_prior.expand(len(candidates_work))
                if w_prior.numel() < len(candidates_work):
                    w_prior = torch.cat(
                        [w_prior, torch.zeros(len(candidates_work) - w_prior.numel(), device=device, dtype=dtype)],
                        dim=0,
                    )
                elif w_prior.numel() > len(candidates_work):
                    w_prior = w_prior[: len(candidates_work)]
                weight_prior_vec = w_prior

                if torch.is_tensor(lambda_weight_prior):
                    lam_prior = torch.as_tensor(lambda_weight_prior, device=device, dtype=dtype).view(-1)
                    if lam_prior.numel() == 1:
                        lam_prior = lam_prior.expand(len(candidates_work))
                    if lam_prior.numel() < len(candidates_work):
                        lam_prior = torch.cat(
                            [lam_prior, torch.zeros(len(candidates_work) - lam_prior.numel(), device=device, dtype=dtype)],
                            dim=0,
                        )
                    elif lam_prior.numel() > len(candidates_work):
                        lam_prior = lam_prior[: len(candidates_work)]
                    if float(lam_prior.max().item()) > 0.0:
                        lambda_weight_prior_vec = lam_prior
                else:
                    lam_scalar = float(lambda_weight_prior)
                    if lam_scalar > 0.0:
                        lambda_weight_prior_vec = lam_scalar
                lam_log = 0.0
                if torch.is_tensor(lambda_weight_prior_vec):
                    lam_log = float(lambda_weight_prior_vec.max().item())
                elif lambda_weight_prior_vec is not None:
                    lam_log = float(lambda_weight_prior_vec)
                logger.info(
                    "Weight-mode prior enabled.",
                    lambda_weight_prior=lam_log,
                    prior_label=weight_prior_label,
                )

            try:
                group_ids_full = compute_group_ids(candidates_work, device=device, dtype=torch.long)
            except Exception:
                group_ids_full = None

            per_elem_reg_vec: Optional[torch.Tensor] = None
            if per_type_reg:
                reg_list = [float(per_type_reg.get(elem.type, reg_l1)) for elem in candidates_work]
                per_elem_reg_vec = torch.tensor(reg_list, device=device, dtype=dtype)

            lambda_group_vec = _build_lambda_group_vector(
                candidates_work,
                device=device,
                dtype=dtype,
                lambda_default=lambda_group,
            )

            colloc_n_points = colloc_n_points_eff
            colloc_ratio = colloc_ratio_base
            rng = _make_collocation_rng()
            if adaptive_rounds <= 1:
                colloc_out = get_collocation_data(
                    spec,
                    logger,
                    device=device,
                    dtype=dtype,
                    return_is_boundary=need_boundary_mask,
                    rng=rng,
                    n_points_override=colloc_n_points,
                    ratio_override=colloc_ratio,
                    subtract_physical_potential=subtract_physical_potential,
                )
                if isinstance(colloc_out, tuple) and len(colloc_out) == 3:
                    colloc_pts, target, is_boundary = colloc_out  # type: ignore[misc]
                else:
                    colloc_pts, target = colloc_out  # type: ignore[misc]
                    is_boundary = None
            else:
                colloc_pts, target, is_boundary = build_adaptive_collocation(
                    spec=spec,
                    logger=logger,
                    device=device,
                    dtype=dtype,
                    n_points=colloc_n_points,
                    ratio_boundary=colloc_ratio,
                    n_rounds=adaptive_rounds,
                    initial_system=None,
                    subtract_physical_potential=subtract_physical_potential,
                )
                if not need_boundary_mask:
                    is_boundary = None

            if colloc_pts.shape[0] == 0:
                return ImageSystem([], torch.zeros(0, device=device, dtype=dtype))
            try:
                frac_boundary = float(is_boundary.float().mean().item()) if is_boundary is not None and bool(is_boundary.numel()) else 0.0
            except Exception:
                frac_boundary = 0.0
            try:
                logger.info(
                    "collocation_stats",
                    n_points=int(colloc_pts.shape[0]),
                    frac_boundary=float(frac_boundary),
                    adaptive_rounds=int(adaptive_rounds),
                )
            except Exception:
                pass

            if lista_model is not None and hasattr(lista_model, "K"):
                try:
                    lista_K = int(getattr(lista_model, "K"))
                    if lista_K != len(candidates_work):
                        logger.error(
                            "LISTA checkpoint dimension mismatch.",
                            lista_K=int(lista_K),
                            n_candidates=len(candidates_work),
                        )
                        return ImageSystem([], torch.zeros(0, device=device, dtype=dtype))
                except Exception:
                    pass

            A_dict = assemble_basis(candidates_work, colloc_pts, use_operator, device, dtype)
            try:
                col_min = float("nan")
                col_max = float("nan")
                if isinstance(A_dict, BasisOperator):
                    col_norms = getattr(A_dict, "col_norms", None)
                    if col_norms is None:
                        col_norms = A_dict.estimate_col_norms(colloc_pts)  # type: ignore[arg-type]
                        A_dict.col_norms = col_norms
                    col_min = float(col_norms.min().item()) if col_norms.numel() else float("nan")
                    col_max = float(col_norms.max().item()) if col_norms.numel() else float("nan")
                else:
                    col_norms = torch.linalg.norm(A_dict, dim=0)
                    col_min = float(col_norms.min().item()) if col_norms.numel() else float("nan")
                    col_max = float(col_norms.max().item()) if col_norms.numel() else float("nan")
                logger.info(
                    "basis_operator_stats",
                    K=len(candidates_work),
                    N=int(colloc_pts.shape[0]),
                    operator_mode=bool(use_operator),
                    col_norm_min=col_min,
                    col_norm_max=col_max,
                )
            except Exception:
                pass

            def _build_system_from_weights(w_vec: torch.Tensor, support: List[int]) -> ImageSystem:
                if w_vec.numel() == 0 or not support:
                    return ImageSystem([], torch.zeros(0, device=device, dtype=dtype))
                support_lim = support[: max(0, int(n_max))]
                elems = [candidates_work[i] for i in support_lim]
                weights_sel = w_vec[support_lim]
                system_out = ImageSystem(elems, weights_sel)
                if restarts > 0:
                    max_iter_lbfgs = 200 if intensive_mode else None
                    extent = _domain_extent_from_spec(spec) if intensive_mode else None
                    system_out = optimize_parameters_lbfgs(
                        system_out,
                        colloc_pts,
                        target,
                        logger,
                        domain_extent=extent,
                        max_iter_override=max_iter_lbfgs,
                    )
                return system_out

            def _run_sparse_once(
                A_in: torch.Tensor | BasisOperator,
                target_in: torch.Tensor,
                support_cap: Optional[int],
                with_stats: bool = False,
            ) -> Tuple[torch.Tensor, List[int]] | Tuple[torch.Tensor, List[int], Dict[str, Any]]:
                return solve_sparse(
                    A_in,
                    colloc_pts,
                    target_in,
                    is_boundary if need_boundary_mask else None,
                    logger,
                    reg_l1=reg_l1,
                    solver=solver_mode,
                    lista_model=lista_model if solver_mode == "lista" else None,
                    aug_lagrange_cfg=aug_cfg,
                    boundary_weight=boundary_weight if aug_cfg is None else None,
                    boundary_mode=boundary_mode,
                    boundary_penalty_default=boundary_penalty_default,
                    per_elem_reg=per_elem_reg_vec,
                    group_ids=group_ids_full,
                    lambda_group=lambda_group_vec if lambda_group_vec is not None else lambda_group,
                    weight_prior=weight_prior_vec,
                    lambda_weight_prior=lambda_weight_prior_vec if lambda_weight_prior_vec is not None else 0.0,
                    normalize_columns=True,
                    ls_refit=True,
                    support_k=support_cap,
                    max_iter=1000,
                    tol=1e-6,
                    lista_refine=True,
                    return_stats=with_stats,
                    constraints=constraint_specs,
                    admm_cfg=admm_cfg,
                    constraint_mode=constraint_mode,
                )

            if two_stage:
                nonlocal_types = [
                    "ring",
                    "ring_gauss",
                    "poloidal_ring",
                    "ring_ladder_inner",
                    "ring_ladder_outer",
                    "toroidal_mode_cluster",
                ]
                non_idx = [i for i, c in enumerate(candidates_work) if c.type in nonlocal_types]
                point_idx = [i for i, c in enumerate(candidates_work) if c.type == "point"]

                weights_full = torch.zeros(len(candidates_work), device=device, dtype=dtype)
                target_res = target

                if non_idx:
                    A_non = assemble_basis([candidates_work[i] for i in non_idx], colloc_pts, use_operator, device, dtype)
                    w_non, _ = _run_sparse_once(A_non, target_res, min(len(non_idx), n_max))
                    for idx_val, w_val in zip(non_idx, w_non):
                        weights_full[idx_val] = w_val
                    try:
                        target_res = target - _matvec(A_non, w_non, colloc_pts if isinstance(A_non, BasisOperator) else None)
                    except Exception as exc:
                        logger.warning("Two-stage residual update failed; using original targets.", error=str(exc))

                if point_idx:
                    A_point = assemble_basis([candidates_work[i] for i in point_idx], colloc_pts, use_operator, device, dtype)
                    w_point, _ = _run_sparse_once(A_point, target_res, min(len(point_idx), n_max))
                    for idx_val, w_val in zip(point_idx, w_point):
                        weights_full[idx_val] = w_val

                w_out = weights_full
                support_full = [
                    int(i) for i in torch.argsort(torch.abs(w_out), descending=True).tolist()
                    if float(torch.abs(w_out[int(i)]).item()) > 0.0
                ]
                system = _build_system_from_weights(w_out, support_full)
                return system

            w_out = _run_sparse_once(A_dict, target, n_max if n_max > 0 else None, with_stats=True)
            if isinstance(w_out, tuple) and len(w_out) == 3:
                w_final, support_final, solver_stats = w_out  # type: ignore[misc]
            else:
                w_final, support_final = w_out  # type: ignore[misc]
                solver_stats = {}
            system = _build_system_from_weights(w_final, support_final)
            if solver_stats:
                try:
                    logger.info(
                        "solver_stats",
                        solver=str(solver_stats.get("solver")),
                        iters=int(solver_stats.get("iters", -1)),
                        converged=bool(solver_stats.get("converged", False)),
                        bc_norm=float(solver_stats.get("bc_norm", float("nan"))),
                        int_norm=float(solver_stats.get("int_norm", float("nan"))),
                        lambda_group=float(solver_stats.get("lambda_group", float("nan"))),
                        aug_rho_final=float(solver_stats.get("aug_rho_final", float("nan"))),
                    )
                except Exception:
                    pass
            # Numeric diagnostics for manifest / Gate 2/3 linkage.
            diagnostics = _numeric_diagnostics(system, colloc_pts, target)
            system.metadata.update(diagnostics)
            return system
        except RuntimeError as exc:
            is_oom = "out of memory" in str(exc).lower()
            if not (intensive_mode and is_oom and attempt == 0):
                raise
            logger.warning(
                "Intensive mode scaled back due to VRAM limits.",
                attempt=int(attempt + 1),
                n_points=int(colloc_n_points_eff),
                n_candidates=len(candidates_work),
                error=str(exc),
            )
            colloc_n_points_eff = max(256, int(colloc_n_points_eff // 2))
            new_k = max(1, len(candidates_work) // 2)
            candidates_work = candidates_work[:new_k]
            attempt += 1
            continue
