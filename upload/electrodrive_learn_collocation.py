from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from electrodrive.utils.config import EPS_0, K_E

from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.core.images import (
    AnalyticSolution,
    potential_plane_halfspace,
    potential_sphere_grounded,
    potential_line_cylinder2d_grounded,
    potential_parallel_planes_subset,
)
from electrodrive.learn.encoding import encode_spec, ENCODING_DIM
from electrodrive.fmm3d.logging_utils import ConsoleLogger
from electrodrive.debug import bem_intercept

try:
    # BEM is optional for CPU-only / minimal installs.
    from electrodrive.core.bem import bem_solve, BEMSolution  # type: ignore
    from electrodrive.utils.config import BEMConfig

    BEM_AVAILABLE = True
except Exception:  # pragma: no cover - handled gracefully at runtime
    from typing import Any as _AnyType  # fallback for type checking

    bem_solve = None  # type: ignore
    BEMSolution = _AnyType  # type: ignore
    BEMConfig = _AnyType  # type: ignore
    BEM_AVAILABLE = False


logger = logging.getLogger("EDE.Learn.Collocation")


class _NullLogger:
    """Minimal no-op logger for BEM oracles when we don't want console spam.

    Matches the .info/.warning/.error interface expected by bem_solve but
    silently drops all messages. Callers can still pass an explicit logger
    via bem_cfg["logger"] when verbose logging is desired.
    """

    def info(self, *args: Any, **kwargs: Any) -> None:
        pass

    def debug(self, *args: Any, **kwargs: Any) -> None:
        pass

    def warning(self, *args: Any, **kwargs: Any) -> None:
        pass

    def error(self, *args: Any, **kwargs: Any) -> None:
        pass


OracleSolution = Union[AnalyticSolution, "BEMSolution"]


# ---------------------------------------------------------------------------
# Analytic oracle selection (mirrors electrodrive.learn.dataset)
# ---------------------------------------------------------------------------


def _solve_analytic(spec: CanonicalSpec) -> Optional[AnalyticSolution]:
    """
    Try to construct an AnalyticSolution for a small set of canonical
    geometries.

    This mirrors the logic historically implemented in the learning
    stack and is intentionally conservative: if any of the structural
    checks fail, we return None and let the caller fall back to BEM.
    """
    ctypes = (
        sorted({c.get("type") for c in spec.conductors})
        if spec.conductors
        else []
    )

    # --- 1) Single grounded plane + single point charge ---
    if (
        ctypes == ["plane"]
        and len(spec.conductors) == 1
        and len(spec.charges) == 1
        and spec.charges[0]["type"] == "point"
    ):
        q = spec.charges[0]["q"]
        r0 = tuple(spec.charges[0]["pos"])
        c = spec.conductors[0]
        if (
            c.get("z", 0.0) == 0.0
            and c.get("potential", 0.0) == 0.0
            and r0[2] > 0
        ):
            return potential_plane_halfspace(q, r0)

    # --- 2) Single grounded sphere + single point charge ---
    if (
        ctypes == ["sphere"]
        and len(spec.conductors) == 1
        and len(spec.charges) == 1
        and spec.charges[0]["type"] == "point"
    ):
        q = spec.charges[0]["q"]
        r0 = tuple(spec.charges[0]["pos"])
        c = spec.conductors[0]
        radius = c.get("radius")
        center = tuple(c.get("center", [0.0, 0.0, 0.0]))
        if radius is not None and c.get("potential", 0.0) == 0.0:
            return potential_sphere_grounded(
                q,
                r0,
                center,
                float(radius),
            )

    # --- 3) Cylinder / line-charge 2D image construction ---
    if (
        ("cylinder" in ctypes or "cylinder2D" in ctypes)
        and len(spec.conductors) == 1
        and len(spec.charges) == 1
        and spec.charges[0]["type"] in ("line_charge", "line")
    ):
        lambda_c = spec.charges[0].get("lambda")
        r0_2d = spec.charges[0].get("pos_2d")
        c = spec.conductors[0]
        radius = c.get("radius")
        potential = c.get("potential", 0.0)
        if (
            lambda_c is not None
            and r0_2d is not None
            and radius is not None
            and potential == 0.0
        ):
            return potential_line_cylinder2d_grounded(
                float(lambda_c),
                float(radius),
                tuple(r0_2d),
            )

    # --- 4) Parallel grounded planes + central point charge ---
    if (
        ctypes == ["plane"]
        and len(spec.conductors) == 2
        and len(spec.charges) == 1
        and spec.charges[0]["type"] == "point"
    ):
        z1 = spec.conductors[0].get("z")
        z2 = spec.conductors[1].get("z")
        p1 = spec.conductors[0].get("potential", 0.0)
        p2 = spec.conductors[1].get("potential", 0.0)
        if (
            z1 is not None
            and z2 is not None
            and p1 == 0.0
            and p2 == 0.0
            and abs(z1 + z2) < 1e-6
        ):
            d = abs(float(z1))
            q = spec.charges[0]["q"]
            r0 = tuple(spec.charges[0]["pos"])
            if abs(r0[2]) < d:
                return potential_parallel_planes_subset(
                    q,
                    r0,
                    d,
                    N_terms=30,
                )

    return None


# ---------------------------------------------------------------------------
# Oracle selection / BEM configuration
# ---------------------------------------------------------------------------


def get_oracle_solution(
    spec: CanonicalSpec,
    mode: str,
    bem_cfg: Dict[str, Any],
) -> Optional[OracleSolution]:
    """
    Select and construct a field oracle for a given spec.

    For analytic / image-charge geometries we expect the analytic shortcut
    and the BEM oracle to agree on a shared set of collocation points.

    For BEM oracles, we bias toward:
      * fp64 for numerical stability
      * GPU if available (do not force CPU)
      * near-field quadrature at evaluation points
        (but not in the GMRES matvec, which is expensive and CPU-only)
    """
    sol: Optional[OracleSolution] = None

    # 1) Analytic path (if explicitly requested or allowed by "auto").
    if mode in ("analytic", "auto"):
        try:
            sol = _solve_analytic(spec)
        except Exception:
            sol = None

    # 2) BEM oracle path.
    if sol is None and mode in ("bem", "auto") and BEM_AVAILABLE:
        try:
            cfg = BEMConfig()

            # Apply caller overrides first.
            if bem_cfg is None:
                bem_cfg = {}

            # Allow callers to override the BEM logger explicitly; by default
            # we use a quiet in-memory logger to avoid stdout/stderr issues.
            log_obj = bem_cfg.get("logger", None)
            if log_obj is None:
                log_obj = _NullLogger()

            has_near_quad_cfg = "use_near_quadrature" in bem_cfg
            has_near_quad_mv_cfg = "use_near_quadrature_matvec" in bem_cfg
            for k, v in bem_cfg.items():
                if k == "logger":
                    # handled separately
                    continue
                try:
                    setattr(cfg, k, v)
                except Exception:
                    continue

            # fp64 for stability.
            cfg.fp64 = True
            # IMPORTANT: do NOT force use_gpu here. Let cfg.use_gpu and the
            # caller config decide. On your machine, GPU is the fast path.

            # Disable near-smoothing; we want the raw Green's function.
            if hasattr(cfg, "near_alpha"):
                cfg.near_alpha = 0.0

            # Refinement: allow at least 3 passes if the caller didn't
            # already ask for more.
            try:
                base_max = int(getattr(cfg, "max_refine_passes", 3) or 3)
            except Exception:
                base_max = 3
            cfg.max_refine_passes = max(base_max, 3)

            try:
                base_min = int(getattr(cfg, "min_refine_passes", 1) or 1)
            except Exception:
                base_min = 1
            cfg.min_refine_passes = max(base_min, 1)

            # Use tight GMRES tolerance; don't loosen it.
            try:
                gmres_tol = float(getattr(cfg, "gmres_tol", 5e-8))
            except Exception:
                gmres_tol = 5e-8
            cfg.gmres_tol = min(gmres_tol, 5e-8)

            # Evaluation near-quadrature: good for potentials near surfaces.
            if hasattr(cfg, "use_near_quadrature") and not has_near_quad_cfg:
                cfg.use_near_quadrature = True
            if hasattr(cfg, "use_near_quadrature_matvec") and not has_near_quad_mv_cfg:
                # Default behaviour: keep GMRES matvec inexpensive unless the
                # caller explicitly opts in via bem_cfg.
                cfg.use_near_quadrature_matvec = False

            # Use either the caller-provided logger or a quiet no-op logger by
            # default. Callers can still pass ConsoleLogger() explicitly via
            # bem_cfg["logger"] if they want stdout JSON logs.
            out = bem_solve(spec, cfg, log_obj)  # type: ignore[arg-type]
            if isinstance(out, dict) and "solution" in out:
                sol = out["solution"]
        except Exception as e:  # pragma: no cover - defensive
            logger.error("BEM oracle failed: %s", e)

    return sol


# ---------------------------------------------------------------------------
# Collocation sampling + oracle evaluation
# ---------------------------------------------------------------------------

# Simple identity-based cache so repeated calls for the *same* CanonicalSpec
# (e.g. across epochs in ElectrostaticsJITDataset) reuse the expensive BEM
# solution instead of re-solving from scratch.
_ORACLE_CACHE: Dict[
    Tuple[int, str, Tuple[Tuple[str, str], ...]], OracleSolution
] = {}


def _bem_cfg_cache_key(bem_cfg: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    """
    Represent a config dict as a sorted tuple of stringified (k, v) pairs,
    which is stable enough for caching purposes.
    """
    return tuple(sorted((str(k), repr(v)) for k, v in bem_cfg.items()))


def _empty_batch(device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """Return an all-empty collocation batch on the requested device/dtype."""
    return {
        "X": torch.zeros(0, 3, device=device, dtype=dtype),
        "V_gt": torch.zeros(0, device=device, dtype=dtype),
        "is_boundary": torch.zeros(0, device=device, dtype=torch.bool),
        "mask_finite": torch.zeros(0, device=device, dtype=torch.bool),
        "encoding": torch.zeros(0, ENCODING_DIM, device=device, dtype=dtype),
    }


def _infer_geom_type_from_spec(spec: CanonicalSpec) -> str:
    """
    Heuristic geometry label used for the fast analytic paths.

    This roughly matches the geometry heuristics used for encoding and
    curriculum synthesis.
    """
    ctypes = (
        sorted({c.get("type") for c in spec.conductors})
        if spec.conductors
        else []
    )
    if ctypes == ["plane"]:
        if len(spec.conductors) == 1:
            return "plane"
        if len(spec.conductors) == 2:
            return "parallel_planes"
    if ctypes == ["sphere"] and len(spec.conductors) == 1:
        return "sphere"
    if ("cylinder" in ctypes or "cylinder2D" in ctypes) and len(spec.conductors) == 1:
        return "cylinder2D"
    if ("torus" in ctypes or "toroid" in ctypes) and len(spec.conductors) == 1:
        return "torus"
    return "unknown"


def _infer_bbox_for_spec(spec: CanonicalSpec) -> float:
    """
    Heuristic bounding-box extent for collocation sampling.

    For planes we mirror the finite BEM plane patch extent used in
    electrodrive.core.bem_mesh.generate_mesh (L = max(8, 16 * dmin)).
    For spheres / cylinders we use a modest multiple of the radius and
    fall back to a global default otherwise.
    """
    bbox = 10.0

    conductors = getattr(spec, "conductors", None) or []
    charges = getattr(spec, "charges", None) or []

    if not conductors:
        return bbox

    for c in conductors:
        ctype = c.get("type")
        if ctype == "plane":
            z = float(c.get("z", 0.0))
            dmin = 0.3
            for ch in charges:
                if ch.get("type") != "point":
                    continue
                try:
                    zq = float(ch["pos"][2])
                except Exception:
                    continue
                dmin = max(dmin, abs(zq - z))
            L = max(8.0, 16.0 * dmin)
            return float(L)

        if ctype == "sphere":
            try:
                r = float(c.get("radius", 1.0))
            except Exception:
                continue
            return float(4.0 * r)

        if ctype in ("cylinder", "cylinder2D"):
            try:
                r = float(c.get("radius", 1.0))
            except Exception:
                continue
            return float(4.0 * r)

        if ctype in ("torus", "toroid"):
            try:
                R = float(c.get("major_radius", c.get("radius", 1.0)))
                a = float(c.get("minor_radius", 0.25 * R))
            except Exception:
                continue
            return float(4.0 * (R + a))

    return bbox


def _sample_points_for_spec(
    spec: CanonicalSpec,
    n_points: int,
    ratio_boundary: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Sample interior + boundary points for a given spec.

    This is a direct, RNG-parameterised transplant of the logic in
    ElectrostaticsJITDataset._sample_and_evaluate, but split out so it
    can be reused by the discovery engine.
    """
    N = int(n_points)
    N_boundary = int(N * float(ratio_boundary))
    N_interior = N - N_boundary
    bbox = _infer_bbox_for_spec(spec)

    # Interior: uniform box sampling
    points_int = rng.uniform(-bbox / 2.0, bbox / 2.0, (N_interior, 3))

    # For finite plane patches (single plane or parallel planes), restrict
    # interior sampling in x/y (and now z) to a central band so that the
    # finite BEM geometry is a good approximation to the infinite-plane
    # analytic shortcuts used in the image-charge formulas.
    try:
        geom = _infer_geom_type_from_spec(spec)
    except Exception:
        geom = "unknown"

    if geom in ("plane", "parallel_planes"):
        # Restrict x/y to the central 40% of the patch in each direction.
        alpha = 0.4
        points_int[:, 0] *= alpha
        points_int[:, 1] *= alpha

        # Additionally, restrict z to a band around the planes and any point
        # charges, rather than the full [-bbox/2, +bbox/2] extent. This avoids
        # sampling deep far-field regions where a finite patch deviates most
        # from the infinite-plane analytic model.
        conductors = getattr(spec, "conductors", None) or []
        charges = getattr(spec, "charges", None) or []
        z_vals: List[float] = []

        for c in conductors:
            if c.get("type") == "plane":
                try:
                    zc = float(c.get("z", 0.0))
                    z_vals.append(zc)
                except Exception:
                    continue

        for ch in charges:
            if ch.get("type") == "point":
                try:
                    pos = ch.get("pos") or ch.get("position")
                    if pos is not None:
                        zq = float(pos[2])
                        z_vals.append(zq)
                except Exception:
                    continue

        if z_vals and N_interior > 0:
            z_min = min(z_vals)
            z_max = max(z_vals)
            z_center = 0.5 * (z_min + z_max)
            # Half-span at least 0.3, plus a modest margin beyond the
            # extremal conductor/charge positions.
            half_span = max(0.3, 0.5 * (z_max - z_min))
            z_lo = z_center - 2.0 * half_span
            z_hi = z_center + 2.0 * half_span
            # Clamp to the original bbox in case of pathological specs.
            z_lo = max(z_lo, -bbox / 2.0)
            z_hi = min(z_hi, bbox / 2.0)
            if z_hi <= z_lo:
                # Fallback: use a central band of the original box.
                z_lo = -0.25 * bbox
                z_hi = 0.25 * bbox
            points_int[:, 2] = rng.uniform(z_lo, z_hi, size=N_interior)

    # Boundary: per-conductor sampling, reusing the same shape heuristics
    points_bnd_list: List[np.ndarray] = []
    conductors = getattr(spec, "conductors", None) or []
    if N_boundary > 0 and conductors:
        N_per = max(1, N_boundary // len(conductors))
        for c in conductors:
            ctype = c.get("type")
            if ctype == "plane":
                z = float(c.get("z", 0.0))
                # For plane/parallel_planes, restrict boundary sampling in x/y
                # to the same central region we use for interior points. This
                # avoids over-sampling near the patch edges where finite-patch
                # BEM and infinite-plane analytic shortcuts differ most.
                if geom in ("plane", "parallel_planes"):
                    alpha = 0.4
                    span = (bbox / 2.0) * alpha
                else:
                    span = bbox / 2.0
                xy = rng.uniform(-span, span, (N_per, 2))
                points_bnd_list.append(
                    np.c_[xy, np.full(N_per, z, dtype=float)]
                )
            elif ctype == "sphere":
                r = float(c.get("radius", 1.0))
                center = np.array(
                    c.get("center", [0.0, 0.0, 0.0]),
                    dtype=float,
                )
                vec = rng.standard_normal((N_per, 3))
                vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
                # Sample slightly off the surface
                eps = 1e-3 * r
                points_bnd_list.append(vec * (r + eps) + center)
            elif ctype in ("cylinder", "cylinder2D"):
                r = float(c.get("radius", 1.0))
                phi = rng.uniform(0.0, 2.0 * math.pi, N_per)
                z = rng.uniform(-bbox / 2.0, bbox / 2.0, N_per)
                points_bnd_list.append(
                    np.stack(
                        [
                            r * np.cos(phi),
                            r * np.sin(phi),
                            z,
                        ],
                        axis=1,
                    )
                )
            elif ctype in ("torus", "toroid"):
                R = float(c.get("major_radius", c.get("radius", 1.0)))
                a = float(c.get("minor_radius", 0.25 * R))
                center = np.array(
                    c.get("center", [0.0, 0.0, 0.0]),
                    dtype=float,
                )
                u = rng.uniform(0.0, 2.0 * math.pi, N_per)
                v = rng.uniform(0.0, 2.0 * math.pi, N_per)
                cosu = np.cos(u)
                sinu = np.sin(u)
                cosv = np.cos(v)
                sinv = np.sin(v)
                x = (R + a * cosv) * cosu
                y = (R + a * cosv) * sinu
                z = a * sinv
                pts = np.stack([x, y, z], axis=1) + center
                # Slightly push outward along approximate normal to avoid duplicate surface points.
                eps = 1e-3 * max(a, R)
                nx = cosv * cosu
                ny = cosv * sinu
                nz = sinv
                pts += np.stack([nx, ny, nz], axis=1) * eps
                points_bnd_list.append(pts)

    if points_bnd_list:
        points_bnd = np.vstack(points_bnd_list)
    else:
        points_bnd = np.empty((0, 3), dtype=float)

    N_bnd = points_bnd.shape[0]
    if N_bnd > 0:
        points_np = np.vstack([points_int, points_bnd])
    else:
        points_np = points_int

    N_total = points_np.shape[0]
    is_boundary = torch.zeros(N_total, dtype=torch.bool)
    if N_bnd > 0:
        is_boundary[-N_bnd:] = True
    return points_np, is_boundary


def _evaluate_oracle_on_points(
    spec: CanonicalSpec,
    solution: OracleSolution,
    geom_type: str,
    points_np: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Evaluate an oracle (analytic or BEM) on a batch of 3D points.

    The analytic path keeps the optimised vectorised implementations for
    the common curriculum geometries (plane, sphere, cylinder2D) and falls
    back to AnalyticSolution.eval otherwise.  The BEM path uses
    BEMSolution.eval_V_E_batched when available.

    Note
    ----
    The fast analytic paths below intentionally work in the same "well-
    scaled" reduced units as the historical learning stack: they omit the
    global Coulomb constant K_E = 1 / (4π ε₀). As a result, BEM and
    analytic results differ by a fixed 1 / EPS_0 scale factor; unit tests
    compensate by rescaling BEM outputs when comparing analytic vs BEM.
    """
    # Analytic solution branch ------------------------------------------------
    if isinstance(solution, AnalyticSolution):
        fast_done = False
        V_gt: Optional[torch.Tensor] = None

        try:
            # 1) Grounded plane at z=0 with one point charge
            if (
                geom_type == "plane"
                and len(spec.conductors) == 1
                and len(spec.charges) == 1
                and spec.charges[0]["type"] == "point"
            ):
                c = spec.conductors[0]
                if (
                    c.get("type") == "plane"
                    and abs(c.get("z", 0.0)) < 1e-12
                    and c.get("potential", 0.0) == 0.0
                ):
                    q = float(spec.charges[0]["q"])
                    x0, y0, z0 = map(float, spec.charges[0]["pos"])
                    R = points_np - np.array([x0, y0, z0], dtype=np.float64)
                    Rm = points_np - np.array([x0, y0, -z0], dtype=np.float64)
                    r = np.linalg.norm(R, axis=1).clip(1e-9, None)
                    rm = np.linalg.norm(Rm, axis=1).clip(1e-9, None)
                    V_gt_np = K_E * (q / r - q / rm)
                    V_gt = torch.from_numpy(V_gt_np).to(device=device, dtype=dtype)
                    fast_done = True

            # 2) Grounded sphere (centered or shifted) with one point charge
            if (
                not fast_done
                and geom_type == "sphere"
                and len(spec.conductors) == 1
                and len(spec.charges) == 1
                and spec.charges[0]["type"] == "point"
            ):
                c = spec.conductors[0]
                if c.get("type") == "sphere" and c.get("potential", 0.0) == 0.0:
                    a = float(c.get("radius"))
                    cx, cy, cz = map(float, c.get("center", [0.0, 0.0, 0.0]))
                    q = float(spec.charges[0]["q"])
                    x0, y0, z0 = map(float, spec.charges[0]["pos"])
                    r0c = np.array([x0 - cx, y0 - cy, z0 - cz], dtype=np.float64)
                    r0_norm = float(np.linalg.norm(r0c)) + 1e-12
                    q_img = -(a / r0_norm) * q
                    r_img = (a * a / (r0_norm * r0_norm)) * r0c
                    r_img_world = np.array([cx, cy, cz], dtype=np.float64) + r_img
                    R = points_np - np.array([x0, y0, z0], dtype=np.float64)
                    Ri = points_np - r_img_world
                    r = np.linalg.norm(R, axis=1).clip(1e-9, None)
                    ri = np.linalg.norm(Ri, axis=1).clip(1e-9, None)
                    # Reduced units: drop the 1/ε₀ factor from K_E
                    V_gt_np = K_E * (q / r + q_img / ri)
                    V_gt = torch.from_numpy(V_gt_np).to(device=device, dtype=dtype)
                    fast_done = True

            # 3) Grounded cylinder (infinite) with 2D line charge
            #    V = (λ / (2π)) * ln(ρ / ρ'), with image at (a^2 / ρ0^2) * r0_2d
            if (
                not fast_done
                and geom_type in ("cylinder2D", "cylinder")
                and len(spec.conductors) == 1
                and len(spec.charges) == 1
                and spec.charges[0]["type"] in ("line_charge", "line")
            ):
                c = spec.conductors[0]
                if c.get("type") in ("cylinder", "cylinder2D") and c.get(
                    "potential", 0.0
                ) == 0.0:
                    a = float(c.get("radius"))
                    lam = float(spec.charges[0].get("lambda"))
                    x0, y0 = map(float, spec.charges[0].get("pos_2d"))
                    rho0 = math.hypot(x0, y0) + 1e-12
                    xi = (a * a / (rho0 * rho0)) * x0
                    yi = (a * a / (rho0 * rho0)) * y0
                    dx = points_np[:, 0] - x0
                    dy = points_np[:, 1] - y0
                    dxi = points_np[:, 0] - xi
                    dyi = points_np[:, 1] - yi
                    rho = np.sqrt(dx * dx + dy * dy).clip(1e-12, None)
                    rhoi = np.sqrt(dxi * dxi + dyi * dyi).clip(1e-12, None)
                    # Reduced units: omit global K_E prefactor
                    V_gt_np = (K_E * lam / EPS_0) * np.log(rho / rhoi)
                    V_gt = torch.from_numpy(V_gt_np).to(device=device, dtype=dtype)
                    fast_done = True
        except Exception:
            # Any failure here just drops us back to the slower scalar path.
            fast_done = False
            V_gt = None

        if not fast_done or V_gt is None:
            # `solution.eval` returns SI potentials that already include K_E.
            # The learning stack expects reduced units with the 1/ε₀ factor
            # stripped, i.e. V_reduced = ε₀ * V_SI.
            V_gt_np = np.array(
                [solution.eval(tuple(p)) for p in points_np],
                dtype=np.float64,
            )
            V_gt = torch.from_numpy(V_gt_np).to(device=device, dtype=dtype)

        return V_gt

    # BEM solution branch -----------------------------------------------------
    if BEM_AVAILABLE and hasattr(solution, "eval_V_E_batched"):
        sol_device = getattr(solution, "_device", "cpu")
        sol_dtype = getattr(solution, "_dtype", torch.float64)
        pts = torch.tensor(points_np, device=sol_device, dtype=sol_dtype)
        with torch.no_grad():
            V_gt_t, _ = solution.eval_V_E_batched(pts)  # type: ignore[operator]
        return V_gt_t.detach().to(device=device, dtype=dtype).view(-1)

    # As a last resort, treat it as a scalar-eval oracle (keep whatever units
    # the underlying solution uses; this path is rarely hit in the learning
    # stack and is not part of the BEM-vs-analytic unit bridge).
    V_gt_np = np.array(
        [solution.eval(tuple(p)) for p in points_np],
        dtype=np.float64,
    )
    return torch.from_numpy(V_gt_np).to(device=device, dtype=dtype)


def make_collocation_batch_for_spec(
    spec: CanonicalSpec,
    n_points: int,
    ratio_boundary: float,
    supervision_mode: str,  # "analytic" | "bem" | "auto"
    device: torch.device,
    dtype: torch.dtype,
    *,
    rng: Optional[np.random.Generator] = None,
    bem_oracle_config: Optional[Dict[str, Any]] = None,
    encoding: Optional[torch.Tensor] = None,
    geom_type: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build a collocation batch (points + oracle targets) for a single spec.

    Parameters
    ----------
    spec:
        CanonicalSpec describing boundary conditions and sources.
    n_points:
        Target number of collocation points (interior + boundary).  The
        actual number may differ by at most O(#conductors) due to
        per-conductor rounding, matching the historical dataset logic.
    ratio_boundary:
        Fraction of points that should lie on the boundary.  Remaining
        points are drawn from a fixed bounding box for the interior.
    supervision_mode:
        ``"analytic"``, ``"bem"`` or ``"auto"`` — forwarded to
        :func:`get_oracle_solution`.
    device, dtype:
        Desired output device and dtype for ``"X"``, ``"V_gt"`` and
        ``"encoding"`` tensors.

    Other Parameters
    ----------------
    rng:
        Optional :class:`numpy.random.Generator` used for sampling.  If
        omitted, a fresh default RNG is constructed.
    bem_oracle_config:
        Optional dict of overrides for :class:`BEMConfig`; if omitted, a
        default config is used.
    encoding:
        Optional precomputed encoding vector for ``spec``.  If omitted,
        :func:`encode_spec` is called internally.
    geom_type:
        Optional geometry label (e.g. ``"plane"`` / ``"sphere"`` /
        ``"parallel_planes"``).  If omitted, we infer a label from
        ``spec`` via :func:`_infer_geom_type_from_spec`.

    Returns
    -------
    batch : Dict[str, torch.Tensor]
        Dictionary with keys:

        - ``"X"``:          [N, 3] collocation points
        - ``"V_gt"``:       [N] oracle potential values
        - ``"is_boundary"``: [N] bool mask
        - ``"mask_finite"``: [N] bool mask of finite targets
        - ``"encoding"``:   [N, ENCODING_DIM] broadcast spec encoding
    """
    device = torch.device(device)
    if rng is None:
        rng = np.random.default_rng()
    if bem_oracle_config is None:
        bem_oracle_config = {}

    # Oracle selection + identity-based cache --------------------------------
    cache_key = (id(spec), str(supervision_mode), _bem_cfg_cache_key(bem_oracle_config))
    solution = _ORACLE_CACHE.get(cache_key)
    if solution is None:
        solution = get_oracle_solution(spec, supervision_mode, bem_oracle_config)
        if solution is None:
            # No oracle available (e.g. BEM not installed and no analytic path).
            return _empty_batch(device=device, dtype=dtype)
        _ORACLE_CACHE[cache_key] = solution

    # Sample collocation points ----------------------------------------------
    points_np, is_boundary = _sample_points_for_spec(
        spec,
        n_points=n_points,
        ratio_boundary=ratio_boundary,
        rng=rng,
    )
    N_total = points_np.shape[0]
    if N_total == 0:
        return _empty_batch(device=device, dtype=dtype)

    is_boundary = is_boundary.to(device=device)

    # Encoding: either reuse caller-provided or build from spec.
    if encoding is None:
        encoding_single = encode_spec(spec).to(device=device, dtype=dtype)
    else:
        encoding_single = encoding.to(device=device, dtype=dtype)
    if encoding_single.dim() == 1:
        encoding_single = encoding_single.unsqueeze(0)

    encoding_batch = encoding_single.repeat(N_total, 1)

    # Evaluate oracle ---------------------------------------------------------
    if geom_type is None:
        geom_type = _infer_geom_type_from_spec(spec)

    V_gt = _evaluate_oracle_on_points(
        spec,
        solution,
        geom_type,
        points_np,
        device=device,
        dtype=dtype,
    )
    mask_finite = torch.isfinite(V_gt)

    X = torch.from_numpy(points_np).to(device=device, dtype=dtype)

    # Live intercept for diagnostics (does not affect correctness).
    try:
        ctx = bem_intercept.maybe_start_intercept(
            spec, test_name="collocation", bem_cfg=None
        )
        if ctx is not None:
            needs_eps_scaling = geom_type in ("plane", " sphere", "parallel_planes")
            bem_intercept.attach_bem_or_analytic_collocation(
                ctx,
                supervision_mode,
                geom_type,
                X,
                V_gt,
                is_boundary,
                mask_finite,
                ratio_boundary=float(ratio_boundary),
                needs_eps_scaling=needs_eps_scaling,
            )
            bem_intercept.finalize(ctx)
    except Exception:
        # Intercept must never break core logic.
        pass

    return {
        "X": X,
        "V_gt": V_gt,
        "is_boundary": is_boundary,
        "mask_finite": mask_finite,
        "encoding": encoding_batch,
    }
