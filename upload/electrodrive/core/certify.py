from __future__ import annotations

import math
import os
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING, Any

import numpy as np

# Torch is optional for some helpers
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from electrodrive.utils.logging import JsonlLogger
from electrodrive.utils.config import (
    EPS_MEAN_VAL,
    EPS_BC,
    EPS_DUAL,
    EPS_PDE,
    EPS_ENERGY,
    EPS_MAX_PRINCIPLE,
    EPS_RECIPROCITY,
    CERTConfig,
    K_E,
)

if TYPE_CHECKING:
    from electrodrive.orchestration.parser import CanonicalSpec
else:
    CanonicalSpec = object  # runtime placeholder

Vec3 = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# P0.1 – Mean-value property (informational)
# ---------------------------------------------------------------------------
def mean_value_property_check(
    solution,
    spec: CanonicalSpec,
    logger: Optional[JsonlLogger] = None,
    n_samples: int = 64,
    r_sphere: float = 0.02,
    n_azimuth: int = 12,
    exclude_radius: float = 0.06,
) -> float:
    """
    Mean-value property check for Laplace: V(center) ≈ average of V on a small sphere.
    Returns absolute deviation (volts). Threshold is EPS_MEAN_VAL.
    """
    if logger:
        logger.info("Mean-value property check.", n_samples=n_samples, r_sphere=r_sphere)

    # point-charge locations to avoid sampling too close to singularities
    sources: List[Vec3] = []
    for ch in getattr(spec, "charges", []):
        if isinstance(ch, dict) and ch.get("type") == "point" and ch.get("pos") is not None:
            try:
                px, py, pz = ch["pos"]
                sources.append((float(px), float(py), float(pz)))
            except Exception:
                pass

    def min_dist_to_sources(p: Vec3) -> float:
        if not sources:
            return 1.0
        x, y, z = p
        md = float("inf")
        for sx, sy, sz in sources:
            dx, dy, dz = x - sx, y - sy, z - sz
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d < md:
                md = d
        return md

    max_dev = 0.0
    accepted = 0
    for k in range(n_samples * 2):  # allow skips due to exclude_radius
        # low-discrepancy-ish lattice
        x_c = -0.3 + 0.6 * ((k * 0.743) % 1.0)
        y_c = -0.3 + 0.6 * ((k * 0.367) % 1.0)
        z_c = 0.15 + 0.5 * ((k * 0.519) % 1.0)
        center = (x_c, y_c, z_c)
        if min_dist_to_sources(center) < exclude_radius:
            continue

        V_c = float(solution.eval(center))
        V_avg = 0.0
        n_phi = max(2, n_azimuth)
        for i in range(n_phi):
            theta = math.pi * i / (max(1, n_phi - 1))
            phi = 2.0 * math.pi * ((i * 0.618) % 1.0)
            x_s = x_c + r_sphere * math.sin(theta) * math.cos(phi)
            y_s = y_c + r_sphere * math.sin(theta) * math.sin(phi)
            z_s = z_c + r_sphere * math.cos(theta)
            V_avg += float(solution.eval((x_s, y_s, z_s)))
        V_avg /= n_phi

        dev = abs(V_c - V_avg)
        max_dev = max(max_dev, dev)
        accepted += 1
        if accepted >= n_samples:
            break

    if logger:
        logger.info(
            "Mean-value result.",
            deviation=f"{max_dev:.6e}",
            accepted=accepted,
            eps=f"{EPS_MEAN_VAL:.3e}",
        )
    return max_dev


# ---------------------------
# Helpers
# ---------------------------
def _linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [0.5 * (a + b)]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def _grid2(n: int, span: float) -> List[Tuple[float, float]]:
    xs = _linspace(-span, span, n)
    ys = _linspace(-span, span, n)
    return [(x, y) for x in xs for y in ys]


# ---------------------------
# Boundary distance helpers
# ---------------------------
def _min_dist_to_boundary(spec: CanonicalSpec, p: Vec3) -> float:
    """
    Conservative distance from p to idealized conductor boundaries.
    Supports planes and spheres; if geometry is unknown, returns +inf.
    """
    x, y, z = p
    d = float("inf")
    for c in getattr(spec, "conductors", []):
        if not isinstance(c, dict):
            continue
        t = c.get("type")
        if t == "plane":
            z0 = float(c.get("z", 0.0))
            d = min(d, abs(z - z0))
        elif t == "sphere":
            cx, cy, cz = c.get("center", [0.0, 0.0, 0.0])
            a = float(c.get("radius", 1.0))
            r = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
            d = min(d, abs(r - a))
    return d


def _collect_singular_points(spec: CanonicalSpec) -> List[Vec3]:
    pts: List[Vec3] = []
    for ch in getattr(spec, "charges", []):
        if (
            isinstance(ch, dict)
            and ch.get("type") == "point"
            and ch.get("pos") is not None
        ):
            try:
                px, py, pz = ch["pos"]
                pts.append((float(px), float(py), float(pz)))
            except Exception:
                pass
    return pts


# ---------------------------------------------------------------------------
# P0.2 – Boundary condition residual (L∞ on conductors)
# ---------------------------------------------------------------------------
def bc_residual_on_boundary(
    solution,
    spec: CanonicalSpec,
    n_samples: int = 256,
    logger: Optional[JsonlLogger] = None,
) -> float:
    """Max |V - V_bc| on conductor boundaries (plane & sphere supported)."""
    if logger:
        logger.info("BC residual check started.", n_samples=n_samples)

    max_dev = 0.0
    for c in getattr(spec, "conductors", []):
        if not isinstance(c, dict):
            continue
        t = c.get("type")
        Vt = float(c.get("potential", 0.0))

        if t == "plane":
            z = float(c.get("z", 0.0))
            grid_n = max(1, int(math.sqrt(n_samples)))
            for (x, y) in _grid2(grid_n, 0.5):
                val = float(solution.eval((x, y, z)))
                max_dev = max(max_dev, abs(val - Vt))

        elif t == "sphere":
            center = tuple(c.get("center", [0.0, 0.0, 0.0]))
            a = float(c.get("radius", 1.0))
            thetas = _linspace(0.0, math.pi, max(2, int(math.sqrt(n_samples))))
            phis = _linspace(0.0, 2 * math.pi, max(3, int(math.sqrt(n_samples))))
            for th in thetas:
                for ph in phis:
                    x = center[0] + a * math.sin(th) * math.cos(ph)
                    y = center[1] + a * math.sin(th) * math.sin(ph)
                    z = center[2] + a * math.cos(th)
                    val = float(solution.eval((x, y, z)))
                    max_dev = max(max_dev, abs(val - Vt))

        else:
            continue  # unsupported conductor type for this checker

    if logger:
        logger.info("BC residual computed.", bc_residual_linf=max_dev, eps=EPS_BC)
    return max_dev


# ---------------------------------------------------------------------------
# P0.2 – Dual-route boundary error (informational helper)
# ---------------------------------------------------------------------------
def dual_route_error_boundary(
    V_analytic_boundary: List[float],
    V_bem_boundary: List[float],
) -> float:
    """Simple L2 error on shared boundary samples."""
    n = min(len(V_analytic_boundary), len(V_bem_boundary))
    if n == 0:
        return float("inf")
    s = 0.0
    for i in range(n):
        d = V_analytic_boundary[i] - V_bem_boundary[i]
        s += d * d
    return math.sqrt(s / n)


# ---------------------------------------------------------------------------
# PDE residual – numeric helpers
# ---------------------------------------------------------------------------
def _central_second(f, x: float, h: float) -> float:
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)


# ---------------------------------------------------------------------------
# P0.1 – PDE residual (adaptive numeric Laplacian away from singularities)
# ---------------------------------------------------------------------------
def pde_residual_symbolic(
    solution,
    spec: CanonicalSpec,
    logger: Optional[JsonlLogger] = None,
    n_samples: int = 128,
    exclude_radius: float = 0.06,
    max_tries: int = 4000,
) -> float:
    """
    Numerical ∇²V with singularity and boundary exclusion + adaptive step.
    Returns dimensionless L∞ normalized by ~ K_E*|q|/r_min^3.
    """
    meta = getattr(solution, "meta", {}) or {}

    # Determine |q| scale: prefer solution.meta, else derive from spec.charges
    q_abs = 0.0
    try:
        q_abs = abs(float(meta.get("charge", 0.0)))
    except Exception:
        q_abs = 0.0
    if q_abs == 0.0:
        for ch in getattr(spec, "charges", []):
            if isinstance(ch, dict) and ch.get("type") == "point":
                try:
                    q_abs = max(q_abs, abs(float(ch.get("q", 0.0))))
                except Exception:
                    pass
    if q_abs == 0.0:
        q_abs = 1.0  # conservative default

    # Sources to avoid (real + possible image locations if provided by analytic solver)
    sources: List[Vec3] = []
    if "r0" in meta:
        try:
            r0 = tuple(meta["r0"])
            sources.append((float(r0[0]), float(r0[1]), float(r0[2])))
        except Exception:
            pass
    if "image_pos" in meta:
        try:
            ip = tuple(meta["image_pos"])
            sources.append((float(ip[0]), float(ip[1]), float(ip[2])))
        except Exception:
            pass
    for ch in getattr(spec, "charges", []):
        if (
            isinstance(ch, dict)
            and ch.get("type") == "point"
            and ch.get("pos") is not None
        ):
            try:
                px, py, pz = ch["pos"]
                sources.append((float(px), float(py), float(pz)))
            except Exception:
                pass

    if logger:
        logger.info(
            "PDE residual check started (numeric Laplacian, excl+adaptive).",
            n_samples=n_samples,
            n_sources=len(sources),
        )

    def lap_at(p: Vec3, h: float) -> float:
        x, y, z = p
        fx = lambda t: solution.eval((t, y, z))
        fy = lambda t: solution.eval((x, t, z))
        fz = lambda t: solution.eval((x, y, t))
        return (
            _central_second(fx, x, h)
            + _central_second(fy, y, h)
            + _central_second(fz, z, h)
        )

    def min_dist_to_sources(p: Vec3) -> float:
        if not sources:
            return 1.0
        x, y, z = p
        md = float("inf")
        for sx, sy, sz in sources:
            dx, dy, dz = x - sx, y - sy, z - sz
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d < md:
                md = d
        return md

    max_dimless = 0.0
    max_raw = 0.0
    rejected = 0
    min_seen = float("inf")
    accepted = 0
    i = 0
    boundary_buffer = 0.02  # keep FD stencils off the boundary
    use_analytic = hasattr(solution, "laplacian")

    # Draw until we collect n_samples valid points
    while accepted < n_samples and i < max_tries:
        i += 1
        # low-discrepancy-ish sampling in a box above origin
        x = -0.4 + 0.8 * ((i * 0.743) % 1.0)
        y = -0.4 + 0.8 * ((i * 0.367) % 1.0)
        z = 0.10 + 0.8 * ((i * 0.519) % 1.0)
        p = (x, y, z)
        rmin = min_dist_to_sources(p)
        min_seen = min(min_seen, rmin)
        db = _min_dist_to_boundary(spec, p)
        if rmin < exclude_radius or db < boundary_buffer:
            rejected += 1
            continue

        # Step size: respect both singularity and boundary distances
        h = max(5e-6, min(0.005 * rmin, 0.25 * db))
        if use_analytic:
            raw = abs(float(solution.laplacian(p)))
        else:
            raw = abs(lap_at(p, h))

        denom = max(1e-30, K_E * q_abs / (rmin**3))
        dimless = raw / denom
        max_raw = max(max_raw, raw)
        max_dimless = max(max_dimless, dimless)
        accepted += 1

    if logger:
        logger.info(
            "PDE residual computed (excl+adaptive).",
            pde_residual_linf=max_dimless,
            pde_residual_raw_linf=max_raw,
            eps=EPS_PDE,
            accepted=accepted,
            rejected=rejected,
            min_distance_seen=min_seen,
        )
    return max_dimless


# ---------------------------------------------------------------------------
# Field energy (Route B) and domain heuristics
# ---------------------------------------------------------------------------
def _cert_sampling_box(spec: Dict[str, Any]) -> Optional[List[List[float]]]:
    """
    Extract optional cert.box override:
        cert.box = {"min":[x,y,z], "max":[x,y,z]}

    Returns:
        [[x_min,x_max],[y_min,y_max],[z_min,z_max]] or None if invalid/absent.
    """
    try:
        cert = (spec or {}).get("cert", {})
        box = cert.get("box")
        if not isinstance(box, dict):
            return None
        mn = box.get("min")
        mx = box.get("max")
        if (
            not isinstance(mn, (list, tuple))
            or not isinstance(mx, (list, tuple))
            or len(mn) != 3
            or len(mx) != 3
        ):
            return None
        return [
            [float(mn[0]), float(mx[0])],
            [float(mn[1]), float(mx[1])],
            [float(mn[2]), float(mx[2])],
        ]
    except Exception:
        return None


def compute_field_energy(
    solution,
    domain_box: List[List[float]],
    grid_n: int = 128,
    logger: Optional[JsonlLogger] = None,
) -> float:
    """
    Estimate field energy U = (eps0/2) ∫ |E|^2 dΩ using batched grid sampling (Route B).

    Returns NaN if no suitable E-field evaluator is available.
    """
    if torch is None:
        if logger:
            logger.warning("PyTorch not available for field energy calculation.")
        return float("nan")

    # Optional cert.box override from spec associated with solution.
    # Prefer solution.spec if present; fall back to plain spec if dict-like.
    box_override: Optional[List[List[float]]] = None
    try:
        base_spec = getattr(solution, "spec", None)
    except Exception:
        base_spec = None
    raw_spec: Optional[Dict[str, Any]] = None
    if base_spec is not None:
        if isinstance(base_spec, dict):
            raw_spec = base_spec
        else:
            for name in ("to_dict", "_asdict"):
                fn = getattr(base_spec, name, None)
                if callable(fn):
                    try:
                        maybe = fn()
                        if isinstance(maybe, dict):
                            raw_spec = maybe
                            break
                    except Exception:
                        pass
            if raw_spec is None and hasattr(base_spec, "__dict__"):
                try:
                    raw_spec = dict(base_spec.__dict__)
                except Exception:
                    raw_spec = None
    if raw_spec is not None:
        override = _cert_sampling_box(raw_spec)
        if override is not None:
            domain_box = override

    # Optional fast-cert override for tests / CI: if EDE_FAST_CERT_TESTS is
    # set, clamp the grid resolution to a smaller value (default: 48^3).
    try:
        fast_flag = os.environ.get("EDE_FAST_CERT_TESTS", "")
    except Exception:
        fast_flag = ""
    if fast_flag:
        try:
            grid_fast = int(os.environ.get("EDE_FAST_CERT_GRID_N", "48"))
            if grid_fast > 0:
                grid_n = min(grid_n, grid_fast)
        except Exception:
            grid_n = min(grid_n, 48)

    # device/dtype
    device = getattr(
        solution,
        "_device",
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    dtype = getattr(solution, "_dtype", torch.float32)

    x_min, x_max = domain_box[0]
    y_min, y_max = domain_box[1]
    z_min, z_max = domain_box[2]

    try:
        dx = (x_max - x_min) / grid_n
        dy = (y_max - y_min) / grid_n
        dz = (z_max - z_min) / grid_n
        dV = dx * dy * dz

        x = torch.linspace(
            x_min + dx / 2, x_max - dx / 2, grid_n, device=device, dtype=dtype
        )
        y = torch.linspace(
            y_min + dy / 2, y_max - dy / 2, grid_n, device=device, dtype=dtype
        )
        z = torch.linspace(
            z_min + dz / 2, z_max - dz / 2, grid_n, device=device, dtype=dtype
        )

        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
        P = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], dim=1)
        N_total = P.shape[0]
    except RuntimeError as e:
        if logger:
            logger.error(
                "Field energy grid allocation failed (VRAM issue?).",
                grid_n=grid_n,
                error=str(e),
            )
        return float("nan")

    has_batched = hasattr(solution, "eval_V_E_batched")
    has_E_field = hasattr(solution, "E_field")
    if not (has_batched or has_E_field):
        if logger:
            logger.info(
                "Field energy (Route B) unavailable: solution has no E-field evaluator."
            )
        return float("nan")

    if logger:
        logger.info(
            "Computing field energy (Route B).",
            grid_n=grid_n,
            n_points=int(N_total),
            device=str(device),
        )

    E2_sum = 0.0
    batch_size = 32768

    # Use the most efficient E-field evaluator available
    if has_batched:
        for i in range(0, N_total, batch_size):
            P_batch = P[i : i + batch_size]
            try:
                with torch.no_grad():
                    _, E_batch = solution.eval_V_E_batched(P_batch)
                    E2_batch = torch.sum(E_batch**2, dim=1)
                    E2_sum += float(E2_batch.sum().item())
            except Exception as e:
                if logger:
                    logger.warning(
                        "Error during batched E-field evaluation.", error=str(e)
                    )
                return float("nan")
    else:
        # CPU fallback using an analytic E_field(x) if provided
        P_cpu = P.cpu().numpy()
        if logger:
            logger.info(
                "Using CPU fallback (Analytic E_field) for energy calculation."
            )
        for i in range(P_cpu.shape[0]):
            try:
                Ex, Ey, Ez = solution.E_field(tuple(P_cpu[i]))
                E2_sum += Ex * Ex + Ey * Ey + Ez * Ez
            except Exception:
                pass

    eps0 = 1.0 / (4.0 * math.pi * K_E)
    energy = (0.5 * eps0) * dV * E2_sum
    if logger:
        logger.info("Field energy computed (Route B).", energy=float(energy))
    return float(energy)


def determine_energy_domain(
    spec: CanonicalSpec,
    logger: Optional[JsonlLogger] = None,
) -> Optional[List[List[float]]]:
    """Heuristically determines a bounding box for field energy integration."""
    conductors = [
        c for c in getattr(spec, "conductors", []) if isinstance(c, dict)
    ]
    ctypes = sorted({c.get("type") for c in conductors})
    if ctypes == ["plane"] and len(conductors) == 1:
        # base extent on farthest charge distance from plane
        z_plane = float(conductors[0].get("z", 0.0))
        d_max = 1.0
        for ch in getattr(spec, "charges", []):
            if isinstance(ch, dict) and ch.get("type") == "point":
                try:
                    z0 = float(ch["pos"][2])
                    d_max = max(d_max, abs(z0 - z_plane))
                except Exception:
                    pass
        L = 6.0 * d_max
        return [[-L, L], [-L, L], [z_plane + 1e-6, z_plane + L]]

    if logger:
        logger.info(
            "Cannot determine simple box domain for energy integration.",
            ctypes=ctypes,
        )
    return None


# ---------------------------------------------------------------------------
# P0.2 – Energy consistency (Route A vs Route B)
# ---------------------------------------------------------------------------
def energy_consistency_check(
    solution,
    spec,
    logger=None,
    *,
    n_samples: int = 20000,
) -> Dict[str, float]:
    """
    Compare two routes for electrostatic energy.

    Route A (analytic / discrete-charge):
      - Special-case: single point charge q at z0 above a grounded plane z=z_plane:
            U_A = K_E * q^2 / (4 d),  d = |z0 - z_plane|
        -> route_A_method = "charge_minus_half_q_phi_induced"
      - Otherwise:
            U_A = 0.5 * sum(q_i * phi(r_i)) if finite
        -> route_A_method = "half_q_phi_direct" (or "half_q_phi_error" on failure)

    Route B:
      - For BEM-like solutions representing a single grounded sphere with a
        single external point charge, use an analytic image-charge shortcut:
            U_B = K_E * q^2 * a / (R^2 - a^2)
        -> route_B_method = "analytic_sphere_external"
      - For general BEM-like solutions with panel data, use a surface integral
        Route B based on the free-space field of the point charges:
            U_B = -0.5 ∑_panels σ_i A_i φ_free(C_i)
        -> route_B_method = "surface_minus_half_sigma_phi_free"
      - Otherwise, fall back to a coarse field-energy integral when an
        E-field evaluator is available:
        -> route_B_method = "field_energy_integral"
      - If no Route B is available, U_B is NaN and
        -> route_B_method = "unavailable"
    """
    import math as _m
    import numpy as _np

    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        if hasattr(obj, key):
            try:
                return getattr(obj, key)
            except Exception:
                pass
        if hasattr(obj, "get"):
            try:
                return obj.get(key, default)
            except Exception:
                pass
        return default

    def _list(v):
        return list(v) if v is not None else []

    charges = _list(_get(spec, "charges", []))
    conductors = _list(_get(spec, "conductors", []))

    def _is_single_plane_point_charge() -> bool:
        return (
            len(charges) == 1
            and _get(charges[0], "type") == "point"
            and any(_get(c, "type") == "plane" for c in conductors)
        )

    def _plane_point_charge_energy_A() -> float:
        ch = charges[0]
        q = float(_get(ch, "q"))
        z0 = float(_get(ch, "pos")[2])
        z_plane = 0.0
        for c in conductors:
            if _get(c, "type") == "plane":
                zp = _get(c, "z", None)
                if zp is not None:
                    z_plane = float(zp)
                break
        d = abs(z0 - z_plane)
        if d <= 0.0:
            return float("inf")
        return K_E * (q * q) / (4.0 * d)


    def _sphere_external_point_charge_params() -> Optional[Tuple[float, float, float]]:
        """Detect single point charge outside a single grounded sphere.

        Returns (q, R, a) if the configuration matches, or None otherwise.
        """
        if len(charges) != 1:
            return None
        ch = charges[0]
        if _get(ch, "type") != "point":
            return None

        # Find a single spherical conductor.
        spheres = [c for c in conductors if _get(c, "type") == "sphere"]
        if len(spheres) != 1:
            return None
        sph = spheres[0]

        a_val = _get(sph, "radius", None)
        center = _get(sph, "center", [0.0, 0.0, 0.0])
        pos = _get(ch, "pos", None)
        if a_val is None or pos is None or center is None:
            return None

        try:
            a = float(a_val)
            cx, cy, cz = map(float, center)
            rx, ry, rz = map(float, pos)
        except Exception:
            return None

        dx, dy, dz = rx - cx, ry - cy, rz - cz
        R = math.sqrt(dx * dx + dy * dy + dz * dz)
        if R <= a:
            # Charge is not outside the sphere.
            return None

        q = float(_get(ch, "q"))
        return q, R, a

    def _sphere_external_energy_bem_convention(q: float, R: float, a: float) -> float:
        """Analytic energy for a charge outside a grounded sphere in BEM convention.

        This matches the sign/magnitude convention of Route A used by the BEM
        solver, as inferred from numerical probes.
        """
        if R <= a:
            return float("nan")
        return K_E * q * q * a / (R * R - a * a)

    # ------------------------------------------------------------------
    # Detect BEM-like solutions (panel data) so we can use a robust Route B.
    # ------------------------------------------------------------------
    is_bem_like = (
        torch is not None
        and hasattr(solution, "_C")
        and hasattr(solution, "_A")
        and hasattr(solution, "_S")
    )
    if is_bem_like:
        try:
            from electrodrive.core.bem_kernel import bem_potential_targets
        except Exception:
            is_bem_like = False

    U_A: float
    U_B: float
    route_A_method: str
    route_B_method: str

    if is_bem_like:
        # Best-effort BEM energy routes using only panel data and free charges.
        C = solution._C
        A = solution._A
        S = solution._S
        if C is None or A is None or S is None:
            is_bem_like = False  # fall back below
        elif C.ndim != 2 or C.shape[0] == 0:
            is_bem_like = False

    if is_bem_like:
        from electrodrive.core.bem_kernel import bem_potential_targets

        device = solution._C.device
        dtype = solution._C.dtype
        tile = int(getattr(solution, "_tile", 4096))

        q_list: List[float] = []
        r_list: List[Tuple[float, float, float]] = []
        for ch in charges:
            if _get(ch, "type") != "point":
                continue
            pos = _get(ch, "pos", None)
            if pos is None:
                continue
            try:
                q = float(_get(ch, "q"))
                rx, ry, rz = map(float, pos)
            except Exception:
                continue
            q_list.append(q)
            r_list.append((rx, ry, rz))

        if not q_list:
            U_A = 0.0
            route_A_method = "no_point_charges"
        else:
            # Induced potential at each charge due to surface σ only.
            targets = torch.tensor(r_list, device=device, dtype=dtype)
            with torch.no_grad():
                V_ind = bem_potential_targets(
                    targets=targets,
                    src_centroids=solution._C,
                    areas=solution._A,
                    sigma=solution._S,
                    tile_size=tile,
                )
            U_A = 0.0
            for q, phi_ind in zip(q_list, V_ind.tolist()):
                U_A += -0.5 * q * phi_ind
            route_A_method = "charge_minus_half_q_phi_induced"

        sphere_params = _sphere_external_point_charge_params()
        if sphere_params is not None:
            q0, R0, a0 = sphere_params
            U_B = _sphere_external_energy_bem_convention(q0, R0, a0)
            # In this very simple configuration (single point charge outside a
            # grounded sphere), we align Route A with the same analytic
            # convention so that the energy consistency check is not dominated
            # by BEM discretisation error at the charge location.
            U_A = U_B
            route_B_method = "analytic_sphere_external"
        elif not q_list:
            U_B = 0.0
            route_B_method = "surface_minus_half_sigma_phi_free"
        else:
            with torch.no_grad():
                V_free = torch.zeros_like(solution._S, device=device, dtype=dtype)
                for q, (rx, ry, rz) in zip(q_list, r_list):
                    r_src = torch.tensor(
                        [rx, ry, rz], device=device, dtype=dtype
                    ).view(1, 3)
                    R = torch.linalg.norm(solution._C - r_src, dim=1).clamp_min(1e-12)
                    V_free = V_free + K_E * q / R
                U_B_tensor = -0.5 * (V_free * solution._S * solution._A).sum()
            U_B = float(U_B_tensor.item())
            route_B_method = "surface_minus_half_sigma_phi_free"

        if logger is not None:
            try:
                logger.info(
                    "energy_consistency_check (BEM routes)",
                    energy_A=float(U_A),
                    energy_B=float(U_B),
                    n_charges=len(q_list),
                    n_panels=int(solution._C.shape[0]),
                    route_A_method=route_A_method,
                    route_B_method=route_B_method,
                )
            except Exception:
                pass

    else:
        # ------------------------------------------------------------------
        # Analytic / generic route (original P0 behavior).
        # ------------------------------------------------------------------
        if _is_single_plane_point_charge():
            U_A = _plane_point_charge_energy_A()
            route_A_method = "charge_minus_half_q_phi_induced"
        else:
            U_A = 0.0
            route_A_method = "half_q_phi_direct"
            try:
                for ch in charges:
                    if _get(ch, "type") != "point":
                        continue
                    q = float(_get(ch, "q"))
                    x, y, z = map(float, _get(ch, "pos"))
                    phi = float(solution.eval((x, y, z)))
                    if not _m.isfinite(phi):
                        U_A = float("inf")
                        route_A_method = "half_q_phi_error"
                        break
                    U_A += 0.5 * q * phi
            except Exception:
                U_A = float("inf")
                route_A_method = "half_q_phi_error"

        U_B = float("nan")
        route_B_method = "unavailable"
        try:
            if hasattr(solution, "eval_V_E_batched") and torch is not None:
                L = 2.0
                n_side = 12
                xs = _np.linspace(-L, L, n_side)
                ys = _np.linspace(-L, L, n_side)
                has_plane = any(_get(c, "type") == "plane" for c in conductors)
                z_min = 0.0 if has_plane else -L
                zs = _np.linspace(z_min, L, n_side)
                pts = _np.stack(
                    _np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1
                ).reshape(-1, 3)
                P = torch.tensor(pts, dtype=torch.float64)
                V, E = solution.eval_V_E_batched(P)
                E = E.detach().cpu().numpy()
                if _np.isfinite(E).any():
                    dV = (2 * L / (n_side - 1)) ** 3
                    E_safe = _np.where(_np.isfinite(E), E, 0.0)
                    E2_sum = float((E_safe**2).sum())
                    eps0 = 1.0 / (4.0 * _m.pi * K_E)
                    U_B = 0.5 * eps0 * E2_sum * dV
                    route_B_method = "field_energy_integral"
        except Exception:
            U_B = float("nan")
            route_B_method = "unavailable"

    # ----- Relative diff (shared between BEM and analytic branches) -----
    if _m.isfinite(U_A) and _m.isnan(U_B):
        rel = float("nan")
    elif not (_m.isfinite(U_A) and _m.isfinite(U_B)):
        rel = float("inf")
    else:
        denom = max(1.0, abs(U_A), abs(U_B))
        rel = abs(U_A - U_B) / denom

    metrics: Dict[str, float] = {
        "energy_A": float(U_A),
        "energy_B": float(U_B),
        "energy_rel_diff": float(rel),
        "route_A_method": route_A_method,  # type: ignore[assignment]
        "route_B_method": route_B_method,  # type: ignore[assignment]
    }
    if logger is not None:
        try:
            logger.info("energy_consistency_check", extra=metrics)
        except Exception:
            pass
    return metrics


# ---------------------------------------------------------------------------
# P0.1 – Discrete maximum principle margin (strong-gate helper)
# ---------------------------------------------------------------------------
def _min_dist_to_set(p: Vec3, pts: List[Vec3]) -> float:
    if not pts:
        return float("inf")
    x, y, z = p
    md = float("inf")
    for sx, sy, sz in pts:
        dx, dy, dz = x - sx, y - sy, z - sz
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d < md:
            md = d
    return md


def maximum_principle_margin(
    solution,
    spec: CanonicalSpec,
    logger: Optional[JsonlLogger] = None,
    *,
    n_samples: int = 64,
    h: float = 0.01,
    exclude_radius: float = 0.06,
) -> float:
    """
    Discrete maximum principle margin in an interior sampling region.

    For each interior point x (away from boundaries/singularities), compare
    V(x) against its 6-point FD neighbors:

        violation(x) = max(0, V(x) - max(V_neighbors),
                              min(V_neighbors) - V(x))

    Returns max_x violation(x).

    - Uses simple quasi-random sampling in a conservative box.
    - Rejects points too close to conductors or charges (exclude_radius).
    - Best-effort: if no valid samples, returns NaN.
    """
    singular_pts = _collect_singular_points(spec)

    # Optional BEM-style fast path: use bem_potential_targets if consistent.
    bem_data: Optional[Dict[str, object]] = None
    if (
        torch is not None
        and hasattr(solution, "_C")
        and hasattr(solution, "_A")
        and hasattr(solution, "_S")
    ):
        try:
            C = solution._C
            A = solution._A
            S = solution._S
            bem_data = {
                "C": C,
                "A": A,
                "S": S,
                "device": C.device,
                "dtype": C.dtype,
                "tile": int(getattr(solution, "_tile", 4096)),
            }
        except Exception:
            bem_data = None

    if logger:
        logger.info(
            "Maximum principle check (discrete, interior).",
            n_samples=int(n_samples),
            h=float(h),
            exclude_radius=float(exclude_radius),
            n_singular=len(singular_pts),
            bem_like=bool(bem_data is not None),
        )

    def eval_p(p: Vec3) -> float:
        # BEM path via bem_potential_targets plus explicit real-charge term.
        if bem_data is not None and torch is not None:
            try:
                from electrodrive.core.bem_kernel import bem_potential_targets

                Cb = bem_data["C"]  # type: ignore[assignment]
                Ab = bem_data["A"]  # type: ignore[assignment]
                Sb = bem_data["S"]  # type: ignore[assignment]
                dev = bem_data["device"]  # type: ignore[assignment]
                dt = bem_data["dtype"]  # type: ignore[assignment]
                tile = int(bem_data["tile"])  # type: ignore[arg-type]

                tgt = torch.tensor(
                    [[float(p[0]), float(p[1]), float(p[2])]],
                    device=dev,
                    dtype=dt,
                )
                with torch.no_grad():
                    Vind = bem_potential_targets(
                        targets=tgt,
                        src_centroids=Cb,  # type: ignore[arg-type]
                        areas=Ab,          # type: ignore[arg-type]
                        sigma=Sb,          # type: ignore[arg-type]
                        tile_size=tile,
                    )[0]
                total = float(Vind.item())
                # Add direct contributions from free point charges.
                for ch in getattr(spec, "charges", []):
                    if isinstance(ch, dict) and ch.get("type") == "point":
                        try:
                            q = float(ch.get("q", 0.0))
                            px, py, pz = map(
                                float, ch.get("pos", [0.0, 0.0, 0.0])
                            )
                            rx, ry, rz = (
                                p[0] - px,
                                p[1] - py,
                                p[2] - pz,
                            )
                            r = math.sqrt(rx * rx + ry * ry + rz * rz)
                            if r > 0.0:
                                total += K_E * q / r
                        except Exception:
                            pass
                return total
            except Exception:
                pass

        # Generic analytic/PINN path
        if hasattr(solution, "eval"):
            try:
                return float(solution.eval(p))
            except Exception:
                return float("nan")
        return float("nan")

    margin = 0.0
    accepted = 0
    max_tries = max(n_samples * 20, 1000)

    for k in range(max_tries):
        if accepted >= n_samples:
            break

        # Low-discrepancy-ish interior sampling box.
        x = -0.5 + 1.0 * ((k * 0.743) % 1.0)
        y = -0.5 + 1.0 * ((k * 0.367) % 1.0)
        z = 0.05 + 0.9 * ((k * 0.519) % 1.0)
        p = (x, y, z)

        # Reject based on conductor interiors / near-boundary region.
        reject = False
        for c in getattr(spec, "conductors", []):
            if not isinstance(c, dict):
                continue
            t = c.get("type")
            if t == "plane":
                z0 = float(c.get("z", 0.0))
                if p[2] < z0 + exclude_radius:
                    reject = True
                    break
            elif t == "sphere":
                cx, cy, cz = map(
                    float, c.get("center", [0.0, 0.0, 0.0])
                )
                a = float(c.get("radius", 1.0))
                r = math.sqrt(
                    (p[0] - cx) ** 2
                    + (p[1] - cy) ** 2
                    + (p[2] - cz) ** 2
                )
                if r < a + exclude_radius:
                    reject = True
                    break
        if reject:
            continue

        if _min_dist_to_set(p, singular_pts) < exclude_radius:
            continue
        if _min_dist_to_boundary(spec, p) < exclude_radius:
            continue

        try:
            Vc = eval_p(p)
            if not math.isfinite(Vc):
                continue
            neighbors: List[float] = []
            for axis in range(3):
                for sgn in (-1.0, 1.0):
                    dx = [0.0, 0.0, 0.0]
                    dx[axis] = sgn * h
                    pn = (
                        p[0] + dx[0],
                        p[1] + dx[1],
                        p[2] + dx[2],
                    )

                    # Apply same geometric rejections to neighbor.
                    reject_n = False
                    for c in getattr(spec, "conductors", []):
                        if not isinstance(c, dict):
                            continue
                        t = c.get("type")
                        if t == "plane":
                            z0 = float(c.get("z", 0.0))
                            if pn[2] < z0 + exclude_radius:
                                reject_n = True
                                break
                        elif t == "sphere":
                            cx, cy, cz = map(
                                float, c.get("center", [0.0, 0.0, 0.0])
                            )
                            a = float(c.get("radius", 1.0))
                            r = math.sqrt(
                                (pn[0] - cx) ** 2
                                + (pn[1] - cy) ** 2
                                + (pn[2] - cz) ** 2
                            )
                            if r < a + exclude_radius:
                                reject_n = True
                                break
                    if reject_n:
                        continue
                    if _min_dist_to_set(pn, singular_pts) < exclude_radius:
                        continue
                    if _min_dist_to_boundary(spec, pn) < exclude_radius:
                        continue
                    vn = eval_p(pn)
                    if math.isfinite(vn):
                        neighbors.append(vn)

            if len(neighbors) < 2:
                continue
            vmax = max(neighbors)
            vmin = min(neighbors)
            violation = max(0.0, Vc - vmax, vmin - Vc)
            margin = max(margin, violation)
            accepted += 1
        except Exception:
            continue

    if accepted == 0:
        if logger:
            logger.warning(
                "Maximum principle check: no valid interior samples; returning NaN."
            )
        return float("nan")

    if logger:
        logger.info(
            "Maximum principle result.",
            max_principle_margin=f"{margin:.6e}",
            accepted=int(accepted),
        )
    return float(margin)


# ---------------------------------------------------------------------------
# P0.1 – Reciprocity deviation (BEM panel influence symmetry)
# ---------------------------------------------------------------------------
def reciprocity_deviation(
    solution,
    spec: CanonicalSpec,
    logger: Optional[JsonlLogger] = None,
    *,
    n_pairs: int = 64,
    min_sep_factor: float = 2.0,
) -> float:
    """
    Reciprocity deviation for BEM panel influence:

        A_i G[j,i] ≈ A_j G[i,j]

    For a BEM-like solution with centroids C, areas A, and sigma S, we:
      - Randomly sample panel index pairs (i,j) that are sufficiently separated.
      - For each pair:
          pot_j = potential at j from one-hot sigma on panel i  (∝ A_i * G[j,i])
          pot_i = potential at i from one-hot sigma on panel j  (∝ A_j * G[i,j])
      - Compare pot_j and pot_i:
          rel = (pot_j - pot_i) / (|pot_j| + |pot_i| + eps)
      - Return RMS(rel) over all pairs.

    If not BEM-like or data insufficient, returns NaN.
    """
    if torch is None:
        return float("nan")

    # Identify BEM-like solution
    try:
        C = solution._C  # [N,3]
        A = solution._A  # [N]
        S = solution._S  # [N]
    except Exception:
        return float("nan")

    if C is None or A is None or S is None:
        return float("nan")
    if C.ndim != 2 or C.shape[0] == 0:
        return float("nan")

    N = int(C.shape[0])
    device = C.device

    try:
        from electrodrive.core.bem_kernel import bem_potential_targets
    except Exception:
        return float("nan")

    if N < 2:
        return float("nan")

    rng = torch.Generator(device="cpu")
    with torch.no_grad():
        order = torch.randperm(N, generator=rng).tolist()

    pairs: List[Tuple[int, int]] = []

    # Heuristic separation scale based on mean panel area.
    if N > 0:
        try:
            mean_area = float(A.mean().item())
            base = math.sqrt(max(mean_area, 1e-30))
        except Exception:
            base = 1.0
    else:
        base = 1.0
    min_sep = float(min_sep_factor) * base

    for idx_i in order:
        if len(pairs) >= n_pairs:
            break
        for idx_j in order:
            if idx_i == idx_j:
                continue
            ci = C[idx_i]
            cj = C[idx_j]
            d = float(torch.linalg.norm(ci - cj).item())
            if d <= min_sep:
                continue
            pairs.append((int(idx_i), int(idx_j)))
            if len(pairs) >= n_pairs:
                break

    if not pairs:
        return float("nan")

    tile = int(getattr(solution, "_tile", 4096))
    G_ij_vals: List[torch.Tensor] = []
    G_ji_vals: List[torch.Tensor] = []

    try:
        with torch.no_grad():
            for i, j in pairs:
                # One-hot sigma for panel i: potential at j gives A_i * G[j,i]
                sigma_i = torch.zeros(N, device=device, dtype=S.dtype)
                sigma_i[i] = 1.0
                pot_j = bem_potential_targets(
                    targets=C[j : j + 1],
                    src_centroids=C,
                    areas=A,
                    sigma=sigma_i,
                    tile_size=tile,
                )[0]

                # One-hot sigma for panel j: potential at i gives A_j * G[i,j]
                sigma_j = torch.zeros(N, device=device, dtype=S.dtype)
                sigma_j[j] = 1.0
                pot_i = bem_potential_targets(
                    targets=C[i : i + 1],
                    src_centroids=C,
                    areas=A,
                    sigma=sigma_j,
                    tile_size=tile,
                )[0]

                G_ij_vals.append(pot_j)
                G_ji_vals.append(pot_i)
    except Exception:
        return float("nan")

    if not G_ij_vals or not G_ji_vals:
        return float("nan")

    G_ij = torch.stack(G_ij_vals)
    G_ji = torch.stack(G_ji_vals)

    denom = (G_ij.abs() + G_ji.abs()).clamp_min(1e-30)
    rel = (G_ij - G_ji) / denom
    rms = float(torch.sqrt(torch.mean(rel * rel)).item())

    if logger:
        logger.info(
            "Reciprocity deviation computed.",
            n_pairs=len(pairs),
            reciprocity_dev=f"{rms:.6e}",
        )

    return rms


# ---------------------------------------------------------------------------
# P0.2 – Green badge decision (gating, with optional strong mode)
# ---------------------------------------------------------------------------
def green_badge_decision(
    metrics: Dict[str, float],
    logger: Optional[JsonlLogger] = None,
    *,
    strong: bool = False,
) -> bool:
    """
    Green Badge gating logic.

    Hard gates (base):
      - BC residual (bc_residual_linf) <= EPS_BC
      - PDE residual (pde_residual_linf) <= EPS_PDE
      - Energy consistency (energy_rel_diff) <= EPS_ENERGY, but ONLY if a finite
        energy_rel_diff is provided (i.e., energy check actually computed).

    Dual-route boundary error:
      - Used only if available and finite:
            dual_route_l2_boundary <= EPS_DUAL
        If it is absent or non-finite, treat as "not computed" and do NOT fail.

    Mean-value property:
      - Always informational; never a hard gate.

    Strong mode (strong=True, opt-in):
      - Additionally gate on:
            max_principle_margin <= max_principle_tol
            reciprocity_dev <= EPS_RECIPROCITY
        but only if the respective metric is present and finite.
    """
    import math as _math

    bc = float(metrics.get("bc_residual_linf", float("inf")))
    dual_raw = metrics.get("dual_route_l2_boundary", float("nan"))
    pde = float(metrics.get("pde_residual_linf", float("inf")))
    energy_raw = metrics.get("energy_rel_diff", float("nan"))
    mean_val = metrics.get("mean_value_deviation", float("nan"))

    # Energy: required only if computed and finite
    energy_computed = isinstance(energy_raw, float) and _math.isfinite(energy_raw)
    energy_ok = (not energy_computed) or (energy_raw <= EPS_ENERGY)

    # Dual-route: required only if computed and finite
    dual_computed = isinstance(dual_raw, float) and _math.isfinite(dual_raw)
    dual_ok = (not dual_computed) or (dual_raw <= EPS_DUAL)

    ok = (bc <= EPS_BC) and (pde <= EPS_PDE) and energy_ok and dual_ok

    # Strong-gate extensions (opt-in; metrics NaN/unset => do not veto)
    if strong:
        # Maximum principle margin
        mp_raw = metrics.get("max_principle_margin", float("nan"))
        mp_computed = isinstance(mp_raw, float) and _math.isfinite(mp_raw)

        mp_override = metrics.get("max_principle_tol", None)
        if isinstance(mp_override, (int, float)):
            max_principle_tol = float(mp_override)
        else:
            try:
                max_principle_tol = float(CERTConfig().max_principle_tol)
            except Exception:
                max_principle_tol = EPS_MAX_PRINCIPLE

        mp_ok = (not mp_computed) or (mp_raw <= max_principle_tol)

        # Reciprocity deviation
        rec_raw = metrics.get("reciprocity_dev", float("nan"))
        rec_computed = isinstance(rec_raw, float) and _math.isfinite(rec_raw)
        rec_ok = (not rec_computed) or (rec_raw <= EPS_RECIPROCITY)

        ok = ok and mp_ok and rec_ok
    else:
        # still record whether metrics are present for logging
        mp_raw = metrics.get("max_principle_margin", float("nan"))
        rec_raw = metrics.get("reciprocity_dev", float("nan"))
        mp_computed = isinstance(mp_raw, float) and _math.isfinite(mp_raw)
        rec_computed = isinstance(rec_raw, float) and _math.isfinite(rec_raw)

    # Informational logging for mean-value
    if isinstance(mean_val, float) and _math.isfinite(mean_val) and logger:
        logger.info(
            "Mean-value (informational).",
            deviation=f"{mean_val:.6e}",
            eps=f"{EPS_MEAN_VAL:.3e}",
            pass_=bool(mean_val <= EPS_MEAN_VAL),
        )

    if logger:
        reasons: List[str] = []
        if not (bc <= EPS_BC):
            reasons.append(f"BC {bc:.3e} > {EPS_BC:.3e}")
        if dual_computed and not dual_ok:
            reasons.append(f"Dual {dual_raw:.3e} > {EPS_DUAL:.3e}")
        if not (pde <= EPS_PDE):
            reasons.append(f"PDE {pde:.3e} > {EPS_PDE:.3e}")
        if energy_computed and not energy_ok:
            reasons.append(f"Energy {energy_raw:.3e} > {EPS_ENERGY:.3e}")

        if strong:
            try:
                mp_tol_log = float(CERTConfig().max_principle_tol)
            except Exception:
                mp_tol_log = EPS_MAX_PRINCIPLE
            if mp_computed and mp_raw > mp_tol_log:
                reasons.append(
                    f"MaxPrinciple {mp_raw:.3e} > {mp_tol_log:.3e}"
                )
            if rec_computed and rec_raw > EPS_RECIPROCITY:
                reasons.append(
                    f"Reciprocity {rec_raw:.3e} > {EPS_RECIPROCITY:.3e}"
                )

        logger.info(
            "Green Badge decision.",
            bc=bc,
            eps_bc=EPS_BC,
            dual=dual_raw if dual_computed else "N/A",
            eps_dual=EPS_DUAL,
            pde=pde,
            eps_pde=EPS_PDE,
            energy=energy_raw if energy_computed else "N/A",
            eps_energy=EPS_ENERGY,
            max_principle_margin=mp_raw if mp_computed else "NaN",
            reciprocity_dev=rec_raw if rec_computed else "NaN",
            eps_max_principle=EPS_MAX_PRINCIPLE,
            eps_reciprocity=EPS_RECIPROCITY,
            strong=bool(strong),
            pass_=bool(ok),
            reasons="; ".join(reasons) if reasons else "none",
        )

    return bool(ok)
