from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
import json
from pathlib import Path
import math

import torch

from electrodrive.utils.config import K_E
from electrodrive.orchestration.parser import CanonicalSpec


@dataclass
class ImageBasisElement:
    """Abstract base class for image-system basis elements.

    Subclasses must implement :meth:`potential`, which evaluates the
    contribution of a *unit-weight* basis element in the same potential
    units used by the learning stack's collocation targets.
    """

    type: str
    params: Dict[str, torch.Tensor]

    def potential(self, targets: torch.Tensor) -> torch.Tensor:
        """Evaluate the basis element potential at a batch of points.

        Parameters
        ----------
        targets:
            [N, 3] tensor of evaluation points.
        """
        raise NotImplementedError

    def serialize(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "type": self.type,
            "params": {k: v.detach().cpu().tolist() for k, v in self.params.items()},
        }

    @staticmethod
    def deserialize(
        data: Dict[str, Any],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "ImageBasisElement":
        """Inverse of :meth:`serialize`."""
        t = data["type"]
        if t == "toroidal_eigen_mode":
            return ToroidalEigenModeBasis({"components": data.get("params", {}).get("components", [])})
        params = {
            k: torch.tensor(v, device=device, dtype=dtype)
            for k, v in data["params"].items()
        }
        if t == "point":
            return PointChargeBasis(params)
        if t == "ring":
            return RingImageBasis(params, type_name="ring")
        if t == "ring_gauss":
            return RingImageBasis(params, type_name="ring_gauss")
        if t == "mirror_stack":
            return MirrorStackBasis(params)
        if t == "poloidal_ring":
            return PoloidalRingBasis(params)
        if t in ("ring_ladder", "ring_ladder_inner", "ring_ladder_outer"):
            return RingLadderBasis(params)
        if t == "toroidal_mode_cluster":
            return ToroidalModeClusterBasis(params)
        if t == "toroidal_eigen_mode":
            return ToroidalEigenModeBasis(params)
        if t == "inner_rim_arc":
            return InnerRimArcBasis(params)
        if t == "inner_rim_ribbon":
            return InnerRimRibbonBasis(params)
        if t == "inner_patch_ring":
            return InnerPatchRingBasis(params)
        raise ValueError(f"Unknown basis element type: {t}")


class PointChargeBasis(ImageBasisElement):
    """Point-charge image basis element.

    The scalar weight associated with this basis element plays the role
    of an effective image charge. The potential returned here is in the
    *reduced* units used by the analytic shortcuts in the collocation
    stack:

        V_reduced = ε₀ * V_SI = ε₀ * (K_E * q / r) = q / (4π r)

    so that a unit weight corresponds to a unit charge in those units.
    This keeps the scales of the dictionary and the collocation targets
    compatible for canonical analytic problems, while remaining
    well-defined for BEM-backed oracles as well.
    """

    def __init__(self, params: Dict[str, torch.Tensor]):
        pos = params.get("position", None)
        if pos is None:
            raise ValueError("PointChargeBasis requires 'position' in params")
        if pos.ndim != 1 or pos.shape[0] != 3:
            # Be tolerant of [1,3] shapes coming from scripts.
            pos = pos.view(3)
        params["position"] = pos
        super().__init__("point", params)

    def potential(self, targets: torch.Tensor) -> torch.Tensor:
        """Potential of a unit-weight point charge in physical units."""
        pos = self.params["position"].to(targets.device, targets.dtype)
        R = torch.linalg.norm(targets - pos, dim=1).clamp_min(1e-12)
        return K_E / R


class RingImageBasis(ImageBasisElement):
    """Continuous ring (loop) basis approximated via fixed quadrature."""

    def __init__(self, params: Dict[str, torch.Tensor], type_name: str = "ring"):
        center = params.get("center", None)
        radius = params.get("radius", None)
        if center is None or radius is None:
            raise ValueError("RingImageBasis requires 'center' and 'radius'")
        center = center.view(3)
        radius = torch.as_tensor(radius).view(())

        n_quad_raw = params.get("n_quad", torch.tensor(64))
        n_quad = int(torch.as_tensor(n_quad_raw).item())
        n_quad = max(4, min(n_quad, 256))

        sigma = params.get("sigma", None)
        sigma_tensor = None
        if sigma is not None:
            sigma_tensor = torch.as_tensor(sigma).view(())

        super().__init__(
            type_name,
            {
                "center": center,
                "radius": radius,
                "n_quad": torch.tensor(n_quad, device=center.device),
                **({"sigma": sigma_tensor} if sigma_tensor is not None else {}),
            },
        )

    def _angles(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        n_quad = int(self.params["n_quad"].item())
        return torch.linspace(0.0, 2.0 * torch.pi, n_quad + 1, device=device, dtype=dtype)[
            :-1
        ]

    def potential(self, targets: torch.Tensor) -> torch.Tensor:
        """Potential of a unit-weight ring using deterministic quadrature."""
        device = targets.device
        dtype = targets.dtype
        center = self.params["center"].to(device=device, dtype=dtype)
        radius = self.params["radius"].to(device=device, dtype=dtype)

        theta = self._angles(device, dtype)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        pts = torch.stack(
            [radius * cos_t, radius * sin_t, torch.zeros_like(theta)], dim=1
        ) + center

        weights = torch.ones_like(theta)
        if "sigma" in self.params and self.type == "ring_gauss":
            sigma = float(self.params["sigma"].item())
            sigma = max(sigma, 1e-6)
            # Wrap angles to [-pi, pi] for the bell weighting.
            ang = torch.remainder(theta + torch.pi, 2.0 * torch.pi) - torch.pi
            weights = torch.exp(-0.5 * (ang / sigma) ** 2)
            weights = weights / weights.sum().clamp_min(1e-12)
        else:
            weights = weights / float(weights.numel())

        R = torch.linalg.norm(targets[:, None, :] - pts[None, :, :], dim=2).clamp_min(
            1e-12
        )
        return torch.sum(weights * (K_E / R), dim=1)


class MirrorStackBasis(ImageBasisElement):
    """Finite mirror-image stack between two parallel planes."""

    MAX_IMAGES = 20

    def __init__(self, params: Dict[str, torch.Tensor]):
        pos = params.get("position", None)
        z_lower = params.get("z_lower", None)
        z_upper = params.get("z_upper", None)
        n_images_raw = params.get("n_images", torch.tensor(6))

        if pos is None or z_lower is None or z_upper is None:
            raise ValueError(
                "MirrorStackBasis requires 'position', 'z_lower', and 'z_upper'"
            )
        pos = pos.view(3)
        z_lower_f = float(torch.as_tensor(z_lower).item())
        z_upper_f = float(torch.as_tensor(z_upper).item())
        n_images = int(torch.as_tensor(n_images_raw).item())
        n_images = max(1, min(n_images, self.MAX_IMAGES))

        super().__init__(
            "mirror_stack",
            {
                "position": pos,
                "z_lower": torch.tensor(z_lower_f, device=pos.device),
                "z_upper": torch.tensor(z_upper_f, device=pos.device),
                "n_images": torch.tensor(n_images, device=pos.device),
            },
        )

        self._images, self._signs = self._build_stack(pos, z_lower_f, z_upper_f, n_images)

    @staticmethod
    def _build_stack(
        pos: torch.Tensor, z_lower: float, z_upper: float, n_images: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, y0, z0 = float(pos[0]), float(pos[1]), float(pos[2])
        d = 0.5 * (z_upper - z_lower)
        if d <= 0.0:
            raise ValueError("MirrorStackBasis requires z_upper > z_lower")

        images: List[List[float]] = []
        signs: List[float] = []
        # Finite truncation of the classic parallel-planes image series.
        for n in range(-n_images, n_images + 1):
            sign_n = (-1.0) ** n
            z_pos = 2.0 * n * d + z0
            images.append([x0, y0, z_pos])
            signs.append(sign_n)

            if n == 0:
                # Skip the duplicate at n=0 for the mirrored branch.
                continue
            z_mirror = 2.0 * n * d - z0
            images.append([x0, y0, z_mirror])
            signs.append(-sign_n)

        return torch.tensor(images), torch.tensor(signs)

    def potential(self, targets: torch.Tensor) -> torch.Tensor:
        device = targets.device
        dtype = targets.dtype
        imgs = self._images.to(device=device, dtype=dtype)
        signs = self._signs.to(device=device, dtype=dtype)
        R = torch.linalg.norm(targets[:, None, :] - imgs[None, :, :], dim=2).clamp_min(
            1e-12
        )
        return torch.sum(signs * (K_E / R), dim=1)


class PoloidalRingBasis(ImageBasisElement):
    """Fixed poloidal multipole ring combination with a single scalar weight."""

    PATTERNS: Dict[int, Tuple[List[float], List[float]]] = {
        0: ([0.0], [1.0]),
        1: ([-1.0, 1.0], [1.0, -1.0]),
        2: ([-1.0, 0.0, 1.0], [1.0, -2.0, 1.0]),
    }

    def __init__(self, params: Dict[str, torch.Tensor]):
        center = params.get("center", None)
        radius = params.get("radius", None)
        delta_r = params.get("delta_r", None)
        order = int(torch.as_tensor(params.get("order", 0)).item())
        n_quad_raw = params.get("n_quad", torch.tensor(96))

        if center is None or radius is None or delta_r is None:
            raise ValueError("PoloidalRingBasis requires 'center', 'radius', and 'delta_r'")
        if order not in self.PATTERNS:
            raise ValueError(f"PoloidalRingBasis order must be one of {list(self.PATTERNS.keys())}")

        center = center.view(3)
        radius = torch.as_tensor(radius).view(())
        delta_r = torch.as_tensor(delta_r).view(())
        n_quad = int(torch.as_tensor(n_quad_raw).item())
        n_quad = max(8, min(n_quad, 256))

        super().__init__(
            "poloidal_ring",
            {
                "center": center,
                "radius": radius,
                "delta_r": delta_r,
                "order": torch.tensor(order, device=center.device),
                "n_quad": torch.tensor(n_quad, device=center.device),
            },
        )

    def _angles(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        n_quad = int(self.params["n_quad"].item())
        return torch.linspace(0.0, 2.0 * torch.pi, n_quad + 1, device=device, dtype=dtype)[:-1]

    def potential(self, targets: torch.Tensor) -> torch.Tensor:
        device = targets.device
        dtype = targets.dtype
        center = self.params["center"].to(device=device, dtype=dtype)
        base_radius = float(self.params["radius"].item())
        delta_r = float(self.params["delta_r"].item())

        theta = self._angles(device, dtype)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        offsets, coeffs = self.PATTERNS[int(self.params["order"].item())]
        norm = max(1e-8, sum(abs(c) for c in coeffs))
        acc = torch.zeros(targets.shape[0], device=device, dtype=dtype)
        for off, coeff in zip(offsets, coeffs):
            r_here = max(1e-6, base_radius + off * delta_r)
            pts = torch.stack(
                [r_here * cos_t, r_here * sin_t, torch.zeros_like(theta)], dim=1
            ) + center
            R = torch.linalg.norm(targets[:, None, :] - pts[None, :, :], dim=2).clamp_min(1e-12)
            ring_pot = torch.sum((K_E / R), dim=1) / float(theta.numel())
            acc = acc + (coeff / norm) * ring_pot
        return acc


class RingLadderBasis(ImageBasisElement):
    """Stack of rings marching radially inward or outward with decaying weights."""

    def __init__(self, params: Dict[str, torch.Tensor]):
        center = params.get("center", None)
        radius = params.get("radius", None)
        minor_radius = params.get("minor_radius", None)
        variant = params.get("variant", "inner")
        n_quad_raw = params.get("n_quad", torch.tensor(96))

        if center is None or radius is None or minor_radius is None:
            raise ValueError("RingLadderBasis requires 'center', 'radius', and 'minor_radius'")
        center = center.view(3)
        radius = torch.as_tensor(radius).view(())
        minor_radius = torch.as_tensor(minor_radius).view(())
        n_quad = int(torch.as_tensor(n_quad_raw).item())
        n_quad = max(8, min(n_quad, 256))

        if isinstance(variant, torch.Tensor):
            try:
                variant = str(variant.item())
            except Exception:
                variant = "inner"
        variant = str(variant)
        if variant not in ("inner", "outer"):
            variant = "inner"

        super().__init__(
            f"ring_ladder_{variant}",
            {
                "center": center,
                "radius": radius,
                "minor_radius": minor_radius,
                "variant": torch.tensor(0 if variant == "inner" else 1, device=center.device),
                "n_quad": torch.tensor(n_quad, device=center.device),
            },
        )

    def _angles(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        n_quad = int(self.params["n_quad"].item())
        return torch.linspace(0.0, 2.0 * torch.pi, n_quad + 1, device=device, dtype=dtype)[:-1]

    def potential(self, targets: torch.Tensor) -> torch.Tensor:
        device = targets.device
        dtype = targets.dtype
        center = self.params["center"].to(device=device, dtype=dtype)
        R_base = float(self.params["radius"].item())
        a = float(self.params["minor_radius"].item())
        variant = "inner" if int(self.params["variant"].item()) == 0 else "outer"

        theta = self._angles(device, dtype)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # Radial offsets marching into or out of the tube.
        offsets = (-0.6 * a, -0.3 * a, 0.0) if variant == "inner" else (0.0, 0.3 * a, 0.6 * a)
        coeffs = (1.0, 0.6, 0.36)
        norm = max(1e-8, sum(abs(c) for c in coeffs))

        acc = torch.zeros(targets.shape[0], device=device, dtype=dtype)
        for off, coeff in zip(offsets, coeffs):
            r_here = max(1e-6, R_base + off)
            pts = torch.stack(
                [r_here * cos_t, r_here * sin_t, torch.zeros_like(theta)], dim=1
            ) + center
            R = torch.linalg.norm(targets[:, None, :] - pts[None, :, :], dim=2).clamp_min(1e-12)
            ring_pot = torch.sum((K_E / R), dim=1) / float(theta.numel())
            acc = acc + (coeff / norm) * ring_pot
        return acc


class ToroidalModeClusterBasis(ImageBasisElement):
    """Azimuthal mode cluster: discrete ring of points with cos(m phi) weights."""

    def __init__(self, params: Dict[str, torch.Tensor]):
        center = params.get("center", None)
        major_radius = params.get("major_radius", None)
        minor_radius = params.get("minor_radius", None)
        mode_m = int(torch.as_tensor(params.get("mode_m", 0)).item())
        n_phi_raw = params.get("n_phi", torch.tensor(12))
        radial_offset = params.get("radial_offset", None)

        if center is None or major_radius is None or minor_radius is None:
            raise ValueError("ToroidalModeClusterBasis requires 'center', 'major_radius', and 'minor_radius'")
        center = center.view(3)
        R_major = torch.as_tensor(major_radius).view(())
        a_minor = torch.as_tensor(minor_radius).view(())
        n_phi = int(torch.as_tensor(n_phi_raw).item())
        n_phi = max(4, min(n_phi, 64))
        radial_offset = torch.as_tensor(radial_offset if radial_offset is not None else 0.5 * a_minor).view(())

        super().__init__(
            "toroidal_mode_cluster",
            {
                "center": center,
                "major_radius": R_major,
                "minor_radius": a_minor,
                "mode_m": torch.tensor(mode_m, device=center.device),
                "n_phi": torch.tensor(n_phi, device=center.device),
                "radial_offset": radial_offset,
            },
        )

    def potential(self, targets: torch.Tensor) -> torch.Tensor:
        device = targets.device
        dtype = targets.dtype
        center = self.params["center"].to(device=device, dtype=dtype)
        R_major = float(self.params["major_radius"].item())
        a_minor = float(self.params["minor_radius"].item())
        m = int(self.params["mode_m"].item())
        n_phi = int(self.params["n_phi"].item())
        r_off = float(self.params["radial_offset"].item())

        phi = torch.linspace(0.0, 2.0 * torch.pi, n_phi + 1, device=device, dtype=dtype)[:-1]
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        # Place points slightly off the major circle to mimic cross-section penetration.
        r_samples = torch.full_like(phi, R_major + r_off)
        z_samples = torch.zeros_like(phi)
        pts = torch.stack(
            [r_samples * cos_phi, r_samples * sin_phi, z_samples],
            dim=1,
        ) + center

        weights = torch.ones_like(phi)
        if m == 1:
            weights = torch.cos(phi)
        elif m == 2:
            weights = torch.cos(2.0 * phi)
        # Normalise weights to keep scale stable.
        norm = torch.sum(torch.abs(weights)).clamp_min(1e-8)
        weights = weights / norm

        R = torch.linalg.norm(targets[:, None, :] - pts[None, :, :], dim=2).clamp_min(1e-12)
        return torch.sum(weights * (K_E / R), dim=1)


def _torus_point_and_normal(
    center: torch.Tensor,
    R_major: float,
    a_minor: float,
    sigma: torch.Tensor,
    phi: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return torus surface point and outward normal for given angles."""
    sigma = sigma.to(device=device, dtype=dtype)
    phi = phi.to(device=device, dtype=dtype)
    cos_s = torch.cos(sigma)
    sin_s = torch.sin(sigma)
    cos_p = torch.cos(phi)
    sin_p = torch.sin(phi)

    # Parametric point on torus surface.
    r_ring = R_major + a_minor * cos_s
    x = r_ring * cos_p
    y = r_ring * sin_p
    z = a_minor * sin_s
    point = torch.stack([x, y, z], dim=-1) + center

    # Outward normal (not normalized to 1 for small efficiency gain).
    n = torch.stack([cos_s * cos_p, cos_s * sin_p, sin_s], dim=-1)
    norm = torch.linalg.norm(n, dim=-1, keepdim=True).clamp_min(1e-12)
    n_hat = n / norm
    return point, n_hat


class InnerRimArcBasis(ImageBasisElement):
    """Short poloidal arc on the inner rim, represented by a few inward-offset points."""

    def __init__(self, params: Dict[str, torch.Tensor]):
        required = ("center", "R", "a", "phi0", "d_sigma")
        for k in required:
            if k not in params:
                raise ValueError(f"InnerRimArcBasis missing param '{k}'")

        center = params["center"].view(3)
        R = float(torch.as_tensor(params["R"]).item())
        a = float(torch.as_tensor(params["a"]).item())
        phi0 = float(torch.as_tensor(params["phi0"]).item())
        d_sigma = float(torch.as_tensor(params["d_sigma"]).item())
        n_pts = int(torch.as_tensor(params.get("n_pts", 8)).item())
        n_pts = max(3, min(32, n_pts))
        offset_frac = float(torch.as_tensor(params.get("offset_frac", 0.3)).item())
        offset_frac = float(min(0.4, max(0.2, offset_frac)))
        taper_raw = params.get("taper", None)
        taper_str = "cos"
        if taper_raw is not None:
            try:
                taper_flag = int(torch.as_tensor(taper_raw).item())
                taper_str = "cos" if taper_flag == 0 else "gauss"
            except Exception:
                t_str = str(taper_raw)
                taper_str = t_str if t_str in ("cos", "gauss") else "cos"
        taper_str = taper_str if taper_str in ("cos", "gauss") else "cos"

        device = center.device
        dtype = center.dtype

        sigma_center = float(torch.as_tensor(params.get("sigma0", math.pi)).item())
        sigma_vals = torch.linspace(
            sigma_center - d_sigma, sigma_center + d_sigma, n_pts, device=device, dtype=dtype
        )
        phi_vals = torch.full_like(sigma_vals, phi0)

        pts, normals = _torus_point_and_normal(center, R, a, sigma_vals, phi_vals, device, dtype)
        inward = pts - offset_frac * a * normals  # move inside conductor

        # Weighting along sigma.
        if taper_str == "gauss":
            rel = (sigma_vals - sigma_center) / max(d_sigma, 1e-6)
            w = torch.exp(-0.5 * rel * rel)
        else:
            rel = torch.abs(sigma_vals - sigma_center) / max(d_sigma, 1e-6)
            w = torch.clamp(torch.cos(0.5 * math.pi * rel), min=0.0)
        w = w / w.sum().clamp_min(1e-12)

        self._points = inward
        self._weights = w
        super().__init__(
            "inner_rim_arc",
            {
                "center": center,
                "R": torch.tensor(R, device=device, dtype=dtype),
                "a": torch.tensor(a, device=device, dtype=dtype),
                "phi0": torch.tensor(phi0, device=device, dtype=dtype),
                "sigma0": torch.tensor(sigma_center, device=device, dtype=dtype),
                "d_sigma": torch.tensor(d_sigma, device=device, dtype=dtype),
                "n_pts": torch.tensor(n_pts, device=device),
                "offset_frac": torch.tensor(offset_frac, device=device, dtype=dtype),
                "taper": torch.tensor(0 if taper_str == "cos" else 1, device=device, dtype=torch.int64),
            },
        )

    def potential(self, targets: torch.Tensor) -> torch.Tensor:
        pts = self._points.to(device=targets.device, dtype=targets.dtype)
        w = self._weights.to(device=targets.device, dtype=targets.dtype)
        Rv = torch.linalg.norm(targets[:, None, :] - pts[None, :, :], dim=2).clamp_min(1e-12)
        return torch.sum(w * (K_E / Rv), dim=1)

    def serialize(self) -> Dict[str, Any]:
        p = {k: v.detach().cpu().tolist() for k, v in self.params.items()}
        # taper stored as int; map back to string for readability
        taper_flag = int(self.params["taper"].item())
        p["taper"] = "cos" if taper_flag == 0 else "gauss"
        return {"type": self.type, "params": p}


class InnerRimRibbonBasis(ImageBasisElement):
    """Short σ–φ strip on the inner rim using a separable taper."""

    def __init__(self, params: Dict[str, torch.Tensor]):
        required = ("center", "R", "a", "phi0", "d_sigma", "d_phi")
        for k in required:
            if k not in params:
                raise ValueError(f"InnerRimRibbonBasis missing param '{k}'")

        center = params["center"].view(3)
        R = float(torch.as_tensor(params["R"]).item())
        a = float(torch.as_tensor(params["a"]).item())
        phi0 = float(torch.as_tensor(params["phi0"]).item())
        d_sigma = float(torch.as_tensor(params["d_sigma"]).item())
        d_phi = float(torch.as_tensor(params["d_phi"]).item())
        n_sigma = int(torch.as_tensor(params.get("n_sigma", 6)).item())
        n_phi = int(torch.as_tensor(params.get("n_phi", 6)).item())
        n_sigma = max(3, min(48, n_sigma))
        n_phi = max(3, min(48, n_phi))
        offset_frac = float(torch.as_tensor(params.get("offset_frac", 0.3)).item())
        offset_frac = float(min(0.4, max(0.2, offset_frac)))
        taper_sigma_raw = params.get("taper_sigma", "cos")
        taper_phi_raw = params.get("taper_phi", "cos")
        try:
            taper_sigma_flag = int(torch.as_tensor(taper_sigma_raw).item())
            taper_sigma = "cos" if taper_sigma_flag == 0 else "gauss"
        except Exception:
            taper_sigma = str(taper_sigma_raw) if str(taper_sigma_raw) in ("cos", "gauss") else "cos"
        try:
            taper_phi_flag = int(torch.as_tensor(taper_phi_raw).item())
            taper_phi = "cos" if taper_phi_flag == 0 else "gauss"
        except Exception:
            taper_phi = str(taper_phi_raw) if str(taper_phi_raw) in ("cos", "gauss") else "cos"

        device = center.device
        dtype = center.dtype

        sigma_center = float(torch.as_tensor(params.get("sigma0", math.pi)).item())
        sigma_vals = torch.linspace(
            sigma_center - d_sigma, sigma_center + d_sigma, n_sigma, device=device, dtype=dtype
        )
        phi_vals = torch.linspace(phi0 - d_phi, phi0 + d_phi, n_phi, device=device, dtype=dtype)
        sigma_grid, phi_grid = torch.meshgrid(sigma_vals, phi_vals, indexing="ij")

        pts, normals = _torus_point_and_normal(center, R, a, sigma_grid, phi_grid, device, dtype)
        inward = pts - offset_frac * a * normals

        def _taper(vals: torch.Tensor, center_val: float, width: float, kind: str) -> torch.Tensor:
            if kind == "gauss":
                rel = (vals - center_val) / max(width, 1e-6)
                return torch.exp(-0.5 * rel * rel)
            rel = torch.abs(vals - center_val) / max(width, 1e-6)
            return torch.clamp(torch.cos(0.5 * math.pi * rel), min=0.0)

        w_sigma = _taper(sigma_vals, sigma_center, d_sigma, taper_sigma)
        w_phi = _taper(phi_vals, phi0, d_phi, taper_phi)
        w = torch.outer(w_sigma, w_phi)
        w = w / w.sum().clamp_min(1e-12)

        self._points = inward.reshape(-1, 3)
        self._weights = w.reshape(-1)
        super().__init__(
            "inner_rim_ribbon",
            {
                "center": center,
                "R": torch.tensor(R, device=device, dtype=dtype),
                "a": torch.tensor(a, device=device, dtype=dtype),
                "phi0": torch.tensor(phi0, device=device, dtype=dtype),
                "sigma0": torch.tensor(sigma_center, device=device, dtype=dtype),
                "d_sigma": torch.tensor(d_sigma, device=device, dtype=dtype),
                "d_phi": torch.tensor(d_phi, device=device, dtype=dtype),
                "n_sigma": torch.tensor(n_sigma, device=device),
                "n_phi": torch.tensor(n_phi, device=device),
                "offset_frac": torch.tensor(offset_frac, device=device, dtype=dtype),
                "taper_sigma": torch.tensor(0 if taper_sigma == "cos" else 1, device=device, dtype=torch.int64),
                "taper_phi": torch.tensor(0 if taper_phi == "cos" else 1, device=device, dtype=torch.int64),
            },
        )

    def potential(self, targets: torch.Tensor) -> torch.Tensor:
        pts = self._points.to(device=targets.device, dtype=targets.dtype)
        w = self._weights.to(device=targets.device, dtype=targets.dtype)
        Rv = torch.linalg.norm(targets[:, None, :] - pts[None, :, :], dim=2).clamp_min(1e-12)
        return torch.sum(w * (K_E / Rv), dim=1)

    def serialize(self) -> Dict[str, Any]:
        p = {k: v.detach().cpu().tolist() for k, v in self.params.items()}
        p["taper_sigma"] = "cos" if int(self.params["taper_sigma"].item()) == 0 else "gauss"
        p["taper_phi"] = "cos" if int(self.params["taper_phi"].item()) == 0 else "gauss"
        return {"type": self.type, "params": p}


class InnerPatchRingBasis(ImageBasisElement):
    """Localized patch on inner rim plus compensating inner ring for near-neutrality."""

    def __init__(self, params: Dict[str, torch.Tensor]):
        required = ("center", "R", "a", "phi0", "d_sigma", "d_phi")
        for k in required:
            if k not in params:
                raise ValueError(f"InnerPatchRingBasis missing param '{k}'")

        center = params["center"].view(3)
        R = float(torch.as_tensor(params["R"]).item())
        a = float(torch.as_tensor(params["a"]).item())
        phi0 = float(torch.as_tensor(params["phi0"]).item())
        d_sigma = float(torch.as_tensor(params["d_sigma"]).item())
        d_phi = float(torch.as_tensor(params["d_phi"]).item())
        n_sigma = int(torch.as_tensor(params.get("n_sigma", 6)).item())
        n_phi = int(torch.as_tensor(params.get("n_phi", 6)).item())
        n_sigma = max(3, min(48, n_sigma))
        n_phi = max(3, min(48, n_phi))
        offset_frac = float(torch.as_tensor(params.get("offset_frac", 0.3)).item())
        offset_frac = float(min(0.4, max(0.2, offset_frac)))
        ring_offset_frac = float(torch.as_tensor(params.get("ring_offset_frac", 0.55)).item())
        ring_offset_frac = float(min(0.9, max(0.1, ring_offset_frac)))
        n_ring = int(torch.as_tensor(params.get("n_ring", 12)).item())
        n_ring = max(4, min(128, n_ring))
        neutral_factor = float(torch.as_tensor(params.get("neutral_factor", 1.0)).item())
        neutral_factor = float(max(0.0, neutral_factor))
        taper_sigma_raw = params.get("taper_sigma", "cos")
        taper_phi_raw = params.get("taper_phi", "cos")
        try:
            taper_sigma_flag = int(torch.as_tensor(taper_sigma_raw).item())
            taper_sigma = "cos" if taper_sigma_flag == 0 else "gauss"
        except Exception:
            taper_sigma = str(taper_sigma_raw) if str(taper_sigma_raw) in ("cos", "gauss") else "cos"
        try:
            taper_phi_flag = int(torch.as_tensor(taper_phi_raw).item())
            taper_phi = "cos" if taper_phi_flag == 0 else "gauss"
        except Exception:
            taper_phi = str(taper_phi_raw) if str(taper_phi_raw) in ("cos", "gauss") else "cos"

        device = center.device
        dtype = center.dtype

        sigma_center = float(torch.as_tensor(params.get("sigma0", math.pi)).item())
        sigma_vals = torch.linspace(
            sigma_center - d_sigma, sigma_center + d_sigma, n_sigma, device=device, dtype=dtype
        )
        phi_vals = torch.linspace(phi0 - d_phi, phi0 + d_phi, n_phi, device=device, dtype=dtype)
        sigma_grid, phi_grid = torch.meshgrid(sigma_vals, phi_vals, indexing="ij")

        patch_pts, patch_normals = _torus_point_and_normal(center, R, a, sigma_grid, phi_grid, device, dtype)
        patch_pts = patch_pts - offset_frac * a * patch_normals

        def _taper(vals: torch.Tensor, center_val: float, width: float, kind: str) -> torch.Tensor:
            if kind == "gauss":
                rel = (vals - center_val) / max(width, 1e-6)
                return torch.exp(-0.5 * rel * rel)
            rel = torch.abs(vals - center_val) / max(width, 1e-6)
            return torch.clamp(torch.cos(0.5 * math.pi * rel), min=0.0)

        w_sigma = _taper(sigma_vals, sigma_center, d_sigma, taper_sigma)
        w_phi = _taper(phi_vals, phi0, d_phi, taper_phi)
        patch_w = torch.outer(w_sigma, w_phi)
        patch_w = patch_w / patch_w.sum().clamp_min(1e-12)

        # Ring centered on inner radius (R - ring_offset_frac * a) in the torus plane.
        ring_radius = max(1e-6, R - ring_offset_frac * a)
        theta = torch.linspace(0.0, 2.0 * math.pi, n_ring + 1, device=device, dtype=dtype)[:-1]
        ring_cos = torch.cos(theta)
        ring_sin = torch.sin(theta)
        ring_pts = torch.stack(
            [
                ring_radius * ring_cos,
                ring_radius * ring_sin,
                torch.zeros_like(theta),
            ],
            dim=1,
        ) + center
        # Approximate inward shift along inner-rim normal (sigma=pi).
        ring_normals = torch.stack(
            [-ring_cos, -ring_sin, torch.zeros_like(theta)],
            dim=1,
        )
        norm_ring = torch.linalg.norm(ring_normals, dim=1, keepdim=True).clamp_min(1e-12)
        ring_normals = ring_normals / norm_ring
        ring_pts = ring_pts - offset_frac * a * ring_normals

        patch_w_flat = patch_w.reshape(-1)
        patch_pts_flat = patch_pts.reshape(-1, 3)
        patch_total = patch_w_flat.sum().item()
        ring_w = torch.full_like(theta, 1.0 / float(n_ring))
        if patch_total != 0.0:
            ring_w = -neutral_factor * patch_total * ring_w

        # Normalise so that patch weights sum to +1, ring to -neutral_factor.
        self._patch_points = patch_pts_flat
        self._patch_weights = patch_w_flat
        self._ring_points = ring_pts
        self._ring_weights = ring_w

        super().__init__(
            "inner_patch_ring",
            {
                "center": center,
                "R": torch.tensor(R, device=device, dtype=dtype),
                "a": torch.tensor(a, device=device, dtype=dtype),
                "phi0": torch.tensor(phi0, device=device, dtype=dtype),
                "sigma0": torch.tensor(sigma_center, device=device, dtype=dtype),
                "d_sigma": torch.tensor(d_sigma, device=device, dtype=dtype),
                "d_phi": torch.tensor(d_phi, device=device, dtype=dtype),
                "n_sigma": torch.tensor(n_sigma, device=device),
                "n_phi": torch.tensor(n_phi, device=device),
                "offset_frac": torch.tensor(offset_frac, device=device, dtype=dtype),
                "ring_offset_frac": torch.tensor(ring_offset_frac, device=device, dtype=dtype),
                "n_ring": torch.tensor(n_ring, device=device),
                "neutral_factor": torch.tensor(neutral_factor, device=device, dtype=dtype),
                "taper_sigma": torch.tensor(0 if taper_sigma == "cos" else 1, device=device, dtype=torch.int64),
                "taper_phi": torch.tensor(0 if taper_phi == "cos" else 1, device=device, dtype=torch.int64),
            },
        )

    def potential(self, targets: torch.Tensor) -> torch.Tensor:
        device = targets.device
        dtype = targets.dtype
        patch_pts = self._patch_points.to(device=device, dtype=dtype)
        patch_w = self._patch_weights.to(device=device, dtype=dtype)
        ring_pts = self._ring_points.to(device=device, dtype=dtype)
        ring_w = self._ring_weights.to(device=device, dtype=dtype)

        R_patch = torch.linalg.norm(targets[:, None, :] - patch_pts[None, :, :], dim=2).clamp_min(1e-12)
        R_ring = torch.linalg.norm(targets[:, None, :] - ring_pts[None, :, :], dim=2).clamp_min(1e-12)

        V_patch = torch.sum(patch_w * (K_E / R_patch), dim=1)
        V_ring = torch.sum(ring_w * (K_E / R_ring), dim=1)
        return V_patch + V_ring

    def serialize(self) -> Dict[str, Any]:
        p = {k: v.detach().cpu().tolist() for k, v in self.params.items()}
        p["taper_sigma"] = "cos" if int(self.params["taper_sigma"].item()) == 0 else "gauss"
        p["taper_phi"] = "cos" if int(self.params["taper_phi"].item()) == 0 else "gauss"
        return {"type": self.type, "params": p}


class ToroidalEigenModeBasis(ImageBasisElement):
    """Fixed linear combination of primitive basis elements representing a learned BEM mode."""

    def __init__(self, params: Dict[str, torch.Tensor]):
        comps = params.get("components", None)
        if comps is None:
            raise ValueError("ToroidalEigenModeBasis requires 'components'")
        comp_list = comps
        elements: List[Tuple[float, ImageBasisElement]] = []
        for entry in comp_list:
            coeff = float(entry.get("coeff", 0.0))
            elem_ser = entry.get("elem", {})
            if elem_ser is None:
                continue
            elem = ImageBasisElement.deserialize(elem_ser, device=params.get("device", "cpu"), dtype=torch.float32)
            elements.append((coeff, elem))
        self.components = elements
        super().__init__(
            "toroidal_eigen_mode",
            {
                "components": comp_list,
            },
        )

    def potential(self, targets: torch.Tensor) -> torch.Tensor:
        if not self.components:
            return torch.zeros(targets.shape[0], device=targets.device, dtype=targets.dtype)
        acc = torch.zeros(targets.shape[0], device=targets.device, dtype=targets.dtype)
        for coeff, elem in self.components:
            acc = acc + coeff * elem.potential(targets)
        return acc

    def serialize(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "params": {
                "components": self.params["components"],
            },
        }

def generate_candidate_basis(
    spec: CanonicalSpec,
    basis_types: List[str],
    n_candidates: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> List[ImageBasisElement]:
    """Generate a list of candidate image basis elements for a spec.

    The current implementation focuses on grounded planes with point
    charges, using simple physics-informed heuristics:

    * For each point charge, include the mirror point across the plane
      as an image candidate.
    * Perturb the mirror position tangentially and along the normal to
      create a small cloud of nearby candidates.

    This is intentionally generic: it does *not* hard-code any closed-
    form solution, but instead proposes a family of plausible basis
    locations that can be pruned by the sparse solver.
    """
    candidates: List[ImageBasisElement] = []

    conductors = getattr(spec, "conductors", []) or []
    point_charges = [
        c for c in getattr(spec, "charges", []) if c.get("type") == "point"
    ]

    wants_point = "point" in basis_types
    wants_ring = "ring" in basis_types
    wants_ring_gauss = "ring_gauss" in basis_types
    wants_mirror_stack = "mirror_stack" in basis_types
    wants_poloidal = "poloidal_ring" in basis_types
    wants_ladder_inner = "ring_ladder_inner" in basis_types
    wants_ladder_outer = "ring_ladder_outer" in basis_types
    wants_mode_cluster = "toroidal_mode_cluster" in basis_types
    wants_eigen_mode = "toroidal_eigen_mode" in basis_types
    wants_eigen_boundary = "toroidal_eigen_mode_boundary" in basis_types
    wants_eigen_offaxis = "toroidal_eigen_mode_offaxis" in basis_types
    wants_inner_arc = "inner_rim_arc" in basis_types
    wants_inner_ribbon = "inner_rim_ribbon" in basis_types
    wants_inner_patch_ring = "inner_patch_ring" in basis_types
    wants_rich_inner = "rich_inner_rim" in basis_types

    # Mirror-stack candidates for parallel planes (experimental).
    planes = [c for c in conductors if c.get("type") == "plane"]
    if wants_mirror_stack and len(planes) == 2 and point_charges:
        try:
            z_vals = sorted(float(p.get("z", 0.0)) for p in planes)
            z_lower, z_upper = z_vals[0], z_vals[1]
            for charge in point_charges:
                pos = torch.tensor(charge["pos"], device=device, dtype=dtype)
                n_img = min(MirrorStackBasis.MAX_IMAGES, max(2, n_candidates // 4))
                candidates.append(
                    MirrorStackBasis(
                        {
                            "position": pos,
                            "z_lower": torch.tensor(z_lower, device=device, dtype=dtype),
                            "z_upper": torch.tensor(z_upper, device=device, dtype=dtype),
                            "n_images": torch.tensor(n_img, device=device, dtype=dtype),
                        }
                    )
                )
        except Exception:
            pass

    # Plane heuristic (original path)
    plane_conductor = None
    for conductor in conductors:
        if conductor.get("type") == "plane":
            plane_conductor = conductor
            break
    if plane_conductor is not None and wants_point:
        z_plane = float(plane_conductor.get("z", 0.0))
        if not point_charges:
            return candidates[:n_candidates]
        for charge in point_charges:
            pos = charge["pos"]
            x0, y0, z0 = float(pos[0]), float(pos[1]), float(pos[2])

            # Include the real charge position so the solver can represent
            # the total field when needed.
            real_pos = torch.tensor([x0, y0, z0], device=device, dtype=dtype)
            candidates.append(PointChargeBasis({"position": real_pos}))

            # Mirror the charge position across the plane z = z_plane.
            z_img = 2.0 * z_plane - z0
            img_pos = torch.tensor([x0, y0, z_img], device=device, dtype=dtype)
            candidates.append(PointChargeBasis({"position": img_pos}))

            # Add a small cloud of perturbed candidates around the mirror.
            dist = abs(z0 - z_plane)
            if dist > 0.0:
                perturb_scale = 0.1 * dist

                remaining = max(0, n_candidates - len(candidates))
                if remaining > 0:
                    n_perturb = min(16, remaining)

                    for _ in range(n_perturb):
                        perturb = (
                            torch.randn(3, device=device, dtype=dtype) * perturb_scale
                        )
                        p = img_pos + perturb
                        # Keep perturbed images on the opposite side of the plane
                        # from the source charge.
                        if (z0 > z_plane and p[2] <= z_plane) or (
                            z0 < z_plane and p[2] >= z_plane
                        ):
                            p[2] = 2.0 * z_plane - p[2]
                        candidates.append(PointChargeBasis({"position": p}))
            if len(candidates) >= n_candidates:
                break
        return candidates[:n_candidates]

    # Torus heuristic (point + ring candidates)
    torus = None
    for conductor in conductors:
        if conductor.get("type") in ("torus", "toroid"):
            torus = conductor
            break
    if torus is None:
        return candidates

    def _torus_tag(major_radius: float, minor_radius: float) -> str:
        aspect = minor_radius / max(major_radius, 1e-6)
        if aspect < 0.25:
            return "thin"
        if aspect < 0.45:
            return "mid"
        return "fat"

    def _load_eigen_modes(tag: str) -> List[Dict[str, Any]]:
        search_paths = [
            Path("runs/torus") / f"toroidal_eigenmodes_{tag}.json",
            Path("runs") / f"toroidal_eigenmodes_{tag}.json",
        ]
        for p in search_paths:
            if p.exists():
                try:
                    data = json.load(p.open())
                    return data.get("modes", [])
                except Exception:
                    continue
        return []

    R = float(torus.get("major_radius", torus.get("radius", 1.0)))
    a = float(torus.get("minor_radius", 0.25 * R))
    center = torch.tensor(
        torus.get("center", [0.0, 0.0, 0.0]), device=device, dtype=dtype
    )
    aspect = a / max(R, 1e-9)

    # Helper to append a ring of point candidates at radius r_ring, height z_off.
    def _add_ring_points(
        n_pts: int, r_ring: float, z_off: float, jitter_scale: float
    ) -> None:
        nonlocal candidates
        if n_pts <= 0 or len(candidates) >= n_candidates:
            return
        theta = torch.linspace(0.0, 2.0 * torch.pi, n_pts + 1, device=device)[:-1]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        base = torch.stack(
            [r_ring * cos_t, r_ring * sin_t, torch.full_like(theta, z_off)], dim=1
        )
        jitter = torch.randn_like(base) * jitter_scale
        pts = base + jitter + center
        for p in pts:
            if len(candidates) >= n_candidates:
                break
            candidates.append(PointChargeBasis({"position": p}))

    # Non-local ring basis elements (experimental) added first so they survive trimming.
    if wants_ring or wants_ring_gauss:
        ring_n_quad = 64
        radii = [max(1e-6, R - 0.5 * a), R, max(1e-6, R + 0.5 * a)]
        z_offsets = [0.0, 0.3 * a, -0.3 * a]
        for r_ring in radii:
            for z_off in z_offsets:
                if wants_ring:
                    candidates.append(
                        RingImageBasis(
                            {
                                "center": center + torch.tensor(
                                    [0.0, 0.0, z_off], device=device, dtype=dtype
                                ),
                                "radius": torch.tensor(r_ring, device=device, dtype=dtype),
                                "n_quad": torch.tensor(ring_n_quad, device=device),
                            },
                            type_name="ring",
                        )
                    )
                if wants_ring_gauss:
                    candidates.append(
                        RingImageBasis(
                            {
                                "center": center + torch.tensor(
                                    [0.0, 0.0, z_off], device=device, dtype=dtype
                                ),
                                "radius": torch.tensor(r_ring, device=device, dtype=dtype),
                                "n_quad": torch.tensor(ring_n_quad, device=device),
                                "sigma": torch.tensor(0.4, device=device, dtype=dtype),
                            },
                            type_name="ring_gauss",
                        )
                    )

    # Poloidal multipole rings (non-local, keep near front).
    if wants_poloidal:
        delta_r = 0.5 * a
        n_quad = 128
        for order in (0, 1, 2):
            candidates.append(
                PoloidalRingBasis(
                    {
                        "center": center,
                        "radius": torch.tensor(R, device=device, dtype=dtype),
                        "delta_r": torch.tensor(delta_r, device=device, dtype=dtype),
                        "order": torch.tensor(order, device=device, dtype=dtype),
                        "n_quad": torch.tensor(n_quad, device=device),
                    }
                )
            )

    # Ring ladders approximating tails.
    if wants_ladder_inner:
        candidates.append(
            RingLadderBasis(
                {
                    "center": center,
                    "radius": torch.tensor(R, device=device, dtype=dtype),
                    "minor_radius": torch.tensor(a, device=device, dtype=dtype),
                    "variant": torch.tensor(0, device=device, dtype=torch.int64),
                    "n_quad": torch.tensor(96, device=device),
                }
            )
        )
    if wants_ladder_outer:
        candidates.append(
            RingLadderBasis(
                {
                    "center": center,
                    "radius": torch.tensor(R, device=device, dtype=dtype),
                    "minor_radius": torch.tensor(a, device=device, dtype=dtype),
                    "variant": torch.tensor(1, device=device, dtype=torch.int64),
                    "n_quad": torch.tensor(96, device=device),
                }
            )
        )

    def _load_family(family: str) -> List[Dict[str, Any]]:
        tag = _torus_tag(R, a)
        fname = f"toroidal_eigenmodes_{tag}_{family}.json"
        search_paths = [
            Path("runs/torus") / fname,
            Path("runs") / fname,
        ]
        for p in search_paths:
            if p.exists():
                try:
                    data = json.load(p.open())
                    return data.get("modes", [])
                except Exception:
                    continue
        return []

    def _append_modes(modes: List[Dict[str, Any]], max_modes: int = 4) -> None:
        for m in modes[:max_modes]:
            comps = m.get("components", [])
            try:
                candidates.append(
                    ToroidalEigenModeBasis(
                        {
                            "components": comps,
                        }
                    )
                )
            except Exception:
                continue

    if wants_eigen_mode:
        modes = _load_eigen_modes(_torus_tag(R, a))
        _append_modes(modes, max_modes=4)
    if wants_eigen_boundary:
        modes = _load_family("boundary")
        _append_modes(modes, max_modes=4)
    if wants_eigen_offaxis:
        modes = _load_family("offaxis")
        _append_modes(modes, max_modes=4)

    # Inner-rim localized primitives (experimental, boundary-layer inspired).
    if wants_inner_arc or wants_inner_ribbon or wants_inner_patch_ring:
        # Span families (half-spans in radians).
        if aspect < 0.25:  # thin
            arc_spans = [math.radians(25.0), math.radians(35.0)]
            ribbon_phi_spans = [0.55, 0.9]
            ribbon_sigma_span = math.radians(35.0)
        elif aspect < 0.45:  # mid
            arc_spans = [math.radians(40.0), math.radians(55.0)]
            ribbon_phi_spans = [0.8, 1.1]
            ribbon_sigma_span = math.radians(50.0)
        else:  # fat fallback
            arc_spans = [math.radians(45.0), math.radians(60.0)]
            ribbon_phi_spans = [0.8, 1.2]
            ribbon_sigma_span = math.radians(55.0)

        offset_frac = 0.3
        sigma_center = math.pi

        def _phi0_for_charge(pos: torch.Tensor) -> float:
            rel = pos - center
            return float(math.atan2(rel[1].item(), rel[0].item()))

        arc_span_extra = math.radians(15.0)
        for charge in point_charges:
            if len(candidates) >= n_candidates:
                break
            pos = torch.tensor(charge["pos"], device=device, dtype=dtype)
            phi0 = _phi0_for_charge(pos)
            rel = pos - center
            rho = float(torch.linalg.norm(rel[:2]).item())
            z_val = float(rel[2].item())
            dist_centerline = math.sqrt((rho - R) ** 2 + z_val * z_val)
            near_centerline = dist_centerline <= (R + 1.1 * a)

            if wants_inner_arc:
                spans_local = list(arc_spans)
                if wants_rich_inner and aspect < 0.25 and near_centerline:
                    spans_local = [arc_span_extra] + spans_local
                for i, span in enumerate(spans_local):
                    if len(candidates) >= n_candidates:
                        break
                    n_pts = 8 if i == 0 else 10
                    candidates.append(
                        InnerRimArcBasis(
                            {
                                "center": center,
                                "R": torch.tensor(R, device=device, dtype=dtype),
                                "a": torch.tensor(a, device=device, dtype=dtype),
                                "phi0": torch.tensor(phi0, device=device, dtype=dtype),
                                "d_sigma": torch.tensor(span, device=device, dtype=dtype),
                                "n_pts": torch.tensor(n_pts, device=device),
                                "offset_frac": torch.tensor(offset_frac, device=device, dtype=dtype),
                                "taper": torch.tensor(0, device=device, dtype=torch.int64),
                            }
                            )
                        )
                if wants_rich_inner and len(candidates) < n_candidates:
                    delta = math.radians(10.0)
                    span_rich = arc_span_extra if (aspect < 0.25 and near_centerline) else arc_spans[0]
                    for sign in (-1.0, 1.0):
                        if len(candidates) >= n_candidates:
                            break
                        sigma0 = math.pi + sign * delta
                        candidates.append(
                            InnerRimArcBasis(
                                {
                                    "center": center,
                                    "R": torch.tensor(R, device=device, dtype=dtype),
                                    "a": torch.tensor(a, device=device, dtype=dtype),
                                    "phi0": torch.tensor(phi0, device=device, dtype=dtype),
                                    "sigma0": torch.tensor(sigma0, device=device, dtype=dtype),
                                    "d_sigma": torch.tensor(span_rich, device=device, dtype=dtype),
                                    "n_pts": torch.tensor(8, device=device),
                                    "offset_frac": torch.tensor(offset_frac, device=device, dtype=dtype),
                                    "taper": torch.tensor(0, device=device, dtype=torch.int64),
                                }
                            )
                        )
            if wants_inner_ribbon and len(candidates) < n_candidates:
                d_phi = ribbon_phi_spans[-1]
                candidates.append(
                    InnerRimRibbonBasis(
                        {
                            "center": center,
                            "R": torch.tensor(R, device=device, dtype=dtype),
                            "a": torch.tensor(a, device=device, dtype=dtype),
                            "phi0": torch.tensor(phi0, device=device, dtype=dtype),
                            "d_sigma": torch.tensor(ribbon_sigma_span, device=device, dtype=dtype),
                            "d_phi": torch.tensor(d_phi, device=device, dtype=dtype),
                            "n_sigma": torch.tensor(6, device=device),
                            "n_phi": torch.tensor(6, device=device),
                            "offset_frac": torch.tensor(offset_frac, device=device, dtype=dtype),
                            "taper_sigma": torch.tensor(0, device=device, dtype=torch.int64),
                            "taper_phi": torch.tensor(0, device=device, dtype=torch.int64),
                        }
                    )
                )
            if wants_inner_patch_ring and len(candidates) < n_candidates:
                candidates.append(
                    InnerPatchRingBasis(
                        {
                            "center": center,
                            "R": torch.tensor(R, device=device, dtype=dtype),
                            "a": torch.tensor(a, device=device, dtype=dtype),
                            "phi0": torch.tensor(phi0, device=device, dtype=dtype),
                            "d_sigma": torch.tensor(ribbon_sigma_span, device=device, dtype=dtype),
                            "d_phi": torch.tensor(ribbon_phi_spans[-1], device=device, dtype=dtype),
                            "n_sigma": torch.tensor(6, device=device),
                            "n_phi": torch.tensor(6, device=device),
                            "offset_frac": torch.tensor(offset_frac, device=device, dtype=dtype),
                            "ring_offset_frac": torch.tensor(0.55, device=device, dtype=dtype),
                            "n_ring": torch.tensor(16, device=device),
                            "neutral_factor": torch.tensor(1.0, device=device, dtype=dtype),
                            "taper_sigma": torch.tensor(0, device=device, dtype=torch.int64),
                            "taper_phi": torch.tensor(0, device=device, dtype=torch.int64),
                        }
                    )
                )
                if wants_rich_inner and len(candidates) < n_candidates:
                    candidates.append(
                        InnerPatchRingBasis(
                            {
                                "center": center,
                                "R": torch.tensor(R, device=device, dtype=dtype),
                                "a": torch.tensor(a, device=device, dtype=dtype),
                                "phi0": torch.tensor(phi0, device=device, dtype=dtype),
                                "d_sigma": torch.tensor(arc_spans[0], device=device, dtype=dtype),
                                "d_phi": torch.tensor(ribbon_phi_spans[0], device=device, dtype=dtype),
                                "n_sigma": torch.tensor(5, device=device),
                                "n_phi": torch.tensor(5, device=device),
                                "offset_frac": torch.tensor(offset_frac, device=device, dtype=dtype),
                                "ring_offset_frac": torch.tensor(0.6, device=device, dtype=dtype),
                                "n_ring": torch.tensor(12, device=device),
                                "neutral_factor": torch.tensor(0.8, device=device, dtype=dtype),
                                "taper_sigma": torch.tensor(0, device=device, dtype=torch.int64),
                                "taper_phi": torch.tensor(0, device=device, dtype=torch.int64),
                            }
                        )
                    )

    if wants_point:
        remaining = max(1, n_candidates - len(candidates))
        target = max(1, remaining)
        n_ring_main = max(8, min(32, target // 2))
        n_ring_offset = max(0, min(24, target // 3))
        n_axis = max(1, min(3, target - (n_ring_main + 2 * n_ring_offset)))

        # Main ring slightly inside the tube to stay within conductor volume.
        r_main = max(1e-6, R - 0.3 * a)
        jitter = 0.1 * a
        _add_ring_points(n_ring_main, r_main, 0.0, jitter)

        # Inner/outer offset rings and z-perturbed rings to capture surface curvature.
        r_inner = max(1e-6, R - 0.6 * a)
        r_outer = max(1e-6, R + 0.6 * a)
        z_off = 0.3 * a
        _add_ring_points(n_ring_offset, r_inner, 0.0, jitter)
        _add_ring_points(n_ring_offset, r_outer, 0.0, jitter)
        _add_ring_points(max(4, n_ring_offset // 2), r_main, z_off, jitter)
        _add_ring_points(max(4, n_ring_offset // 2), r_main, -z_off, jitter)

        # Axial helpers near the torus hole / centerline to act like effective ring charges.
        if n_axis > 0:
            rho_axis = 0.25 * R
            z_positions = torch.linspace(-0.2 * a, 0.2 * a, n_axis, device=device)
            for z_val in z_positions:
                for sign in (-1.0, 1.0):
                    if len(candidates) >= n_candidates:
                        break
                    pos = torch.tensor(
                        [sign * rho_axis, 0.0, float(z_val)], device=device, dtype=dtype
                    ) + center
                    candidates.append(PointChargeBasis({"position": pos}))

    if wants_mode_cluster:
        for m in (0, 1, 2):
            candidates.append(
                ToroidalModeClusterBasis(
                    {
                        "center": center,
                        "major_radius": torch.tensor(R, device=device, dtype=dtype),
                        "minor_radius": torch.tensor(a, device=device, dtype=dtype),
                        "mode_m": torch.tensor(m, device=device, dtype=torch.int64),
                        "n_phi": torch.tensor(16, device=device, dtype=torch.int64),
                        "radial_offset": torch.tensor(0.5 * a, device=device, dtype=dtype),
                    }
                )
            )

    if len(candidates) > n_candidates:
        candidates = candidates[:n_candidates]
    return candidates


def build_dictionary(
    basis: Sequence[ImageBasisElement],
    X: torch.Tensor,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a dictionary matrix Φ with columns Φ[:, k] = basis[k].potential(X).

    Parameters
    ----------
    basis:
        Sequence of image basis elements.
    X:
        [N, 3] tensor of evaluation points.
    device, dtype:
        Device and dtype for the returned matrix and the point cloud.
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)

    X = X.to(device=device, dtype=dtype)
    N = X.shape[0]
    K = len(basis)

    if K == 0:
        return torch.zeros(N, 0, device=device, dtype=dtype)
    if N == 0:
        return torch.zeros(0, K, device=device, dtype=dtype)

    Phi = torch.empty(N, K, device=device, dtype=dtype)
    for k, elem in enumerate(basis):
        Phi[:, k] = elem.potential(X).to(device=device, dtype=dtype)
    return Phi
