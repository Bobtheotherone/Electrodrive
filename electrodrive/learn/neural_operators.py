from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _default_device_dtype(
    device: Optional[torch.device],
    dtype: Optional[torch.dtype],
) -> Tuple[torch.device, torch.dtype]:
    dev = torch.device("cpu") if device is None else torch.device(device)
    dt = torch.float32 if dtype is None else dtype
    return dev, dt


def make_unit_sphere_grid(
    n_theta: int,
    n_phi: int,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a deterministic (theta, phi) grid on the unit sphere.

    Theta spans [0, pi], phi spans [0, 2*pi) with periodic wrap in phi.
    """
    device, dtype = _default_device_dtype(device, dtype)
    theta = torch.linspace(0.0, math.pi, n_theta, device=device, dtype=dtype)
    phi = (
        torch.arange(n_phi, device=device, dtype=dtype)
        * (2.0 * math.pi / float(n_phi))
    )
    theta_g, phi_g = torch.meshgrid(theta, phi, indexing="ij")
    return theta_g, phi_g


def _angle_encoding(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Return channel-first trig encoding for spherical angles."""
    return torch.stack(
        [
            torch.sin(theta),
            torch.cos(theta),
            torch.sin(phi),
            torch.cos(phi),
        ],
        dim=0,
    )


class SpectralConv2d(nn.Module):
    """Minimal 2D Fourier layer used by SphereFNO."""

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        scale = 1.0 / (in_channels * out_channels)
        weight_shape = (in_channels, out_channels, modes_x, modes_y, 2)
        self.weights = nn.Parameter(scale * torch.randn(*weight_shape))

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # input: (B, in_c, H, W//2+1); weights: (in_c, out_c, modes_x, modes_y, 2)
        real = weights[..., 0]
        imag = weights[..., 1]
        # einsum over complex components manually to avoid dtype mismatches.
        out_real = torch.einsum("bixy,ioxy->boxy", input.real, real) - torch.einsum(
            "bixy,ioxy->boxy", input.imag, imag
        )
        out_imag = torch.einsum("bixy,ioxy->boxy", input.real, imag) + torch.einsum(
            "bixy,ioxy->boxy", input.imag, real
        )
        return torch.complex(out_real, out_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            height,
            width // 2 + 1,
            device=x.device,
            dtype=x_ft.dtype,
        )

        mx = min(self.modes_x, x_ft.shape[2])
        my = min(self.modes_y, x_ft.shape[3])
        out_ft[:, :, :mx, :my] = self.compl_mul2d(
            x_ft[:, :, :mx, :my],
            self.weights[:, :, :mx, :my],
        )

        return torch.fft.irfft2(out_ft, s=(height, width))


class FNOBlock(nn.Module):
    """One Fourier block: spectral conv + 1x1 residual projection."""

    def __init__(self, width: int, modes_x: int, modes_y: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x, modes_y)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x)
        y = y + self.pointwise(x)
        return F.gelu(y)


class SphereFNO(nn.Module):
    """
    Lightweight Fourier Neural Operator for grounded sphere potentials.

    Inputs: broadcasted (q, z0, a) scalars + trig encoding of (theta, phi).
    Output: potential on the spherical grid in reduced units.
    """

    def __init__(
        self,
        n_theta: int = 64,
        n_phi: int = 128,
        modes_theta: int = 16,
        modes_phi: int = 16,
        width: int = 64,
        n_layers: int = 4,
        param_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.width = width

        theta_g, phi_g = make_unit_sphere_grid(n_theta, n_phi)
        self.register_buffer("theta_grid", theta_g, persistent=False)
        self.register_buffer("phi_grid", phi_g, persistent=False)
        pos_enc = _angle_encoding(theta_g, phi_g)
        self.register_buffer("pos_enc", pos_enc, persistent=False)

        self.param_mlp = nn.Sequential(
            nn.Linear(3, param_hidden),
            nn.GELU(),
            nn.Linear(param_hidden, width),
        )

        in_channels = 3 + pos_enc.shape[0] + width
        self.input_proj = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNOBlock(width, modes_theta, modes_phi) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, 1, kernel_size=1),
        )

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        params : torch.Tensor
            Shape [B, 3] containing (q, z0, a).
        """
        if params.dim() == 1:
            params = params.unsqueeze(0)
        params = params.to(device=self.pos_enc.device, dtype=self.pos_enc.dtype)
        batch = params.shape[0]

        pos = self.pos_enc.unsqueeze(0).expand(batch, -1, -1, -1)
        param_raw = params.view(batch, 3, 1, 1).expand(-1, -1, self.n_theta, self.n_phi)
        param_embed = (
            self.param_mlp(params)
            .view(batch, self.width, 1, 1)
            .expand(-1, -1, self.n_theta, self.n_phi)
        )
        x = torch.cat([param_raw, pos, param_embed], dim=1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        out = self.head(x)
        return out.squeeze(1)  # [B, n_theta, n_phi]


def _pull_model_state(state: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("model_state_dict", "state_dict", "model"):
        if key in state and isinstance(state[key], dict):
            return state[key]
    return state


def _extract_metric(
    metrics: Dict[str, Any], keys: Tuple[str, ...]
) -> Optional[float]:
    for k in keys:
        if k in metrics:
            try:
                return float(metrics[k])
            except Exception:
                continue
    return None


@dataclass
class SphereFNOSurrogate:
    """Wrapper that manages SphereFNO inference + light validation."""

    model: SphereFNO
    ckpt_path: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    val_rel_l2: Optional[float] = None
    val_rel_linf: Optional[float] = None
    validated: bool = False
    allow_unvalidated: bool = False
    radial_extension: str = "inv_r"  # inv_r | clamp_zero

    def is_ready(self) -> bool:
        return self.validated or self.allow_unvalidated

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        l2_tol: float = 1e-3,
        linf_tol: float = 1e-2,
        allow_unvalidated: bool = False,
        radial_extension: str = "inv_r",
    ) -> "SphereFNOSurrogate":
        dev, dt = _default_device_dtype(device, dtype)
        state = torch.load(ckpt_path, map_location=dev)
        model = SphereFNO()
        model.load_state_dict(_pull_model_state(state), strict=False)
        model.to(device=dev, dtype=dt).eval()

        metrics: Dict[str, float] = {}
        for key in ("metrics", "val_metrics", "summary", "validation"):
            if key in state and isinstance(state[key], dict):
                metrics.update(
                    {k: float(v) for k, v in state[key].items() if isinstance(v, (int, float))}
                )

        val_rel_l2 = _extract_metric(
            metrics,
            ("val_rel_l2", "rel_l2", "val_l2", "val_relative_l2"),
        )
        val_rel_linf = _extract_metric(
            metrics,
            ("val_rel_linf", "rel_linf", "val_linf", "val_relative_linf"),
        )
        validated = (
            val_rel_l2 is not None
            and val_rel_l2 <= l2_tol
            and val_rel_linf is not None
            and val_rel_linf <= linf_tol
        )

        return cls(
            model=model,
            ckpt_path=str(ckpt_path),
            metrics=metrics or None,
            val_rel_l2=val_rel_l2,
            val_rel_linf=val_rel_linf,
            validated=validated,
            allow_unvalidated=allow_unvalidated,
            radial_extension=radial_extension,
        )

    @torch.no_grad()
    def forward_grid(self, params: torch.Tensor) -> torch.Tensor:
        """Run the model on the canonical grid."""
        return self.model(params)

    @torch.no_grad()
    def evaluate_points(
        self,
        params: Tuple[float, float, float],
        points: torch.Tensor,
        *,
        center: Tuple[float, float, float],
    ) -> torch.Tensor:
        """
        Evaluate the surrogate on arbitrary 3D points.

        Parameters
        ----------
        params:
            Tuple ``(q, z0, a)`` describing the Stage-0 sphere task.
        points:
            [N, 3] tensor of positions in world coordinates.
        center:
            Sphere centre used to compute local spherical angles.
        """
        if points.numel() == 0:
            return torch.zeros(0, device=self.model.pos_enc.device, dtype=self.model.pos_enc.dtype)

        points = points.to(device=self.model.pos_enc.device, dtype=self.model.pos_enc.dtype)
        q, z0, a = params
        params_t = torch.tensor(
            [[float(q), float(z0), float(a)]],
            device=self.model.pos_enc.device,
            dtype=self.model.pos_enc.dtype,
        )
        grid_values = self.forward_grid(params_t)[0]  # [n_theta, n_phi]

        pts_local = points - torch.tensor(
            center, device=points.device, dtype=points.dtype
        ).view(1, 3)
        r = torch.linalg.norm(pts_local, dim=1).clamp_min(1e-8)
        theta = torch.acos(torch.clamp(pts_local[:, 2] / r, -1.0, 1.0))
        phi = torch.atan2(pts_local[:, 1], pts_local[:, 0])
        phi = torch.remainder(phi, 2.0 * math.pi)

        theta_idx = theta / math.pi * max(1, self.model.n_theta - 1)
        phi_idx = phi / (2.0 * math.pi) * float(self.model.n_phi)
        theta0 = torch.floor(theta_idx).long().clamp(0, self.model.n_theta - 1)
        theta1 = torch.clamp(theta0 + 1, max=self.model.n_theta - 1)
        phi0 = torch.floor(phi_idx).long()
        phi1 = (phi0 + 1) % self.model.n_phi
        wt = theta_idx - theta0.float()
        wp = phi_idx - phi0.float()

        grid = grid_values
        v00 = grid[theta0, phi0]
        v01 = grid[theta0, phi1]
        v10 = grid[theta1, phi0]
        v11 = grid[theta1, phi1]
        v0 = v00 * (1 - wp) + v01 * wp
        v1 = v10 * (1 - wp) + v11 * wp
        v_surface = v0 * (1 - wt) + v1 * wt

        a_safe = max(float(a), 1e-6)
        r_norm = r / a_safe
        if self.radial_extension == "clamp_zero":
            scale = torch.where(r_norm < 1.0, torch.zeros_like(r_norm), torch.ones_like(r_norm))
        else:  # inv_r decay with conductor interior clamped to zero
            scale = torch.where(r_norm < 1.0, torch.zeros_like(r_norm), 1.0 / r_norm)

        return v_surface * scale.to(device=v_surface.device, dtype=v_surface.dtype)


def extract_stage0_sphere_params(
    spec: Any,
    *,
    axis_tol: float = 1e-6,
) -> Optional[Tuple[float, float, float, Tuple[float, float, float]]]:
    """
    Parse CanonicalSpec for the Stage-0 grounded sphere + on-axis charge task.
    """
    conductors = getattr(spec, "conductors", None) or []
    charges = getattr(spec, "charges", None) or []
    if len(conductors) != 1 or len(charges) != 1:
        return None
    c = conductors[0]
    ch = charges[0]
    if c.get("type") != "sphere" or c.get("potential", 0.0) != 0.0:
        return None
    if ch.get("type") != "point":
        return None

    q = float(ch.get("q", 0.0))
    pos = ch.get("pos") or ch.get("position")
    if pos is None:
        return None
    x0, y0, z0_abs = map(float, pos)
    center = tuple(map(float, c.get("center", [0.0, 0.0, 0.0])))
    cx, cy, cz = center
    if abs(x0 - cx) > axis_tol or abs(y0 - cy) > axis_tol:
        return None
    radius = c.get("radius")
    if radius is None:
        return None
    a = float(radius)
    z0 = z0_abs - cz
    return q, z0, a, center


def load_spherefno_from_env(
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    *,
    l2_tol: float = 1e-3,
    linf_tol: float = 1e-2,
) -> Optional[SphereFNOSurrogate]:
    """
    Convenience loader driven by environment variables.

    - EDE_SPHEREFNO_CKPT: path to checkpoint
    - EDE_SPHEREFNO_ALLOW_UNVALIDATED: allow use even without metrics
    - EDE_SPHEREFNO_RADIAL: "inv_r" (default) or "clamp_zero"
    """
    ckpt = os.getenv("EDE_SPHEREFNO_CKPT", "").strip()
    if not ckpt:
        return None
    allow_unvalidated = os.getenv("EDE_SPHEREFNO_ALLOW_UNVALIDATED", "").lower() in (
        "1",
        "true",
        "yes",
    )
    radial_mode = os.getenv("EDE_SPHEREFNO_RADIAL", "inv_r").strip().lower()
    try:
        surrogate = SphereFNOSurrogate.from_checkpoint(
            ckpt,
            device=device,
            dtype=dtype,
            l2_tol=l2_tol,
            linf_tol=linf_tol,
            allow_unvalidated=allow_unvalidated,
            radial_extension=radial_mode if radial_mode in ("inv_r", "clamp_zero") else "inv_r",
        )
    except FileNotFoundError:
        return None
    except Exception:
        return None
    if not surrogate.is_ready():
        return None
    return surrogate
