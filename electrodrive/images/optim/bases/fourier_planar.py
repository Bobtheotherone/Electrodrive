from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch


def build_planar_grid(
    grid_shape: Sequence[int],
    *,
    x_extent: Sequence[float] = (-1.0, 1.0),
    y_extent: Sequence[float] = (-1.0, 1.0),
    z: float = 0.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    h, w = int(grid_shape[0]), int(grid_shape[1])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or torch.float32
    x = torch.linspace(float(x_extent[0]), float(x_extent[1]), w, device=device, dtype=dtype)
    y = torch.linspace(float(y_extent[0]), float(y_extent[1]), h, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    zz = torch.full_like(xx, float(z))
    pts = torch.stack([xx, yy, zz], dim=-1)
    return pts.reshape(-1, 3)


class PlanarFFTConstraintOp:
    def __init__(
        self,
        *,
        grid_shape: Sequence[int],
        mode_indices: Optional[Iterable[Sequence[int]]] = None,
        mask: Optional[torch.Tensor] = None,
        fft_shift: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_complex: bool = False,
    ) -> None:
        self.grid_h = int(grid_shape[0])
        self.grid_w = int(grid_shape[1])
        self.fft_shift = bool(fft_shift)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32
        self.use_complex = bool(use_complex)
        self._complex_dtype = torch.complex128 if self.dtype == torch.float64 else torch.complex64

        if mask is not None:
            mask_t = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
            if mask_t.shape != (self.grid_h, self.grid_w):
                raise ValueError("FFT mask must match grid shape.")
            self.mask = mask_t
            self.mode_indices = None
        else:
            if mode_indices is None:
                self.mask = torch.ones((self.grid_h, self.grid_w), device=self.device, dtype=torch.bool)
                self.mode_indices = None
            else:
                idx = torch.as_tensor(list(mode_indices), device=self.device, dtype=torch.long)
                if idx.ndim != 2 or idx.shape[1] != 2:
                    raise ValueError("mode_indices must be shape (M, 2).")
                self.mode_indices = idx
                self.mask = None

    def _reshape(self, r: torch.Tensor) -> torch.Tensor:
        if r.ndim == 1:
            if r.numel() != self.grid_h * self.grid_w:
                raise ValueError("Residual length does not match grid size.")
            return r.view(self.grid_h, self.grid_w)
        if r.ndim == 2 and r.shape == (self.grid_h, self.grid_w):
            return r
        raise ValueError("Residual must be a flattened grid or (H, W) tensor.")

    def _select(self, f: torch.Tensor) -> torch.Tensor:
        if self.mask is not None:
            return f[self.mask]
        assert self.mode_indices is not None
        return f[self.mode_indices[:, 0], self.mode_indices[:, 1]]

    def _scatter(self, coeffs: torch.Tensor) -> torch.Tensor:
        grid = torch.zeros((self.grid_h, self.grid_w), device=self.device, dtype=self._complex_dtype)
        if self.mask is not None:
            grid[self.mask] = coeffs
        else:
            assert self.mode_indices is not None
            grid[self.mode_indices[:, 0], self.mode_indices[:, 1]] = coeffs
        return grid

    def apply(self, r: torch.Tensor) -> torch.Tensor:
        r_grid = self._reshape(r.to(device=self.device, dtype=self.dtype))
        f = torch.fft.fft2(r_grid.to(self._complex_dtype))
        if self.fft_shift:
            f = torch.fft.fftshift(f)
        coeffs = self._select(f)
        if self.use_complex:
            return coeffs
        return torch.cat([coeffs.real, coeffs.imag], dim=0)

    def adjoint(self, c: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(c):
            coeffs = c
        else:
            c = c.to(device=self.device, dtype=self.dtype).view(-1)
            if self.mask is not None:
                n_coeffs = int(self.mask.sum().item())
            else:
                assert self.mode_indices is not None
                n_coeffs = int(self.mode_indices.shape[0])
            if c.numel() != 2 * n_coeffs:
                raise ValueError("Coefficient vector has unexpected length.")
            coeffs = c[:n_coeffs] + 1j * c[n_coeffs:]
        f_grid = self._scatter(coeffs.to(self._complex_dtype))
        if self.fft_shift:
            f_grid = torch.fft.ifftshift(f_grid)
        r_grid = torch.fft.ifft2(f_grid)
        return r_grid.real.reshape(-1)
