from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch


class CylindricalFourierConstraintOp:
    def __init__(
        self,
        *,
        grid_shape: Sequence[int],
        mode_indices: Optional[Iterable[Sequence[int]]] = None,
        mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_complex: bool = False,
    ) -> None:
        self.grid_phi = int(grid_shape[0])
        self.grid_z = int(grid_shape[1])
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32
        self.use_complex = bool(use_complex)
        self._complex_dtype = torch.complex128 if self.dtype == torch.float64 else torch.complex64

        if mask is not None:
            mask_t = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
            if mask_t.shape != (self.grid_phi, self.grid_z):
                raise ValueError("Cylindrical FFT mask must match grid shape.")
            self.mask = mask_t
            self.mode_indices = None
        else:
            if mode_indices is None:
                self.mask = torch.ones((self.grid_phi, self.grid_z), device=self.device, dtype=torch.bool)
                self.mode_indices = None
            else:
                idx = torch.as_tensor(list(mode_indices), device=self.device, dtype=torch.long)
                if idx.ndim != 2 or idx.shape[1] != 2:
                    raise ValueError("mode_indices must be shape (M, 2).")
                self.mode_indices = idx
                self.mask = None

    def _reshape(self, r: torch.Tensor) -> torch.Tensor:
        if r.ndim == 1:
            if r.numel() != self.grid_phi * self.grid_z:
                raise ValueError("Residual length does not match grid size.")
            return r.view(self.grid_phi, self.grid_z)
        if r.ndim == 2 and r.shape == (self.grid_phi, self.grid_z):
            return r
        raise ValueError("Residual must be a flattened grid or (N_phi, N_z) tensor.")

    def _select(self, f: torch.Tensor) -> torch.Tensor:
        if self.mask is not None:
            return f[self.mask]
        assert self.mode_indices is not None
        return f[self.mode_indices[:, 0], self.mode_indices[:, 1]]

    def _scatter(self, coeffs: torch.Tensor) -> torch.Tensor:
        grid = torch.zeros((self.grid_phi, self.grid_z), device=self.device, dtype=self._complex_dtype)
        if self.mask is not None:
            grid[self.mask] = coeffs
        else:
            assert self.mode_indices is not None
            grid[self.mode_indices[:, 0], self.mode_indices[:, 1]] = coeffs
        return grid

    def apply(self, r: torch.Tensor) -> torch.Tensor:
        r_grid = self._reshape(r.to(device=self.device, dtype=self.dtype))
        f = torch.fft.fft2(r_grid.to(self._complex_dtype))
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
        r_grid = torch.fft.ifft2(f_grid)
        return r_grid.real.reshape(-1)
