from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import torch

from .pole_types import PoleTerm


def _poly_eval(coeffs: np.ndarray, z: complex) -> complex:
    return np.polyval(coeffs, z)


def _poly_derivative(coeffs: np.ndarray) -> np.ndarray:
    degree = len(coeffs) - 1
    return np.array([coeffs[i] * (degree - i) for i in range(degree)], dtype=coeffs.dtype)


def find_poles_rational_fit(
    reflection_fn,
    *,
    k_samples: Iterable[float] | None = None,
    max_poles: int = 4,
    order_override: Optional[int] = None,
    stability_half_plane: Optional[str] = None,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.complex128,
) -> List[PoleTerm]:
    k_grid = (
        torch.as_tensor(list(k_samples), device=device, dtype=dtype)
        if k_samples is not None
        else torch.linspace(1e-4, 20.0, 256, device=device, dtype=dtype)
    )
    values = reflection_fn(k_grid)
    values = values.to(device=device, dtype=dtype)

    order = order_override if order_override is not None else min(max_poles, 2)
    m = max(2, int(order))

    # Build linear system for P(k)/Q(k) ~ F(k); Q leading coefficient normalized to 1.
    k_cpu = k_grid.detach().cpu().numpy()
    v_cpu = values.detach().cpu().numpy()

    A_rows = []
    b_rows = []
    for ki, fi in zip(k_cpu, v_cpu):
        num_terms = [ki ** p for p in range(m)]  # a0..a_{m-1}
        den_terms = [-(fi * (ki ** p)) for p in range(1, m + 1)]  # b1..b_m
        A_rows.append(np.concatenate([num_terms, den_terms]))
        b_rows.append(fi)
    A = np.vstack(A_rows)
    b = np.asarray(b_rows)

    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a = sol[:m]
    b_coeff = np.concatenate([np.array([1.0 + 0j], dtype=complex), sol[m:]])

    # Build full coefficient arrays highest degree first for polyval
    P_coeff = np.flip(a)
    Q_coeff = np.flip(b_coeff)

    poles_raw = np.roots(Q_coeff)
    poles: List[PoleTerm] = []

    Q_deriv = _poly_derivative(Q_coeff)
    def _stable(p: complex) -> bool:
        if stability_half_plane:
            plane = stability_half_plane.lower()
            if plane.startswith("re>0") and p.real <= 0:
                return False
            if plane.startswith("im>0") and p.imag <= 0:
                return False
        return True

    filtered = [p for p in poles_raw if _stable(p)]
    if not filtered:
        filtered = list(poles_raw)
    filtered = sorted(filtered, key=lambda p: (p.real < 0, abs(p.imag), abs(p.real)))

    for pole in filtered:
        resid = _poly_eval(P_coeff, pole) / _poly_eval(Q_deriv, pole)
        poles.append(
            PoleTerm(
                pole=complex(pole),
                residue=complex(resid),
                kind="rational_fit",
                meta={"fit_order": int(m)},
            )
        )
        if len(poles) >= max_poles:
            break

    return poles
