from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

import torch

from .pole_types import PoleTerm


def _default_k_samples(k_rect: float, n_samples: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.linspace(1e-6, k_rect, n_samples, device=device, dtype=dtype)


def _complex_derivative(fn: Callable[[torch.Tensor], torch.Tensor], z: torch.Tensor, h: float = 1e-5) -> torch.Tensor:
    h_c = torch.as_tensor(h, device=z.device, dtype=z.dtype)
    return (fn(z + 1j * h_c) - fn(z - 1j * h_c)) / (2j * h_c)


def find_poles_denominator_roots(
    denominator_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    *,
    k_samples: Iterable[float] | None = None,
    max_poles: int = 4,
    k_rect: float = 20.0,
    n_samples: int = 256,
    newton_tol: float = 1e-8,
    newton_max_iter: int = 16,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.complex128,
) -> List[PoleTerm]:
    k_line = (
        torch.as_tensor(list(k_samples), device=device, dtype=dtype)
        if k_samples is not None
        else _default_k_samples(k_rect, n_samples, device, dtype)
    )
    denom_vals, numer_vals = denominator_fn(k_line)
    if denom_vals.device.type != "cuda":
        denom_vals = denom_vals.to(device=device, dtype=dtype)
    if numer_vals.device.type != "cuda":
        numer_vals = numer_vals.to(device=device, dtype=dtype)

    mag = torch.abs(denom_vals)
    topk = min(int(max_poles * 3), mag.numel())
    _, candidate_idxs = torch.topk(-mag, k=topk)
    if mag.numel() > 2:
        local_min = (mag[1:-1] < mag[:-2]) & (mag[1:-1] <= mag[2:])
        local_idxs = torch.nonzero(local_min).flatten() + 1
        candidate_idxs = torch.unique(torch.cat([candidate_idxs, local_idxs]))
    candidate_idxs = candidate_idxs[: min(len(candidate_idxs), max_poles * 4)]

    poles: List[PoleTerm] = []
    for idx in candidate_idxs[:max_poles]:
        k0 = k_line[idx]

        def f(z: torch.Tensor) -> torch.Tensor:
            d, _ = denominator_fn(z)
            return d

        def numer(z: torch.Tensor) -> torch.Tensor:
            _, n = denominator_fn(z)
            return n

        k_cur = k0
        iters = 0
        for it in range(newton_max_iter):
            iters = it + 1
            denom_val = f(k_cur)
            deriv = _complex_derivative(f, k_cur)
            if torch.abs(deriv) < 1e-14:
                break
            step = denom_val / deriv
            k_next = k_cur - step
            if torch.abs(step) < newton_tol:
                k_cur = k_next
                break
            k_cur = k_next

        # Deduplicate
        if any(abs(k_cur.item() - p.pole) < newton_tol * 10 for p in poles):
            continue

        denom_deriv = _complex_derivative(f, k_cur)
        resid_num = numer(k_cur)
        if torch.abs(denom_deriv) < 1e-14:
            continue
        residue = resid_num / denom_deriv
        poles.append(
            PoleTerm(
                pole=complex(k_cur.item()),
                residue=complex(residue.item()),
                kind="denominator_root",
                meta={"iterations": int(iters)},
            )
        )
        if len(poles) >= max_poles:
            break

    return poles
