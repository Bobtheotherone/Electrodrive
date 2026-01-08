from __future__ import annotations

from typing import Optional

import torch


def stable_subtract_reference(
    values: torch.Tensor,
    reference: torch.Tensor,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Subtract reference in float64 and return in out_dtype."""
    vals64 = values.to(device=values.device, dtype=torch.float64)
    ref64 = reference.to(device=values.device, dtype=torch.float64)
    out = vals64 - ref64
    return out.to(dtype=out_dtype)


def add_reference(pred_corr: torch.Tensor, reference: Optional[torch.Tensor]) -> torch.Tensor:
    """Add reference after casting to pred dtype to avoid unintended upcasts."""
    if reference is None:
        return pred_corr
    ref = reference.to(device=pred_corr.device, dtype=pred_corr.dtype)
    return pred_corr + ref


__all__ = ["stable_subtract_reference", "add_reference"]
