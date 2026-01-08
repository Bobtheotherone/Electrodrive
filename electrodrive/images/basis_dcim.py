from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Dict, Tuple

import torch

from electrodrive.images.basis import ImageBasisElement
from electrodrive.layers.dcim_types import ComplexImageTerm, DCIMBlock


@dataclass
class DCIMBlockBasis(ImageBasisElement):
    """
    Basis element that evaluates one or more DCIM complex images.

    The basis emits a *real-valued* potential using either the real or
    imaginary component of the underlying complex image contribution.
    """

    component: str
    images: Sequence[ComplexImageTerm]
    block: DCIMBlock
    amplitude: complex

    def __init__(
        self,
        block: DCIMBlock,
        images: Optional[Sequence[ComplexImageTerm]] = None,
        component: str = "real",
        amplitude: complex = 1.0 + 0.0j,
    ):
        if component not in ("real", "imag"):
            raise ValueError("component must be 'real' or 'imag'")
        self.block = block
        self.images = tuple(images) if images is not None else tuple(block.images)
        self.component = component
        self.amplitude = amplitude
        self._cache: Dict[Tuple[torch.device, torch.dtype], Dict[str, object]] = {}
        # Store lightweight params for (de)serialization.
        comp_flag = 0 if component == "real" else 1
        super().__init__(
            f"dcim_block_{component}",
            {
                "component": torch.tensor(comp_flag),
                "amplitude": torch.tensor([float(amplitude.real), float(amplitude.imag)]),
            },
        )

    def _get_cache(self, device: torch.device, real_dtype: torch.dtype) -> Dict[str, object]:
        key = (device, real_dtype)
        cached = self._cache.get(key, None)
        if cached is not None:
            return cached

        meta = self.block.certificate.meta or {}
        source_pos = meta.get("source_pos", None)
        if source_pos is None or len(source_pos) < 3:
            raise ValueError("DCIMBlockBasis requires source_pos metadata with three components.")
        src_xy = torch.tensor(source_pos[:2], device=device, dtype=real_dtype)
        z0 = float(source_pos[2])
        z_ref_val = float(meta.get("z_ref", 0.0))
        z_ref_tensor = torch.as_tensor(z_ref_val, device=device, dtype=real_dtype)
        z0_ref = torch.as_tensor(z0 - z_ref_val, device=device, dtype=real_dtype)

        eps1 = float(self.block.stack.layers[0].eps.real)
        q = float(meta.get("source_charge", 1.0))
        pref = q / (4.0 * math.pi * eps1)

        complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
        depths = torch.as_tensor([img.depth for img in self.images], device=device, dtype=complex_dtype)
        weights = torch.as_tensor([img.weight for img in self.images], device=device, dtype=complex_dtype)
        amp = torch.as_tensor(self.amplitude, device=device, dtype=complex_dtype)
        weights = weights * amp * pref

        zero_like_depths = torch.zeros((), device=device, dtype=depths.real.dtype)
        all_real = bool(
            torch.allclose(torch.imag(depths), zero_like_depths, atol=0.0)
            and torch.allclose(torch.imag(weights), zero_like_depths, atol=0.0)
        )

        cache = {
            "src_xy": src_xy,
            "z0_ref": z0_ref,
            "z_ref_tensor": z_ref_tensor,
            "z_ref_val": z_ref_val,
            "eps1": eps1,
            "q": q,
            "pref": pref,
            "depths": depths,
            "weights": weights,
            "all_real": all_real,
        }
        self._cache[key] = cache
        return cache

    def potential(self, targets: torch.Tensor) -> torch.Tensor:
        if targets.device.type != "cuda":
            raise ValueError("DCIMBlockBasis requires CUDA targets (GPU-first).")
        if len(self.images) == 0:
            return torch.zeros(targets.shape[0], device=targets.device, dtype=targets.dtype)
        real_dtype = targets.dtype
        cache = self._get_cache(targets.device, real_dtype)

        if os.getenv("DCIM_BASIS_DEBUG_ASSERT", ""):
            assert cache["src_xy"].device.type == "cuda"
            assert cache["depths"].device.type == "cuda"
            assert cache["weights"].device.type == "cuda"

        src_xy = cache["src_xy"]  # type: ignore[assignment]
        z0_ref = cache["z0_ref"]  # type: ignore[assignment]
        z_ref_tensor = cache["z_ref_tensor"]  # type: ignore[assignment]
        z_ref_val = cache["z_ref_val"]  # type: ignore[assignment]
        depths = cache["depths"]  # type: ignore[assignment]
        weights = cache["weights"]  # type: ignore[assignment]
        all_real = bool(cache["all_real"])  # type: ignore[arg-type]

        # rho via direct elementwise math to avoid linalg overhead.
        dx = targets[:, 0] - src_xy[0]
        dy = targets[:, 1] - src_xy[1]
        rho2 = dx * dx + dy * dy
        rho2 = rho2.to(dtype=real_dtype)

        z_rel = targets[:, 2]
        if z_ref_val != 0.0:
            z_rel = z_rel - z_ref_tensor
        z_rel = z_rel.to(dtype=real_dtype)
        d_base = z_rel + z0_ref

        if all_real:
            depths_r = torch.real(depths)
            weights_r = torch.real(weights)
            d = d_base[:, None] + depths_r[None, :]
            denom = rho2[:, None] + d * d + 1e-24
            inv_r = torch.rsqrt(denom)
            inv_r3 = inv_r * inv_r * inv_r
            V = torch.sum(weights_r[None, :] * d * inv_r3, dim=1)
            return V.to(dtype=targets.dtype)

        d_complex = d_base[:, None].to(dtype=depths.dtype) + depths[None, :]
        r_sq = rho2[:, None].to(dtype=depths.dtype) + d_complex * d_complex
        r = torch.sqrt(r_sq + 1e-24)
        inv_r3 = 1.0 / (r * r * r)
        V_complex = torch.sum(weights[None, :] * d_complex * inv_r3, dim=1)
        out = V_complex.real if self.component == "real" else V_complex.imag
        return out.to(dtype=targets.dtype)

    def serialize(self) -> dict:
        return {
            "type": self.type,
            "params": {
                "component": self.component,
                "amplitude": [float(self.amplitude.real), float(self.amplitude.imag)],
                "block": self.block.to_json(),
                "images": [img.to_json() for img in self.images],
            },
        }

    @staticmethod
    def deserialize(data: dict) -> "DCIMBlockBasis":
        params = data.get("params", {})
        block_json = params.get("block", {})
        block = DCIMBlock.from_json(block_json)
        component = str(params.get("component", "real"))
        amp_raw = params.get("amplitude", [1.0, 0.0])
        amplitude = complex(float(amp_raw[0]), float(amp_raw[1]))
        imgs_json = params.get("images", None)
        imgs = None
        if imgs_json is not None:
            imgs = tuple(ComplexImageTerm.from_json(j) for j in imgs_json)
        return DCIMBlockBasis(block=block, images=imgs, component=component, amplitude=amplitude)


def dcim_basis_from_block(block: DCIMBlock) -> List[DCIMBlockBasis]:
    """Return real/imag split basis elements for each image term in stable order (real, imag)."""
    elems: List[DCIMBlockBasis] = []
    for img in block.images:
        elems.append(DCIMBlockBasis(block=block, images=[img], component="real", amplitude=1.0 + 0.0j))
        elems.append(DCIMBlockBasis(block=block, images=[img], component="imag", amplitude=1.0 + 0.0j))
    return elems
