from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

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
        # Store lightweight params for (de)serialization.
        comp_flag = 0 if component == "real" else 1
        super().__init__(
            f"dcim_block_{component}",
            {
                "component": torch.tensor(comp_flag),
                "amplitude": torch.tensor([float(amplitude.real), float(amplitude.imag)]),
            },
        )

    def potential(self, targets: torch.Tensor) -> torch.Tensor:
        if targets.device.type != "cuda":
            raise ValueError("DCIMBlockBasis requires CUDA targets (GPU-first).")
        if len(self.images) == 0:
            return torch.zeros(targets.shape[0], device=targets.device, dtype=targets.dtype)
        dtype = torch.complex128
        real_dtype = torch.empty((), dtype=dtype).real.dtype
        rho = torch.linalg.norm(targets[:, :2], dim=1).to(dtype=real_dtype)
        z = targets[:, 2].to(dtype=real_dtype)

        eps1 = float(self.block.stack.layers[0].eps.real)
        meta = self.block.certificate.meta or {}
        z0 = float(meta.get("source_z", 0.0))
        q = float(meta.get("source_charge", 1.0))

        depths = torch.as_tensor([img.depth for img in self.images], device=targets.device, dtype=dtype)
        weights = torch.as_tensor([img.weight for img in self.images], device=targets.device, dtype=dtype)
        weights = weights * torch.as_tensor(self.amplitude, device=targets.device, dtype=dtype)

        d = z[:, None].to(dtype=dtype) + z0 + depths[None, :]
        r_sq = rho[:, None].to(dtype=dtype) ** 2 + d * d
        r = torch.sqrt(r_sq + 1e-24)
        pref = (q / (2.0 * math.pi * 2.0 * eps1))
        V_complex = pref * torch.sum(weights[None, :] * d / (r * r * r), dim=1)
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
    """Return real/imag split basis elements for each image term."""
    elems: List[DCIMBlockBasis] = []
    for img in block.images:
        elems.append(DCIMBlockBasis(block=block, images=[img], component="real", amplitude=1.0 + 0.0j))
        elems.append(DCIMBlockBasis(block=block, images=[img], component="imag", amplitude=1.0 + 0.0j))
    return elems
