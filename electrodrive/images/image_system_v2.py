from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from electrodrive.images.basis import ImageBasisElement, PointChargeBasis
from electrodrive.images.basis_dcim import DCIMBlockBasis
from electrodrive.utils.config import K_E


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ImageSystemV2 (GPU-first repo).")


def _normalize_flag(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


class ImageSystemV2:
    """Batched evaluator for image systems (GPU-first)."""

    def __init__(
        self,
        elements: Sequence[ImageBasisElement],
        weights: torch.Tensor,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        _require_cuda()
        self.elements = list(elements)
        self.metadata: Dict[str, object] = metadata or {}

        if weights.numel() > 0:
            self.device = weights.device
            self.dtype = weights.dtype
        else:
            self.device = torch.device("cuda")
            self.dtype = torch.float32

        if self.device.type != "cuda":
            self.device = torch.device("cuda")
        self.weights = weights.to(device=self.device, dtype=self.dtype)

        self._point_real_pos: Optional[torch.Tensor] = None
        self._point_real_idx: Optional[torch.Tensor] = None
        self._point_complex_real_pos: Optional[torch.Tensor] = None
        self._point_complex_real_idx: Optional[torch.Tensor] = None
        self._point_complex_real_z_imag: Optional[torch.Tensor] = None
        self._point_complex_imag_pos: Optional[torch.Tensor] = None
        self._point_complex_imag_idx: Optional[torch.Tensor] = None
        self._point_complex_imag_z_imag: Optional[torch.Tensor] = None
        self._dcim_elems: List[DCIMBlockBasis] = []
        self._dcim_idx: List[int] = []
        self._fallback_elems: List[ImageBasisElement] = []
        self._fallback_idx: List[int] = []

        self._pack_elements()

    def _pack_elements(self) -> None:
        point_real_pos: List[torch.Tensor] = []
        point_real_idx: List[int] = []
        point_complex_real_pos: List[torch.Tensor] = []
        point_complex_real_idx: List[int] = []
        point_complex_real_z: List[torch.Tensor] = []
        point_complex_imag_pos: List[torch.Tensor] = []
        point_complex_imag_idx: List[int] = []
        point_complex_imag_z: List[torch.Tensor] = []

        for idx, elem in enumerate(self.elements):
            if isinstance(elem, DCIMBlockBasis):
                self._dcim_elems.append(elem)
                self._dcim_idx.append(idx)
                continue

            if isinstance(elem, PointChargeBasis):
                pos = elem.params["position"].to(device=self.device, dtype=self.dtype).view(3)
                use_complex = bool(getattr(elem, "_use_complex", False))
                if use_complex:
                    z_imag = elem.params.get("z_imag")
                    z_imag_t = torch.as_tensor(z_imag if z_imag is not None else 0.0, device=self.device, dtype=self.dtype).view(())
                    component = elem.params.get("component")
                    is_imag = False
                    if component is not None:
                        try:
                            if torch.is_tensor(component):
                                is_imag = bool(float(component.item()) >= 0.5)
                            elif isinstance(component, (int, float)):
                                is_imag = bool(float(component) >= 0.5)
                            elif isinstance(component, str):
                                is_imag = component.strip().lower().startswith("imag")
                        except Exception:
                            is_imag = False
                    if is_imag:
                        point_complex_imag_pos.append(pos)
                        point_complex_imag_z.append(z_imag_t)
                        point_complex_imag_idx.append(idx)
                    else:
                        point_complex_real_pos.append(pos)
                        point_complex_real_z.append(z_imag_t)
                        point_complex_real_idx.append(idx)
                else:
                    point_real_pos.append(pos)
                    point_real_idx.append(idx)
                continue

            self._fallback_elems.append(elem)
            self._fallback_idx.append(idx)

        if point_real_pos:
            self._point_real_pos = torch.stack(point_real_pos, dim=0).contiguous()
            self._point_real_idx = torch.tensor(point_real_idx, device=self.device, dtype=torch.long)
        if point_complex_real_pos:
            self._point_complex_real_pos = torch.stack(point_complex_real_pos, dim=0).contiguous()
            self._point_complex_real_idx = torch.tensor(point_complex_real_idx, device=self.device, dtype=torch.long)
            self._point_complex_real_z_imag = torch.stack(point_complex_real_z, dim=0).contiguous()
        if point_complex_imag_pos:
            self._point_complex_imag_pos = torch.stack(point_complex_imag_pos, dim=0).contiguous()
            self._point_complex_imag_idx = torch.tensor(point_complex_imag_idx, device=self.device, dtype=torch.long)
            self._point_complex_imag_z_imag = torch.stack(point_complex_imag_z, dim=0).contiguous()

    def _gather_weights(self, idx: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if idx is None:
            return None
        return self.weights.index_select(0, idx)

    def evaluate(self, targets: torch.Tensor) -> torch.Tensor:
        return self.potential(targets)

    def potential(self, targets: torch.Tensor) -> torch.Tensor:
        orig_device = targets.device
        orig_dtype = targets.dtype
        compute_dtype = torch.promote_types(self.dtype, targets.dtype)

        X = targets.to(device=self.device, dtype=compute_dtype)
        V = torch.zeros(X.shape[0], device=self.device, dtype=compute_dtype)

        if self._point_real_pos is not None:
            pos = self._point_real_pos.to(dtype=compute_dtype)
            w = self._gather_weights(self._point_real_idx)
            if w is not None:
                w = w.to(dtype=compute_dtype)
            diff = X[:, None, :] - pos[None, :, :]
            r = torch.linalg.norm(diff, dim=-1).clamp_min(1e-12)
            phi = (K_E / r)
            if w is not None:
                V = V + phi @ w

        if self._point_complex_real_pos is not None and self._point_complex_real_z_imag is not None:
            pos = self._point_complex_real_pos.to(dtype=compute_dtype)
            w = self._gather_weights(self._point_complex_real_idx)
            if w is not None:
                w = w.to(dtype=compute_dtype)
                complex_real_dtype = torch.float64 if compute_dtype == torch.float64 else torch.float32
                Xc = X.to(dtype=complex_real_dtype)
                pos_c = pos.to(dtype=complex_real_dtype)
                z_imag = self._point_complex_real_z_imag.to(dtype=complex_real_dtype)

                dx = Xc[:, None, 0] - pos_c[None, :, 0]
                dy = Xc[:, None, 1] - pos_c[None, :, 1]
                dz = Xc[:, None, 2] - pos_c[None, :, 2]
                dz_complex = torch.complex(dz, -z_imag.view(1, -1).expand_as(dz))
                r2_complex = dx * dx + dy * dy + dz_complex * dz_complex
                inv_r = 1.0 / torch.sqrt(r2_complex)
                phi = 2.0 * inv_r.real
                phi = (K_E * phi).to(dtype=compute_dtype)
                V = V + phi @ w

        if self._point_complex_imag_pos is not None and self._point_complex_imag_z_imag is not None:
            pos = self._point_complex_imag_pos.to(dtype=compute_dtype)
            w = self._gather_weights(self._point_complex_imag_idx)
            if w is not None:
                w = w.to(dtype=compute_dtype)
                complex_real_dtype = torch.float64 if compute_dtype == torch.float64 else torch.float32
                Xc = X.to(dtype=complex_real_dtype)
                pos_c = pos.to(dtype=complex_real_dtype)
                z_imag = self._point_complex_imag_z_imag.to(dtype=complex_real_dtype)

                dx = Xc[:, None, 0] - pos_c[None, :, 0]
                dy = Xc[:, None, 1] - pos_c[None, :, 1]
                dz = Xc[:, None, 2] - pos_c[None, :, 2]
                dz_complex = torch.complex(dz, -z_imag.view(1, -1).expand_as(dz))
                r2_complex = dx * dx + dy * dy + dz_complex * dz_complex
                inv_r = 1.0 / torch.sqrt(r2_complex)
                phi = -2.0 * inv_r.imag
                phi = (K_E * phi).to(dtype=compute_dtype)
                V = V + phi @ w

        if self._dcim_elems:
            for elem, idx in zip(self._dcim_elems, self._dcim_idx):
                w = self.weights[idx].to(dtype=compute_dtype)
                V = V + w * elem.potential(X)

        if self._fallback_elems:
            for elem, idx in zip(self._fallback_elems, self._fallback_idx):
                w = self.weights[idx].to(dtype=compute_dtype)
                V = V + w * elem.potential(X)

        return V.to(device=orig_device, dtype=orig_dtype)


def maybe_wrap_imagesystem_v2(system: object) -> object:
    if not hasattr(system, "elements") or not hasattr(system, "weights"):
        return system
    flag = _normalize_flag(os.getenv("EDE_IMAGE_SYSTEM_V2", ""))
    if not flag:
        return system
    if isinstance(system, ImageSystemV2):
        return system
    return ImageSystemV2(system.elements, system.weights, metadata=getattr(system, "metadata", None))
