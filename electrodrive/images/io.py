from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from electrodrive.images.basis import ImageBasisElement
from electrodrive.images.search import ImageSystem


def save_image_system(
    system: ImageSystem,
    path: Path,
    metadata: Dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    weights_cpu = system.weights.detach().cpu().tolist()
    images: List[Dict[str, Any]] = []
    for elem, w in zip(system.elements, weights_cpu):
        entry = elem.serialize()
        entry["weight"] = w
        images.append(entry)

    data = {
        "metadata": metadata or {},
        "system_metadata": getattr(system, "metadata", {}),
        "images": images,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_image_system(
    path: Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ImageSystem:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    elements: List[ImageBasisElement] = []
    weights: List[float] = []

    for entry in images:
        w = float(entry.pop("weight", 1.0))
        elem = ImageBasisElement.deserialize(entry, device=device, dtype=dtype)
        elements.append(elem)
        weights.append(w)

    w_tensor = torch.tensor(weights, device=device, dtype=dtype)
    sys_meta = data.get("system_metadata", {}) if isinstance(data, dict) else {}
    return ImageSystem(elements, w_tensor, metadata=sys_meta)
