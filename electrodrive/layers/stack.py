from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


@dataclass(frozen=True)
class Layer:
    name: str
    eps: complex
    z_min: float
    z_max: float


@dataclass(frozen=True)
class LayerInterface:
    z: float
    upper: int
    lower: int


@dataclass(frozen=True)
class LayerStack:
    layers: Tuple[Layer, ...]
    interfaces: Tuple[LayerInterface, ...]
    z_bounds: Tuple[float, ...]

    def layer_index(self, z: float, *, tol: float = 1e-9) -> int:
        """Return the layer index that contains coordinate z (top -> bottom ordering)."""
        for idx, layer in enumerate(self.layers):
            if (z <= layer.z_max + tol) and (z >= layer.z_min - tol):
                return idx
        raise ValueError(f"z={z} outside stack bounds [{self.z_bounds[0]}, {self.z_bounds[-1]}].")

    def thickness(self, i: int) -> float:
        """Return thickness of layer i (math.inf for semi-infinite layers)."""
        layer = self.layers[i]
        if math.isinf(layer.z_max) or math.isinf(layer.z_min):
            return math.inf
        return float(layer.z_max - layer.z_min)


def _coerce_bound(val, default: float) -> float:
    if val is None:
        return default
    if isinstance(val, str):
        lower = val.lower()
        if lower in ("inf", "+inf", "infinity"):
            return math.inf
        if lower in ("-inf", "-infinity"):
            return -math.inf
    try:
        return float(val)
    except Exception as exc:
        raise ValueError(f"Invalid layer bound '{val}'") from exc


def _coerce_eps(val) -> complex:
    if isinstance(val, complex):
        return val
    if isinstance(val, (float, int)):
        return complex(float(val))
    if isinstance(val, Sequence) and not isinstance(val, (str, bytes)) and len(val) == 2:
        try:
            return complex(float(val[0]), float(val[1]))
        except Exception:
            pass
    raise ValueError(f"Invalid epsilon value '{val}' (expected numeric or complex pair).")


def _validate_layers(layers: Sequence[Layer], tol: float = 1e-9) -> None:
    if not layers:
        raise ValueError("LayerStack must contain at least one layer.")
    for idx, layer in enumerate(layers):
        if not (layer.z_max > layer.z_min):
            raise ValueError(f"Layer '{layer.name}' has non-positive thickness (z_min={layer.z_min}, z_max={layer.z_max}).")
        if idx < len(layers) - 1:
            lower = layers[idx + 1]
            if upper_exceeds := (lower.z_max > layer.z_min + tol):
                raise ValueError(
                    f"Layer ordering/overlap error between '{layer.name}' and '{lower.name}': "
                    f"upper z_min={layer.z_min}, lower z_max={lower.z_max}."
                )
            gap = layer.z_min - lower.z_max
            if gap > tol:
                raise ValueError(
                    f"Gap detected between '{layer.name}' (z_min={layer.z_min}) and "
                    f"'{lower.name}' (z_max={lower.z_max})."
                )
            if layer.z_max < lower.z_max - tol:
                raise ValueError(
                    f"Layers not ordered top->bottom: '{layer.name}' z_max={layer.z_max} below "
                    f"'{lower.name}' z_max={lower.z_max}."
                )
            if layer.z_min < lower.z_min - tol:
                raise ValueError(
                    f"Layers not ordered top->bottom: '{layer.name}' z_min={layer.z_min} below "
                    f"'{lower.name}' z_min={lower.z_min}."
                )


def _unique_bounds(layers: Sequence[Layer]) -> Tuple[float, ...]:
    bounds = []
    for layer in layers:
        bounds.extend([layer.z_min, layer.z_max])
    # Preserve top->bottom ordering while dropping near-duplicates.
    bounds_sorted = sorted(bounds, reverse=True)
    uniq: list[float] = []
    for b in bounds_sorted:
        if not uniq or abs(uniq[-1] - b) > 1e-12:
            uniq.append(b)
    return tuple(uniq)


def layerstack_from_spec(spec) -> LayerStack:
    """
    Parse a CanonicalSpec-like object into a LayerStack.

    Expects `spec.dielectrics` to define planar layers with keys:
      - name (str)
      - epsilon / eps (numeric or complex)
      - z_min, z_max (float or +/-inf sentinel/None)
    """
    dielectrics = getattr(spec, "dielectrics", None)
    if not dielectrics:
        raise ValueError("Spec missing 'dielectrics' entries required for layered stack.")

    parsed: list[Layer] = []
    for idx, d in enumerate(dielectrics):
        if not isinstance(d, dict):
            raise ValueError(f"Dielectric entry at index {idx} is not a mapping: {d}")
        name = str(d.get("name", f"layer_{idx}"))
        eps_val = d.get("epsilon", d.get("eps", d.get("permittivity")))
        if eps_val is None:
            raise ValueError(f"Layer '{name}' missing epsilon/eps field.")
        eps = _coerce_eps(eps_val)
        z_min = _coerce_bound(d.get("z_min", None), -math.inf)
        z_max = _coerce_bound(d.get("z_max", None), math.inf)
        parsed.append(Layer(name=name, eps=eps, z_min=z_min, z_max=z_max))

    # Deterministic ordering: top (largest z_max) to bottom (smallest z_min).
    parsed_sorted = tuple(sorted(parsed, key=lambda l: (-l.z_max, -l.z_min)))
    _validate_layers(parsed_sorted)

    interfaces = []
    for i in range(len(parsed_sorted) - 1):
        interfaces.append(
            LayerInterface(z=parsed_sorted[i].z_min, upper=i, lower=i + 1)
        )
    z_bounds = _unique_bounds(parsed_sorted)
    return LayerStack(layers=parsed_sorted, interfaces=tuple(interfaces), z_bounds=z_bounds)
