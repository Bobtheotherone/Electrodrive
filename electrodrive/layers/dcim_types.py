from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from electrodrive.layers.poles import PoleTerm
from electrodrive.layers.spectral_kernels import SpectralKernelSpec
from electrodrive.layers.stack import LayerStack


def _complex_to_tuple(z: complex) -> Tuple[float, float]:
    return (float(z.real), float(z.imag))


def _tuple_to_complex(p: Tuple[float, float]) -> complex:
    return complex(float(p[0]), float(p[1]))


def _jsonify_meta(meta: Dict[str, object]) -> Dict[str, object]:
    packed: Dict[str, object] = {}
    for k, v in meta.items():
        if isinstance(v, complex):
            packed[k] = _complex_to_tuple(v)
        elif isinstance(v, dict):
            packed[k] = _jsonify_meta(v)
        else:
            packed[k] = v
    return packed


def _unjsonify_meta(meta: Dict[str, object]) -> Dict[str, object]:
    unpacked: Dict[str, object] = {}
    for k, v in meta.items():
        if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            unpacked[k] = _tuple_to_complex((float(v[0]), float(v[1])))
        elif isinstance(v, dict):
            unpacked[k] = _unjsonify_meta(v)
        else:
            unpacked[k] = v
    return unpacked


@dataclass(frozen=True)
class ComplexImageTerm:
    depth: complex
    weight: complex
    family: str = "exp_fit"
    meta: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> Dict[str, object]:
        return {
            "depth": _complex_to_tuple(self.depth),
            "weight": _complex_to_tuple(self.weight),
            "family": self.family,
            "meta": self.meta,
        }

    @staticmethod
    def from_json(d: Dict[str, object]) -> "ComplexImageTerm":
        return ComplexImageTerm(
            depth=_tuple_to_complex(tuple(d["depth"])),  # type: ignore[arg-type]
            weight=_tuple_to_complex(tuple(d["weight"])),  # type: ignore[arg-type]
            family=str(d.get("family", "exp_fit")),
            meta=dict(d.get("meta", {})),
        )


@dataclass(frozen=True)
class DCIMCertificate:
    k_grid: Tuple[float, ...]
    fit_residual_L2: float
    fit_residual_Linf: float
    spatial_check_rel_L2: float
    spatial_check_rel_Linf: float
    stable: bool
    meta: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> Dict[str, object]:
        return {
            "k_grid": [float(x) for x in self.k_grid],
            "fit_residual_L2": float(self.fit_residual_L2),
            "fit_residual_Linf": float(self.fit_residual_Linf),
            "spatial_check_rel_L2": float(self.spatial_check_rel_L2),
            "spatial_check_rel_Linf": float(self.spatial_check_rel_Linf),
            "stable": bool(self.stable),
            "meta": _jsonify_meta(self.meta),
        }

    @staticmethod
    def from_json(d: Dict[str, object]) -> "DCIMCertificate":
        return DCIMCertificate(
            k_grid=tuple(float(x) for x in d.get("k_grid", [])),
            fit_residual_L2=float(d.get("fit_residual_L2", 0.0)),
            fit_residual_Linf=float(d.get("fit_residual_Linf", 0.0)),
            spatial_check_rel_L2=float(d.get("spatial_check_rel_L2", 0.0)),
            spatial_check_rel_Linf=float(d.get("spatial_check_rel_Linf", 0.0)),
            stable=bool(d.get("stable", False)),
            meta=_unjsonify_meta(dict(d.get("meta", {}))),
        )


@dataclass(frozen=True)
class DCIMBlock:
    stack: LayerStack
    kernel: SpectralKernelSpec
    poles: Tuple[PoleTerm, ...]
    images: Tuple[ComplexImageTerm, ...]
    certificate: DCIMCertificate

    def to_json(self) -> Dict[str, object]:
        return {
            "stack": {
                "layers": [
                    {
                        "name": l.name,
                        "eps": _complex_to_tuple(l.eps),
                        "z_min": l.z_min,
                        "z_max": l.z_max,
                    }
                    for l in self.stack.layers
                ]
            },
            "kernel": {
                "source_region": self.kernel.source_region,
                "obs_region": self.kernel.obs_region,
                "component": self.kernel.component,
                "bc_kind": self.kernel.bc_kind,
            },
            "poles": [
                {
                    "pole": _complex_to_tuple(p.pole),
                    "residue": _complex_to_tuple(p.residue),
                    "kind": p.kind,
                    "meta": p.meta,
                }
                for p in self.poles
            ],
            "images": [img.to_json() for img in self.images],
            "certificate": self.certificate.to_json(),
        }

    @staticmethod
    def from_json(d: Dict[str, object]) -> "DCIMBlock":
        from electrodrive.layers.stack import Layer, LayerStack, LayerInterface, _validate_layers  # local import to avoid cycles
        layers_raw = d.get("stack", {}).get("layers", [])
        layers = []
        for l in layers_raw:
            layers.append(
                Layer(
                    name=str(l.get("name", "")),
                    eps=_tuple_to_complex(tuple(l.get("eps", (0.0, 0.0)))),
                    z_min=float(l.get("z_min", 0.0)),
                    z_max=float(l.get("z_max", 0.0)),
                )
            )
        layers_tuple = tuple(layers)
        _validate_layers(layers_tuple)
        interfaces = tuple(
            LayerInterface(z=layers_tuple[i].z_min, upper=i, lower=i + 1)
            for i in range(len(layers_tuple) - 1)
        )
        # Rebuild z_bounds as in stack._unique_bounds: descending, deduped.
        z_vals = []
        for l in layers_tuple:
            z_vals.extend([l.z_min, l.z_max])
        z_bounds = []
        for z in sorted(z_vals, reverse=True):
            if not z_bounds or abs(z_bounds[-1] - z) > 1e-12:
                z_bounds.append(z)
        stack = LayerStack(layers=layers_tuple, interfaces=interfaces, z_bounds=tuple(z_bounds))
        kernel_d = d.get("kernel", {})
        kernel = SpectralKernelSpec(
            source_region=int(kernel_d.get("source_region", 0)),
            obs_region=int(kernel_d.get("obs_region", 0)),
            component=str(kernel_d.get("component", "potential")),
            bc_kind=str(kernel_d.get("bc_kind", "dielectric_interfaces")),
        )
        poles = []
        for p in d.get("poles", []):
            poles.append(
                PoleTerm(
                    pole=_tuple_to_complex(tuple(p.get("pole", (0.0, 0.0)))),
                    residue=_tuple_to_complex(tuple(p.get("residue", (0.0, 0.0)))),
                    kind=str(p.get("kind", "guided")),
                    meta=dict(p.get("meta", {})),
                )
            )
        images = tuple(ComplexImageTerm.from_json(x) for x in d.get("images", []))
        certificate = DCIMCertificate.from_json(d.get("certificate", {}))
        return DCIMBlock(
            stack=stack,
            kernel=kernel,
            poles=tuple(poles),
            images=images,
            certificate=certificate,
        )
