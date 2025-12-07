"""
Canonical spec registry for Stage-0/Stage-1 tasks.

This module centralises the canonical specs so that training, regression,
and smoke tools do not drift when new variants are introduced. The original
files remain untouched; variants live alongside them as additional JSONs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from electrodrive.orchestration.parser import CanonicalSpec, parse_spec


_SPEC_ROOT = Path(__file__).resolve().parents[2] / "specs"

# Canonical specs (do not edit these paths without coordination).
_STAGE0_SPHERE_AXIS_EXTERNAL = _SPEC_ROOT / "sphere_axis_point_external.json"
_STAGE1_SPHERE_DIMER_INSIDE = _SPEC_ROOT / "stage1_sphere_dimer_axis_point_inside.json"


@dataclass(frozen=True)
class SpecInfo:
    """Lightweight descriptor for a spec file and its role."""

    name: str
    path: Path
    role: str
    note: str = ""


def stage0_sphere_external_path() -> Path:
    """Canonical Stage-0 grounded sphere spec (axis, external point charge)."""
    return _STAGE0_SPHERE_AXIS_EXTERNAL


def stage1_sphere_dimer_inside_path() -> Path:
    """Canonical Stage-1 sphere dimer spec (axis, midpoint point charge)."""
    return _STAGE1_SPHERE_DIMER_INSIDE


def load_stage0_sphere_external() -> CanonicalSpec:
    return parse_spec(_STAGE0_SPHERE_AXIS_EXTERNAL)


def load_stage1_sphere_dimer_inside() -> CanonicalSpec:
    return parse_spec(_STAGE1_SPHERE_DIMER_INSIDE)


def list_stage0_variants(include_canonical: bool = False) -> List[SpecInfo]:
    """Available Stage-0 specs (external baseline + optional variants)."""
    variants: List[SpecInfo] = [
        SpecInfo(
            name="stage0_sphere_axis_point_external_r1.5",
            path=_SPEC_ROOT / "stage0_sphere_axis_point_external_r1.5.json",
            role="variant",
            note="Larger sphere to widen charge/surface scale for meta-learning.",
        ),
    ]
    if include_canonical:
        variants.insert(
            0,
            SpecInfo(
                name="sphere_axis_point_external",
                path=_STAGE0_SPHERE_AXIS_EXTERNAL,
                role="canonical",
                note="Primary Stage-0 regression/training spec.",
            ),
        )
    return variants


def list_stage1_variants(include_canonical: bool = False) -> List[SpecInfo]:
    """Available Stage-1 specs (inside-lens baseline + optional variants)."""
    variants: List[SpecInfo] = [
        SpecInfo(
            name="stage1_sphere_dimer_axis_point_inside_D2.5",
            path=_SPEC_ROOT / "stage1_sphere_dimer_axis_point_inside_D2.5.json",
            role="variant",
            note="Slightly wider separation to probe gap-size sensitivity (baseline d=2.4).",
        ),
    ]
    if include_canonical:
        variants.insert(
            0,
            SpecInfo(
                name="stage1_sphere_dimer_axis_point_inside",
                path=_STAGE1_SPHERE_DIMER_INSIDE,
                role="canonical",
                note="Primary Stage-1 regression/experiment spec.",
            ),
        )
    return variants


def all_spec_paths(include_variants: bool = True) -> Iterable[Path]:
    """Helper for quick validation sweeps."""
    yield _STAGE0_SPHERE_AXIS_EXTERNAL
    yield _STAGE1_SPHERE_DIMER_INSIDE
    if include_variants:
        for info in list_stage0_variants():
            yield info.path
        for info in list_stage1_variants():
            yield info.path
