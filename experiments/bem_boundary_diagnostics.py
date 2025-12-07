import argparse
import json
import math
import pathlib
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from electrodrive.core.bem import _bc_vector, bem_solve
from electrodrive.fmm3d.logging_utils import ConsoleLogger
from electrodrive.utils.config import BEMConfig
from tests.test_bem_quadrature import (
    _build_parallel_planes_spec,
    _build_plane_spec,
    _build_sphere_spec,
)


GeomEntry = Tuple[str, str, Any]


def _percentile(t: torch.Tensor, q: float) -> float:
    if t.numel() == 0:
        return float("nan")
    return float(torch.quantile(t, q / 100.0).item())


def _histogram(data: np.ndarray, bins: int = 30) -> Dict[str, Any]:
    if data.size == 0:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(data, bins=bins)
    return {"bins": edges.tolist(), "counts": counts.astype(int).tolist()}


def _configure_bem() -> BEMConfig:
    cfg = BEMConfig()
    cfg.use_gpu = False
    cfg.fp64 = True
    cfg.max_refine_passes = max(int(getattr(cfg, "max_refine_passes", 3) or 3), 3)
    cfg.min_refine_passes = 1
    cfg.near_alpha = 0.0
    cfg.use_near_quadrature = True
    cfg.use_near_quadrature_matvec = False
    return cfg


def _select_best_pass(history: List[Dict[str, Any]], target_bc: float) -> Dict[str, Any]:
    if not history:
        return {}
    for p in history:
        try:
            if abs(float(p.get("bc_resid_linf", math.inf)) - target_bc) < 1e-12:
                return p
        except Exception:
            continue
    return min(history, key=lambda p: p.get("bc_resid_linf", float("inf")))


def _offset_scale(spec, mesh_stats: Dict[str, Any]) -> float:
    if "patch_L" in mesh_stats and mesh_stats["patch_L"] is not None:
        return 1e-3 * float(mesh_stats["patch_L"])
    for c in spec.conductors:
        if c.get("type") == "sphere":
            return 1e-3 * float(c.get("radius", 1.0))
    area = float(mesh_stats.get("total_area", 1.0) or 1.0)
    return 1e-3 * math.sqrt(max(area, 1e-12))


def _gather_panel_samples(
    C: torch.Tensor,
    bc: torch.Tensor,
    V_tot: torch.Tensor,
    sigma: torch.Tensor,
    abs_resid: torch.Tensor,
    limit: int,
) -> List[Dict[str, Any]]:
    k = min(limit, abs_resid.numel())
    if k <= 0:
        return []
    idxs = torch.topk(abs_resid, k=k).indices
    samples: List[Dict[str, Any]] = []
    for idx in idxs.tolist():
        samples.append(
            {
                "centroid": [float(v) for v in C[idx].tolist()],
                "bc": float(bc[idx].item()),
                "V_tot": float(V_tot[idx].item()),
                "sigma": float(sigma[idx].item()),
                "abs_resid": float(abs_resid[idx].item()),
            }
        )
    return samples


def _run_geometry(
    name: str,
    builder,
    geom_type: str,
    sample_limit: int,
    output_dir: pathlib.Path,
) -> None:
    spec = builder()
    cfg = _configure_bem()
    result = bem_solve(spec, cfg, ConsoleLogger())

    if "solution" not in result:
        payload = {
            "geometry": name,
            "error": result.get("error", "BEM solve failed"),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"bem_boundary_{name}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"{name}: BEM solve failed -> {out_path}")
        return

    solution = result["solution"]
    mesh_stats = dict(result.get("mesh_stats", {}))
    gmres_stats = dict(result.get("gmres_stats", {}))
    history = result.get("refinement_history", [])
    target_bc = float(mesh_stats.get("bc_residual_linf", math.inf))
    best = _select_best_pass(history, target_bc)

    if not best:
        raise RuntimeError("BEM result missing refinement history/artifacts.")
    if "artifacts" not in best:
        raise RuntimeError("Selected refinement pass missing artifacts payload.")

    spec_b, C_b, A_b, Nrm_b, sigma_b, Vtot_b, mesh_b = best["artifacts"]
    bc_vec = _bc_vector(spec_b, mesh_b, C_b.device, C_b.dtype)
    abs_resid = torch.abs(Vtot_b - bc_vec)

    centroid_stats = {
        "max": float(abs_resid.max().item()) if abs_resid.numel() else float("nan"),
        "mean": float(abs_resid.mean().item()) if abs_resid.numel() else float("nan"),
        "p95": _percentile(abs_resid, 95),
        "histogram": _histogram(abs_resid.detach().cpu().numpy(), bins=30),
    }

    # Off-surface samples along panel normals
    eps = _offset_scale(spec_b, mesh_stats)
    n_panels = C_b.shape[0]
    max_offsets = min(sample_limit, n_panels)
    idxs = torch.linspace(0, n_panels - 1, steps=max_offsets).long()
    C_sel = C_b[idxs]
    N_sel = Nrm_b[idxs]
    bc_sel = bc_vec[idxs]

    offsets = torch.cat(
        [C_sel + eps * N_sel, C_sel - eps * N_sel],
        dim=0,
    )
    bc_offsets = torch.cat([bc_sel, bc_sel], dim=0)

    with torch.no_grad():
        V_off, _ = solution.eval_V_E_batched(offsets.to(solution._device))
    off_resid = torch.abs(V_off - bc_offsets.to(V_off.device))

    offset_stats = {
        "max": float(off_resid.max().item()) if off_resid.numel() else float("nan"),
        "mean": float(off_resid.mean().item()) if off_resid.numel() else float("nan"),
        "p95": _percentile(off_resid, 95),
    }

    samples = _gather_panel_samples(
        C_b, bc_vec, Vtot_b, sigma_b, abs_resid, sample_limit
    )

    payload = {
        "geometry": name,
        "mesh_stats": mesh_stats,
        "gmres_stats": gmres_stats,
        "centroid_bc_residuals": centroid_stats,
        "surface_offset_residuals": offset_stats,
        "per_panel_samples": samples,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"bem_boundary_{name}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"{name}: bc_resid_linf={mesh_stats.get('bc_residual_linf')} -> {out_path}"
    )


def _parse_geometries(sel: Iterable[str]) -> List[GeomEntry]:
    mapping = {
        "plane": ("plane", _build_plane_spec, "plane"),
        "sphere": ("sphere", _build_sphere_spec, "sphere"),
        "parallel_planes": ("parallel_planes", _build_parallel_planes_spec, "parallel_planes"),
    }
    if "all" in sel:
        return list(mapping.values())
    chosen: List[GeomEntry] = []
    for key in sel:
        if key not in mapping:
            raise ValueError(f"Unknown geometry: {key}")
        chosen.append(mapping[key])
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Boundary condition diagnostics for BEM solutions."
    )
    parser.add_argument(
        "--geometry",
        choices=["all", "plane", "sphere", "parallel_planes"],
        nargs="+",
        default=["all"],
        help="Geometries to run (default: all).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=512,
        help="Maximum number of panels to include in per-panel samples and offset checks.",
    )
    args = parser.parse_args()

    geometries = _parse_geometries(args.geometry)
    out_dir = pathlib.Path(__file__).resolve().parent / "_agent_outputs"

    for name, builder, geom_type in geometries:
        _run_geometry(
            name=name,
            builder=builder,
            geom_type=geom_type,
            sample_limit=args.sample_limit,
            output_dir=out_dir,
        )


if __name__ == "__main__":
    main()
