"""
Mid-torus ribbon/patch mini-grid (Stage4-style collocation runs, no BEM).

Runs a small set of basis combinations for the mid torus only, focused on
inner_rim_ribbon and inner_patch_ring primitives alongside toroidal eigen
modes. Results are saved to runs/torus/stage4_metrics_mid_ribbon_patch.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

from electrodrive.orchestration.parser import CanonicalSpec
from tools.run_grandchallenge_experiments import run_single


def build_belts(spec: CanonicalSpec) -> List[Tuple[float, float]]:
    torus = next(c for c in spec.conductors if c.get("type") in ("torus", "toroid"))
    R = float(torus.get("major_radius", torus.get("radius", 1.0)))
    a = float(torus.get("minor_radius", 0.25 * R))
    belts = []
    for r in (R - a, R - 0.75 * a, R - 0.5 * a):
        for z in (0.0, 0.2 * a, -0.2 * a):
            belts.append((r, z))
    return belts


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runs_root = root / "runs" / "torus"
    spec = CanonicalSpec.from_json(json.load(open(root / "specs" / "torus_axis_point_mid.json")))
    belts_inner = build_belts(spec)

    configs = []
    base_regs = [8e-4, 1e-3]

    def per_reg(reg: float, include_patch: bool, include_ribbon: bool) -> Dict[str, float]:
        d: Dict[str, float] = {
            "point": 2e-3,
            "toroidal_eigen_mode_boundary": reg,
        }
        if include_ribbon:
            d["inner_rim_ribbon"] = reg
        if include_patch:
            d["inner_patch_ring"] = reg
        return d

    # A: eigen + ribbon
    for reg in base_regs:
        configs.append(
            (
                ["point", "toroidal_eigen_mode_boundary", "inner_rim_ribbon"],
                12,
                reg,
                per_reg(reg, include_patch=False, include_ribbon=True),
                0.8,
            )
        )
    # B: eigen + ribbon + patch
    for reg in base_regs:
        configs.append(
            (
                ["point", "toroidal_eigen_mode_boundary", "inner_rim_ribbon", "inner_patch_ring"],
                12,
                reg,
                per_reg(reg, include_patch=True, include_ribbon=True),
                0.8,
            )
        )
    # C: eigen + patch only
    for reg in base_regs:
        configs.append(
            (
                ["point", "toroidal_eigen_mode_boundary", "inner_patch_ring"],
                12,
                reg,
                per_reg(reg, include_patch=True, include_ribbon=False),
                0.75,
            )
        )

    results = []
    for basis, n_max, reg, per_type_reg, bw in configs:
        tag = f"torus_mid_{'_'.join(basis)}_n{n_max}_reg{reg}_bw{bw}_tsFalse_midpatch"
        print(f"[mid-ribbon] running {tag}")
        t0 = time.time()
        eval_res = run_single(
            spec,
            basis_types=basis,
            n_max=n_max,
            reg_l1=reg,
            restarts=1,
            per_type_reg=per_type_reg,
            boundary_weight=bw,
            two_stage=False,
            belts=belts_inner,
        )
        rec = {
            "spec": "torus_mid",
            "run": tag,
            "basis_types": basis,
            "n_max": n_max,
            "reg_l1": reg,
            "boundary_weight": bw,
            "restarts": 1,
            "two_stage": False,
            "per_type_reg": per_type_reg,
            "metrics": eval_res.metrics,
            "type_counts": eval_res.type_counts,
            "elapsed_s": time.time() - t0,
        }
        results.append(rec)

    out_path = runs_root / "stage4_metrics_mid_ribbon_patch.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, out_path.open("w"), indent=2)
    print(f"Saved {len(results)} records to {out_path}")


if __name__ == "__main__":
    main()
