"""
Random torus exploration (Stage4 collocation) with simple novelty detection.

Generates random basis/hyperparameter configurations for torus specs (thin, mid,
optionally off-axis if present), runs discovery, logs metrics, and prints
“interesting” candidates vs baselines. No BEM is run here; diagnostics can be
launched separately for the shortlisted runs.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from electrodrive.orchestration.parser import CanonicalSpec
from tools.run_grandchallenge_experiments import run_single


# ------------------------ Helpers ------------------------ #


def load_spec(path: Path) -> Optional[CanonicalSpec]:
    if not path.exists():
        return None
    try:
        return CanonicalSpec.from_json(json.loads(path.read_text()))
    except Exception:
        return None


def belts_inner(spec: CanonicalSpec) -> List[Tuple[float, float]]:
    torus = next(c for c in spec.conductors if c.get("type") in ("torus", "toroid"))
    R = float(torus.get("major_radius", torus.get("radius", 1.0)))
    a = float(torus.get("minor_radius", 0.25 * R))
    belts = []
    for r in (R - a, R - 0.75 * a, R - 0.5 * a, R, R + 0.5 * a):
        for z in (0.0, 0.2 * a, -0.2 * a):
            belts.append((r, z))
    return belts


def belts_default(spec: CanonicalSpec) -> List[Tuple[float, float]]:
    torus = next(c for c in spec.conductors if c.get("type") in ("torus", "toroid"))
    R = float(torus.get("major_radius", torus.get("radius", 1.0)))
    a = float(torus.get("minor_radius", 0.25 * R))
    return [
        (R - 0.5 * a, -0.2 * a),
        (R - 0.5 * a, 0.0),
        (R - 0.5 * a, 0.2 * a),
        (R, 0.0),
        (R + 0.5 * a, -0.2 * a),
        (R + 0.5 * a, 0.0),
        (R + 0.5 * a, 0.2 * a),
        (R + a, 0.0),
    ]


def choose_per_type_reg(reg: float, basis: Sequence[str]) -> Dict[str, float]:
    rng = random.random
    def rand_scale(lo: float, hi: float) -> float:
        return reg * random.uniform(lo, hi)

    per: Dict[str, float] = {}
    per["point"] = reg * random.uniform(2.0, 5.0)
    for b in basis:
        if b == "point":
            continue
        if b in ("toroidal_eigen_mode_boundary", "toroidal_eigen_mode_offaxis", "poloidal_ring", "ring_ladder_inner", "toroidal_mode_cluster"):
            per[b] = rand_scale(0.8, 1.2)
        elif b in ("inner_rim_arc", "inner_rim_ribbon", "inner_patch_ring"):
            per[b] = rand_scale(0.5, 1.5)
    return per


@dataclass
class TrialResult:
    spec_key: str
    run: str
    basis_types: List[str]
    n_max: int
    reg_l1: float
    boundary_weight: Optional[float]
    two_stage: bool
    restarts: int
    per_type_reg: Dict[str, float]
    metrics: Dict[str, float]
    type_counts: Dict[str, int]
    elapsed_s: float
    seed: int


# ------------------------ Random sweep ------------------------ #


def sample_basis(pool_global: List[str], pool_local: List[str]) -> List[str]:
    basis = ["point"]
    # Ensure one global element
    basis.append(random.choice(pool_global))
    # Optionally add another global
    if random.random() < 0.35:
        basis.append(random.choice(pool_global))
    # Local primitives with independent probabilities
    if random.random() < 0.4:
        basis.append("inner_rim_arc")
    if random.random() < 0.5:
        basis.append("inner_rim_ribbon")
    if random.random() < 0.35:
        basis.append("inner_patch_ring")
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for b in basis:
        if b not in seen:
            uniq.append(b)
            seen.add(b)
    return uniq


def run_random_trials(specs: Dict[str, CanonicalSpec], trials_per_spec: int, out_path: Path) -> List[TrialResult]:
    pool_global = [
        "poloidal_ring",
        "ring_ladder_inner",
        "toroidal_mode_cluster",
        "toroidal_eigen_mode_boundary",
        "toroidal_eigen_mode_offaxis",
    ]
    pool_local = ["inner_rim_arc", "inner_rim_ribbon", "inner_patch_ring"]
    n_max_choices = [4, 6, 8, 12, 16, 20, 24]
    reg_choices = [3e-4, 8e-4, 1e-3, 3e-3, 1e-2]
    bw_choices = [0.5, 0.7, 0.8, 0.9]

    results: List[TrialResult] = []
    for spec_key, spec in specs.items():
        belts = belts_inner(spec)
        for i in range(trials_per_spec):
            seed = random.randint(0, 10_000_000)
            random.seed(seed)
            basis = sample_basis(pool_global, pool_local)
            n_max = random.choice(n_max_choices)
            reg = random.choice(reg_choices)
            bw = random.choice(bw_choices)
            two_stage = random.random() < 0.3
            restarts = 1 if random.random() < 0.5 else 0
            per_type_reg = choose_per_type_reg(reg, basis)
            run_slug = f"{spec_key}_{'_'.join(basis)}_n{n_max}_reg{reg}_bw{bw}_ts{two_stage}_rs{restarts}_seed{seed}"

            t0 = time.time()
            eval_res = run_single(
                spec=spec,
                basis_types=basis,
                n_max=n_max,
                reg_l1=reg,
                restarts=restarts,
                per_type_reg=per_type_reg,
                boundary_weight=bw,
                two_stage=two_stage,
                belts=belts,
            )
            elapsed = time.time() - t0
            results.append(
                TrialResult(
                    spec_key=spec_key,
                    run=run_slug,
                    basis_types=basis,
                    n_max=n_max,
                    reg_l1=reg,
                    boundary_weight=bw,
                    two_stage=two_stage,
                    restarts=restarts,
                    per_type_reg=per_type_reg,
                    metrics=eval_res.metrics,
                    type_counts=eval_res.type_counts,
                    elapsed_s=elapsed,
                    seed=seed,
                )
            )
            print(f"[random] {run_slug} done in {elapsed:.1f}s (boundary_mae={eval_res.metrics.get('boundary_mae')})")

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump([r.__dict__ for r in results], out_path.open("w"), indent=2)
    print(f"Saved {len(results)} random records to {out_path}")
    return results


# ------------------------ Interestingness ------------------------ #


def _baseline_map() -> Dict[str, Dict[str, float]]:
    base_path = Path("runs/torus/stage4_metrics_modes_families.json")
    data = json.loads(base_path.read_text()) if base_path.exists() else []
    baseline: Dict[str, Dict[str, float]] = {}
    for rec in data:
        spec = rec.get("spec")
        if spec not in ("torus_thin", "torus_mid"):
            continue
        m = rec.get("metrics", {})
        b_mae = m.get("boundary_mae", math.inf)
        belt = m.get("offaxis_belt_rel", None)
        n_img = m.get("n_images", None)
        prev = baseline.get(spec)
        if prev is None or b_mae < prev["boundary_mae"]:
            baseline[spec] = {
                "boundary_mae": b_mae,
                "belt_rel": belt,
                "n_images": n_img,
            }
    return baseline


def find_interesting(records: List[TrialResult]) -> Dict[str, List[TrialResult]]:
    baseline = _baseline_map()
    by_spec: Dict[str, List[TrialResult]] = {}
    for r in records:
        by_spec.setdefault(r.spec_key, []).append(r)

    interesting: Dict[str, List[TrialResult]] = {}
    for spec, recs in by_spec.items():
        base = baseline.get(spec, {"boundary_mae": math.inf, "belt_rel": None, "n_images": None})
        b_base = base.get("boundary_mae", math.inf)
        belt_base = base.get("belt_rel", None)
        n_base = base.get("n_images", None)
        for r in recs:
            m = r.metrics
            b = m.get("boundary_mae", math.inf)
            belt = m.get("offaxis_belt_rel", None)
            n_img = m.get("n_images", r.type_counts.get("n_images"))

            b_impr = b / b_base if b_base and b_base < math.inf else math.inf
            belt_impr = (belt / belt_base) if (belt_base and belt is not None) else math.inf

            flag = False
            if b_impr < 0.5 or belt_impr < 0.5:
                flag = True
            if n_base and n_img and n_img <= max(4, 0.5 * n_base) and b_impr < 1.5:
                flag = True
            if any(k in r.basis_types for k in ("inner_rim_arc", "inner_rim_ribbon", "inner_patch_ring")) and b_impr < 1.0:
                flag = True
            if flag:
                interesting.setdefault(spec, []).append(r)

        # Rank by boundary_mae
        if spec in interesting:
            interesting[spec].sort(key=lambda x: x.metrics.get("boundary_mae", math.inf))
    return interesting


def print_interesting(interesting: Dict[str, List[TrialResult]]) -> None:
    for spec, recs in interesting.items():
        print(f"\n[interesting] {spec} top {len(recs)}")
        for i, r in enumerate(recs[:10]):
            m = r.metrics
            belt = m.get("offaxis_belt_rel")
            print(
                f"{i+1:02d} {r.run} | b_mae={m.get('boundary_mae'):.3e} off_rel={m.get('offaxis_rel')} "
                f"belt={belt} n={m.get('n_images')} types={r.type_counts}"
            )


# ------------------------ Main ------------------------ #


def main() -> None:
    ap = argparse.ArgumentParser(description="Random torus Stage4 explorer with novelty hints.")
    ap.add_argument("--trials-per-spec", type=int, default=10, help="Number of random trials per spec.")
    ap.add_argument("--analyze-only", action="store_true", help="Only analyze existing random_sweep.json.")
    args = ap.parse_args()

    out_path = Path("runs/torus/stage4_metrics_random_sweep.json")

    specs: Dict[str, CanonicalSpec] = {}
    root = Path(__file__).resolve().parents[1]
    thin = load_spec(root / "specs" / "torus_axis_point_thin.json")
    mid = load_spec(root / "specs" / "torus_axis_point_mid.json")
    offaxis = load_spec(root / "specs" / "torus_offaxis_point_thin.json")
    if thin:
        specs["torus_thin"] = thin
    if mid:
        specs["torus_mid"] = mid
    if offaxis:
        specs["torus_offaxis_thin"] = offaxis

    records: List[TrialResult] = []
    if args.analyze_only:
        if out_path.exists():
            data = json.loads(out_path.read_text())
            for rec in data:
                records.append(TrialResult(**rec))
    else:
        # Cap total trials to ~40
        total_specs = len(specs)
        trials = max(1, min(args.trials_per_spec, 20))
        max_trials_total = 40
        if trials * total_specs > max_trials_total and total_specs > 0:
            trials = max_trials_total // total_specs
        records = run_random_trials(specs, trials, out_path)

    if not records:
        print("No records to analyze.")
        return

    interesting = find_interesting(records)
    print_interesting(interesting)


if __name__ == "__main__":
    main()
