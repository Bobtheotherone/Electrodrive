import math
import pathlib
import torch

from electrodrive.utils.config import K_E, BEMConfig
from electrodrive.learn.collocation import BEM_AVAILABLE
from electrodrive.core.bem import bem_solve
from electrodrive.core.bem_kernel import bem_potential_targets
from electrodrive.orchestration.parser import CanonicalSpec

SPHERE_OUT = CanonicalSpec.from_json(
    {
        "domain": "R3",
        "conductors": [
            {"type": "sphere", "center": [0.0, 0.0, 0.0], "radius": 1.0, "potential": 0.0},
        ],
        "charges": [
            {"type": "point", "q": 1.0e-9, "pos": [0.0, 0.0, 2.0]},
        ],
        "queries": ["potential"],
    }
)


def solve_bem(spec):
    cfg = BEMConfig()
    cfg.fp64 = True
    cfg.use_gpu = False
    cfg.max_refine_passes = 1
    cfg.min_refine_passes = 1
    cfg.use_near_quadrature = True
    cfg.use_near_quadrature_matvec = False

    class DummyLogger:
        def info(self, *args, **kwargs):
            pass
        def debug(self, *args, **kwargs):
            pass
        def warning(self, *args, **kwargs):
            pass
        def error(self, *args, **kwargs):
            print("ERROR", args, kwargs)

    out = bem_solve(spec, cfg, logger=DummyLogger())
    if isinstance(out, dict) and "solution" in out:
        return out["solution"]
    raise RuntimeError(f"BEM solve failed: {out}")


def phi_induced_at_charge(sol, r_q):
    tgt = torch.tensor([list(r_q)], device=sol._device, dtype=sol._dtype)
    with torch.no_grad():
        V_ind = bem_potential_targets(
            targets=tgt,
            src_centroids=sol._C,
            areas=sol._A,
            sigma=sol._S,
            tile_size=sol._tile,
        )[0]
    return float(V_ind.item())


def main():
    if not BEM_AVAILABLE:
        raise SystemExit("BEM not available")

    sol = solve_bem(SPHERE_OUT)
    q = 1.0e-9
    r_q = (0.0, 0.0, 2.0)
    a = 1.0
    R = math.sqrt(sum(x * x for x in r_q))

    phi_ind = phi_induced_at_charge(sol, r_q)
    U_A = -0.5 * q * phi_ind
    U_B = -0.5 * K_E * q * q * a / (R * R - a * a)
    denom = max(0.5 * (abs(U_A) + abs(U_B)), 1e-30)
    rel = abs(U_A - U_B) / denom

    lines = [
        "# Sphere External Energy Probe",
        f"phi_induced_bem = {phi_ind:.6e}",
        f"U_A = {U_A:.6e}",
        f"U_B = {U_B:.6e}",
        f"energy_rel_diff = {rel:.6e}",
        f"signs: U_A {'+' if U_A>0 else '-'} , U_B {'+' if U_B>0 else '-'}",
    ]

    out_path = pathlib.Path("experiments/energy_sphere_out_probe_results.md")
    out_path.write_text("\n".join(lines), encoding="utf-8")

    for ln in lines:
        print(ln)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
