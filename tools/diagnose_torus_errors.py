"""
Diagnostic: evaluate image-system vs BEM on an r-z grid for torus specs.

Outputs an NPZ with potentials and errors; optionally dumps PNG heatmaps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.images.search import discover_images
from electrodrive.learn.collocation import get_oracle_solution


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _eval_system(spec: CanonicalSpec, system, pts: torch.Tensor) -> torch.Tensor:
    return system.potential(pts)


def _eval_bem(spec: CanonicalSpec, pts: torch.Tensor) -> torch.Tensor:
    sol = get_oracle_solution(spec, mode="bem", bem_cfg={})  # type: ignore[arg-type]
    if sol is None:
        raise RuntimeError("BEM solution unavailable.")
    if hasattr(sol, "eval_V_E_batched"):
        V, _ = sol.eval_V_E_batched(pts)  # type: ignore[attr-defined]
        return V
    if hasattr(sol, "eval"):
        return sol.eval(pts)  # type: ignore[attr-defined]
    raise RuntimeError("BEM solution has no eval interface.")


def build_rz_grid(R: float, a: float, nr: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    r_min = max(1e-6, R - 1.5 * a)
    r_max = R + 1.5 * a
    z_min = -1.5 * a
    z_max = 1.5 * a
    r = np.linspace(r_min, r_max, nr)
    z = np.linspace(z_min, z_max, nz)
    return r, z


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose torus image vs BEM errors on r-z grid.")
    ap.add_argument("--spec", required=True, type=str)
    ap.add_argument("--basis-types", required=True, type=str, help="Comma-separated basis types")
    ap.add_argument("--n-max", type=int, default=12)
    ap.add_argument("--reg-l1", type=float, default=1e-3)
    ap.add_argument("--boundary-weight", type=float, default=None)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--per-type-reg", type=str, default=None, help="JSON string or path to JSON with per-type reg dict")
    ap.add_argument("--nr", type=int, default=200)
    ap.add_argument("--nz", type=int, default=200)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    basis_types = [b.strip() for b in args.basis_types.split(",") if b.strip()]
    per_type_reg: Optional[Dict[str, float]] = None
    if args.per_type_reg:
        try:
            if Path(args.per_type_reg).exists():
                per_type_reg = json.loads(Path(args.per_type_reg).read_text())
            else:
                per_type_reg = json.loads(args.per_type_reg)
        except Exception:
            per_type_reg = None

    spec = CanonicalSpec.from_json(json.load(open(args.spec)))
    device = _device()
    dtype = torch.float32

    system = discover_images(
        spec=spec,
        basis_types=basis_types,
        n_max=args.n_max,
        reg_l1=args.reg_l1,
        restarts=args.restarts,
        logger=type("L", (), {"info": lambda *a, **k: None, "warning": lambda *a, **k: None, "error": lambda *a, **k: None})(),
        per_type_reg=per_type_reg,
        boundary_weight=args.boundary_weight,
    )

    torus = next(c for c in spec.conductors if c.get("type") in ("torus", "toroid"))
    R = float(torus.get("major_radius", torus.get("radius", 1.0)))
    a = float(torus.get("minor_radius", 0.25 * R))

    r_arr, z_arr = build_rz_grid(R, a, args.nr, args.nz)
    rr, zz = np.meshgrid(r_arr, z_arr, indexing="ij")
    xx = rr
    yy = np.zeros_like(rr)
    pts_np = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    pts = torch.tensor(pts_np, device=device, dtype=dtype)

    with torch.no_grad():
        V_img = _eval_system(spec, system, pts).view(args.nr, args.nz).cpu().numpy()
        V_bem = _eval_bem(spec, pts).view(args.nr, args.nz).cpu().numpy()

    abs_err = np.abs(V_img - V_bem)
    rel_err = abs_err / (np.abs(V_bem) + 1e-12)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        r=r_arr,
        z=z_arr,
        V_img=V_img,
        V_bem=V_bem,
        abs_err=abs_err,
        rel_err=rel_err,
        meta={
            "spec": args.spec,
            "basis_types": basis_types,
            "n_max": args.n_max,
            "reg_l1": args.reg_l1,
            "boundary_weight": args.boundary_weight,
            "restarts": args.restarts,
            "per_type_reg": per_type_reg,
        },
    )
    print(f"Saved diagnostics to {out_path}")

    if args.plot:
        import matplotlib.pyplot as plt

        for name, data, cmap, vmax in [
            ("abs_err", abs_err, "magma", None),
            ("rel_err", rel_err, "viridis", None),
        ]:
            plt.figure(figsize=(6, 4))
            plt.pcolormesh(z_arr, r_arr, data, shading="auto", cmap=cmap, vmax=vmax)
            plt.xlabel("z")
            plt.ylabel("r")
            plt.title(name)
            plt.colorbar()
            png_path = out_path.with_suffix(f".{name}.png")
            plt.savefig(png_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
