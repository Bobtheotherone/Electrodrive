"""
Plot r–z cross section of the mid-torus geometry with rings and points annotated.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from electrodrive.images.io import load_image_system
from electrodrive.orchestration.parser import CanonicalSpec


def torus_outline(R: float, a: float, n: int = 400):
    theta = np.linspace(0, 2 * np.pi, n)
    r = R + a * np.cos(theta)
    z = a * np.sin(theta)
    return r, z


def extract_rings_points(sys):
    rings = []
    points = []
    for elem in sys.elements:
        if elem.type == "poloidal_ring":
            p = elem.params
            rings.append({"radius": float(p["radius"]), "delta_r": float(p.get("delta_r", 0.0)), "order": int(p.get("order", 0))})
        elif elem.type == "point":
            pos = torch.as_tensor(elem.params["position"]).view(-1).cpu().numpy()
            rho = float(math.hypot(pos[0], pos[1]))
            points.append({"rho": rho, "z": float(pos[2])})
    rings.sort(key=lambda r: r["radius"])
    return rings, points


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot mid-torus geometry cross-section with rings and points.")
    ap.add_argument("--spec", type=Path, default=Path("specs/torus_axis_point_mid.json"))
    ap.add_argument("--system", type=Path, default=Path("runs/torus/discovered/mid_bem_highres_trial02/discovered_system.json"))
    ap.add_argument("--out", type=Path, default=Path("staging/figures/paper/fig1b_geometry_cross_section.png"))
    args = ap.parse_args()

    spec = CanonicalSpec.from_json(json.load(open(args.spec)))
    torus = next(c for c in spec.conductors if c.get("type") in ("torus", "toroid"))
    R = float(torus.get("major_radius", torus.get("radius", 1.0)))
    a = float(torus.get("minor_radius", 0.25 * R))

    sys = load_image_system(args.system, device="cpu", dtype=torch.float32)
    rings, points = extract_rings_points(sys)

    fig, ax = plt.subplots(figsize=(7, 5))
    r_out, z_out = torus_outline(R, a)
    ax.plot(r_out, z_out, color="white", linewidth=1.5, alpha=0.9, label="torus outline")

    ring_color = "magenta"
    for idx, rinfo in enumerate(rings):
        ax.axvline(rinfo["radius"], color=ring_color, linestyle="--", linewidth=2.0, alpha=0.9, label="ring" if idx == 0 else None)
        ax.text(rinfo["radius"], 0.9 * a, f"ring {idx+1}", color=ring_color, ha="center", va="bottom", fontsize=9)

    point_color = "tab:orange"
    for j, p in enumerate(points):
        ax.scatter(p["rho"], p["z"], color=point_color, edgecolors="k", s=60, zorder=5, label="point" if j == 0 else None)
        ax.text(p["rho"], p["z"], f"P{j+1}", color="k", fontsize=8, ha="left", va="bottom")

    ax.set_xlabel("r")
    ax.set_ylabel("z")
    ax.set_title("Mid-torus geometry: 2 poloidal rings + 4 points")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
