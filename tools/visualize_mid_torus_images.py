"""
Visualize a torus image system (rings + points) with 3D rendering and r–z equipotential contours.

Usage:
    python tools/visualize_mid_torus_images.py \
        --spec specs/torus_axis_point_mid.json \
        --system runs/torus/discovered/mid_local_seed5500358_trial003/discovered_system.json \
        --out mid_torus_best.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch

from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.images.io import load_image_system


def torus_surface(R: float, a: float, n_theta: int = 80, n_phi: int = 80) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    x = (R + a * np.cos(tt)) * np.cos(pp)
    y = (R + a * np.cos(tt)) * np.sin(pp)
    z = a * np.sin(tt)
    return x, y, z


def cross_section_potential(system, R: float, a: float, nr: int = 200, nz: int = 200):
    r_min, r_max = max(1e-6, R - 1.5 * a), R + 1.5 * a
    z_min, z_max = -1.5 * a, 1.5 * a
    r = np.linspace(r_min, r_max, nr)
    z = np.linspace(z_min, z_max, nz)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    pts = np.stack([rr, np.zeros_like(rr), zz], axis=-1).reshape(-1, 3)
    with torch.no_grad():
        V = system.potential(torch.tensor(pts, device=system.weights.device, dtype=system.weights.dtype))
    V = V.cpu().numpy().reshape(nr, nz)
    return r, z, V


def extract_elements(system) -> Tuple[List[dict], List[dict]]:
    rings: List[dict] = []
    points: List[dict] = []
    for elem, w in zip(system.elements, system.weights.tolist()):
        if elem.type == "poloidal_ring":
            p = elem.params
            rings.append(
                {
                    "radius": float(p["radius"]),
                    "delta_r": float(p.get("delta_r", 0.0)),
                    "order": int(p.get("order", 0)),
                    "weight": w,
                }
            )
        elif elem.type == "point":
            pos = torch.as_tensor(elem.params["position"]).view(-1).cpu().numpy()
            points.append({"pos": pos, "weight": w})
    return rings, points


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize torus image system with 3D surface and equipotentials.")
    ap.add_argument("--spec", type=str, required=True, help="Path to spec JSON.")
    ap.add_argument("--system", type=str, required=True, help="Path to discovered_system.json.")
    ap.add_argument("--out", type=str, default=None, help="Optional output image path.")
    ap.add_argument("--no-show", action="store_true", help="Skip interactive display.")
    ap.add_argument("--nr", type=int, default=200)
    ap.add_argument("--nz", type=int, default=200)
    args = ap.parse_args()

    spec = CanonicalSpec.from_json(json.load(open(args.spec)))
    system = load_image_system(Path(args.system), device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32)
    torus = next(c for c in spec.conductors if c.get("type") in ("torus", "toroid"))
    R = float(torus.get("major_radius", torus.get("radius", 1.0)))
    a = float(torus.get("minor_radius", 0.25 * R))

    # Build torus surface
    X, Y, Z = torus_surface(R, a, n_theta=120, n_phi=120)

    # Equipotential cross-section
    r, z, V = cross_section_potential(system, R, a, nr=args.nr, nz=args.nz)

    rings, points = extract_elements(system)

    # Styling
    cmap_surface = plt.get_cmap("plasma")
    cmap_contour = plt.get_cmap("viridis")
    norm_weights = mpl.colors.SymLogNorm(linthresh=1e-3, vmin=-max(abs(p["weight"]) for p in points + rings) if points or rings else -1, vmax=max(abs(p["weight"]) for p in points + rings) if points or rings else 1)

    fig = plt.figure(figsize=(14, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax3d.plot_surface(X, Y, Z, rstride=2, cstride=2, facecolors=cmap_surface((Z - Z.min()) / (Z.max() - Z.min() + 1e-9)), linewidth=0, antialiased=True, alpha=0.65, shade=True)
    ax3d.set_box_aspect([1, 1, 0.6])
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    ax3d.set_title("Torus + Images")

    # Rings
    phi_ring = np.linspace(0, 2 * np.pi, 400)
    ring_color = "magenta"
    for i, rinfo in enumerate(rings):
        r_ring = rinfo["radius"]
        x_ring = r_ring * np.cos(phi_ring)
        y_ring = r_ring * np.sin(phi_ring)
        z_ring = np.zeros_like(phi_ring)
        ax3d.plot(x_ring, y_ring, z_ring, color=ring_color, linewidth=4.0, alpha=0.9, label="poloidal_ring" if i == 0 else None)

    # Points
    if points:
        pts = np.stack([p["pos"] for p in points], axis=0)
        w = np.array([p["weight"] for p in points])
        point_color = "tab:orange"
        ax3d.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=(120 + 240 * np.abs(w) / (np.abs(w).max() + 1e-9)),
            c=point_color,
            depthshade=True,
            edgecolors="k",
            linewidths=1.0,
            label="points",
        )

    # Equipotential contour plot
    ax2 = fig.add_subplot(1, 2, 2)
    cf = ax2.contourf(r, z, V.T, levels=40, cmap=cmap_contour)
    cs = ax2.contour(r, z, V.T, levels=10, colors="k", linewidths=0.6, alpha=0.5)
    torus_r = R + a * np.cos(np.linspace(0, 2 * np.pi, 400))
    torus_z = a * np.sin(np.linspace(0, 2 * np.pi, 400))
    ax2.plot(torus_r, torus_z, color="white", linewidth=1.5, alpha=0.9)
    # Overlay rings and points in r-z
    for idx, rinfo in enumerate(rings):
        ax2.axvline(rinfo["radius"], color=ring_color, linestyle="--", linewidth=1.5, alpha=0.9, label="ring" if idx == 0 else None)
    if points:
        rho_pts = np.linalg.norm(pts[:, :2], axis=1)
        ax2.scatter(rho_pts, pts[:, 2], color=point_color, edgecolors="k", s=50, zorder=5, label="points (r,z)")
    ax2.set_xlabel("r (x)"); ax2.set_ylabel("z"); ax2.set_title("Equipotential contours (y=0)")
    fig.colorbar(cf, ax=ax2, shrink=0.8, label="Potential (arb)")

    handles, labels = [], []
    h3, l3 = ax3d.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    handles.extend(h3); handles.extend(h2)
    labels.extend(l3); labels.extend(l2)
    if handles:
        ax3d.legend(loc="upper right")
        ax2.legend(loc="upper right")
    fig.suptitle("Mid-torus image system visualization", fontsize=14)
    fig.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=300, bbox_inches="tight")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
