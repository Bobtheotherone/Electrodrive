# electrodrive/tools/discover_moi.py
from __future__ import annotations
import os, math, json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from electrodrive.core.images import FastImagesSolution

def build_parallel_planes_images(q0: float, pos: np.ndarray, sep: float, max_images: int):
    """
    Point charge q0 at (x0,y0,z0) between two grounded planes z=±sep/2.
    Classic MoI: infinite alternating images across both planes; we truncate.
    Returns (charges[M], positions[M,3]) including the real charge.
    """
    x0, y0, z0 = pos.tolist()
    charges = [q0]
    positions = [pos.copy()]
    # Reflect across top and bottom repeatedly
    z_top, z_bot = +sep/2.0, -sep/2.0
    z_imgs = [(q0, z0)]
    # Each reflection flips sign for grounded Dirichlet planes
    sign = -1.0
    z_curr = [z0]
    for k in range(1, max_images+1):
        new_z = []
        for zc in z_curr:
            # reflect in top
            zt = 2*z_top - zc
            charges.append(sign * q0)
            positions.append(np.array([x0, y0, zt], dtype=np.float64))
            new_z.append(zt)
            # reflect in bottom
            zb = 2*z_bot - zc
            charges.append(sign * q0)
            positions.append(np.array([x0, y0, zb], dtype=np.float64))
            new_z.append(zb)
        z_curr = new_z
        sign *= -1.0
    return np.array(charges, dtype=np.float64), np.stack(positions, axis=0)

@torch.no_grad()
def boundary_error(solution: FastImagesSolution, sep: float, n_pts=512):
    """
    Measure |V| on both planes (Dirichlet 0). Sample random (x,y) square.
    """
    L = sep  # sample a square of side ~sep
    xy = (torch.rand(n_pts, 2, device=solution.device, dtype=solution.dtype) - 0.5) * (2.0*L)
    z_top = torch.full((n_pts,1), sep/2.0, device=solution.device, dtype=solution.dtype)
    z_bot = torch.full((n_pts,1),-sep/2.0, device=solution.device, dtype=solution.dtype)
    pts_top = torch.cat([xy, z_top], dim=1)
    pts_bot = torch.cat([xy, z_bot], dim=1)
    Vt = solution.eval_batch(pts_top)
    Vb = solution.eval_batch(pts_bot)
    # L2 boundary error
    return torch.sqrt((Vt.abs().mean()**2 + Vb.abs().mean()**2) / 2.0)

def run(out_dir: str = "runs/discover_moi",
        sep: float = 2.0,
        q0: float = 1.0,
        z0: float = 0.25,
        max_images_cap: int = 24,
        tol: float = 1e-2,
        grid_n: int = 128,
        device: str = "cuda",
        dtype: str = "float32"):
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    dt = getattr(torch, dtype)

    # Fixed charge location & quick sweep over #images
    x0 = y0 = 0.0
    charges, positions = build_parallel_planes_images(q0, np.array([x0, y0, z0]), sep, 0)
    best = {"images": 0, "err": float("inf")}
    history = []

    # Prepare grid for visualization slice (y=0 plane)
    xs = torch.linspace(-sep, sep, grid_n, device=dev, dtype=dt)
    zs = torch.linspace(-sep/2.0, sep/2.0, grid_n, device=dev, dtype=dt)
    XX, ZZ = torch.meshgrid(xs, zs, indexing="xy")
    YY = torch.zeros_like(XX)
    grid_pts = torch.stack([XX.reshape(-1), YY.reshape(-1), ZZ.reshape(-1)], dim=1)

    # Matplotlib fig
    fig, ax = plt.subplots(figsize=(6,5), constrained_layout=True)
    im = None
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")
    ax.set_title("Parallel planes, y=0 slice")
    ax.set_xlabel("x"); ax.set_ylabel("z")
    ax.axhline(+sep/2, lw=1, ls="--"); ax.axhline(-sep/2, lw=1, ls="--")

    frames = []

    for k in range(0, max_images_cap+1):
        # Extend image set by one reflection “layer” every k>0
        if k > 0:
            charges_k, positions_k = build_parallel_planes_images(q0, np.array([x0,y0,z0]), sep, k)
        else:
            charges_k, positions_k = charges, positions

        sol = FastImagesSolution.from_numpy(charges_k, positions_k, device=device, dtype=dtype)
        err = float(boundary_error(sol, sep, n_pts=1024).detach().cpu())
        history.append({"k": k, "err": err, "M": len(charges_k)})

        if err < best["err"]:
            best = {"images": k, "err": err, "M": len(charges_k)}

        # One visualization frame
        with torch.no_grad():
            V = sol.eval_batch(grid_pts).reshape(grid_n, grid_n).detach().cpu().numpy()
        ax.clear()
        ax.set_title("Parallel planes, y=0 slice")
        ax.set_xlabel("x"); ax.set_ylabel("z")
        ax.axhline(+sep/2, lw=1, ls="--"); ax.axhline(-sep/2, lw=1, ls="--")
        im = ax.contourf(XX.detach().cpu().numpy(), ZZ.detach().cpu().numpy(), V, levels=30)
        txt = ax.text(0.02, 0.98, f"k={k}  images={len(charges_k)}  |V|_BC≈{err:.3e}",
                      transform=ax.transAxes, va="top", ha="left")
        fig.canvas.draw()
        frames.append([*im.collections, txt])

        if err <= tol:
            break

    # Save animation (GIF; MP4 if ffmpeg is present)
    anim = FuncAnimation(fig, lambda i: frames[i], frames=len(frames), blit=True, repeat=False)
    gif_path = os.path.join(out_dir, "discover.gif")
    mp4_path = os.path.join(out_dir, "discover.mp4")
    try:
        anim.save(mp4_path, fps=2)
    except Exception:
        anim.save(gif_path, writer=PillowWriter(fps=2))
    # Save summary
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"best": best, "history": history}, f, indent=2)
    print("DISCOVERY:", best)
