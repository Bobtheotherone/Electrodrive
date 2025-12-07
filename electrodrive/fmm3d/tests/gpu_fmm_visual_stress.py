"""
GPU-only FMM visual stress test + timelapse renderer.

Drop this somewhere inside your repo, e.g.:

    electrodrive/fmm3d/demos/gpu_fmm_visual_stress.py

and run:

    python -m electrodrive.fmm3d.demos.gpu_fmm_visual_stress \
        --n-points 40000 --n-frames 240 --fps 24

This will:
  * Build a large random FMM problem (points + BEM-like charges).
  * Optionally run a GPU-only FMM matvec once and check it against a GPU
    direct P2P subset for sanity (unless --skip-check is set).
  * Then run a loop of GPU FMM solves with time-varying charges,
    recording a 3D scatterplot of the potential field as an MP4,
    while showing a live 3D view of the animation.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Tuple

import torch
from torch import Tensor

# Matplotlib imports for 3D rendering / animation
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (side-effect: registers 3D proj)

from tqdm.auto import tqdm

from electrodrive.fmm3d.config import FmmConfig
from electrodrive.fmm3d.tree import build_fmm_tree
from electrodrive.fmm3d.interaction_lists import build_interaction_lists
from electrodrive.fmm3d.kernels_gpu import (
    p2m_gpu,
    m2m_gpu,
    m2l_gpu,
    l2l_gpu,
    l2p_gpu,
    apply_p2p_gpu,
)
from electrodrive.fmm3d.kernels_cpu import _p2p_block


def _disable_fmm_logging() -> None:
    """
    Turn off all the noisy FMM debug / spectral logging so we get
    clean performance numbers and a quiet console.

    This patches the already-imported modules in-place; it only affects
    this process.
    """
    import electrodrive.fmm3d.multipole_operators as mop
    import electrodrive.fmm3d.kernels_cpu as kcpu
    import electrodrive.fmm3d.kernels_gpu as kgpu

    class _NullLogger:
        def debug(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    def _silent_get_logger(*args, **kwargs):
        return _NullLogger()

    # Multipole operators
    mop.get_logger = _silent_get_logger
    mop.log_spectral_stats = lambda *a, **k: None
    mop.debug_tensor_stats = lambda *a, **k: None
    mop.want_verbose_debug = lambda: False

    # CPU kernels
    kcpu.get_logger = _silent_get_logger
    kcpu.log_spectral_stats = lambda *a, **k: None
    kcpu.debug_tensor_stats = lambda *a, **k: None
    kcpu.want_verbose_debug = lambda: False

    # GPU kernels
    kgpu.get_logger = _silent_get_logger
    kgpu.debug_tensor_stats = lambda *a, **k: None


def make_visual_problem(
    n_points: int,
    *,
    seed: int = 2025,
    dtype: torch.dtype = torch.float64,
    expansion_order: int = 10,
    mac_theta: float = 0.5,
) -> Tuple[Tensor, Tensor, FmmConfig]:
    """
    Build a 'BEM-like' random problem for stress testing + visualization.

    - Geometry: random points in [0, 1]^3, same as other tests.
    - Charges: sigma * area, like a simple BEM patch model.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # Uniform box [0, 1]^3
    points = torch.rand((n_points, 3), generator=g, dtype=dtype)

    # Fake "BEM-like" charges: sigma * area
    sigma = torch.randn(n_points, generator=g, dtype=dtype)
    areas = torch.rand(n_points, generator=g, dtype=dtype)
    charges = sigma * areas

    cfg = FmmConfig(
        expansion_order=expansion_order,
        mac_theta=mac_theta,
        leaf_size=64,
        dtype=dtype,
    )
    return points, charges, cfg


def gpu_fmm_potential_tree_order(
    tree,
    lists,
    charges_tree: Tensor,
    cfg: FmmConfig,
) -> Tensor:
    """
    Run the full GPU FMM pipeline and return potentials in *tree order*.

    Parameters
    ----------
    tree:
        FmmTree that already lives on a CUDA device.
    lists:
        InteractionLists built on CPU.
    charges_tree:
        (N,) tensor of charges in *tree order*, on the same device as tree.
    cfg:
        FMM configuration.
    """
    # Upward / far-field
    multipoles = p2m_gpu(tree=tree, charges=charges_tree, cfg=cfg)
    multipoles = m2m_gpu(tree=tree, multipoles=multipoles, cfg=cfg)

    locals_ = m2l_gpu(
        source_tree=tree,
        target_tree=tree,
        multipoles=multipoles,
        lists=lists,
        cfg=cfg,
    )
    locals_ = l2l_gpu(tree=tree, locals_=locals_, cfg=cfg)
    phi_far_tree = l2p_gpu(tree=tree, locals_=locals_, cfg=cfg)

    # Near-field P2P (still on GPU)
    p2p_result = apply_p2p_gpu(
        source_tree=tree,
        target_tree=tree,
        charges_src=multipoles.charges,
        lists=lists,
        cfg=cfg,
    )
    phi_near_tree = p2p_result.potential

    phi_total_tree = phi_far_tree + phi_near_tree
    return phi_total_tree


def gpu_direct_subset_check(
    points: Tensor,
    charges: Tensor,
    phi_fmm: Tensor,
    subset_size: int = 1024,
    seed: int = 7,
) -> Tuple[float, float]:
    """
    Sanity check: compare GPU FMM potentials against a GPU direct 1/r
    computation on a random subset of targets.

    All inputs are assumed to be on CPU; computation is moved to CUDA.
    """
    device = torch.device("cuda")
    dtype = points.dtype

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    n_points = points.shape[0]
    idx = torch.randint(0, n_points, (subset_size,), generator=g)

    x_tgt = points[idx].to(device=device, dtype=dtype)
    x_src = points.to(device=device, dtype=dtype)
    q_src = charges.to(device=device, dtype=dtype)

    # Direct GPU P2P via the same kernel used by the CPU FMM,
    # but evaluated on CUDA tensors.
    phi_direct_block = _p2p_block(
        x_tgt=x_tgt,
        x_src=x_src,
        q_src=q_src,
        exclude_self=True,
    )
    phi_direct = phi_direct_block.sum(dim=1)  # sum over sources

    # FMM potentials on the same targets.
    phi_fmm_subset = phi_fmm[idx].to(device=device, dtype=dtype)

    denom = torch.linalg.norm(phi_direct)
    rel_l2 = torch.linalg.norm(phi_direct - phi_fmm_subset) / (denom + 1e-16)
    max_rel = torch.max(
        torch.abs(phi_direct - phi_fmm_subset) / (torch.abs(phi_direct) + 1e-16)
    )

    return float(rel_l2.item()), float(max_rel.item())


def _setup_figure():
    # Dark, high-contrast canvas for "sci-fi" vibes.
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    fig.patch.set_facecolor("#050510")
    ax.set_facecolor("#050510")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Small margins for a tight, cinematic framing.
    ax.margins(0.05)

    return fig, ax


def _render_frame(
    ax,
    points_vis_cpu: Tensor,
    phi_vis_cpu: Tensor,
    frame_idx: int,
    n_frames: int,
    title: str,
) -> None:
    """
    Render a single animation frame.

    points_vis_cpu: (M, 3) CPU tensor of points to display.
    phi_vis_cpu:    (M,) CPU tensor of potentials at those points.
    """
    import numpy as np

    ax.cla()
    ax.set_facecolor("#050510")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)

    # Camera path: smooth orbit with gentle bobbing.
    azim = 20.0 + 360.0 * (frame_idx / max(1, n_frames))
    elev = 20.0 + 10.0 * math.sin(2.0 * math.pi * frame_idx / max(1, n_frames))
    ax.view_init(elev=elev, azim=azim)

    xyz = points_vis_cpu.numpy()
    phi = phi_vis_cpu.numpy()

    # Normalize potential for stable colors, using robust percentiles.
    vmin, vmax = np.percentile(phi, [2.0, 98.0])
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0

    norm = (phi - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)

    # "Glow" effect by layering two scatters:
    #  - a large, soft underlay
    #  - a sharper, bright core
    cmap = cm.get_cmap("plasma")
    colors = cmap(norm)

    # Underlay: bigger, faint points for halo.
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=colors,
        s=4.0,
        alpha=0.25,
        linewidths=0,
        edgecolors="none",
    )

    # Core: smaller, bright points.
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=colors,
        s=1.0,
        alpha=0.9,
        linewidths=0,
        edgecolors="none",
    )

    # Subtle title and stats overlay.
    ax.set_title(
        title,
        color="#f8f8ff",
        fontsize=10,
        pad=12,
    )

    # Slight vignette via axis limits.
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU-only FMM visual stress test + timelapse renderer",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=40000,
        help="Number of points in the FMM problem (sources = targets).",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=240,
        help="Number of frames in the output animation.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second for the animation (duration ~= n_frames / fps).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gpu_fmm_timelapse.mp4",
        help="Output MP4 filename.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="CUDA device to use (e.g. 'cuda', 'cuda:0').",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="Floating point precision for the FMM computation.",
    )
    parser.add_argument(
        "--subset-vis",
        type=int,
        default=10000,
        help="Number of points to display in the scatter (for rendering speed).",
    )
    parser.add_argument(
        "--subset-check",
        type=int,
        default=1024,
        help="Number of points for the direct P2P accuracy check.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip the initial global FMM sanity check and go straight to the animation.",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available, but this script is GPU-only.")

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    _disable_fmm_logging()

    print(f"[setup] n_points      = {args.n_points}")
    print(f"[setup] n_frames      = {args.n_frames}")
    print(f"[setup] fps           = {args.fps}  (duration ≈ {args.n_frames / args.fps:.1f}s)")
    print(f"[setup] dtype         = {dtype}")
    print(f"[setup] device        = {device}")
    print(f"[setup] vis subset    = {args.subset_vis}")
    print(f"[setup] check subset  = {args.subset_check}")
    print(f"[setup] output video  = {args.output}")
    print(f"[setup] skip_check    = {args.skip_check}")

    # ------------------------------------------------------------------
    # 1) Build problem + FMM structures
    # ------------------------------------------------------------------
    points_cpu, charges_cpu, cfg = make_visual_problem(
        n_points=args.n_points,
        dtype=dtype,
    )

    # Build tree + interaction lists on CPU.
    print("[build] Constructing FMM tree + interaction lists on CPU...")
    tree = build_fmm_tree(points_cpu, leaf_size=int(cfg.leaf_size))
    lists = build_interaction_lists(tree, tree, mac_theta=float(cfg.mac_theta))

    # IMPORTANT: reorder charges on CPU first, then move to device.
    charges_tree_cpu = tree.map_to_tree_order(charges_cpu)

    # Move tree to CUDA; keep dtype consistent with cfg.
    tree.to(device, dtype=dtype)

    # Now move reordered charges to device.
    charges_tree = charges_tree_cpu.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # 2) One-shot GPU FMM solve + GPU direct subset check
    # ------------------------------------------------------------------
    if not args.skip_check:
        print("[check] Running a single GPU FMM solve for sanity...")
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize(device=device)

        phi_tree = gpu_fmm_potential_tree_order(
            tree=tree,
            lists=lists,
            charges_tree=charges_tree,
            cfg=cfg,
        )
        torch.cuda.synchronize(device=device)

        phi_cpu = tree.map_to_original_order(phi_tree).cpu()

        # Basic "no NaNs/infs" sanity check.
        if not torch.isfinite(phi_cpu).all():
            raise RuntimeError("FMM potential contains NaNs or infs; aborting.")

        # Direct GPU subset check.
        print(f"[check] Comparing against direct 1/r on a subset of {args.subset_check} targets...")
        rel_l2, max_rel = gpu_direct_subset_check(
            points=points_cpu,
            charges=charges_cpu,
            phi_fmm=phi_cpu,
            subset_size=args.subset_check,
        )
        print(f"[check] rel_l2 = {rel_l2:.3e}, max_rel = {max_rel:.3e}")
    else:
        print("[check] Skipping initial global FMM sanity check (--skip-check).")

    # ------------------------------------------------------------------
    # 3) Timelapse: repeated GPU FMM solves with time-varying charges
    # ------------------------------------------------------------------
    # Pre-select a subset of points for visualization to keep rendering fast.
    n_points = points_cpu.shape[0]
    subset_vis = min(args.subset_vis, n_points)

    g_vis = torch.Generator(device="cpu")
    g_vis.manual_seed(123)
    idx_vis = torch.randperm(n_points, generator=g_vis)[:subset_vis]

    points_vis_cpu = points_cpu[idx_vis]

    # Base charges on device in tree order.
    base_charges_tree = charges_tree  # already tree-ordered on device

    # Setup figure + writer
    fig, ax = _setup_figure()
    metadata = dict(
        title="Electrodrive GPU FMM — 3D potential field",
        artist="R.J.Tech",
        comment="GPU-only FMM visual stress test",
    )
    writer = FFMpegWriter(fps=args.fps, metadata=metadata, bitrate=12000)

    # Enable interactive mode so we get a live view.
    plt.ion()
    fig.show()
    fig.canvas.draw()

    print("[anim] Rendering timelapse frames...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with writer.saving(fig, args.output, dpi=150):
        for frame_idx in tqdm(range(args.n_frames), desc="frames"):
            # Time-varying charge pattern (simple breathing / rotation in charge space).
            t = frame_idx / max(1, args.n_frames - 1)
            phase = 2.0 * math.pi * t

            # Modulate charges smoothly to give the field some life.
            # We keep everything on the device.
            modulation = 0.75 + 0.25 * torch.sin(
                torch.linspace(
                    0.0,
                    4.0 * math.pi,
                    base_charges_tree.numel(),
                    device=device,
                )
                + phase
            )
            charges_frame = base_charges_tree * modulation

            # GPU FMM matvec for this frame.
            phi_tree_frame = gpu_fmm_potential_tree_order(
                tree=tree,
                lists=lists,
                charges_tree=charges_frame,
                cfg=cfg,
            )

            # Map to original order on device, then pick visualization subset.
            phi_frame_cpu = tree.map_to_original_order(phi_tree_frame).detach().cpu()
            phi_vis_cpu = phi_frame_cpu[idx_vis]

            title = (
                f"Electrodrive GPU FMM • N = {n_points:,d} • "
                f"frame {frame_idx+1}/{args.n_frames}"
            )
            _render_frame(
                ax=ax,
                points_vis_cpu=points_vis_cpu,
                phi_vis_cpu=phi_vis_cpu,
                frame_idx=frame_idx,
                n_frames=args.n_frames,
                title=title,
            )

            # Update live window for this frame.
            fig.canvas.draw()
            plt.pause(0.001)

            # Save frame to the MP4.
            writer.grab_frame()

    print(f"[done] Saved animation to: {os.path.abspath(args.output)}")

    # Turn off interactive mode and keep the final frame open
    # until the user closes the window.
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
