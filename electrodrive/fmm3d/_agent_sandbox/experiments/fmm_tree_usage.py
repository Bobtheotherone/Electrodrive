import argparse
import datetime
import json
import math
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Dict, Tuple

import torch

from electrodrive.fmm3d.bem_fmm import make_laplace_fmm_backend
from electrodrive.fmm3d.multipole_operators import _get_or_init_global_scale

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"
REPO_ROOT = Path(__file__).resolve().parents[4]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def get_git_commit() -> str:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        commit = "UNKNOWN"
    return commit


def log_jsonl(record: Dict, filename: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def make_random_points(n: int, seed: int, mode: str = "uniform") -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    if mode == "uniform":
        x = 2.0 * torch.rand(n, 3, dtype=torch.float64) - 1.0
    elif mode == "clusters":
        n1 = n // 2
        n2 = n - n1
        c1 = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
        c2 = torch.tensor([-0.5, 0.0, 0.0], dtype=torch.float64)
        x1 = c1 + 0.1 * torch.randn(n1, 3, dtype=torch.float64)
        x2 = c2 + 0.1 * torch.randn(n2, 3, dtype=torch.float64)
        x = torch.cat([x1, x2], dim=0)
    elif mode == "shell":
        r = 0.8
        u = torch.randn(n, 3, dtype=torch.float64)
        u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True)
        x = r * u + 0.05 * torch.randn(n, 3, dtype=torch.float64)
    else:
        raise ValueError(f"Unknown mode={mode!r}")

    q = torch.randn(n, dtype=torch.float64)
    return x, q


def collect_l2l_usage_for_backend(backend, meta: Dict) -> None:
    tree = backend.tree
    cfg = backend.cfg
    centers = tree.node_centers
    levels = tree.node_levels
    parents = tree.node_parents
    p = int(cfg.expansion_order)
    scale = float(_get_or_init_global_scale(cfg, tree))

    now = datetime.datetime.utcnow().isoformat() + "Z"
    provenance = {
        "git_commit": get_git_commit(),
        "entrypoint": "LaplaceFmm3D",
        "python": sys.version,
        "platform": sys.platform,
    }

    for idx in range(tree.n_nodes):
        parent = int(parents[idx].item())
        if parent < 0:
            continue
        t_vec = centers[idx] - centers[parent]
        t_norm = float(torch.linalg.vector_norm(t_vec).item())
        record = {
            "experiment_name": "l2l_usage_stats",
            "run_id": str(uuid.uuid4()),
            "timestamp": now,
            "fmm_stage": "L2L",
            "parameters": {
                "p": p,
                "level": int(levels[idx].item()),
                "translation_norm_over_scale": t_norm / scale if scale != 0 else math.nan,
                "translation_norm": t_norm,
                "scale": scale,
                "translation_vec": [float(x) for x in t_vec.tolist()],
            },
            "metrics": {
                "call_weight": 1,
            },
            "provenance": provenance,
            "notes": meta.get("notes"),
            **{k: v for k, v in meta.items() if k != "notes"},
        }
        log_jsonl(record, "l2l_usage_stats.jsonl")


def run_usage_case(n_points: int, mode: str, p: int, theta: float, max_leaf_size: int, seed: int, tag: str) -> None:
    device = torch.device("cpu")
    set_seed(seed)
    x, q = make_random_points(n_points, seed=seed, mode=mode)
    x = x.to(device=device, dtype=torch.float64)
    q = q.to(device=device, dtype=torch.float64)
    areas = torch.ones_like(q)

    backend = make_laplace_fmm_backend(
        src_centroids=x,
        areas=areas,
        max_leaf_size=int(max_leaf_size),
        theta=float(theta),
        expansion_order=int(p),
    )

    meta = {
        "n_points": n_points,
        "mode": mode,
        "theta": theta,
        "max_leaf_size": max_leaf_size,
        "tag": tag,
        "notes": f"L2L usage for n={n_points}, mode={mode}, p={p}, theta={theta}, leaf={max_leaf_size}",
    }
    collect_l2l_usage_for_backend(backend, meta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect L2L usage statistics from FMM trees.")
    parser.add_argument("--n-points", type=int, default=2048)
    parser.add_argument("--mode", type=str, default="uniform")
    parser.add_argument("--expansion-order", type=int, default=8)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--max-leaf-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tag", type=str, default="default")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_usage_case(
        n_points=args.n_points,
        mode=args.mode,
        p=args.expansion_order,
        theta=args.theta,
        max_leaf_size=args.max_leaf_size,
        seed=args.seed,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
