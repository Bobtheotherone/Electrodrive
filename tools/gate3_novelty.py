from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict
import json

import torch

from electrodrive.discovery.novelty import update_manifest_with_novelty
from electrodrive.images.io import load_image_system
from electrodrive.orchestration.parser import CanonicalSpec


def _find_repo_root(start: Path) -> Path:
    """Walk upward to locate repo root (identified by .git)."""
    for p in [start, *start.parents]:
        if (p / ".git").exists():
            return p
    return start


def _load_manifest(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _resolve_spec_path(spec_arg: str | None, manifest: Dict[str, Any], run_path: Path) -> Path:
    """Resolve the spec path robustly using CLI arg or manifest spec_path."""
    candidates = []
    if spec_arg:
        candidates.append(Path(spec_arg))
    manifest_spec = manifest.get("spec_path")
    if manifest_spec:
        candidates.append(Path(manifest_spec))

    repo_root = _find_repo_root(Path(__file__).resolve())
    for cand in candidates:
        if cand.is_absolute() and cand.exists():
            return cand
        if cand.exists():
            return cand
        alt = repo_root / cand
        if alt.exists():
            return alt
    if not candidates:
        raise FileNotFoundError("No spec path provided and manifest missing spec_path.")
    raise FileNotFoundError(f"Could not resolve spec path from candidates: {candidates}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate 3 novelty updater for discovered image systems.")
    ap.add_argument("--run", required=True, help="Path to discovered_system.json")
    ap.add_argument(
        "--spec",
        required=False,
        default=None,
        help="Path to CanonicalSpec JSON (defaults to manifest spec_path when omitted)",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest path to update (defaults to run_dir/discovery_manifest.json)",
    )
    ap.add_argument("--device", default="cpu", help="Device for loading system (default: cpu)")
    ap.add_argument("--dtype", default="float32", help="Torch dtype for loading system (default: float32)")
    args = ap.parse_args()

    run_path = Path(args.run)
    manifest_path = args.manifest or (run_path.parent / "discovery_manifest.json")
    manifest = _load_manifest(manifest_path)
    spec_path = _resolve_spec_path(args.spec, manifest, run_path)

    dtype = getattr(torch, args.dtype)
    system = load_image_system(run_path, device=args.device, dtype=dtype)
    spec = CanonicalSpec.from_json(json.loads(Path(spec_path).read_text(encoding="utf-8")))

    novelty, gate3_status = update_manifest_with_novelty(system, spec, manifest_path)
    print(json.dumps({"novelty_score": novelty, "gate3_status": gate3_status}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
