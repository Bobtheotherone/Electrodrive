from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyYAML is required for patch_discovery_config.py") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def _dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyYAML is required for patch_discovery_config.py") from exc
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _set_nested(data: Dict[str, Any], keys: Iterable[str], value: object) -> None:
    cur = data
    keys = list(keys)
    for key in keys[:-1]:
        nxt = cur.get(key, None)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _patch_config(data: Dict[str, Any], *, args: argparse.Namespace) -> Dict[str, Any]:
    patches: Tuple[Tuple[Tuple[str, ...], object], ...] = (
        (("run", "generations"), int(args.generations)),
        (("run", "population_B"), int(args.population_B)),
        (("run", "topK_fast"), int(args.topK_fast)),
        (("run", "topk_mid"), int(args.topk_mid)),
        (("run", "topk_final"), int(args.topk_final)),
        (("paths", "gfn_checkpoint"), str(args.gfn_checkpoint)),
        (("paths", "flow_checkpoint"), str(args.flow_checkpoint)),
        (("run", "layered_sampling"), True),
        (("run", "use_reference_potential"), True),
        (("run", "use_gate_proxies"), True),
        (("run", "layered_prefer_dcim"), True),
        (("run", "layered_allow_real_primitives"), False),
        (("run", "layered_exclusion_radius"), float(args.layered_exclusion_radius)),
        (("run", "layered_interface_delta"), float(args.layered_interface_delta)),
        (("run", "layered_interface_band"), float(args.layered_interface_band)),
        (("run", "layered_stability_delta"), float(args.layered_stability_delta)),
        (("run", "layered_complex_boost", "enabled"), False),
        (("run", "layered_complex_boost", "programs"), 0),
    )
    for keys, value in patches:
        _set_nested(data, keys, value)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch discovery config for Stage 9 throughput tuning.")
    parser.add_argument(
        "--input",
        default="configs/discovery_black_hammer_push.yaml",
        help="Input discovery YAML (default: discovery_black_hammer_push.yaml).",
    )
    parser.add_argument(
        "--output",
        default="configs/stage9/discovery_stage9_push.yaml",
        help="Output patched YAML (default: configs/stage9/discovery_stage9_push.yaml).",
    )
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population-B", dest="population_B", type=int, default=4096)
    parser.add_argument("--topk-fast", dest="topK_fast", type=int, default=4096)
    parser.add_argument("--topk-mid", dest="topk_mid", type=int, default=2048)
    parser.add_argument("--topk-final", dest="topk_final", type=int, default=2048)
    parser.add_argument(
        "--gfn-checkpoint",
        default="artifacts/stage9_gfn_rich/gfn_ckpt.pt",
        help="Path to Stage 9 GFN checkpoint.",
    )
    parser.add_argument(
        "--flow-checkpoint",
        default="artifacts/step10_gfn_flow_smoke/flow_ckpt.pt",
        help="Path to flow checkpoint (override if retrained).",
    )
    parser.add_argument("--layered-exclusion-radius", type=float, default=5e-2)
    parser.add_argument("--layered-interface-delta", type=float, default=1e-2)
    parser.add_argument("--layered-interface-band", type=float, default=1e-2)
    parser.add_argument("--layered-stability-delta", type=float, default=1e-2)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    data = _load_yaml(input_path)
    patched = _patch_config(data, args=args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _dump_yaml(output_path, patched)
    print(f"Patched Stage 9 discovery config written to {output_path}")


if __name__ == "__main__":
    main()
