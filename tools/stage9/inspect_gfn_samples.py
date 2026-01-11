from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch

from electrodrive.flows.schemas import SCHEMA_COMPLEX_DEPTH
from electrodrive.gfn.dsl import AddBranchCutBlock, AddPoleBlock, AddPrimitiveBlock, StopProgram
from electrodrive.gfn.integration import GFlowNetProgramGenerator
from electrodrive.gfn.integration.gfn_basis_generator import _spec_metadata_from_spec
from electrodrive.gfn.rollout import SpecBatchItem, rollout_on_policy
from electrodrive.images.learned_generator import SimpleGeoEncoder
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.device import get_default_device


def _default_layered_spec() -> CanonicalSpec:
    return CanonicalSpec.from_json(
        {
            "domain": {"bbox": [[-1, -1, -2], [1, 1, 2]]},
            "dielectrics": [
                {"name": "region1", "epsilon": 1.0, "z_min": 0.5, "z_max": 2.0},
                {"name": "slab", "epsilon": 4.0, "z_min": 0.0, "z_max": 0.5},
                {"name": "region3", "epsilon": 1.0, "z_min": -2.0, "z_max": 0.0},
            ],
            "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.2]}],
            "BCs": "dielectric_interfaces",
        }
    )


def _load_spec(path: Path | None) -> CanonicalSpec:
    if path is None:
        return _default_layered_spec()
    raw = path.read_text(encoding="utf-8")
    data: Dict[str, Any]
    if path.suffix.lower() in {".json"}:
        import json

        data = json.loads(raw)
    else:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyYAML is required for non-JSON specs.") from exc
        data = yaml.safe_load(raw) or {}
    spec_payload = data.get("spec", data)
    return CanonicalSpec.from_json(spec_payload)


def _program_length(program: Any) -> int:
    nodes = getattr(program, "nodes", []) or []
    return int(sum(1 for node in nodes if not isinstance(node, StopProgram)))


def _signature(program: Any, generator: GFlowNetProgramGenerator) -> str:
    tokens = []
    for node in getattr(program, "nodes", []) or []:
        if isinstance(node, StopProgram):
            continue
        token = generator.grammar.action_to_token(node)
        tokens.append(str(token))
    return "-".join(tokens) if tokens else "empty"


def _sample_programs(
    generator: GFlowNetProgramGenerator,
    spec: CanonicalSpec,
    *,
    n_programs: int,
    batch_size: int,
    seed: int,
) -> Iterable[Any]:
    encoder = SimpleGeoEncoder(latent_dim=generator.policy.config.spec_dim, hidden_dim=32)
    spec_embedding, _, _ = encoder.encode(spec, device=generator.device, dtype=torch.float32)
    spec_meta = _spec_metadata_from_spec(spec, extra_overrides={"allow_real_primitives": False})
    collected = 0
    attempt = 0
    while collected < n_programs:
        batch_n = min(batch_size, n_programs - collected)
        spec_batch = [
            SpecBatchItem(
                spec=spec,
                spec_meta=spec_meta,
                spec_embedding=spec_embedding,
                seed=seed + attempt + idx,
            )
            for idx in range(batch_n)
        ]
        rollout = rollout_on_policy(
            generator.env,
            generator.policy,
            spec_batch,
            max_steps=generator.env.max_length,
            generator=torch.Generator(device=generator.device).manual_seed(seed + attempt),
        )
        for state in rollout.final_states or []:
            yield state.program
            collected += 1
            if collected >= n_programs:
                break
        attempt += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect GFN checkpoint samples for DCIM/complex coverage.")
    parser.add_argument("--checkpoint", required=True, help="Path to GFN checkpoint.")
    parser.add_argument("--spec", help="Path to spec JSON/YAML (defaults to layered spec).")
    parser.add_argument("--n-programs", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    device = get_default_device()
    generator = GFlowNetProgramGenerator(
        checkpoint_path=args.checkpoint,
        device=device,
        debug_keep_states=False,
    )
    spec = _load_spec(Path(args.spec)) if args.spec else _load_spec(None)

    length_hist: Counter[int] = Counter()
    sig_hist: Counter[str] = Counter()
    dcim_count = 0
    complex_count = 0
    total = 0

    for program in _sample_programs(
        generator,
        spec,
        n_programs=int(args.n_programs),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    ):
        total += 1
        length_hist[_program_length(program)] += 1
        sig_hist[_signature(program, generator)] += 1
        if any(isinstance(node, (AddPoleBlock, AddBranchCutBlock)) for node in program.nodes):
            dcim_count += 1
        if any(
            isinstance(node, AddPrimitiveBlock) and int(node.schema_id or 0) == SCHEMA_COMPLEX_DEPTH
            for node in program.nodes
        ):
            complex_count += 1

    frac_dcim = dcim_count / max(1, total)
    frac_complex = complex_count / max(1, total)

    print(f"sampled_programs={total}")
    print(f"fraction_dcim={frac_dcim:.3f}")
    print(f"fraction_complex={frac_complex:.3f}")
    print(f"length_hist={dict(sorted(length_hist.items()))}")
    print("top_signatures:")
    for sig, count in sig_hist.most_common(10):
        print(f"  {sig}: {count}")


if __name__ == "__main__":
    main()
