# Step 10: gfn_flow (GFlowNet + Flow)

## What gfn_flow does
- Uses a GFlowNet policy to sample discrete program structure (grammar tokens).
- Uses a ParamFlowNet to sample continuous parameters for parametric nodes.
- Compiles program + parameters into basis elements for the solver stack.

## Required checkpoints
- GFlowNet checkpoint (`--gfn-checkpoint`) is mandatory.
- Flow checkpoint (`--flow-checkpoint`) is mandatory unless `--allow-random-flow` is set.
- Flow checkpoint must include model state + model_config (and optional sampler_config).

## Example CLI commands
Minimal run:
```bash
python -m electrodrive.tools.images_discover discover \
  --spec specs/plane_point.json \
  --basis point \
  --nmax 1 \
  --reg-l1 1e-3 \
  --basis-generator gfn_flow \
  --basis-generator-mode gfn_flow \
  --gfn-checkpoint /path/to/gfn.pt \
  --flow-checkpoint /path/to/flow.pt \
  --flow-steps 2 \
  --flow-solver euler \
  --flow-temp 1.0 \
  --flow-dtype fp32 \
  --out runs/step10_gfn_flow/demo
```

With explicit seeds:
```bash
python -m electrodrive.tools.images_discover discover \
  --spec specs/plane_point.json \
  --basis point \
  --nmax 1 \
  --reg-l1 1e-3 \
  --basis-generator gfn_flow \
  --basis-generator-mode gfn_flow \
  --gfn-checkpoint /path/to/gfn.pt \
  --flow-checkpoint /path/to/flow.pt \
  --gfn-seed 123 \
  --flow-seed 456 \
  --out runs/step10_gfn_flow/seeded
```

## GPU-first policy
- gfn_flow requires CUDA; it does not fall back to CPU.
- Set `EDE_DEVICE=cuda` (and optionally `EDE_DTYPE=float32`).
- If CUDA is unavailable, gfn_flow will raise immediately.

## Debugging
- Determinism / seeds:
  - `--gfn-seed` controls program sampling.
  - `--flow-seed` controls parameter sampling.
  - Batched flow sampling is deterministic per program seed; identical seeds yield identical latent samples.
- Cache invalidation:
  - Reward compile cache keys include program hash + flow config + seed + checkpoint id.
  - If you change flow config/checkpoint, restart the process or clear caches.
- Strict payload mode:
  - Set `EDE_STRICT_PARAM_PAYLOAD=1` to require ParamPayload during compilation.
  - This catches missing flow sampling or accidental CPU paths early.
