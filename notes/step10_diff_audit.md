# Step 10 diff audit

## Touched files (this run)
- electrodrive/flows/device_guard.py: add EDE_FLOW_COMPILE gating helper.
- electrodrive/flows/sampler.py: vectorized schema dim masks, batched AST conditioning reuse, per-program seed handling for batched sampling.
- electrodrive/flows/types.py: allow batched seed typing for ParamSampler.
- electrodrive/gfn/integration/gfn_flow_generator.py: optional torch.compile toggle via EDE_FLOW_COMPILE.
- electrodrive/gfn/reward/reward.py: add compute_batch, accept precomputed ParamPayloads, harden compile cache key + CPU cache storage.
- electrodrive/gfn/train/train_gfn.py: use compute_batch when param_sampler is enabled.
- tests/test_step10_backward_compat.py: instantiate legacy diffusion + gfn generators.
- docs/step10_gfn_flow.md: gfn_flow runbook and debug notes.

## Scope check
- Changes are confined to Step-10 flow/gfn integration, sampler performance, and documentation/testing.
- No defaults flipped; gfn_flow remains opt-in.

## Out-of-scope touches
- None in legacy/non-step10 modules.

## New artifacts
- runs/step10_gfn_flow_smoke/run_1766285538 (failed smoke run)
- runs/step10_gfn_flow_smoke/run_1766285577 (successful smoke run)
