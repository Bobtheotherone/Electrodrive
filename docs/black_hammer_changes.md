# Black Hammer Change Log

## Phase 0: CUDA tooling + smoke test
- change: added `scripts/print_cuda_env.py` and `tests/test_cuda_smoke.py` for GPU environment sanity checks.
- why: confirm CUDA visibility and basic GPU matmul without relying on full test runs.
- reproduce:
  - `python scripts/print_cuda_env.py`
  - `pytest -q tests/test_cuda_smoke.py -vv -rs --maxfail=1`
- validate: smoke test passes and reports CUDA device capability.

## Phase 1: GFDSL import contract
- change: added `electrodrive/gfdsl/tests/test_import_contract.py` to lock down `electrodrive.gfdsl` imports.
- why: ensure GFDSL is importable from the primary package namespace and prevent collection failures.
- reproduce:
  - `python -c "import electrodrive.gfdsl as g; print('gfdsl ok', g.__file__)"`
  - `pytest -q electrodrive/gfdsl/tests -vv -rs --maxfail=1`
- validate: all GFDSL tests pass and import path resolves to `electrodrive/gfdsl`.

## Phase 2.1: BEM/FMM CUDA matvec residency
- change: allow CUDA `sigma` in `LaplaceFmm3D.matvec` and keep outputs on the input device; add CUDA matvec test.
- why: enforce GPU-first execution without implicit CPU fallback in hot paths.
- reproduce:
  - `pytest -q electrodrive/fmm3d/tests/test_accuracy.py -k "laplace_bem_fmm" -vv -rs --maxfail=1`
- validate: CUDA matvec test passes and returns CUDA output.

## Phase 2.2: Vectorized group-lasso prox
- change: replaced per-group Python loops with vectorized scatter reductions in group prox utilities; added CPU/CUDA parity tests.
- why: remove Python-loop hot paths and keep group operations on GPU.
- reproduce:
  - `pytest -q electrodrive/images/optim/tests/test_group_prox.py -k "group_prox" -vv -rs --maxfail=1`
- validate: group prox tests pass on CPU and CUDA.

## Prompt 2 updates
- change: batched ImageSystem point-charge evaluation; removed explicit .cpu() scalar transfer in global_search hot loop.
- change: added green_checks verification utilities and GreenDecomposition helper; added gated boundary/PDE penalty hooks in outer solve config.
- change: added multi-fidelity ladder module and distillation CLI for template clustering.
- reproduce (distill): `python scripts/black_hammer/distill_templates.py <input_dir> <output_dir>`
- reproduce (verification checks): `pytest -q electrodrive/verify/tests/test_green_checks.py -vv -rs --maxfail=1`

## Phase 3: Discovery preflight + gate-ready template
- change: added discovery preflight counters/report (`preflight.json`) to explain candidate throughput and drop reasons.
- change: added `configs/discovery_black_hammer_gate_ready.yaml` as verifier-aligned template for large runs.
- reproduce: run discovery with `run.preflight_enabled: true` and inspect `<RUN_DIR>/preflight.json`.
- validate: report includes counters (compiled_ok, solved_ok, fast_scored, verified_written) and run metadata.

## Phase 3.1: Preflight modes + push template
- change: added `run.preflight_mode` ("off" | "lite" | "full") to control preflight overhead.
- change: added `configs/discovery_black_hammer_push.yaml` with `preflight_mode: lite` for large runs.
- guidance: use gate-ready (full) for pilots/debug; use push (lite) for scale.

## Phase 3.2: Gate A proxy prioritization + stability mix
- change: added `proxy_gateA` (Laplacian residual proxy) aligned to Gate A sampling/thresholds; integrated into discovery proxy ranking.
- change: added `run.use_gateA_proxy` (default on when proxies enabled) and mixed stability proxy points across boundary + interior.
- change: added `build_proxy_stability_points` helper and tests for proxy mixing and Gate A proxy behavior.
- why: surface harmonicity failures earlier and avoid spending verifier budget on A-hopeless candidates; reduce stability proxy bias.
- reproduce:
  - `pytest -q tests/test_gate_proxies.py -vv -rs --maxfail=1`
  - `pytest -q tests/test_discovery_proxy_points.py -vv -rs --maxfail=1`
  - `pytest -q tests/test_fast_weights_stability.py -vv -rs --maxfail=1`
  - `pytest -q tests/test_reference_decomposition.py -vv -rs --maxfail=1`

## Phase 4: Gate A proxy alignment + balanced ranking
- change: aligned `proxy_gateA` autograd/FD sampling with verifier Gate A and added method/point-count diagnostics.
- change: added bounded Gate A proxy transform (`run.proxyA_transform`, `run.proxyA_cap`, `run.proxyA_weight`) and balanced proxy ranking (`run.proxy_ranking_mode`).
- why: prevent proxy_gateA saturation from dominating ranking and restore B/C/D frontier search before optimizing Gate A.
- reproduce:
  - `pytest -q tests/test_gate_proxies.py -vv -rs --maxfail=1`
  - `pytest -q tests/test_proxyA_transform.py -vv -rs --maxfail=1`
- validate: proxy Gate A diagnostics report method/n_used, transform remains finite for extreme ratios, and balanced mode prefers candidates that pass B/C/D.

## Phase 4.1: Gate A precision + stencil-safe sampling
- change: Gate A Laplacian runs in float64, autograd handles constant-gradient second derivatives, and FD sampling resamples until stencil-safe counts are met.
- change: point-charge complex evaluation respects input dtype (float32→complex64, float64→complex128) in ImageSystem/PointCharge/DCIM paths.
- why: eliminate NaN/Inf blowups from autograd/FD cancellation and prevent dtype downcasts from corrupting Gate A metrics.
- reproduce:
  - `pytest -q tests/test_verifier_gates.py -k "gate_a_pde_autograd_constant_linear_pass or gate_a_fd_precision_improves" -vv -rs --maxfail=1`
  - `pytest -q tests/test_imagesystem_dtype.py -vv -rs --maxfail=1`
  - `pytest -q tests/test_gate_proxies.py -vv -rs --maxfail=1`
- validate: Gate A passes for linear/constant harmonic functions, FD precision improves in float64, and float64 candidate_eval remains differentiable.
