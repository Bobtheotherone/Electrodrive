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

## Phase 4.2: Nonfinite weight rejection + finite proxy fails
- change: added weight validation guards before mid/refine/verification; reject nonfinite weights and record reasons in preflight.
- change: proxy_gateA failure payloads now keep `proxy_gateA_worst_ratio` finite (1e12).
- change: added tests covering nonfinite weight rejection and summary serialization.
- reproduce:
  - `pytest -q tests/test_gate_proxies.py -vv -rs --maxfail=1`
  - `pytest -q tests/test_weights_validation.py -vv -rs --maxfail=1`
- validate: nonfinite weights are rejected before verification, summaries never contain non-numeric weights, and proxy failures remain finite.

## Phase 4.3: Holdout metric finiteness contract
- change: enforce finite holdout/interior/laplacian metrics in discovery scoring; nonfinite candidates receive finite penalties and are excluded from best/ramp tracking; DCIM baseline cannot be chosen if nonfinite.
- change: added preflight counters for holdout nonfinite and DCIM baseline nonfinite plus holdout reject reasons.
- why: prevent NaN/Inf metrics from corrupting readiness/refine selection and console summaries.
- reproduce:
  - `pytest -q tests/test_holdout_metric_finiteness.py -vv -rs --maxfail=1`
- validate: pilot run shows no `score=nan used_as_best=True` or `best_in_abs=inf` and preflight includes holdout counters.

## Phase 4.4: Holdout partition + denom stability
- change: compute oracle holdout mean-abs denominators in float64; use float64 fallback for holdout mean errors to avoid overflow; track holdout partition sizes and denom stats in preflight; resample interior holdout if filtered empty and flag empty partitions.
- why: prevent denom_in/lap_denom overflow to inf and keep holdout metrics meaningful during discovery scoring.
- reproduce:
  - `pytest -q tests/test_holdout_partitioning.py -vv -rs --maxfail=1`
- validate: pilot run logs finite denom values, holdout_interior_empty_count stays near 0, and DCIM baseline is either finite or reported unavailable.

## Phase 4.5: Proxy finiteness contract (Gate B/D + proxy score)
- change: proxy_gateB/proxy_gateD now compute stability metrics in float64 and sanitize nonfinite inputs to large finite penalties; emit `proxy_gateB_nonfinite`/`proxy_gateD_nonfinite` flags.
- change: proxy_score sanitizes nonfinite proxy inputs to a finite penalty and records `proxy_score_nonfinite_sanitized` in preflight.
- change: verifier Gate D stability metrics now use float64, clamp denominators, and coerce nonfinite outputs to finite fail values.
- why: prevent float32 overflow (e.g., 1e35 scale → variance inf) from propagating NaN/Infinity proxy metrics into summaries during push runs.
- reproduce:
  - `pytest -q tests/test_gate_proxies.py -vv -rs --maxfail=1`
  - `pytest -q tests/test_proxyA_transform.py -vv -rs --maxfail=1`
- validate: proxy Gate B/D metrics and proxy_score remain finite under extreme magnitudes and large runs no longer emit NaN/Infinity proxy fields.

## Phase 5: Complex/DCIM boost + fast proxy alignment
- change: added layered complex/DCIM boost generator with multi-block DCIM sampling, depth clustering, and interface exclusion; wired via `run.layered_complex_boost` and `run.dcim_diversity`.
- change: added fast proxy metrics (far-field ratio, interface jump, condition ratio, speed proxy) to fast scoring; optional hard rejection for pathological candidates.
- change: preflight now records complex/DCIM fractions plus histograms for pole/block counts, max |Im(z)|, max |weights|, and condition ratio; includes baseline backend name.
- why: expand expressive complex/DCIM search space, align fast scoring with gates B-E, and diagnose DCIM/complex coverage in push runs.
- reproduce:
  - `pytest -q tests/test_layered_complex_candidates_determinism.py -vv -rs --maxfail=1`
  - `pytest -q tests/test_fast_proxy_metrics.py -vv -rs --maxfail=1`
  - `pytest -q tests/test_complex_pair_real_cuda.py -vv -rs --maxfail=1`
  - `pytest -q tests/test_dcim_block_eval_shape.py -vv -rs --maxfail=1`
- validate: preflight reports nontrivial complex/DCIM fractions, fast proxy metrics are finite, and CUDA-only tests pass.

## Phase 6: GFN expressivity audit (root causes)
- grammar defaults enumerate only baseline/connector/pade with single interface_id=0 and fixed pole/branch budgets, so the discrete action space is effectively 1-choice per block.
- grammar schema_ids are empty by default, so add_primitive emits no schema_id and cannot represent complex-depth primitives when real primitives are disallowed.
- branch-cut enumeration only uses approx_types[0], collapsing discrete branch types into a single token.
- action masking ignores spec_meta.n_dielectrics, so interface_id actions are not filtered for layered specs with fewer interfaces.
- tokenization collapses all nodes to type-only tokens (add_pole/add_branch_cut/etc.), losing interface_id/schema_id/budget/family distinctions.
- GFN checkpoints load only families/motifs/approx_types/budgets; schema_ids and expanded grammar choices are dropped on load.
- action_vocab/token mapping is not persisted, so tokenization cannot stay stable across checkpoint reloads.

## Phase 6.1: GFN generator expressivity fixes
- change: expanded Grammar with interface/budget/schema choices, interface-aware masking, and conjugate ref options to unlock rich DCIM/complex action space.
- change: tokenization now maps discrete action args (interface_id/schema_id/budget/etc.) via grammar action vocab; policy token embeddings sized from vocab.
- change: checkpoints persist full grammar fields plus action_vocab mapping; loader restores schema ids and warns when vocab is missing.
- change: complex-only primitives supported end-to-end (env gating + schema_id metadata + compile path); group_info now records interface_id/schema_id/n_poles/budget/approx_type.
- change: structural fingerprint/novelty includes DCIM families, complex-depth families, interface/schema stats, and DCIM arg summaries.
- change: added `electrodrive/gfn/train/run_train.py` minimal training entrypoint and CUDA smoke test for rich program sampling.
- reproduce:
  - `pytest -q tests/test_grammar_rich_action_space.py -vv -rs --maxfail=1`
  - `pytest -q tests/test_tokenize_distinguishes_discrete_args.py -vv -rs --maxfail=1`
  - `pytest -q tests/test_checkpoint_grammar_roundtrip.py -vv -rs --maxfail=1`
  - `pytest -q tests/test_complex_only_primitives_allowed.py -vv -rs --maxfail=1`
  - `pytest -q tests/test_structural_features_dcim_complex.py -vv -rs --maxfail=1`
  - `pytest -q tests/test_train_and_sample_rich_programs_smoke.py -vv -rs --maxfail=1`
- train (example): `python -m electrodrive.gfn.train.run_train --config <yaml>`

## Phase 9: Gate-proxy reward + Stage 9 tooling
- change: added gate-proxy reward computer for GFN training using Gate A-D proxies, speed proxy, and DCIM/complex bonuses with finite clamps.
- change: `run_train.py` now supports `reward.type: gate_proxy` plus Stage 9 rich-grammar training config.
- change: added Stage 9 discovery patch script, GFN sample inspector, and verifier analysis script (histograms + near-miss report).
- reproduce:
  - `pytest -q tests/test_gate_proxy_reward_finite.py -vv -rs --maxfail=1`
  - `pytest -q tests/test_gate_proxy_reward_prefers_better_proxy.py -vv -rs --maxfail=1`
  - `python -m electrodrive.gfn.train.run_train --config configs/stage9/train_gfn_rich_gate_proxy.yaml`
  - `python tools/stage9/patch_discovery_config.py --input configs/discovery_black_hammer_push.yaml --output configs/stage9/discovery_stage9_push.yaml`
  - `python tools/stage9/analyze_verifier_results.py <RUN_DIR>`

## Phase 9.1: Align Stage 9 spec_dim with flow checkpoint
- change: set `spec_dim: 32` and moved Stage 9 rich GFN checkpoints to `artifacts/stage9_gfn_rich_spec32/` to match the flow checkpoint spec embedding width.
- why: prevent flow parameter sampler shape mismatches during Stage 9 discovery with gfn_flow.
- reproduce:
  - `python -m electrodrive.gfn.train.run_train --config configs/stage9/train_gfn_rich_gate_proxy.yaml`
  - `python tools/stage9/inspect_gfn_samples.py --checkpoint artifacts/stage9_gfn_rich_spec32/gfn_ckpt.pt --n-programs 512 --batch-size 32 --seed 123`

## Phase 9.2: Stabilize flow sampling for Stage 9 pilot/push
- change: reduced Stage 9 flow temperature to 0.5, set flow sampling dtype to fp32, and added `latent_clip: 8.0` for flow latents in Stage 9 discovery configs.
- why: mitigate nonfinite weights observed during pilot discovery (preflight_first_offender: weights_nonfinite).
- reproduce:
  - `python -m electrodrive.experiments.run_discovery --config configs/stage9/discovery_stage9_pilot.yaml`

## Phase 9.3: Clamp complex-depth imag range for flow schemas
- change: clamp complex-depth imag scale to `[1e-3, 8.0]` in `ComplexDepthPointSchema`.
- why: avoid near-singular complex images that overflow scoring metrics during Stage 9 discovery.
- reproduce:
  - `pytest -q tests/test_complex_depth_schema_clamp.py -vv -rs --maxfail=1`
