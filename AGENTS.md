

## 2) Updated AGENTS.md (GPU-first + discovery workflow + ramp + refinement)

````markdown
# AGENTS.md — Electrodrive (GPU-First Discovery + Repo Health)

This document defines how humans and AI agents must work on Electrodrive.
Primary objectives:
1) Run **GPU-first** discovery experiments safely and reproducibly.
2) Preserve **repo health** (tests, APIs, CLIs).
3) Enable **credible discovery** (verified metrics, not just “it ran”).

---

## 0) Non-negotiables

### 0.1 GPU-first doctrine
- All heavy compute must be on **CUDA**.
- If `torch.cuda.is_available()` is false: **STOP**. Do not “try CPU anyway.”
- Forbidden in hot paths:
  - large CPU numpy loops
  - `.cpu()` / `.numpy()` on big tensors
  - CPU fallback for flows / gfn_flow

### 0.2 Flows require CUDA
- The flows subsystem enforces CUDA (device_guard.ensure_cuda).
- Any path that constructs flow-based generators must pass `device="cuda"` when CUDA is available.
- If environment variables request CPU but flow mode is active, code must override to CUDA or fail loudly.

### 0.3 Repo health never declines
- Do not break existing CLIs/APIs.
- Prefer additive changes behind flags/config.
- Tests should continue to pass (skips are allowed only for optional deps).

---

## 1) Environment conventions

- Use `python3` explicitly (do not assume `python` exists).
- Prefer a venv: `.venv/bin/python`.
- Always confirm GPU is visible:

```bash
python3 - <<'PY'
import torch
print(torch.__version__)
print("cuda:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print("gpu:", torch.cuda.get_device_name(0))
PY
nvidia-smi
````

---

## 2) Discovery runner contract

Canonical entrypoint:

```bash
python3 -m electrodrive.experiments.run_discovery --config <yaml>
```

All runs must produce:

```
runs/<timestamp>_<tag>/
  config.yaml
  env.json
  git.json
  metrics.jsonl
  best.jsonl
  artifacts/certificates/
```

---

## 3) Required metrics for credibility

Every best candidate must log at minimum:

* Absolute:

  * `mean_bc_err_holdout`, `max_bc_err_holdout`
  * `mean_pde_err_holdout`, `max_pde_err_holdout` (can be laplacian proxy)
* Relative:

  * `rel_bc_err_holdout`
  * `rel_lap_holdout` (or `rel_pde_err_holdout` if laplacian not available)
* Denominators:

  * `oracle_bc_mean_abs_holdout`
  * `oracle_in_mean_abs_holdout`
* Complexity / structure:

  * `n_terms`, `complex_count`
* Timing:

  * `eval_time_us`, `solve_time_us`, `total_time_us`

If relative errors are not logged, discovery claims are not credible.

---

## 4) Ramp protocol (monster-run readiness gate)

### 4.1 Ramp config purpose

Ramp is a “cheap” run to determine whether scaling is worth it.
Ramp must:

* be deterministic or controlled (see 4.2)
* abort if not improving
* produce a readiness report

### 4.2 Deterministic ramp

To avoid noise from changing specs:

* Ramp should support `run.fixed_spec: true`
* If enabled, the same physical spec is reused for all generations.
* Log `spec_hash` per generation to prove it is constant.

### 4.3 Ramp pass criteria

A run is “READY for monster” only if:

* no ramp abort triggered
* rel metrics improve by configured thresholds
* final `rel_bc_err_holdout < 1e-3` (or a clearly justified alternate)
* empty compilation fraction is low
* element typing indicates intended basis (DCIM elements present for layered runs)

---

## 5) Layered media (DCIM-like) rules

For `spec.BCs == "dielectric_interfaces"`:

### 5.1 Translation invariance

* Image terms should share the source `(x,y)` in layered planar geometry.
* Refinement must **not** change x/y for layered problems.

### 5.2 Complex images

* `z_imag` must be nonzero for DCIM pole/branch-cut images.
* Guardrails should prevent “all z_imag == 0” populations in layered runs.

### 5.3 Element identity must be preserved

* `DCIMPoleImageBasis` and `DCIMBranchCutImageBasis` must stay distinct.
* Serialization/deserialization must keep their types.

---

## 6) Refinement policy (if enabled)

Refinement is allowed only if:

* GPU-only
* config-gated (`run.refine_enabled`)
* bounded compute (few steps, few candidates)
* preserves invariants:

  * layered x/y fixed
  * z_imag clamped to >0 minimum
  * all params remain CUDA tensors

Refinement must update:

* cand["elements"], cand["weights"], cand["metrics"], and cand["score"]
  atomically when it accepts an improvement.

---

## 7) Testing policy

* Avoid full test suite unless requested.

* Always run targeted tests when fixing regressions:

  * step10 GPU device smoke:

    ```bash
    pytest -q -vv --maxfail=1 tests/test_step10_integration_e2e.py::test_discover_images_gfn_flow_smoke
    ```

* If optional deps are missing (pykeops, xitorch), tests should skip cleanly.

---

## 8) Performance guidelines

* No `torch.cuda.synchronize()` inside inner loops (only around timing).
* Use microbatch knobs where implemented (`run.score_microbatch`).
* Cache expensive intermediates for topK only (e.g., holdout matrices).
* Prefer BF16/FP16 for proposal nets; keep verification in FP32.

---

## 9) “Monster run” definition and when to do it

A monster run means:

* population_B in the thousands (4096–8192)
* hundreds to thousands of generations
* large point counts

Do **not** run a monster configuration until:

* ramp is passing (no abort)
* relative errors are trending down
* repo health tests for critical paths are green

---


