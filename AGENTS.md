# AGENTS.md — Operation Black Hammer (Discovery Push Edition)

This repository is an active research codebase for **Operation Black Hammer**: discovering **new analytical Green’s functions** via a **method-of-images / programmatic AI** approach. Codex must keep the repo **healthy, reproducible, GPU-first, and verifier-aligned**.

This file is the single source of truth for how Codex should operate during any discovery push.

---

## 1) Non-negotiables

### 1.1 GPU-first doctrine
Target machine:
- ROG Zephyrus G16 GU605CX_GU605CX
- Intel Core Ultra 9 285H
- NVIDIA Blackwell RTX 5090 (24 GB VRAM)
- 32 GB RAM

Rules:
1. **GPU is primary.** Prefer GPU implementations even if it requires extra engineering.
2. **No implicit CPU fallbacks in hot paths.** `.cpu()`, `.numpy()`, Python loops over tensors, CPU-only kernels in critical loops are treated as defects unless explicitly justified.
3. Mixed precision policy:
   - Proposal/generative models: BF16/FP16 allowed
   - Solver/verification: FP32 default
   - Certification / numerically sensitive transforms: FP64 where required

### 1.2 Do not weaken gates
The verifier gates (A–D and beyond) are the target. **Do not reduce gate strictness**, thresholds, or sample sizes to “get a pass.” Fix methodology/representation/sampling instead.

### 1.3 Always operate in the venv
Every shell session begins with:
```bash
source .venv/bin/activate
````

---

## 2) Repo health contract

### 2.1 What “repo health” means

Repo health includes:

* No new test failures (relative to baseline)
* No broken imports / packaging
* No silent numerical meaning changes without tests + docs
* No new CPU/GPU transfer regressions in core loops
* Discovery runs must be diagnosable via **preflight.json** (see §5)

### 2.2 Clean working tree requirement

Before and after any commit, and before/after any run:

```bash
git status --porcelain
```

If non-empty:

1. If the only entries are **obvious artifacts** (e.g., `runs/**`, `__pycache__/`, `*.pyc`, `.DS_Store`, `.ipynb_checkpoints/`, logs): **note them once and continue.**
2. Otherwise, **checkpoint automatically**, then continue from a clean tree:

   ```bash
   git stash push -u -m "checkpoint: dirty tree before <next action>"
   git status --porcelain  # must be empty
   ```

### 2.3 Full pytest command (canonical)

Run full pytest only when explicitly instructed or immediately before merging a major change:

```bash
source .venv/bin/activate
pytest --ignore=staging --ignore=electrodrive/fmm3d/tests/test_kernels_gpu_stress.py --ignore=temp_AI_upgrade -vv -rs -q --maxfail=1
```

Otherwise, use targeted tests only.

---

## 3) Safety system: emergency stop & no-thrashing

### 3.1 Absolute prohibitions

* **No `rm -rf` anywhere** (including `runs/`).
* Do not delete/overwrite run artifacts. If consolidating, write to a new directory.
* Do not mass-reformat or rename unrelated files.
* Do not “fix” failures by disabling tests unless explicitly instructed.

### 3.2 Emergency stop conditions

Trigger EMERGENCY STOP if:

* New widespread failures appear and root cause is unclear
* You enter an edit→fail→edit loop without clear progress
* You suspect you’ve corrupted core infrastructure (imports, verifier, solver, device placement)

Emergency stop procedure:

1. Revert protected code:

   ```bash
   git reset --hard HEAD
   git clean -fd
   ```
2. Write an incident report:

   * `.blackhammer/emergency_stop_report.md`
   * include what changed, what broke, and why you reverted

After EMERGENCY STOP, you may continue only by adding **new experimental code** in a new namespace until instructed otherwise.

---

## 4) Change discipline (how Codex edits)

### 4.1 Small commits only

* Make surgical edits.
* Commit in small, reviewable chunks.
* After each commit: run targeted tests relevant to the change.

### 4.2 No behavioral changes without tests

If you change any of:

* scoring / ranking behavior
* sampling distributions
* solver behavior
* verification wiring
  you must add/adjust tests and update docs.

### 4.3 Determinism & reproducibility

* Seeds must be recorded in run directories.
* Runs must record:

  * git SHA
  * config (as executed)
  * device info
  * preflight.json (when enabled)

---

## 5) Discovery runs: required instrumentation

### 5.1 Preflight is mandatory for pushes

Discovery runs must use `preflight_mode`:

* `off`: no counters
* `lite`: low overhead (for scale)
* `full`: heavy diagnostics (for debugging/pilots)

**Push runs default to `lite`**. Debug/pilot runs may use `full`.

Artifacts:

* `<RUN_DIR>/preflight.json` must exist when preflight is enabled.
* If failures occur, `<RUN_DIR>/preflight_first_offender.json` may be written (once per run).

### 5.2 Interpreting preflight.json (minimum signals)

A healthy run typically has:

* `compiled_ok / sampled_programs_total` not tiny (often >0.5)
* `solved_ok > 0`
* `fast_scored > 0` every generation
* `verified_written > 0` at least once per short run and frequently in longer runs
* `nonfinite_pred_fraction ≈ 0`
* `fraction_dcim_candidates` not tiny (push runs should show substantial DCIM presence)
* `fraction_complex_candidates` not tiny (push runs should show substantial complex-image presence)
* `baseline_speed_backend_name` (or equivalent) present and meaningful

If these are violated, treat it as an operational or algorithmic diagnosis task:

* Do not guess. Use preflight counters + first offender snapshot.

### 5.3 DCIM/complex push sanity (mandatory for layered pushes)

Preflight must include:

* `fraction_dcim_candidates`
* `fraction_complex_candidates`
* `dcim_block_count_hist`
* `dcim_pole_count_hist`
* `max_abs_imag_depth_hist`
* `weight_magnitude_hist` (or `max_abs_weight_hist`)
* `baseline_speed_backend_name` (or equivalent)

Operational rule:

* If `fraction_dcim_candidates < 0.30` OR `fraction_complex_candidates < 0.30` for >3 consecutive generations in a push run,
  treat as misconfiguration/bug and stop to fix representation emission (do NOT “just run longer”).

---

## 6) Gate-ready operational checklist (must be satisfied before large pushes)

For layered dielectric discovery targeting strict verifier A–D, configs should enable:

* `run.layered_sampling: true`
* `run.use_reference_potential: true`
* `run.use_gate_proxies: true`
* `run.layered_prefer_dcim: true`
* `run.layered_allow_real_primitives: false`
* `run.layered_exclusion_radius` aligned to verifier exclusion (~5e-2 typical)
* `run.layered_interface_delta` aligned to verifier delta (~1e-2 typical)
* `run.layered_interface_band` aligned to interface delta (~1e-2 typical)
* `run.layered_stability_delta: 1e-2` (perturbation magnitude)
* `solver.fast_column_normalize: true`
* `run.allow_not_ready: true` (avoid early termination during exploration)
* Confirm speed baseline is real and distinct (avoid baseline==candidate situations that force speedup≈1); record the baseline backend name in preflight.

**Do not edit verifier thresholds to make passing easier.**

---

## 7) Canonical configs

Repository provides templates:

* Debug/pilot (more diagnostics): `configs/discovery_black_hammer_gate_ready.yaml`
* Scale/push (low overhead): `configs/discovery_black_hammer_push.yaml`

Use these as baselines. Do not fork ad-hoc configs without recording why.

---

## 8) Performance rules for push readiness

### 8.1 Avoid CPU sync in hot loops

Inside candidate loops:

* avoid `.item()` calls unless necessary
* batch operations
* avoid printing per-candidate logs

### 8.2 Stable math first

* Use column normalization in fast solve when enabled.
* Reject nonfinite candidates early; do not allow NaN scores into ranking.

---

## 9) Required documentation updates

Any change that affects discovery behavior must be logged in:

* `docs/black_hammer_changes.md`

Include:

* what changed
* why it changed
* how to validate (tests + minimal run recipe)

---

## 10) Command reference

Activate venv:

```bash
source .venv/bin/activate
```

Targeted tests (example patterns):

```bash
pytest -q tests/test_fast_weights_stability.py -vv -rs --maxfail=1
pytest -q tests/test_gate_proxies.py -vv -rs --maxfail=1
pytest -q tests/test_reference_decomposition.py -vv -rs --maxfail=1
```

Full test suite (canonical, only when instructed):

```bash
pytest --ignore=staging --ignore=electrodrive/fmm3d/tests/test_kernels_gpu_stress.py --ignore=temp_AI_upgrade -vv -rs -q --maxfail=1
```

CPU fallback scan (use sparingly):

```bash
rg -n "\.cpu\(|\.numpy\(|\.item\(|to\('cpu'\)" electrodrive
```

---

## 11) Final rule: protect the repo

Black Hammer succeeds only if:

* the repo remains healthy and testable
* discovery runs are reproducible and diagnosable
* GPU-first doctrine is upheld
* strict A–D gate passes are pursued by **methodology improvements**, not threshold relaxation
