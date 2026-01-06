````markdown
# AGENTS.md — Operation Black Hammer (Codex Surgical Playbook)

This repository is an active research/codebase for **Operation Black Hammer**: intensively repair, upgrade, and dramatically increase the scientific precision of the repo while enabling experiments whose core directive is:

> **Discover a completely new analytical Green’s Function using a method-of-images–based AI implementation.**

Codex must **prioritize GPU-first execution** and **must not allow repo health to decline**. If repo health declines and Codex cannot confidently identify/fix the cause, Codex must **EMERGENCY STOP** to prevent thrashing and damage to existing code.

---

## 0) Hardware + performance doctrine (non-negotiable)

Target machine:

- Laptop: **ROG Zephyrus G16 GU605CX_GU605CX**
- CPU: **Intel(R) Core(TM) Ultra 9 285H (2.90 GHz)**
- GPU: **NVIDIA Blackwell RTX 5090, 24 GB VRAM**
- RAM: **32 GB**

Doctrine:

1. **GPU is the primary compute device.** Prefer GPU even if it requires more engineering (e.g., custom CUDA/Triton kernels).
2. **CPU is only allowed** when:
   - the operation is not supported on GPU, or
   - the CPU is certainly more efficient (rare), or
   - it is strictly non-hot-path (I/O, orchestration, logging).
3. Any **implicit CPU fallback** in hot paths is a **bug**. Examples: `.cpu()` in inner loops, `.numpy()` conversions, host-device ping-pong, Python loops over large tensors.
4. Mixed precision policy:
   - Proposal / generative models may use **BF16/FP16**.
   - Solver / verification uses **FP32**.
   - Certification / final numerical validation uses **FP64** when necessary.

---

## 1) Environment: ALWAYS run inside venv

**Every command must start with:**
```bash
source .venv/bin/activate
````

Sanity checks (run early, and whenever GPU issues arise):

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda?', torch.cuda.is_available()); print('dev', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('cap', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"
```

If building CUDA extensions:

* Prefer setting:

  * `TORCH_CUDA_ARCH_LIST="12.0+PTX"` (Blackwell target) when appropriate in your environment.
* Keep builds reproducible; never hardcode machine-local absolute paths.

---

## 2) Repo health definition + gatekeeping

### 2.1 Repo health = “no regressions”

Repo health includes:

* Tests: existing test suite must not gain new failures.
* Behavior: no silent changes to numerical meaning without explicit versioning and docs.
* Performance: no new CPU/GPU transfer regressions in hot loops.
* Structure: no breaking imports, packaging, or CLI entrypoints unless explicitly upgraded with migration notes.

### 2.2 Full test command (canonical)

When instructed to run full pytest (or before merging major changes), use **exactly**:

```bash
source .venv/bin/activate
pytest --ignore=staging --ignore=electrodrive/fmm3d/tests/test_kernels_gpu_stress.py --ignore=temp_AI_upgrade -vv -rs -q --maxfail=1
```

### 2.3 “Baseline first” protocol (MANDATORY)

At the start of any run:

1. Ensure a clean working tree:

   ```bash
   git status --porcelain
   ```
2. Record baseline:

   * Save the current commit hash: `git rev-parse HEAD`
   * Run the full test command once and save output to a file:

     * `.blackhammer/baseline_pytest.txt`
3. Record “protected file set”:

   ```bash
   mkdir -p .blackhammer
   git ls-files > .blackhammer/protected_files_at_start.txt
   ```

---

## 3) Emergency stop (anti-thrashing safety system)

### 3.1 When to trigger EMERGENCY STOP

Trigger an EMERGENCY STOP if **any** occurs and you cannot confidently fix within the next minimal edit:

* New test failures appear that:

  * are outside the intended change scope, or
  * are widespread / non-localized, or
  * imply you may have broken core infrastructure (imports, packaging, device placement, numerics).
* Performance collapses due to accidental CPU fallback and the cause isn’t obvious.
* You detect repetitive edit-test-fail cycles without clear progress.

### 3.2 What EMERGENCY STOP means

When EMERGENCY STOP triggers:

1. **Immediately revert** changes to existing (“protected”) code:

   ```bash
   git reset --hard HEAD
   git clean -fd
   ```
2. **Freeze modifications** to any file that existed at the start of the run:

   * Those are listed in `.blackhammer/protected_files_at_start.txt`.
3. You may continue working **only** on:

   * New files created during this run, and/or
   * Files explicitly created under a new experimental namespace (recommended):

     * `staging/black_hammer_experiments/` (or similar)
4. Add a short incident report in:

   * `.blackhammer/emergency_stop_report.md`
     including:
   * what you changed,
   * what broke (copy errors),
   * hypotheses,
   * what you reverted.

**Do not resume editing protected code** unless a future instruction explicitly expands scope or you have a surgical, test-backed fix.

---

## 4) Surgical workflow (how Codex must operate)

### 4.1 Change discipline

* Make **small, atomic commits**.
* After each commit:

  * run the narrowest relevant tests (if available),
  * then periodically run the full test command.
* Avoid “refactor everything” edits.
* Never mass-format unrelated files.
* Never change public APIs without a migration note and tests.

### 4.2 GPU-first implementation discipline

When touching performance-critical code:

* Add a device assertion in hot paths when appropriate:

  * e.g., `assert x.is_cuda` (or explicit `device` plumbing).
* Avoid Python loops over tensor groups; prefer fused kernels or vectorized ops.
* Avoid host synchronization (`.item()`, implicit prints, frequent timing calls) inside loops.
* Minimize kernel launch overhead by batching operations.

### 4.3 Numerical rigor discipline

* Prefer **relative error metrics** and report them.
* Add explicit tests for:

  * symmetry/reciprocity (where physically required),
  * boundary condition satisfaction,
  * near-singularity behavior (x≈y),
  * far-field asymptotics (sanity scaling).

---

## 5) Operation Black Hammer phases (0 → 5) — Implementation playbook

Codex must be able to execute tasks from any phase below. Each task must include:

* code changes,
* tests,
* documentation updates (brief),
* and a rollback path.

### Phase 0 — Toolchain lock-in for Blackwell

**Goal:** guarantee the stack runs natively on the GPU and builds kernels for Blackwell.

Tasks may include:

* Add a script `scripts/print_cuda_env.py` that prints:

  * torch version, cuda availability, device name, capability, dtype support.
* Ensure any custom extension build config targets Blackwell (`sm_120`/`12.0`) and includes PTX fallback where appropriate.
* Add a minimal GPU smoke test in `tests/` that:

  * allocates tensors on CUDA,
  * runs a representative kernel,
  * verifies output deterministically.

Acceptance:

* GPU smoke test passes.
* No CPU fallback in core hot paths.

---

### Phase 1 — Repo hardening (packaging, imports, determinism)

**Goal:** everything is importable, tested, deterministic.

Key targets (search and repair):

* Broken imports / dead modules.
* Test collection failures.
* Non-deterministic seeds in experiment loops.

Typical surgical tasks:

* If a module exists but is not importable (packaging/layout mismatch), fix package structure:

  * create/repair `__init__.py`,
  * update import paths,
  * update `pyproject.toml` / `setup.cfg` if needed,
  * add tests that enforce importability.

Acceptance:

* Full test command passes.
* `python -c "import <top-level-package>"` passes reliably.

---

### Phase 2 — GPU-first performance refactor

**Goal:** remove structural CPU bottlenecks.

High-priority patterns to locate with ripgrep:

```bash
rg -n "\.cpu\(|\.numpy\(|torch\.unique\(|for .* in range\(|\.item\(" electrodrive
```

Typical upgrades:

* Remove forced `.cpu()` in matvec/solver paths.
* Replace Python loops over groups with vectorized scatter/segment operations.
* Batch basis evaluation and fuse operations (Triton/CUDA if needed).
* Optional: CUDA graphs / `torch.compile` once shapes are stabilized.

Acceptance:

* No new CPU↔GPU transfers introduced.
* Performance microbench improves or remains stable.
* Numerics remain within verified tolerance.

---

### Phase 3 — Scientific precision + verification

**Goal:** make “discovered Green’s functions” defensible.

Add/upgrade verification utilities:

* multi-region checks (near-singular / boundary / far-field),
* reciprocity / symmetry tests,
* FP64 certification mode for final candidates.

Acceptance:

* Verification tests exist and run on CI (or local full pytest command).
* Failures are informative and localized.

---

### Phase 4 — Integrate best academic patterns without losing interpretability

**Goal:** adopt SOTA ideas (operator learning, integral constraints, singular handling) while preserving “program/images” interpretability.

Implementation patterns:

* Support decomposition: `G = G_singular + G_smooth`.
* Support constraint-based training objectives (boundary-integral residuals, PDE residuals).
* Keep outputs interpretable by recording program templates, motif usage, and learned parameters.

Acceptance:

* New capabilities gated behind flags/configs.
* Backwards compatibility maintained (unless explicitly versioned).

---

### Phase 5 — Discovery campaign pipeline (new analytic Green’s function)

**Goal:** scale search and distill stable analytic structure.

Expected deliverables:

* A multi-fidelity evaluation ladder (cheap → expensive oracle).
* Reward shaping that favors:

  * low complexity,
  * stable motifs,
  * symmetry compliance,
  * low error.
* A distillation tool that:

  * clusters solutions by structural fingerprints,
  * fits parameter laws across families of source conditions,
  * exports datasets for symbolic regression.

Acceptance:

* A reproducible experiment entrypoint exists (script/CLI).
* Outputs are saved with:

  * program templates,
  * parameters,
  * solver stats,
  * verification stats,
  * git commit hash.

---

## 6) Logging + artifacts (required for credible science)

Every experimental run must record:

* git commit hash,
* command invocation,
* seeds,
* device details,
* dtype policy,
* config,
* summary metrics (error, latency, complexity, novelty),
* verification outcomes.

Recommended directory:

* `runs/black_hammer/<timestamp>_<short_hash>/`

---

## 7) Documentation rule

If you change behavior, add a short note in:

* `docs/black_hammer_changes.md` (create if missing)
  including:
* what changed,
* why it changed,
* how to reproduce,
* how to validate.

---

## 8) What Codex must NEVER do

* Never “fix” failing tests by disabling them unless explicitly instructed and justified.
* Never introduce CPU fallbacks in hot paths as a convenience.
* Never mass-reformat or rename unrelated files.
* Never repeatedly thrash protected code after EMERGENCY STOP.
* Never remove numerical checks/certification in the name of speed.

---

## 9) Quick command reference

Activate venv:

```bash
source .venv/bin/activate
```

Full test suite (canonical):

```bash
pytest --ignore=staging --ignore=electrodrive/fmm3d/tests/test_kernels_gpu_stress.py --ignore=temp_AI_upgrade -vv -rs -q --maxfail=1
```

Search for CPU fallbacks / performance smells:

```bash
rg -n "\.cpu\(|\.numpy\(|\.item\(|torch\.unique\(" electrodrive
```

GPU capability sanity:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
```

---

## 10) Final rule: protect the repo

Operation Black Hammer succeeds only if:

* the repo remains healthy,
* changes are surgical and test-backed,
* GPU-first doctrine is enforced,
* and discovered Green’s function candidates are numerically and scientifically defensible.

If uncertain: **stop, revert, and contain changes to new experimental code only.**

```
::contentReference[oaicite:0]{index=0}
```
