# AGENTS.md — Operation Black Hammer (Repo Standard + Experimental Discovery Protocol)

This repository is an active research + engineering codebase for **Operation Black Hammer**.

**Core directive:** discover **new analytical (or analytically distillable) Green’s functions** using an **AI + method-of-images program representation**, with **GPU-first** implementation and **scientific-grade verification**.

This file is written for Codex. Follow it exactly.

---

## 0) Hardware + performance doctrine (non-negotiable)

Target machine:
- Laptop: ROG Zephyrus G16 GU605CX_GU605CX
- CPU: Intel(R) Core(TM) Ultra 9 285H (2.90 GHz)
- GPU: NVIDIA Blackwell RTX 5090 Laptop GPU, 24 GB VRAM (capability 12.0)
- RAM: 32 GB

Doctrine:
1. **GPU-first** for anything performance-relevant. CPU is allowed only for I/O, orchestration, or when GPU support is impossible.
2. **No silent CPU fallbacks in hot paths.** Any `.cpu()`, `.numpy()`, host sync `.item()` in inner loops is a bug unless explicitly justified.
3. Mixed precision policy:
   - Proposal / generative models: BF16/FP16 allowed
   - Solver + verification: FP32
   - Certification / final numeric checks: FP64 as needed
4. Prefer vectorization, batching, fused kernels (Triton/CUDA) over Python loops.

---

## 1) Environment + commands (always use venv)

Every shell session begins with:
```bash
source .venv/bin/activate
````

GPU sanity:

```bash
python scripts/print_cuda_env.py
pytest -q tests/test_cuda_smoke.py -vv -rs --maxfail=1
```

**Canonical full pytest command** (use only at milestone checkpoints and before large experiments):

```bash
pytest --ignore=staging --ignore=electrodrive/fmm3d/tests/test_kernels_gpu_stress.py --ignore=temp_AI_upgrade -vv -rs -q --maxfail=1
```

---

## 2) Repo health guardrails

### 2.1 Definition of “repo health”

Repo health means:

* tests do not regress,
* imports/packaging do not break,
* numerical meaning does not silently change,
* no new GPU→CPU transfer regressions in hot loops,
* new features are gated (default OFF) unless explicitly intended.

### 2.2 Working discipline

* Keep changes **surgical**.
* Make **small, atomic commits**.
* Run **targeted tests** after each commit.
* Avoid mass refactors and mass formatting.

---

## 3) Emergency stop (anti-thrashing, mandatory)

### 3.1 Trigger conditions

If any of these occur and you cannot confidently fix immediately:

* new failures appear outside your intended scope,
* failures are widespread or unclear,
* GPU/CPU device behavior becomes inconsistent,
* you are stuck in edit-test-fail loops.

### 3.2 Emergency stop procedure

1. Revert protected code to clean state:

```bash
git reset --hard HEAD
git clean -fd
```

2. Freeze modifications to any file that existed at the start of the run.
3. You may continue only in **new files** or a clearly isolated experimental namespace.
4. Write an incident report:

* `.blackhammer/emergency_stop_report.md` (what changed, what broke, why you stopped)

---

## 4) Black Hammer experimental goal: quick analytical win

We want a **fast analytical surrogate** that:

* matches a trusted oracle (BEM / DCIM / verifier) to high accuracy,
* is **significantly faster** than the oracle for repeated queries,
* is compact enough to **distill to an analytic form** (or near-analytic template),
* is valuable for engineering workflows (e.g., accelerating repeated solves in FEA/BEM contexts).

**A “quick win” target** should have:

* a strong existing oracle in this repo,
* repeated-query value (many sources/targets),
* a compact image-template likely exists (finite or small structured family).

Examples of good early targets (choose what the repo already supports best):

* planar / layered-media Green’s surrogates (compact image templates approximating a known expensive oracle),
* repeated source evaluation scenarios where speedups are monetizable.

---

## 5) Experimental run protocol (Codex must follow)

### 5.1 Preflight (required)

Before ANY discovery run:

1. Clean tree:

```bash
git status --porcelain
```

Must be empty or STOP.
2. GPU sanity:

```bash
python scripts/print_cuda_env.py
pytest -q tests/test_cuda_smoke.py -vv -rs --maxfail=1
```

3. Run minimal correctness suite for recently touched critical parts (examples; adjust if files move):

```bash
pytest -q electrodrive/gfdsl/tests -vv -rs --maxfail=1
pytest -q electrodrive/images/optim/tests/test_group_prox.py -vv -rs --maxfail=1
pytest -q electrodrive/images/tests/test_imagesystem_batched_potential.py -vv -rs --maxfail=1
pytest -q electrodrive/verify/tests/test_green_checks.py -vv -rs --maxfail=1
```

### 5.2 Run structure + artifacts

All discovery runs must write to:

* `runs/black_hammer/<timestamp>_<tag>/`

Each run directory must include:

* `config.yaml` (or equivalent)
* `env.json` (CUDA, torch, device)
* `git.json` (commit SHA, branch)
* `best.jsonl` or equivalent candidate trace
* certificates / verifier output where available

### 5.3 Multi-fidelity ladder (preferred)

Use a staged evaluation strategy:

* F0 (cheap proxy), F1 (mid), F2 (expensive oracle)
  Promotion policy must be explicit and recorded.

If ladder tooling exists (e.g., `electrodrive/verify/ladder.py`), use it or follow its pattern.

### 5.4 Verification requirements for any claimed win

A candidate may be called “promising” only if it:

* passes core verification checks (reciprocity/symmetry if applicable, far-field sanity, boundary checks where relevant),
* demonstrates stable accuracy on holdout points (not only training points),
* shows reproducibility (seeded rerun yields similar results).

---

## 6) The_Vault protocol (mandatory for potentially novel discoveries)

### 6.1 The_Vault purpose

**The_Vault** is for storing and documenting any image system, program template, or analytic formula that appears **potentially novel relative to this repo** and could be productized as an exclusive service (e.g., major speedups for firms doing FEA/BEM).

**Important:** “novel” here means *not obviously present in this repo’s existing baselines and templates*.
Codex must not claim global novelty without external validation.

### 6.2 Storage location

At repo root:

* `The_Vault/`
* `The_Vault/Vault_Report.md` (index + setup narrative)

Create `The_Vault/` if missing.

### 6.3 When to vault

Vault a discovery if all are true:

* accuracy: meets a strict threshold vs oracle on holdout (define threshold in the vault entry; typical target: relerr ≤ 1e-3 or better),
* speed: demonstrates material speedup vs oracle (define benchmark method; typical target: ≥ 10× on GPU),
* stability: result persists across at least 2 seeds or 2 nearby configurations,
* interpretability: program/template is compact enough to explain and potentially distill.

### 6.4 Vault entry structure

For each vault-worthy discovery, create:

* `The_Vault/<YYYYMMDD_HHMMSS>_<short_tag>/`

Inside it, include at minimum:

* `README.md` — one-page human summary:

  * what problem it solves (physics + geometry + BCs),
  * what is new relative to repo baselines,
  * measured accuracy + speed,
  * how to reproduce in ≤ 10 minutes.
* `Vault_Entry.json` — machine-readable metadata:

  * git SHA, branch,
  * seeds,
  * config parameters,
  * program canonical hash/bytes reference,
  * metrics, verification summary,
  * oracle fidelity used.
* `Reproduce.sh` — exact reproduction commands (venv activation included).
* `Program/` — canonical program representation:

  * canonical JSON/bytes,
  * motif description,
  * any derived analytic simplification notes.
* `Benchmarks/` — benchmark scripts + output:

  * timing CSV,
  * hardware + dtype info.
* `Verification/` — verification results:

  * gate outputs / certificates,
  * summary JSON and/or logs.
* `Distillation/` — if distillation is attempted:

  * clustered templates,
  * exported datasets (CSV/NPZ),
  * notes about inferred closed forms.

**Do not** put huge checkpoints in The_Vault. Prefer pointers to `runs/` artifacts and store only compact essentials.

### 6.5 Vault_Report.md requirements

`The_Vault/Vault_Report.md` must:

* list all vault entries (timestamp + tag),
* summarize each (problem, accuracy, speed, status),
* include a **“Discovery Setup Narrative”**:

  * exactly what scripts/configs were run,
  * how GPU-first constraints were ensured,
  * what verification gates were used,
  * what made the discovery emerge (search space, reward shaping, priors).

---

## 7) Distillation + analytic extraction (required for “analytical win”)

If a run produces a stable template:

1. Cluster/aggregate candidates by structural fingerprint.
2. Fit parameter laws across a family of source locations/configs.
3. Export datasets for symbolic regression.
4. Record the entire pipeline in the vault entry.

If a distillation tool exists (example):

```bash
python scripts/black_hammer/distill_templates.py <RUN_DIR> <OUT_DIR>
```

Then place `<OUT_DIR>` (or a curated subset) under the vault entry.

---

## 8) Codex must NOT do

* Do not disable tests to “fix” failures.
* Do not introduce CPU fallbacks in hot paths for convenience.
* Do not claim global novelty; only “appears novel relative to repo.”
* Do not thrash protected code after emergency stop.
* Do not store massive artifacts in The_Vault.

---

## 9) Go/No-Go before expensive discovery runs

Go only if:

* GPU sanity passes,
* targeted suites pass,
* **canonical full pytest** has passed at least once on the current branch after recent changes,
* run directory structure + env/git logging is in place.

No-Go if:

* layered oracle returns empty batches unexpectedly for your intended target,
* device mismatches appear (CUDA sigma on CPU backend, etc.),
* verification is not being recorded.

---

## 10) Quick-win default experiment recipe (use unless instructed otherwise)

1. Choose a target scenario already supported by existing oracles.
2. Run a **small “smoke discovery”**:

   * tiny budgets, shallow templates, cheap fidelity
3. Promote only if it passes gates and beats baseline.
4. Distill templates immediately.
5. If vault criteria are met → create vault entry + update Vault_Report.md.

````

---

### Optional: a single Codex “do the experiment” prompt you can paste
(If you want Codex to immediately run a quick-win attempt and vault anything promising, paste this to Codex after you install the AGENTS.md above.)

```text
CODEX EXPERIMENT EXECUTION — Quick Analytical Win + The_Vault

Follow AGENTS.md strictly.

1) Preflight
- source .venv/bin/activate
- git status --porcelain must be clean or STOP.
- python scripts/print_cuda_env.py
- pytest -q tests/test_cuda_smoke.py -vv -rs --maxfail=1
- pytest -q electrodrive/gfdsl/tests -vv -rs --maxfail=1
- pytest -q electrodrive/images/optim/tests/test_group_prox.py -vv -rs --maxfail=1
- pytest -q electrodrive/images/tests/test_imagesystem_batched_potential.py -vv -rs --maxfail=1
- pytest -q electrodrive/verify/tests/test_green_checks.py -vv -rs --maxfail=1

2) Identify the best “quick win” target already supported by the repo
- Search for existing discovery entrypoints/configs:
  rg -n "discovery|gfn|gflow|run.*discovery|oracle" experiments scripts electrodrive | head
  ls runs | head
- Choose ONE target scenario with:
  - strong oracle support
  - repeated-query value
  - likely compact image template
- Write a short plan into: .blackhammer/experiment_plan.md (target, oracle, metrics, thresholds)

3) Run a small smoke discovery run (GPU-first)
- Create run dir: runs/black_hammer/<timestamp>_quickwin_<tag>/
- Ensure env/git info written (torch/cuda device, git SHA)
- Run a short discovery budget (few generations, small candidate pool)
- Record command lines verbatim in the run dir.

4) Verify + benchmark top candidate(s)
- Use verifier/gates/certification tools already in repo where available.
- Measure speed vs oracle on GPU using consistent timing protocol.
- Summarize results in runs/black_hammer/.../summary.md

5) Distill templates
- If available:
  python scripts/black_hammer/distill_templates.py <RUN_DIR> <RUN_DIR>/distilled
- Otherwise implement a minimal deterministic distillation script only if needed.

6) Vaulting
If a candidate meets vault criteria (accuracy threshold + speedup + stability + interpretability):
- Create The_Vault/ if missing.
- Create The_Vault/<YYYYMMDD_HHMMSS>_<tag>/
- Populate:
  README.md, Vault_Entry.json, Reproduce.sh, Program/, Benchmarks/, Verification/, Distillation/
- Update The_Vault/Vault_Report.md:
  - add an entry in the index
  - include “Discovery Setup Narrative”: commands, configs, GPU settings, verification gates, why it worked
- Do NOT store huge checkpoints; store compact artifacts + pointers to runs/.

7) Final safety
- If any repo health regression occurs and you cannot localize it, EMERGENCY STOP per AGENTS.md.
- Leave working tree clean.