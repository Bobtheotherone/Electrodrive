````markdown
# AGENTS.md — Gate B/C Sprint Guide (Dielectric Interfaces, GPU-First)

This repo is a GPU-first electrostatics / Green’s-function discovery system. **Our current bottleneck is not Gate A anymore.** Gate A is passing (autograd + fp64 verification points), but **Gate B and Gate C are failing for all generated GFDSL candidates** on the three-layer planar dielectric spec.

This AGENTS.md is specialized for **making Gate B and Gate C pass** as fast as possible, with minimal thrash and maximal reuse of existing working code.

---

## 0) Non-Negotiables

### GPU-first, always
- CUDA is mandatory for any real run.
- Never add CPU fallbacks for heavy compute.
- Keep tensors on CUDA; avoid `.cpu()`/`.numpy()` in hot paths.

### Focus: gate passing, not unit tests
- Do **not** prioritize full test suites right now.
- Use **gfdsl_verify** and gate artifacts as the primary feedback loop.

### Emergency stop (anti-thrashing)
If Gate B/C work causes unclear regressions in unrelated systems:
1) Stop touching unrelated code.
2) Revert risky edits.
3) Continue only in isolated modules related to gates or GFDSL evaluation.

---

## 1) Current Known State (Do Not Re-Debug)

### Target Spec (primary)
`specs/planar_three_layer_eps2_80_sym_h04_region1.json`

Key facts (from verifier config output):
- BCs: `dielectric_interfaces`
- Dielectrics:
  - region1: eps=1.0, z ∈ [0, 10]
  - slab: eps=80.0, z ∈ [-0.4, 0]
  - region3: eps=1.0, z ∈ [-10, -0.4]
- Source: point charge at z=0.2 in region1
- Symmetry: `rot_z`
- Domain bbox: [-1.2,1.2]^3

### Gate results (as observed)
For ~200 GFDSL programs in `runs/hard_discovery_01/program_bank/`:
- Gate A: **PASS** for all (after fixes)
- Gate B: **FAIL** for all
- Gate C: **FAIL** for all
- ABC passers: **0**

This means:
- We have harmonic (PDE) building blocks,
- but we do **not** satisfy **dielectric interface conditions** (Gate B) or **near+far 1/r behavior** (Gate C).

---

## 2) What Gates B and C Actually Require

### Gate B (Boundary Conditions / Interface Continuity)
Located at: `electrodrive/verify/gates/gateB_bc.py`

For `BCs=dielectric_interfaces`, Gate B typically checks:
- **Potential continuity across each interface** (e.g., z=0, z=-0.4)
- **Normal displacement continuity**: eps * dV/dn continuous (no free surface charge)
- May also compute `dirichlet_max_err` (even if conductors list is empty; confirm logic in gateB code)

Outputs to watch (in certificate metrics):
- `interface_max_v_jump`
- `interface_max_d_jump`
- `dirichlet_max_err`
- sample counts

**Implication:** candidate evaluation must be **region-aware and physically consistent across interfaces**, not just harmonic in a single region.

### Gate C (Asymptotics / Spurious Behavior)
Located at: `electrodrive/verify/gates/gateC_asymptotics.py`

Gate C checks:
- Slope near radius (default ~0.5) and far radius (~10) should be close to **-1** (1/r potential)
- It may reject spurious growth / non-decaying behavior

Outputs to watch:
- `near_slope`, `far_slope`
- `spurious_fraction`
- `near_radius`, `far_radius`

**Implication:** candidate must have a correct **monopole** behavior near and far. In practice this often requires:
- including a correct “direct/reference” term (fixed term) OR
- enforcing a net-charge constraint / moment constraint on the learned correction terms.

---

## 3) Fast Repro Commands (Use These Instead of Tests)

Always run inside venv and GPU knobs:

```bash
source .venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export EDE_IMAGE_SYSTEM_V2=1
````

### A-only sanity (should pass quickly)

```bash
python -m electrodrive.tools.gfdsl_verify \
  --spec specs/planar_three_layer_eps2_80_sym_h04_region1.json \
  --program runs/hard_discovery_01/control/conjpair.json \
  --out runs/_debug/conjpair_A \
  --dtype fp32 \
  --gates A \
  --n-points 256
```

### Gate B debug (small)

```bash
python -m electrodrive.tools.gfdsl_verify \
  --spec specs/planar_three_layer_eps2_80_sym_h04_region1.json \
  --program runs/hard_discovery_01/program_bank/dcim_full_000.json \
  --out runs/_debug/dcim_full_000_B \
  --dtype fp32 \
  --gates B \
  --n-points 128
```

### Gate C debug (small)

```bash
python -m electrodrive.tools.gfdsl_verify \
  --spec specs/planar_three_layer_eps2_80_sym_h04_region1.json \
  --program runs/hard_discovery_01/program_bank/dcim_full_000.json \
  --out runs/_debug/dcim_full_000_C \
  --dtype fp32 \
  --gates C \
  --n-points 128
```

---

## 4) Artifact-Driven Debugging (Don’t Guess)

Every run writes:

* `<out>/verification_certificate.json`
* Gate artifacts under:

  * `<out>/gates/B/...`
  * `<out>/gates/C/...`

Use these to diagnose what is failing:

* Which interface is worst (z=0 vs z=-0.4)?
* Is the failure dominated by potential jump or displacement jump?
* Are C slopes wrong near, far, or both?

Recommended quick inspection:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("runs/_debug/dcim_full_000_B/verification_certificate.json")
d = json.loads(p.read_text())
print("B:", (d.get("gates", {}) or {}).get("B", {}))
PY
```

---

## 5) Repo Map (Where To Look First)

### Gate implementations

* `electrodrive/verify/gates/gateB_bc.py`  ← interface checks
* `electrodrive/verify/gates/gateC_asymptotics.py` ← slope checks
* `electrodrive/verify/verifier.py` ← wiring + default configs

### GFDSL pipeline

* `electrodrive/tools/gfdsl_verify.py` ← the runnable pipeline (spec → compile → solve → verify)
* `electrodrive/gfdsl/ast/nodes.py` ← lowering of layered nodes / DCIM blocks
* `electrodrive/gfdsl/eval/layered.py` ← current layered evaluators (likely too naive for B/C)
* `electrodrive/gfdsl/eval/kernels_complex.py` ← complex conjugate pair kernel (A is fixed here)

### “Known-good” layered/image implementations (use as baseline)

These already exist in the repo and should contain physics that passes interface checks:

* `electrodrive/images/basis.py`
* `electrodrive/images/basis_dcim.py`
* `electrodrive/images/dcim_types.py`
* Reference stratified solver:

  * `electrodrive/core/planar_stratified_reference.py`

If Codex is stuck, the fastest path is:
**copy the physics from the known-good basis implementations into GFDSL lowering**, rather than inventing new pole/branchcut math from scratch.

---

## 6) The Likely Root Causes of B/C Failure (Prioritized)

### (1) Candidate is missing an explicit “direct/reference” term

Gate C near-slope will fail if the representation does not correctly reproduce the local 1/r singular behavior (even away from the exclusion radius).
Even if A passes (harmonic away from charges), C can fail if near behavior is wrong.

**Fix pattern:** include a fixed-term baseline (reference Green’s function) and solve only for a correction:

* `V_total = V_ref + V_correction`
* V_ref should be simple and stable, e.g. homogeneous Coulomb in eps1 with correct K_E scaling.
* The correction terms enforce interface BCs.

Implement/refine this in:

* `electrodrive/tools/gfdsl_verify.py` (fixed_term composition)
* and/or layered node lowering.

### (2) Layered evaluators are not region-aware / not enforcing dielectric reflection/transmission physics

A harmonic function is not enough. For layered media, the representation must match:

* continuity of V
* continuity of eps*dV/dn

Your current `InterfacePoleNode` / `BranchCutApproxNode` lowering in GFDSL is likely a generic linear basis, not tied to eps1/eps2/eps3 or interface locations properly.

**Fix pattern:** implement physically correct planar stratified Green’s representation:

* for each observation region (region1, slab, region3), the representation differs
* coefficients depend on eps ratios and slab thickness h
* existing DCIM/three-layer basis code in `electrodrive/images/` likely already embodies this

Implement/refine in:

* `electrodrive/gfdsl/eval/layered.py`
* `electrodrive/gfdsl/ast/nodes.py` lowering

### (3) Gate B may be applying a dirichlet check unexpectedly (even with no conductors)

If `dirichlet_max_err` dominates while conductors are empty, confirm Gate B logic:

* Is it checking a plane boundary as “dirichlet” by default?
* Is it interpreting “interfaces” incorrectly?

Fix in `gateB_bc.py` only if the current behavior is demonstrably wrong for `dielectric_interfaces`.

---

## 7) Fast Strategy for Codex: “Make One Candidate Pass B + C”

### Step 1: Establish a baseline that should pass B/C

Codex should build or reuse a known-good candidate evaluator:

* e.g., run the existing stratified reference backend and measure B/C on it.
* If the baseline fails, the gate logic or thresholds may be wrong for this spec.

### Step 2: Introduce a reference + correction decomposition

* Add `V_ref` fixed term (direct Coulomb in eps1, with K_E scaling consistent with collocation/oracles).
* Solve correction coefficients with GFDSL images/poles/branchcut.
* This should immediately improve:

  * Gate C near slope (by giving correct local 1/r behavior)
  * numerical stability / coefficient magnitudes, which helps B as well.

### Step 3: Make layered correction terms physically parameterized

* Use eps ratios and interface z positions (0 and -0.4) directly from spec.
* Prefer implementing a GFDSL node that wraps known-good three-layer basis rather than “free” residues.

**Pragmatic path:** add a GFDSL node like `ThreeLayerDCIMNode` that lowers to an evaluator built from existing DCIM basis code.

---

## 8) Performance / Iteration Guidance (So It Doesn’t Take Forever Again)

* Debug B and C **separately** first with `--gates B` or `--gates C`
* Use `--n-points 64–256` until metrics move meaningfully
* Only then run `--gates A,B,C` (A should remain pass)
* For final confirmation use higher points (512–2048)

---

## 9) Working Definition of “Done” for This Sprint

For the target spec:

* A: pass (already achieved)
* B: pass (interface continuity within thresholds)
* C: pass (near and far slope near -1 within tolerance)
* Once ABC passes exist, we can resume D/E and “novel discovery” selection.

---

## 10) Immediate Next Action for Codex (When You Start a Session)

1. Read gate code:

   * `electrodrive/verify/gates/gateB_bc.py`
   * `electrodrive/verify/gates/gateC_asymptotics.py`

2. Read known-good layered basis code:

   * `electrodrive/images/basis_dcim.py`
   * `electrodrive/core/planar_stratified_reference.py`

3. Modify ONLY the minimal path needed to create one ABC-passing candidate:

   * add reference fixed term (V_ref) in `gfdsl_verify.py`
   * implement region-aware correction terms in GFDSL layered lowering
   * iterate using `--gates B` and `--gates C` with small `--n-points`

4. Keep everything CUDA-first.
