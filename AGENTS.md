````markdown
# AGENTS.md — Electrodrive (Codex / Repo Health)

This file defines how coding agents (including Codex) should work in this repository.

## Primary goals (in order)
1. **Repo health never declines**: keep the full test suite green (expected skips for optional deps are OK).
2. **Fix the user-reported regression**: `tests/test_images_diffusion_cli.py::test_diffusion_no_checkpoint_creates_random_generator`.
3. **Minimize blast radius**: smallest correct change, no new hard dependencies, no behavior changes outside the intended fix.

---

## Non-negotiables
- **Do not “fix” by weakening tests**: no `xfail`, no blanket skips, no removing assertions.
- **No new required dependencies** (stdlib + existing deps only). Optional extras are fine only if already present in repo conventions.
- **Prefer defensive, explicit error handling** over broad `except:` that hides root causes.
- **Keep interfaces stable**: don’t break public APIs/CLIs; avoid renaming CLI flags.
- **Leave the tree clean** (no stray debug files, no temporary scripts committed).

---

## Quick reproduction (the current failure)
Run this first and keep iterating until it passes:

```bash
python3 -m pytest -q -vv -s tests/test_images_diffusion_cli.py::test_diffusion_no_checkpoint_creates_random_generator
````

Then confirm both diffusion CLI tests:

```bash
python3 -m pytest -q -vv -s tests/test_images_diffusion_cli.py
```

Finally, confirm the repo-level pytest run used by the maintainer (matches local workflow):

```bash
pytest --ignore=staging \
       --ignore=temp_AI_upgrade \
       --ignore=electrodrive/fmm3d/tests/test_kernels_gpu_stress.py \
       -q -vv -rs --maxfail=1
```

---

## Debugging workflow for this failure

The failing assertion is that `cli.run_discover(args)` should return **0** when:

* `basis_generator="diffusion"`
* `basis_generator_mode="diffusion"`
* `model_checkpoint=None`

### Where to look

* CLI entrypoint: `electrodrive/tools/images_discover.py` (`run_discover`)
* Diffusion generator: `electrodrive/images/diffusion_generator.py` (`DiffusionBasisGenerator`)
* Image search pipeline: `electrodrive/images/search.py` (`discover_images`, `ImageSystem`)
* Logger output: `out_dir/events.jsonl` (written by `JsonlLogger`)

### How to get the real exception

If `run_discover` returns `1`, it should have logged an `ERROR` record.
Open the run directory produced by the test (a tmp path) and inspect:

* `<out_dir>/events.jsonl`
* Look for `"level":"ERROR"` and the `"trace"` field (logger supports `exc_info=True`)

Use the trace to identify the exact failing line and fix the underlying cause.

---

## Behavioral contract (what “correct” means)

### Diffusion generator behavior

* If `--basis-generator diffusion` (or `hybrid_diffusion`) is requested **without** a checkpoint:

  * **Create a fresh `DiffusionBasisGenerator`** with a reasonable default config.
  * **Proceed normally** and return `0` if downstream steps succeed.
  * Emit a **warning-level log** that weights are random / exploratory.

* If a checkpoint is provided but does **not** contain diffusion weights:

  * Treat as a **hard error** and return `1` (the existing test expects this).

### Exit codes

* `0` = successful run (including no-checkpoint diffusion generator initialization)
* `1` = expected user/config error or runtime failure (with a logged error message)
* Never call `sys.exit()` inside library functions; return codes are required for testability.

### Artifact writes

`run_discover` should remain safe and deterministic about output:

* It may create the output directory.
* It may write manifests/logs (`events.jsonl`, `discovery_manifest.json`, etc.).
* Tests may monkeypatch `save_image_system`; do not assume serialization always happens.

---

## Implementation guidelines (to prevent repo health regression)

* Make the smallest change that restores the contract above.
* Prefer **localized fixes** in the CLI (`run_discover`) rather than sweeping changes across the solver stack.
* If the failure is due to a missing attribute on a returned object, fix it **at the source** (preferred), or add a **backwards-compatible fallback** (acceptable) without breaking type expectations.
* Add/adjust tests only when:

  * The desired behavior is not already covered, or
  * You’re preventing a future regression with a clear, minimal test.

---

## Testing policy

* For this task, always run:

  * the single failing test (fast loop),
  * the whole `tests/test_images_diffusion_cli.py`,
  * then the maintainer’s full pytest invocation shown above.
* Skips are acceptable only for optional dependencies (e.g., `pykeops`, `xitorch`) and must remain clean (no errors).

---

## Style / maintainability

* Keep changes readable and well-commented where logic is non-obvious (especially around CLI exit codes).
* Avoid adding global state or environment-variable side effects unless strictly necessary.
* If you touch logging, keep it JSON-serializable and stable (no huge tensors dumped).

---

## Definition of done

* ✅ `tests/test_images_diffusion_cli.py::test_diffusion_no_checkpoint_creates_random_generator` passes.
* ✅ Maintainer pytest command passes (same ignores).
* ✅ No new failing tests, no new required dependencies, no weakened assertions.
* ✅ Logs and manifests (if written) remain valid JSON and informative.

```
::contentReference[oaicite:0]{index=0}
```
