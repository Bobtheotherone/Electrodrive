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
