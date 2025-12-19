# Final summary — three_layer_complex basis scaffolding

Changes:
- Added a CUDA-only layered basis helper inside the package (`electrodrive/images/layered_basis_complex.py`) and wired the opt-in `three_layer_complex` basis type in `electrodrive/images/basis.py`.
- Kept CLI/help unchanged except for documenting the new flag; default behavior is unchanged (no layered candidates unless the flag is provided).
- Added GPU tests (`tests/test_images_three_layer_basis.py`) to assert candidate coverage and opt-in behavior, plus notes/checker updates for usage.

Defaults:
- Baseline discovery behavior is unchanged unless `three_layer_complex` is included in `--basis`.
- GPU is required for the experimental helper; it fails fast if CUDA is unavailable.

How to run:
- Default path (unchanged): e.g., `python -m electrodrive.tools.images_discover discover --spec <spec> --basis point --nmax 8 --reg-l1 1e-3`.
- Experimental basis: `python -m electrodrive.tools.images_discover discover --spec specs/planar_three_layer_eps2_80_sym_h04_region1.json --basis point,axis_point,three_layer_images,three_layer_complex --nmax 12 --reg-l1 1e-3 --out runs/demo_three_layer_complex`.
- Checker: `EDE_DEVICE=cuda EDE_DTYPE=float32 python notes/check_three_layer_complex.py` (writes `runs/checks/three_layer_complex/run_<timestamp>/summary.json`).

Tests run:
- `python -c "import electrodrive; print('import_ok')"`
- `pytest -q tests/test_images_three_layer_basis.py`
- `python notes/repro_step2_gpu.py`
- `python notes/check_three_layer_complex.py`
- `python -m electrodrive.tools.images_discover discover --help`
