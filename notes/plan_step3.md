# Step 3 plan — three-layer basis upgrade (GPU-first)

Goal:
- Add an opt-in three-layer basis generator that does not get starved by `axis_point` and proposes structured slab-aware depths (mirrors, slab interior anchors, evanescent tails) on CUDA. Defaults stay unchanged unless the new flag is supplied.

Interfaces / toggles:
- Library: extend `electrodrive/images/basis.py::generate_candidate_basis` to recognize a new basis type flag `three_layer_complex`.
- New helper module in `temp_AI_upgrade/layered_basis.py` to assemble three-layer candidate positions on the provided `device`/`dtype`.
- CLI opt-in: users pass `--basis ... ,three_layer_complex` (comma-separated) to enable; without the flag the old heuristics run exactly as before.

GPU/dtype policy:
- Require `device.type == "cuda"` for the new generator (fail fast otherwise); respect the passed `dtype` (default float32).
- Keep tensors on GPU; only detach to CPU for small logs/tests.

Integration strategy:
- When `three_layer_complex` is requested, build a small curated set of positions (top interface mirror + ladder, slab interior anchors, below-slab decay taps) via the helper before adding generic `axis_point` seeds so they are not pruned by budget limits.
- Group-info tags for new elements (family/motif) to keep solver metadata tidy.
- Leave existing `three_layer_images` and other basis paths untouched; no default flips.

Test plan (targeted, GPU-only):
- Add a GPU-skipped-when-no-CUDA test (e.g., `tests/test_images_three_layer_basis.py`) that requests `three_layer_complex` for `planar_three_layer_eps2_80_sym_h04_region1.json` and asserts we get non-empty layered candidates spanning top/slab/bottom regions and that axis_point no longer monopolizes the budget.
- Keep the test small (<= 32 candidates) and deterministic.

Health gates to run after changes:
- `python -c "import electrodrive; print('import_ok')"`
- `pytest -q tests/test_images_three_layer_basis.py`
