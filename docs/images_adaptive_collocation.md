# Adaptive collocation knobs for MOI discovery

Environment variables consumed by `electrodrive.images.search.discover_images`:

- `EDE_IMAGES_ADAPTIVE_ROUNDS` (default: `1`): number of oracle collocation rounds. `1` keeps the legacy single-round `get_collocation_data` path; `>=2` triggers the residual-driven builder that calls `make_collocation_batch_for_spec` once per round and keeps the top-residual points.
- Legacy `EDE_IMAGES_ADAPTIVE_PASSES` is still honoured for compatibility (`rounds = passes + 1` when provided).
- `EDE_IMAGES_N_POINTS` / `EDE_IMAGES_RATIO_BOUNDARY` still control the per-round collocation batch size.

Stage-0 sphere tool flags (see `tools/stage0_sphere_axis_moi.py`):

- `--adaptive-collocation-rounds`: forwards to `discover_images(adaptive_collocation_rounds=...)`.

Safety notes:

- Each round makes exactly one oracle call with `n_points` samples; total collocation points are bounded by `rounds * n_points`.
- Adaptive sampling reuses a deterministic RNG per run (`_make_collocation_rng`), so runs can be reproduced.
