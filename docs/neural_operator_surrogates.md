# 4.9 Neural Operator Surrogates (SphereFNO)

Neural operator surrogates approximate the Stage-0 BEM/analytic mapping for a
grounded sphere with a single on-axis point charge. They are opt-in and only
used when `supervision_mode="neural"` is requested.

## Concept
- Learn `SphereFNO: (q, z0, a) -> V(theta, phi)` on a fixed unit-sphere grid.
- Architecture choices: Fourier Neural Operator (default) or DeepONet variants.
- Surrogates replace oracle calls during training but remain opt-in; validation
  and fallbacks keep legacy behaviour intact.

## Input / Output Encoding
- Parameters: `q` (charge), `z0` (axis position relative to centre), `a`
  (radius).
- Domain: fixed angular grid `(theta, phi)` on the unit sphere.
- Channels: broadcast `(q, z0, a)`, trig positional encoding
  `(sin theta, cos theta, sin phi, cos phi)`, plus an MLP embedding of the
  parameters.
- Output: potential on the spherical grid in the same reduced units as the
  analytic/BEM oracle.

## Reference SphereFNO Architecture
- Grid: `n_theta = 64`, `n_phi = 128`.
- Layers: 4 Fourier blocks, 16 modes per angular dimension.
- Width: 64 channels with GELU nonlinearity.
- Training recipe: 50k random triples with `q in [0.5, 2.0]`,
  `z0/a in [1.05, 3.0]`, `a in [0.5, 2.0]`; oracle targets from the Stage-0
  analytic/BEM tools; Adam `lr=1e-3` with L2 loss.
- A surrogate is considered **validated** when `rel_L2 <= 1e-3` and
  `rel_Linf <= 1e-2` on a held-out grid.

## Dataset and Training
- Dataset: `Stage0SphereAxisDataset` in `electrodrive.learn.datasets_stage0`
  samples on-axis grounded spheres from the above ranges and produces analytic
  grid targets for training.
- Training entrypoint: `tools/train_spherefno_stage0.py` (also exposed via the
  CLI subcommand `train_spherefno_stage0`), default config
  `configs/train_spherefno_stage0.yaml`.
- Validation metrics (`val_rel_l2`, `val_rel_linf`) are stored in checkpoints
  and consumed by `SphereFNOSurrogate.from_checkpoint`.

## Runtime Integration
- `make_collocation_batch_for_spec(..., supervision_mode="neural", ...)` tries
  the SphereFNO surrogate when:
  - geometry matches the Stage-0 grounded sphere with one on-axis charge, and
  - a checkpoint is provided via `EDE_SPHEREFNO_CKPT`.
- Validation: a small random subset compares surrogate predictions against the
  analytic (or optional BEM) oracle; tolerances default to
  `EDE_SPHEREFNO_L2_TOL=1e-3` and `EDE_SPHEREFNO_LINF_TOL=1e-2`.
- Fallback: if validation fails or no surrogate is available, collocation
  reverts to `EDE_NEURAL_FALLBACK_MODE` (default `analytic`).

## CLI smoke test
- `smoke_spherefno --ckpt <path> [--spec specs/sphere_axis_point_external.json]`
  compares surrogate predictions against the analytic oracle on random points
  outside the sphere and prints relative errors.

## Environment Variables
- `EDE_SPHEREFNO_CKPT`: path to the SphereFNO checkpoint.
- `EDE_SPHEREFNO_ALLOW_UNVALIDATED=1`: permit checkpoints without metrics.
- `EDE_SPHEREFNO_RADIAL`: `inv_r` (default) or `clamp_zero` radial extension.
- `EDE_SPHEREFNO_VAL_SAMPLES`: number of points for runtime validation
  (default: 64).
- `EDE_SPHEREFNO_L2_TOL` / `EDE_SPHEREFNO_LINF_TOL`: validation thresholds.
- `EDE_SPHEREFNO_VALIDATE_WITH_BEM=1`: enable BEM-based validation when the
  analytic shortcut is unavailable.
- `EDE_NEURAL_FALLBACK_MODE`: fallback oracle mode (`analytic`|`bem`|`auto`).
