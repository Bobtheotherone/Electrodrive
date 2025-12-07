# Training Curricula (MOI 3.0)

This file tracks the staged curricula for the MOI discovery stack. The training entry point is `electrodrive.images.training.train_stage0` (Stage-0/1) via the CLI wrapper `python -m electrodrive.cli learn mo3_train ...`.

## Stage-0 (Plane & Sphere)
- Geometries: grounded plane + point, grounded sphere + external point; optional grounded sphere + internal point.
- Oracle: analytic solutions via `learn.collocation` (`supervision_mode=auto` falls back to BEM if needed).
- Sampling: z-draws in radii (`Stage0Ranges.plane_z`, `sphere_external`, `sphere_internal`) with charge magnitudes in `Stage0Ranges.q`; sign is random.
- Goals: LISTA learns mirrored location/weight for the plane and Kelvin-like placement for the sphere. Candidate generator can be enabled with `--n-learned`.
- Status note: Stage-0 MOI discovery is currently an anomaly tracker (bc_rel_mean can be extremely large); use it as a stress test, not as a correctness benchmark, until the BC/basis stack is overhauled.

## Stage-1 (Sphere Dimer Lens)
- Geometry: two grounded spheres on the z-axis (canonical `stage1_sphere_dimer_axis_point_inside.json` plus optional variants).
- Oracle: BEM via `learn.collocation`.
- Sampling: charge drawn along the lens gap using `Stage1Ranges.charge_frac` and `gap_margin`; magnitude in `Stage1Ranges.q` with random sign. Variants toggle with `--no-stage1-variants`.
- Goals: small boundary/axis errors on both spheres; LISTA sparsity tracks shared low-rank weight structure. Use `--basis-dimer` to tune candidate families.

## Stage-2 (Periodic Gratings) — TODO
- Geometry: periodic sphere arrays / gratings.
- Oracle: periodic BEM/FMM (not wired yet).
- Sampling + losses: pending; placeholder `Stage2Ranges` exists for future wiring.

## Quick CLI knobs
- Stage selector: `--stage {0,1}` (Stage-2 raises NotImplementedError).
- Basis families: `--basis-plane`, `--basis-sphere`, `--basis-dimer`.
- Collocation budget: `--n-points`, `--n-points-val`, `--ratio-boundary`, `--ratio-boundary-val`.
- Variants: `--no-stage1-variants` to limit Stage-1 to the canonical lens.
