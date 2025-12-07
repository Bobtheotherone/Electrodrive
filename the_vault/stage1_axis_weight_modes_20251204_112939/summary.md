Stage-1 axis sweep weight modes

Geometry: Stage-1 sphere dimer (spec=stage1_sphere_dimer_axis_point_inside.json)
Basis: sphere_kelvin_ladder, sphere_equatorial_ring
Axis sweep z-grid: 0.7, 1.4, 2

Spectral summary:
- sigma_norm: 1, 0.275, 0.00527
- effective_rank (1e-1 / 1e-2): 2 / 2
- reconstruction rel_fro: rank1_rel_fro:0.265, rank2_rel_fro:0.00508, rank3_rel_fro:5.12e-16

Symbolic mode laws:
- mode 0 [poly] rmse=2.49e-14 rel_rmse=5.95e-16: -2.48572 * z^4 + -2.49591 * z^3 + 0.222867 * z^2 + 6.58082 * z + 3.85997
- mode 1 [rational] rmse=4.96e-15 rel_rmse=4.32e-16: (-2.25464 + -0.986937 * z^1 + 1.71997 * z^2) / (1 + -1.16728 * z^1)
- mode 2 [poly] rmse=1.44e-16 rel_rmse=6.44e-16: -0.0536527 * z^4 + -0.0135617 * z^3 + 0.141463 * z^2 + 0.329795 * z + -0.272151

research_wishlist:
- Connect leading mode to Kelvin ladder for symmetric dimers; derive decay rate of singular values from symmetry.
- Relate rational fit poles to effective image distances; test against analytic two-sphere Neumann series.
- Extend controller beyond axis by coupling to off-axis collocation or low-order spherical harmonics.
