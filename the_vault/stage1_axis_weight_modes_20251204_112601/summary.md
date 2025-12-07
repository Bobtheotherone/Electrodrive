Stage-1 axis sweep weight modes

Geometry: Stage-1 sphere dimer (spec=stage1_sphere_dimer_axis_point_inside.json)
Basis: sphere_kelvin_ladder, sphere_equatorial_ring
Axis sweep z-grid: 0.5, 1, 1.5, 2

Spectral summary:
- sigma_norm: 1, 0.00881, 0.00512, 0.000478
- effective_rank (1e-1 / 1e-2): 1 / 1
- reconstruction rel_fro: rank1_rel_fro:0.0102, rank2_rel_fro:0.00514, rank3_rel_fro:0.000478

Symbolic mode laws:
- mode 0 [poly] rmse=6.63e-14 rel_rmse=1.76e-15: -13.0909 * z^4 + 15.5844 * z^3 + 35.4935 * z^2 + -56.9294 * z + 18.9568
- mode 1 [rational] rmse=3.59e-16 rel_rmse=1.1e-15: (-10.1371 + 16.3265 * z^1 + -5.68022 * z^2) / (1 + -22.7014 * z^1)
- mode 2 [poly] rmse=1.06e-15 rel_rmse=5.61e-15: -0.208098 * z^4 + 0.330984 * z^3 + 0.656194 * z^2 + -1.1871 * z + 0.430241

research_wishlist:
- Connect leading mode to Kelvin ladder for symmetric dimers; derive decay rate of singular values from symmetry.
- Relate rational fit poles to effective image distances; test against analytic two-sphere Neumann series.
- Extend controller beyond axis by coupling to off-axis collocation or low-order spherical harmonics.
