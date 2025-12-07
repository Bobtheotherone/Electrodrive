Stage-0 axis sweep weight modes (n_max=1)

Geometry: Stage-0 grounded sphere (radius=1, center_z=0)
Basis: sphere_kelvin_ladder
Axis sweep z-grid: 1.05, 1.15, 1.3, 1.5, 1.9

Spectral summary:
- sigma_norm: 1
- effective_rank (1e-1 / 1e-2): 1 / 1
- reconstruction rel_fro: rank1_rel_fro:1.72e-16

Symbolic mode laws:
- mode 0 [poly] rmse=1.99e-13 rel_rmse=7.02e-13: -19.6396 * z^4 + 105.501 * z^3 + -207.243 * z^2 + 176.438 * z + -55.0519

research_wishlist:
- Relate leading mode to Kelvin inversion law and quantify deviations near the surface.
- Prove spectral gap along axis using boundary integral operator symmetry.
- Map rational fit poles/zeros to candidate ladder positions for controller design.
