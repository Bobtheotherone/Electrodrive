Stage-0 axis sweep weight modes (n_max=3)

Geometry: Stage-0 grounded sphere (radius=1, center_z=0)
Basis: sphere_kelvin_ladder
Axis sweep z-grid: 1.25, 1.5, 2

Spectral summary:
- sigma_norm: 1, 0.258, 0.00213
- effective_rank (1e-1 / 1e-2): 2 / 2
- reconstruction rel_fro: rank1_rel_fro:0.249, rank2_rel_fro:0.00207, rank3_rel_fro:4.01e-16

Symbolic mode laws:
- mode 0 [poly] rmse=5.3e-15 rel_rmse=1.17e-15: -2.07144 * z^4 + 0.562385 * z^3 + 5.97591 * z^2 + 7.8017 * z + -15.3881
- mode 1 [rational] rmse=1.99e-15 rel_rmse=1.81e-15: (-0.119149 + -0.0836756 * z^1 + 0.0900659 * z^2) / (1 + -0.641217 * z^1)
- mode 2 [poly] rmse=3.04e-17 rel_rmse=3.25e-15: 0.00793453 * z^4 + -0.00430754 * z^3 + -0.0279507 * z^2 + -0.0321715 * z + 0.0822851

research_wishlist:
- Relate leading mode to Kelvin inversion law and quantify deviations near the surface.
- Prove spectral gap along axis using boundary integral operator symmetry.
- Map rational fit poles/zeros to candidate ladder positions for controller design.
