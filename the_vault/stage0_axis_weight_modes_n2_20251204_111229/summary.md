Stage-0 axis sweep weight modes (n_max=2)

Geometry: Stage-0 grounded sphere (radius=1, center_z=0)
Basis: sphere_kelvin_ladder
Axis sweep z-grid: 1.05, 1.15, 1.3, 1.5, 1.9

Spectral summary:
- sigma_norm: 1, 0.0452
- effective_rank (1e-1 / 1e-2): 1 / 2
- reconstruction rel_fro: rank1_rel_fro:0.0452, rank2_rel_fro:1.38e-16

Symbolic mode laws:
- mode 0 [poly] rmse=9.46e-13 rel_rmse=1.35e-12: -77.7917 * z^4 + 428.068 * z^3 + -863.956 * z^2 + 758.771 * z + -245.3
- mode 1 [poly] rmse=5.87e-14 rel_rmse=2.4e-12: -4.79809 * z^4 + 26.6203 * z^3 + -54.3122 * z^2 + 48.3955 * z + -15.9197

research_wishlist:
- Relate leading mode to Kelvin inversion law and quantify deviations near the surface.
- Prove spectral gap along axis using boundary integral operator symmetry.
- Map rational fit poles/zeros to candidate ladder positions for controller design.
