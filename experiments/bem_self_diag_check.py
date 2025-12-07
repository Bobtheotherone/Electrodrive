import math
import random
import pathlib
import numpy as np

from electrodrive.utils.config import EPS_0, K_E
from electrodrive.core.bem_quadrature import self_integral_correction

# Single triangle with centroid at origin, edges ~0.2
v0 = np.array([-0.1, -0.1, 0.0], dtype=float)
v1 = np.array([0.1, -0.1, 0.0], dtype=float)
v2 = np.array([0.0, 0.1, 0.0], dtype=float)
verts = np.stack([v0, v1, v2], axis=0)

# Area
area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
centroid = verts.mean(axis=0)

# Monte Carlo integration of K_E/|r-r'| over triangle
rng = random.Random(12345)
N = 500000
sum_val = 0.0
for _ in range(N):
    # uniform barycentric sampling
    r1 = rng.random()
    r2 = rng.random()
    if r1 + r2 > 1.0:
        r1, r2 = 1.0 - r1, 1.0 - r2
    r3 = 1.0 - r1 - r2
    p = r1 * v0 + r2 * v1 + r3 * v2
    r = np.linalg.norm(p - centroid)
    r = max(r, 1e-12)
    sum_val += K_E / r
I_MC = area * (sum_val / N)

I_self = self_integral_correction(area)
sigma = 1.0
K_diag_used = I_self
V_diag_used = K_diag_used * sigma * area

ratios = {
    "I_self_over_I_MC": I_self / I_MC,
    "I_self_over_A_over_I_MC": (I_self / area) / I_MC,
    "V_diag_over_I_MC": V_diag_used / I_MC,
}

out_text = []
out_text.append("# BEM Self Integral Check")
out_text.append(f"Area A = {area:.6e} m^2")
out_text.append(f"I_MC (MC integral) = {I_MC:.6e}")
out_text.append(f"I_self (equal-area disk) = {I_self:.6e}")
out_text.append(f"I_self / A = {I_self/area:.6e}")
out_text.append(f"V_diag_used = {V_diag_used:.6e} (K_diag * sigma * A)")
out_text.append("")
out_text.append("Ratios:")
for k, v in ratios.items():
    out_text.append(f"- {k}: {v:.6e}")

out_path = pathlib.Path("experiments/bem_self_diag_check_results.md")
out_path.write_text("\n".join(out_text), encoding="utf-8")

print("\n".join(out_text))
