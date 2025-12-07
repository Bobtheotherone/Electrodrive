import math
from typing import Callable, Tuple, Optional, Dict, Any
try:
    import torch
except ImportError:
    torch = None
from electrodrive.utils.config import EPS_0, K_E
from electrodrive.utils.logging import JsonlLogger

Vec3 = Tuple[float, float, float]

# Vector helpers for analytic paths
def _vsub(a: Vec3, b: Vec3) -> Vec3: return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def _vmul(a: Vec3, s: float) -> Vec3: return (a[0]*s, a[1]*s, a[2]*s)
def _norm(a: Vec3) -> float: return math.sqrt(a[0]**2 + a[1]**2 + a[2]**2 + 1e-300)

def maxwell_force_on_conductor(solution, logger: Optional[JsonlLogger] = None) -> Tuple[float, float, float]:
    """
    P2.6: Compute the total force on a conductor using the electrostatic pressure (for BEM).
    F = ∫_S P n dA, where P = σ^2 / (2 eps0). This is derived from the Maxwell Stress Tensor.
    """
    # Check if the solution is a BEM solution and has required components (sigma, areas, normals).
    if hasattr(solution, '_S') and hasattr(solution, '_A') and hasattr(solution, '_N') and solution._N is not None:
        # BEM path
        sigma = solution._S
        areas = solution._A
        normals = solution._N

        # Pressure P = σ^2 / (2 eps0)
        pressure = (sigma**2) / (2.0 * EPS_0)

        # Force F = Σ P_i n_i A_i
        # [N] * [N, 3] * [N] -> [N, 3]
        force_vectors = pressure[:, None] * normals * areas[:, None]
        total_force = torch.sum(force_vectors, dim=0)

        F = total_force.detach().cpu().numpy()
        if logger:
            logger.info("Force on conductor computed via surface pressure (BEM).", Fx=float(F[0]), Fy=float(F[1]), Fz=float(F[2]))

        return (float(F[0]), float(F[1]), float(F[2]))

    else:
        if logger:
            logger.warning("Force calculation on conductor only implemented for BEM solutions with stored normals.")
        return (float('nan'), float('nan'), float('nan'))

def force_on_point_charge(solution, r0: Vec3, q: float, logger: Optional[JsonlLogger] = None) -> Tuple[float, float, float]:
    """
    P2.6: Compute the force on a point charge q at r0 due to the induced field.
    F = q * E_induced(r0).
    """
    # For analytic solutions (method of images), E_induced is the field from the image charges.
    meta = getattr(solution, 'meta', {})
    # Simplified check for single image systems (plane/sphere)
    if meta and 'image_charge' in meta and 'image_pos' in meta:
        q_img = meta['image_charge']
        r_img = meta['image_pos']

        R = _vsub(r0, r_img)
        dist = _norm(R)
        if dist < 1e-12:
            return (0.0, 0.0, 0.0)

        # Coulomb force between q and q_img
        F_mag = K_E * q * q_img / (dist**2)
        F_vec = _vmul(R, F_mag / dist)
        return F_vec

    # For BEM/PINN, extracting E_induced requires complex singularity handling or MST integration.
    if logger:
        logger.warning("Force on point charge only implemented for simple analytic image solutions.")
    return (float('nan'), float('nan'), float('nan'))

def dep_metric_field(solution, probe_points: 'torch.Tensor', logger: Optional[JsonlLogger]=None) -> 'torch.Tensor':
    """
    P2.6: Compute the DEP metric sampler ∇|E|^2 at probe points (GPU-batched).
    Uses Autograd on the E-field evaluator.
    """
    if torch is None:
        raise ImportError("PyTorch required for DEP metric field calculation.")

    if not hasattr(solution, 'eval_V_E_batched'):
        if logger: logger.warning("DEP metric calculation requires batched V/E evaluation (eval_V_E_batched).")
        return torch.full((probe_points.shape[0], 3), float('nan'))

    # Ensure points are on the correct device/dtype and require gradients
    P = probe_points.to(solution._device, solution._dtype).requires_grad_(True)

    try:
        # Evaluate E-field, allowing gradients to flow back to P
        # We must enable create_graph=True if we needed higher derivatives, but False here is fine.
        _, E = solution.eval_V_E_batched(P)

        # Compute |E|^2
        E2 = torch.sum(E**2, dim=1)

        # Compute ∇|E|^2
        grad_E2 = torch.autograd.grad(E2, P, grad_outputs=torch.ones_like(E2), create_graph=False)[0]

        return grad_E2.detach()

    except Exception as e:
        if logger: logger.error("Error during Autograd for DEP metric.", error=str(e))
        return torch.full((probe_points.shape[0], 3), float('nan'))

