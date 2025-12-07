import torch
from electrodrive.core.pinn_data import InteriorSampler, BoundarySampler

def test_interior_sampler_shapes_and_seed():
    s1 = InteriorSampler(domain=[[-1,1],[-2,2],[0,3]], seed=7)
    s2 = InteriorSampler(domain=[[-1,1],[-2,2],[0,3]], seed=7)
    a = s1.sample(10, device=torch.device("cpu"), dtype=torch.float32)
    b = s2.sample(10, device=torch.device("cpu"), dtype=torch.float32)
    assert a.shape == (10,3)
    assert torch.allclose(a, b)

def test_boundary_plane_and_sphere():
    plane = BoundarySampler(plane_L=3.0)
    P = plane.sample(8, device=torch.device("cpu"), dtype=torch.float32)
    assert P.shape == (8,3) and torch.allclose(P[:,2], torch.zeros(8))

    sphere = BoundarySampler(sphere_radius=2.5, sphere_center=(0.5,-0.5,1.0))
    S = sphere.sample(100, device=torch.device("cpu"), dtype=torch.float64)
    r = torch.linalg.norm(S - torch.tensor([0.5,-0.5,1.0]), dim=1)
    assert torch.allclose(r, torch.full_like(r, 2.5), atol=1e-6, rtol=0)

def test_interior_avoid_points():
    s = InteriorSampler(domain=[[-1,1],[-1,1],[-1,1]], seed=0)
    pts = s.sample(256, device=torch.device("cpu"), dtype=torch.float32,
                   avoid_points=[(0.0,0.0,0.0)], avoid_radius=0.05)
    d = torch.linalg.norm(pts, dim=1)
    assert (d >= 0.05).float().mean() > 0.9
