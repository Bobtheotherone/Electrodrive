import torch

from electrodrive.images.basis_dcim import DCIMBlockBasis
from electrodrive.layers.dcim_types import ComplexImageTerm, DCIMBlock, DCIMCertificate
from electrodrive.layers.spectral_kernels import SpectralKernelSpec
from electrodrive.layers.stack import Layer, LayerInterface, LayerStack
from electrodrive.utils.device import ensure_cuda_available_or_skip


def _make_block() -> DCIMBlock:
    layers = (
        Layer(name="region1", eps=1.0 + 0.0j, z_min=0.0, z_max=1.0),
        Layer(name="slab", eps=4.0 + 0.0j, z_min=-0.5, z_max=0.0),
        Layer(name="region3", eps=1.0 + 0.0j, z_min=-1.0, z_max=-0.5),
    )
    interfaces = (
        LayerInterface(z=0.0, upper=0, lower=1),
        LayerInterface(z=-0.5, upper=1, lower=2),
    )
    z_bounds = (1.0, 0.0, -0.5, -1.0)
    stack = LayerStack(layers=layers, interfaces=interfaces, z_bounds=z_bounds)
    kernel = SpectralKernelSpec(
        source_region=0,
        obs_region=0,
        component="potential",
        bc_kind="dielectric_interfaces",
    )
    images = (
        ComplexImageTerm(depth=0.6 + 0.2j, weight=0.1 - 0.05j),
    )
    certificate = DCIMCertificate(
        k_grid=(0.1, 0.2),
        fit_residual_L2=0.0,
        fit_residual_Linf=0.0,
        spatial_check_rel_L2=0.0,
        spatial_check_rel_Linf=0.0,
        stable=True,
        meta={"source_pos": (0.0, 0.0, 0.2), "source_charge": 1.0, "z_ref": 0.0},
    )
    return DCIMBlock(stack=stack, kernel=kernel, poles=tuple(), images=images, certificate=certificate)


def test_dcim_block_eval_shape_device_dtype():
    ensure_cuda_available_or_skip("dcim block basis evaluation")
    device = torch.device("cuda")
    targets = torch.randn(12, 3, device=device, dtype=torch.float32)
    block = _make_block()
    for component in ("real", "imag"):
        elem = DCIMBlockBasis(block=block, images=block.images, component=component)
        out = elem.potential(targets)
        assert out.is_cuda
        assert out.dtype == targets.dtype
        assert out.shape == (targets.shape[0],)
        assert torch.isfinite(out).all()
