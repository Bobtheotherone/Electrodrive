"""Integration bridges between the GFlowNet generator and existing solvers."""

from electrodrive.gfn.integration.compile import compile_program_to_basis
from electrodrive.gfn.integration.gfn_basis_generator import GFlowNetProgramGenerator

__all__ = ["compile_program_to_basis", "GFlowNetProgramGenerator"]
