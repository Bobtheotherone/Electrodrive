"""GFlowNet program generation scaffold (Step-6 experimental stack).

This package is intentionally self-contained to avoid disturbing existing
discovery modes. Public boundaries:

- :mod:`electrodrive.gfn.dsl`: program AST, actions, grammar, canonicalization.
- :mod:`electrodrive.gfn.env`: rollout state containers and cached embeddings.
- :mod:`electrodrive.gfn.integration`: adapters into the existing solver stack.
"""

from electrodrive.gfn.dsl.action import Action
from electrodrive.gfn.dsl.grammar import Grammar
from electrodrive.gfn.dsl.program import Program
from electrodrive.gfn.env import ElectrodriveProgramEnv, PartialProgramState, SpecMetadata
from electrodrive.gfn.integration.compile import compile_program_to_basis
from electrodrive.gfn.integration.gfn_basis_generator import GFlowNetProgramGenerator

__all__ = [
    "Action",
    "Grammar",
    "ElectrodriveProgramEnv",
    "PartialProgramState",
    "Program",
    "SpecMetadata",
    "compile_program_to_basis",
    "GFlowNetProgramGenerator",
]
