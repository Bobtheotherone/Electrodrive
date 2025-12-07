"""MPI-based domain decomposition and communication.

Target size: ~1800 LOC.

Responsibilities
----------------
- Partition the global tree across MPI ranks:
    * space-filling-curve (Morton/Hilbert) based partitioning
    * load-balanced assignment of tree nodes / particles
- Manage halo / ghost regions:
    * data exchange for M2L interactions across rank boundaries
    * P2P near-field corrections across domains
- Provide:
    * convenience wrappers around ``mpi4py``
    * collective operations for timing / diagnostics.

For now this module only declares light scaffolding so the rest
of the system can import it without actually requiring MPI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover
    MPI = None  # type: ignore


@dataclass
class MpiContext:
    """Lightweight wrapper around an MPI communicator."""

    comm: Optional[object]  # MPI.Comm in real use
    rank: int = 0
    size: int = 1


def get_mpi_context() -> MpiContext:
    """Return an MPI context if mpi4py is available, else a dummy one."""
    if MPI is None:
        return MpiContext(comm=None, rank=0, size=1)
    comm = MPI.COMM_WORLD
    return MpiContext(comm=comm, rank=comm.Get_rank(), size=comm.Get_size())
