# electrodrive.gfn

Experimental GFlowNet stack used in Step-6 structure discovery. The package is
kept import-safe and GPU-first, with subpackages owning narrow boundaries:

- WARNING: Step-6 development assumes a CUDA-enabled PyTorch install. The
  current venv reports CUDA unavailable; GPU-first behaviors and future prompts
  cannot be validated without a working CUDA runtime.

- `dsl`: program AST, actions, grammar, canonicalization, and hashing.
- `env`: rollout state containers and cached embeddings.
- `policy`, `losses`, `rollout`, `replay`, `reward`, `train`: reserved for
  learning, sampling, and training loops.
- `integration`: bridge from a compiled program to the existing solver
  pipeline; currently stubbed.

All heavy logic stays within this tree to avoid disrupting existing discovery
modes until the GFlowNet path is ready.
