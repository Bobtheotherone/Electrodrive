# Stage 10 Gate B root cause triage

## Run analyzed
- run: `runs/20260112_001447_black_hammer_push`
- triage tool: `tools/stage10/triage_gateB.py`
- outputs: `stage10/audit/20260112_001447_black_hammer_push/triage/gateB_triage.md`

## Evidence summary
Across top candidates (gen0-2 ranks), Gate B fails with dominant **interface displacement** residuals. Representative values:

- candidate_only: interface_max_d_jump ~2.5e9 to 8.2e9; interface_max_v_jump ~3.9e5 to 1.4e6; dirichlet_max_err ~0
- candidate_plus_reference: interface_max_d_jump ~6.6e10; interface_max_v_jump ~9.8e6; worse by ~10x
- candidate_minus_reference: interface_max_d_jump ~6.6e10; interface_max_v_jump ~9.8e6; worse by ~10x

This pattern is consistent across sampled candidates and generations, so the failure is **global**, not localized to a single singularity.

## Conclusion
The reference +/- toggle **does not** reduce Gate B (it worsens by an order of magnitude), so this is **not** a reference mismatch. The dominant failure is epsilon-weighted displacement continuity at the interface, indicating the solver fit did not enforce interface physics.

## Selected fix (Case B)
Introduce explicit interface continuity constraints in the layered weight solve:
- enforce `phi_plus - phi_minus = 0`
- enforce `eps_plus * dphi_dn_plus - eps_minus * dphi_dn_minus = 0`

Expose via Stage 10 config flags:
- `run.layered_enforce_interface_constraints`
- `run.layered_interface_constraint_weight`
- `run.layered_interface_constraint_points`

This targets the Gate B displacement continuity failure without weakening verifier gates.
