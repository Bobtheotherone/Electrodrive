# Step 10 final checklist

## Import gate
Command:
python -c "import electrodrive; print('import_ok')"
Output:
import_ok

## CUDA gate
Command:
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
Output:
NVIDIA GeForce RTX 5090 Laptop GPU

## Targeted pytest (step10)
Command:
pytest -k "step10_" -q
Output (failed):
ERROR temp_AI_upgrade/electrodrive/gfdsl/tests/test_ast_roundtrip.py
ERROR temp_AI_upgrade/electrodrive/gfdsl/tests/test_compile_context_fields.py
ERROR temp_AI_upgrade/electrodrive/gfdsl/tests/test_eval_complex_pair.py
ERROR temp_AI_upgrade/electrodrive/gfdsl/tests/test_eval_real_primitives.py
ERROR temp_AI_upgrade/electrodrive/gfdsl/tests/test_fixed_nodes_roundtrip.py
ERROR temp_AI_upgrade/electrodrive/gfdsl/tests/test_fixed_term_convention.py
ERROR temp_AI_upgrade/electrodrive/gfdsl/tests/test_gpu_first_params.py
ERROR temp_AI_upgrade/electrodrive/gfdsl/tests/test_gradients.py
ERROR temp_AI_upgrade/electrodrive/gfdsl/tests/test_hash_stability.py
ERROR temp_AI_upgrade/electrodrive/gfdsl/tests/test_layered_nodes_placeholder.py
ERROR temp_AI_upgrade/electrodrive/gfdsl/tests/test_legacy_adapter.py
ERROR temp_AI_upgrade/electrodrive/gfdsl/tests/test_lower_to_dense_operator_parity.py
ERROR temp_AI_upgrade/electrodrive/gfdsl/tests/test_macro_mirror_ladder.py
ERROR temp_AI_upgrade/electrodrive/gfdsl/tests/test_opaque_unknown_node.py
ERROR temp_AI_upgrade/electrodrive/gfdsl/tests/test_operator_no_dense_fallback.py
ERROR temp_AI_upgrade/electrodrive/gfdsl/tests/test_validation_errors.py
ModuleNotFoundError: No module named 'electrodrive.gfdsl'
Interrupted: 16 errors during collection

## CLI help
Command:
python -m electrodrive.tools.images_discover --help
Output:
usage: images_discover.py [-h] {discover,eval} ...

Electrodrive sparse Method-of-Images discovery CLI

positional arguments:
  {discover,eval}
    discover       Run sparse image discovery for a given CanonicalSpec.
    eval           Evaluate a saved image system against a spec
                   (experimental).

options:
  -h, --help       show this help message and exit

## gfn_flow smoke run
Command:
python - <<'PY' ... (see run log in shell history)
Output:
smoke_ok elements=1 out=runs/step10_gfn_flow_smoke/run_1766285577

## Reward cache CPU cleanup
Lines removed: group_ids_cached = group_ids_cached.to("cpu"); device="cpu" if serialize else device
Tests: pytest --ignore=temp_AI_upgrade -k "step10_" -q (11 passed)
Grep audit: clean (no cpu patterns in diff)
