## Iteration 1 (baseline)
- Action: Ran pytest tests/test_bem_quadrature.py::test_analytic_matches_bem_up_to_units[_build_plane_spec-plane-True] -q to reproduce issue.
- Result: Warning emitted '[FMM-SPECTRAL-WARN] Failed to set up near-field quadrature; continuing without it.' on both refine passes; test failed with rel_err ~0.1587 > 1e-2.
- Notes: Warning originates from electrodrive/core/bem.py try/except around near-pair setup (~lines 1475-1555). Need to instrument except to surface actual exception next.
- Status: Baseline reproduced, warning present, accuracy poor.
## Iteration 2 (instrumented failure)
- Action: Added traceback logging/re-raise in near-field setup and re-ran pytest tests/test_bem_quadrature.py::test_analytic_matches_bem_up_to_units[_build_plane_spec-plane-True] -q.
- Result: BEM aborted; captured error: ConsoleLogger.info() takes 2 positional arguments but 3 were given. Stack shows near-field setup failing; no collocation points returned.
- Notes: Error originates from logger.info call in bem.py near-field matvec setup (GPU fallback path) using positional device argument; logger expects keyword fields.
- Status: Root cause identified (logging call crash), near-field disabled.
## Iteration 3 (logging fix + precompute)
- Action: Fixed logger call in near-field setup, restored warning handling, added precomputation of near-field matvec correction weights to avoid per-iteration quadrature, reran pytest tests/test_bem_quadrature.py::test_analytic_matches_bem_up_to_units[_build_plane_spec-plane-True].
- Result: No near-field setup warnings; test completed (~121s) but still fails with rel_err ~0.159. Near-field corrections logged as enabled on CPU after precomputing weights.
- Notes: Precomputing near-field weights for pass1 (3200 panels, 49k pairs) takes ~24.5s; pass2 (4802 panels, 74k pairs) estimated ~40s. Accuracy gap persists despite near quadrature.
- Status: Warning resolved; accuracy issue remains.
