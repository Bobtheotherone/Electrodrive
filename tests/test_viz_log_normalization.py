import pytest

from electrodrive.viz import ai_solve, iter_viz


def test_ai_solve_handles_msg_and_resid_variants():
    records = [
        {"msg": "gmres_iter", "iter": 5, "resid_precond": 1e-3},
        {"event": "gmres_iter", "step": 6, "resid_true": 1e-4},
    ]
    iters, resids, _ = ai_solve._extract_solver_trace(records)  # type: ignore[attr-defined]

    assert iters == [5, 6]
    assert resids == [pytest.approx(1e-3), pytest.approx(1e-4)]


def test_iter_viz_parses_iter_and_resid_variants():
    rec = {"message": "GMRES Iter.", "k": 7, "resid_true_l2": 5e-5}
    sample = iter_viz._parse_iter_event(rec)

    assert sample is not None
    assert sample.iter == 7
    assert sample.resid == pytest.approx(5e-5)
