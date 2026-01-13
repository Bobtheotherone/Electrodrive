from electrodrive.experiments.run_discovery import _proxy_score


def test_proxy_score_monotonicity() -> None:
    metrics_good = {
        "proxy_gateA_worst_ratio": 0.1,
        "proxy_gateB_max_v_jump": 1e-4,
        "proxy_gateB_max_d_jump": 1e-4,
        "proxy_gateC_far_slope": -1.0,
        "proxy_gateC_near_slope": -1.0,
        "proxy_gateC_spurious_fraction": 0.0,
        "proxy_gateD_rel_change": 1e-3,
    }
    metrics_bad = {
        "proxy_gateA_worst_ratio": 10.0,
        "proxy_gateB_max_v_jump": 1e-1,
        "proxy_gateB_max_d_jump": 1e-1,
        "proxy_gateC_far_slope": 0.5,
        "proxy_gateC_near_slope": 0.2,
        "proxy_gateC_spurious_fraction": 0.5,
        "proxy_gateD_rel_change": 0.5,
    }
    score_good = _proxy_score(metrics_good, a_weight=1.0, a_cap=100.0, a_transform="logcap")
    score_bad = _proxy_score(metrics_bad, a_weight=1.0, a_cap=100.0, a_transform="logcap")
    assert score_bad > score_good
