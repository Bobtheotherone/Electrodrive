import math

from electrodrive.experiments.run_discovery import (
    _proxyA_effective_ratio,
    _proxy_fail_count_noA,
    _proxy_score,
)


def test_proxyA_effective_ratio_logcap_inf() -> None:
    cap = 1e6
    metrics = {"proxy_gateA_worst_ratio": float("inf")}
    a_eff = _proxyA_effective_ratio(metrics, cap, "logcap")
    assert math.isfinite(a_eff)
    assert math.isclose(a_eff, math.log10(1.0 + cap))


def test_proxyA_effective_ratio_logcap_large() -> None:
    cap = 1e6
    metrics = {"proxy_gateA_worst_ratio": 1e40}
    a_eff = _proxyA_effective_ratio(metrics, cap, "logcap")
    assert math.isfinite(a_eff)
    assert math.isclose(a_eff, math.log10(1.0 + cap))


def test_proxyA_effective_ratio_monotonic_small() -> None:
    cap = 1e6
    vals = [0.0, 1.0, 10.0]
    effs = [_proxyA_effective_ratio({"proxy_gateA_worst_ratio": v}, cap, "logcap") for v in vals]
    assert effs[0] < effs[1] < effs[2]


def test_proxy_ranking_balanced_prioritizes_noA() -> None:
    thresholds = {"bc_continuity": 5e-3, "slope_tol": 0.15, "stability": 5e-2}
    cap = 1e6
    transform = "logcap"
    weight = 1.0

    m1 = {
        "proxy_gateA_worst_ratio": 1e12,
        "proxy_gateB_max_v_jump": 0.0,
        "proxy_gateB_max_d_jump": 0.0,
        "proxy_gateC_far_slope": -1.0,
        "proxy_gateC_near_slope": -1.0,
        "proxy_gateC_spurious_fraction": 0.0,
        "proxy_gateD_rel_change": 0.0,
    }
    m2 = {
        "proxy_gateA_worst_ratio": 1.0,
        "proxy_gateB_max_v_jump": 1e-2,
        "proxy_gateB_max_d_jump": 0.0,
        "proxy_gateC_far_slope": -1.0,
        "proxy_gateC_near_slope": -1.0,
        "proxy_gateC_spurious_fraction": 0.0,
        "proxy_gateD_rel_change": 0.0,
    }

    key1 = (
        _proxy_fail_count_noA(m1, thresholds),
        _proxyA_effective_ratio(m1, cap, transform),
        _proxy_score(m1, a_weight=weight, a_cap=cap, a_transform=transform),
        0.0,
    )
    key2 = (
        _proxy_fail_count_noA(m2, thresholds),
        _proxyA_effective_ratio(m2, cap, transform),
        _proxy_score(m2, a_weight=weight, a_cap=cap, a_transform=transform),
        0.0,
    )

    assert key1 < key2


def test_proxy_score_sanitizes_nonfinite_metrics() -> None:
    metrics = {
        "proxy_gateA_worst_ratio": 1.0,
        "proxy_gateB_max_v_jump": "Infinity",
        "proxy_gateB_max_d_jump": 0.0,
        "proxy_gateC_far_slope": -1.0,
        "proxy_gateC_near_slope": -1.0,
        "proxy_gateC_spurious_fraction": 0.0,
        "proxy_gateD_rel_change": "NaN",
    }
    score = _proxy_score(metrics, a_weight=1.0, a_cap=1e6, a_transform="logcap")
    assert math.isfinite(score)
