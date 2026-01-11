import json
import sys
from pathlib import Path

from tools.stage9 import analyze_verifier_results


def test_stage9_analyze_verifier_results_smoke(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    cert_dir = run_dir / "artifacts" / "certificates"
    cert_dir.mkdir(parents=True)

    verify_dir = run_dir / "verify"
    verify_dir.mkdir(parents=True)

    cert = {
        "gates": {
            "A": {
                "status": "pass",
                "metrics": {"linf": 0.001, "l2": 0.001},
                "thresholds": {"linf": 0.01, "l2": 0.01},
            },
            "B": {
                "status": "pass",
                "metrics": {"interface_max_v_jump": 0.001, "interface_max_d_jump": 0.001},
                "thresholds": {"continuity": 0.01},
            },
            "C": {
                "status": "fail",
                "metrics": {"far_slope": -0.8, "near_slope": -0.9, "spurious_fraction": 0.1},
                "thresholds": {"slope_tol": 0.05},
            },
            "D": {
                "status": "pass",
                "metrics": {"relative_change": 0.001},
                "thresholds": {"stability_tol": 0.01},
            },
            "E": {
                "status": "pass",
                "metrics": {"speedup": 2.0},
                "thresholds": {"min_speedup": 1.2},
            },
        },
        "final_status": "fail",
    }
    (verify_dir / "discovery_certificate.json").write_text(json.dumps(cert), encoding="utf-8")

    summary = {"verification": {"path": str(verify_dir)}, "generation": 0, "rank": 0}
    (cert_dir / "candidate_0_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["analyze_verifier_results.py", str(run_dir)])
    analyze_verifier_results.main()

    summary_path = run_dir / "analysis" / "analysis_summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    hist = data.get("hist_pass_k", {})
    assert hist.get("4", hist.get(4)) == 1
