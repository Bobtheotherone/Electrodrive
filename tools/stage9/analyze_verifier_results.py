from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


GATE_ORDER = ("A", "B", "C", "D", "E")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_summaries(run_dir: Path) -> List[Path]:
    cert_dir = run_dir / "artifacts" / "certificates"
    if not cert_dir.exists():
        return []
    return sorted(cert_dir.glob("*_summary.json"))


def _gate_statuses(cert: Dict[str, Any]) -> Dict[str, str]:
    gates = cert.get("gates", {}) if isinstance(cert.get("gates", {}), dict) else {}
    statuses: Dict[str, str] = {}
    for name in GATE_ORDER:
        gate = gates.get(name, {})
        statuses[name] = str(gate.get("status", "unknown"))
    return statuses


def _gate_metrics(cert: Dict[str, Any], name: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    gates = cert.get("gates", {}) if isinstance(cert.get("gates", {}), dict) else {}
    gate = gates.get(name, {}) if isinstance(gates.get(name, {}), dict) else {}
    metrics = gate.get("metrics", {}) if isinstance(gate.get("metrics", {}), dict) else {}
    thresholds = gate.get("thresholds", {}) if isinstance(gate.get("thresholds", {}), dict) else {}
    return (
        {str(k): float(v) for k, v in metrics.items()},
        {str(k): float(v) for k, v in thresholds.items()},
    )


def _gate_margin(name: str, cert: Dict[str, Any]) -> float:
    metrics, thresholds = _gate_metrics(cert, name)
    if name == "A":
        linf = metrics.get("linf", float("inf"))
        l2 = metrics.get("l2", float("inf"))
        t_linf = thresholds.get("linf", float("inf"))
        t_l2 = thresholds.get("l2", t_linf)
        return max(linf - t_linf, l2 - t_l2)
    if name == "B":
        v_jump = metrics.get("interface_max_v_jump", float("inf"))
        d_jump = metrics.get("interface_max_d_jump", float("inf"))
        tol = thresholds.get("continuity", float("inf"))
        return max(v_jump - tol, d_jump - tol)
    if name == "C":
        far_slope = metrics.get("far_slope", float("inf"))
        near_slope = metrics.get("near_slope", float("inf"))
        slope_tol = thresholds.get("slope_tol", float("inf"))
        slope_margin = max(abs(far_slope + 1.0), abs(near_slope + 1.0)) - slope_tol
        spurious = metrics.get("spurious_fraction", float("inf"))
        spurious_margin = spurious - 0.05
        return max(slope_margin, spurious_margin)
    if name == "D":
        rel_change = metrics.get("relative_change", float("inf"))
        tol = thresholds.get("stability_tol", float("inf"))
        return rel_change - tol
    if name == "E":
        speedup = metrics.get("speedup", float("-inf"))
        tol = thresholds.get("min_speedup", float("inf"))
        return tol - speedup
    return float("inf")


def _hist(values: List[float], bins: int = 40) -> Dict[str, Any]:
    if not values:
        return {"bins": [], "counts": []}
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return {"bins": [], "counts": []}
    hist, edges = np.histogram(values, bins=bins)
    return {"bins": [float(x) for x in edges.tolist()], "counts": [int(x) for x in hist.tolist()]}


def _plot_hist(values: List[float], out_path: Path, title: str) -> None:
    if not values:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(values, bins=40)
    ax.set_title(title)
    ax.set_xlabel(title)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_pass_k_hist(hist: Dict[int, int], out_path: Path) -> None:
    if not hist:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return
    labels = sorted(hist.keys())
    counts = [hist[k] for k in labels]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, counts)
    ax.set_title("hist_pass_k")
    ax.set_xlabel("gates_passed")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage 9 verifier results.")
    parser.add_argument("run_dir", help="Run directory with artifacts/certificates.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summaries = _find_summaries(run_dir)
    if not summaries:
        print("No summary files found.")
        return

    counts = {"A": 0, "AB": 0, "ABC": 0, "ABCD": 0, "ABCDE": 0}
    hist_pass_k = {k: 0 for k in range(len(GATE_ORDER) + 1)}
    lap_vals: List[float] = []
    bc_vals: List[float] = []
    asym_vals: List[float] = []
    stab_vals: List[float] = []
    speed_vals: List[float] = []
    near_misses: List[Dict[str, Any]] = []
    best_candidates: List[Dict[str, Any]] = []
    best_partial: Dict[str, Any] | None = None

    for summary_path in summaries:
        summary = _load_json(summary_path)
        verification = summary.get("verification", {}) if isinstance(summary.get("verification", {}), dict) else {}
        verify_path = verification.get("path", None)
        if not verify_path:
            continue
        cert_path = Path(verify_path) / "discovery_certificate.json"
        if not cert_path.exists():
            continue
        cert = _load_json(cert_path)
        statuses = _gate_statuses(cert)
        status_list = [statuses.get(g, "unknown") for g in GATE_ORDER]
        pass_flags = [s == "pass" for s in status_list]
        pass_k = int(sum(pass_flags))
        hist_pass_k[pass_k] = hist_pass_k.get(pass_k, 0) + 1
        gates_passed = [g for g, ok in zip(GATE_ORDER, pass_flags) if ok]
        failed_gates = [g for g, ok in zip(GATE_ORDER, pass_flags) if not ok]
        fail_margins = {g: _gate_margin(g, cert) for g in failed_gates}
        fail_margin_sum = float(sum(fail_margins.values())) if fail_margins else 0.0

        if pass_flags[0]:
            counts["A"] += 1
        if all(pass_flags[:2]):
            counts["AB"] += 1
        if all(pass_flags[:3]):
            counts["ABC"] += 1
        if all(pass_flags[:4]):
            counts["ABCD"] += 1
        if all(pass_flags):
            counts["ABCDE"] += 1

        metrics_a, _ = _gate_metrics(cert, "A")
        metrics_b, _ = _gate_metrics(cert, "B")
        metrics_c, _ = _gate_metrics(cert, "C")
        metrics_d, _ = _gate_metrics(cert, "D")
        metrics_e, _ = _gate_metrics(cert, "E")

        if "linf" in metrics_a and math.isfinite(metrics_a["linf"]):
            lap_vals.append(metrics_a["linf"])
        if "interface_max_v_jump" in metrics_b and math.isfinite(metrics_b["interface_max_v_jump"]):
            bc_vals.append(metrics_b["interface_max_v_jump"])
        if "far_slope" in metrics_c and "near_slope" in metrics_c:
            asym = abs(metrics_c["far_slope"] + 1.0) + abs(metrics_c["near_slope"] + 1.0)
            if math.isfinite(asym):
                asym_vals.append(asym)
        if "relative_change" in metrics_d and math.isfinite(metrics_d["relative_change"]):
            stab_vals.append(metrics_d["relative_change"])
        if "speedup" in metrics_e and math.isfinite(metrics_e["speedup"]):
            speed_vals.append(metrics_e["speedup"])

        failed = [g for g, ok in zip(GATE_ORDER, pass_flags) if not ok]
        if len(failed) == 1:
            gate = failed[0]
            margin = _gate_margin(gate, cert)
            near_misses.append(
                {
                    "generation": summary.get("generation"),
                    "rank": summary.get("rank"),
                    "gate": gate,
                    "margin": float(margin),
                    "summary_path": str(summary_path),
                    "verification_path": str(verify_path),
                }
            )

        candidate_summary = {
            "generation": summary.get("generation"),
            "rank": summary.get("rank"),
            "summary_path": str(summary_path),
            "verification_path": str(verify_path),
            "pass_k": pass_k,
            "gates_passed": gates_passed,
            "failed_gates": failed_gates,
            "fail_margin_sum": fail_margin_sum,
            "fail_margins": fail_margins,
        }
        if best_partial is None:
            best_partial = candidate_summary
        elif pass_k > int(best_partial.get("pass_k", -1)):
            best_partial = candidate_summary
        elif pass_k == int(best_partial.get("pass_k", -1)) and fail_margin_sum < float(
            best_partial.get("fail_margin_sum", float("inf"))
        ):
            best_partial = candidate_summary

        if cert.get("final_status") == "pass":
            best_candidates.append(
                {
                    "generation": summary.get("generation"),
                    "rank": summary.get("rank"),
                    "summary_path": str(summary_path),
                    "verification_path": str(verify_path),
                    "final_status": cert.get("final_status"),
                }
            )

    near_misses.sort(key=lambda x: x.get("margin", float("inf")))
    best_candidates.sort(key=lambda x: (x.get("generation", 0), x.get("rank", 0)))

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _plot_hist(lap_vals, analysis_dir / "hist_gateA_linf.png", "gateA_linf")
    _plot_hist(bc_vals, analysis_dir / "hist_gateB_interface_jump.png", "gateB_interface_jump")
    _plot_hist(asym_vals, analysis_dir / "hist_gateC_asym_error.png", "gateC_asym_error")
    _plot_hist(stab_vals, analysis_dir / "hist_gateD_relative_change.png", "gateD_relative_change")
    _plot_hist(speed_vals, analysis_dir / "hist_gateE_speedup.png", "gateE_speedup")
    _plot_pass_k_hist(hist_pass_k, analysis_dir / "hist_pass_k.png")

    summary_out = {
        "counts": counts,
        "hist_pass_k": hist_pass_k,
        "histograms": {
            "gateA_linf": _hist(lap_vals),
            "gateB_interface_jump": _hist(bc_vals),
            "gateC_asym_error": _hist(asym_vals),
            "gateD_relative_change": _hist(stab_vals),
            "gateE_speedup": _hist(speed_vals),
        },
        "near_misses": near_misses[:10],
        "best_candidates": best_candidates[:10],
        "best_partial": best_partial,
    }
    (analysis_dir / "analysis_summary.json").write_text(
        json.dumps(summary_out, indent=2), encoding="utf-8"
    )

    print(f"analysis_dir={analysis_dir}")
    print(f"counts={counts}")
    print(f"hist_pass_k={hist_pass_k}")
    if near_misses:
        print("near_misses:")
        for entry in near_misses[:5]:
            print(f"  gate={entry['gate']} margin={entry['margin']:.3g} summary={entry['summary_path']}")
    if best_partial:
        print(
            "best_partial="
            f"pass_k={best_partial.get('pass_k')} "
            f"gates={best_partial.get('gates_passed')} "
            f"summary={best_partial.get('summary_path')}"
        )
    if best_candidates:
        print(f"best_candidates={len(best_candidates)}")


if __name__ == "__main__":
    main()
