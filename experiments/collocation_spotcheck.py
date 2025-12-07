import json
import pathlib
import numpy as np
import torch

from electrodrive.learn.collocation import make_collocation_batch_for_spec
from electrodrive.utils.config import EPS_0
from tests.test_bem_quadrature import _build_plane_spec, _build_sphere_spec


def run_case(name, spec_builder, geom_type):
    spec = spec_builder()
    n_points = 256
    ratio_boundary = 0.5
    device = torch.device("cpu")
    dtype = torch.float64
    rng1 = np.random.default_rng(1234)
    rng2 = np.random.default_rng(1234)

    batch_a = make_collocation_batch_for_spec(
        spec=spec,
        n_points=n_points,
        ratio_boundary=ratio_boundary,
        supervision_mode="analytic",
        device=device,
        dtype=dtype,
        rng=rng1,
        geom_type=geom_type,
    )
    batch_b = make_collocation_batch_for_spec(
        spec=spec,
        n_points=n_points,
        ratio_boundary=ratio_boundary,
        supervision_mode="bem",
        device=device,
        dtype=dtype,
        rng=rng2,
        geom_type=geom_type,
    )

    X = batch_a["X"]
    assert torch.allclose(X, batch_b["X"]), "collocation points mismatch"
    mask = batch_a["mask_finite"] & batch_b["mask_finite"]
    Va = batch_a["V_gt"][mask]
    Vb = batch_b["V_gt"][mask] * EPS_0

    abs_err = torch.abs(Va - Vb)
    rel_err = abs_err / (torch.abs(Va) + torch.abs(Vb) + torch.tensor(1e-9, dtype=Va.dtype))

    def percentile(t, q):
        return float(torch.quantile(t, q / 100.0).item()) if t.numel() > 0 else float('nan')

    stats = {
        "num_points": int(mask.sum().item()),
        "abs_err_max": float(abs_err.max().item()),
        "abs_err_mean": float(abs_err.mean().item()),
        "abs_err_median": percentile(abs_err, 50),
        "rel_err_max": float(rel_err.max().item()),
        "rel_err_mean": float(rel_err.mean().item()),
        "rel_err_median": percentile(rel_err, 50),
        "rel_err_p95": percentile(rel_err, 95),
    }

    idx = torch.arange(rel_err.numel())
    step = max(1, rel_err.numel() // 10)
    sel = idx[::step][:10]
    samples = []
    Xsel = X[mask]
    for i in sel.tolist():
        p = Xsel[i]
        samples.append(
            {
                "point": [float(x) for x in p.tolist()],
                "Va": float(Va[i].item()),
                "Vb_comp": float(Vb[i].item()),
                "rel_err": float(rel_err[i].item()),
            }
        )

    return stats, samples


def main():
    results = {}
    summaries = []
    for name, builder, geom in [
        ("plane", _build_plane_spec, "plane"),
        ("sphere", _build_sphere_spec, "sphere"),
    ]:
        stats, samples = run_case(name, builder, geom)
        results[name] = {"stats": stats, "samples": samples}
        summaries.append(
            f"{name}: rel_err max={stats['rel_err_max']:.3e}, mean={stats['rel_err_mean']:.3e}, median={stats['rel_err_median']:.3e}, p95={stats['rel_err_p95']:.3e}"
        )

    out_path = pathlib.Path("experiments/collocation_spotcheck_results.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n".join(summaries))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
