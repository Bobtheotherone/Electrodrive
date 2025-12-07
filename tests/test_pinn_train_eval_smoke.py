import types
import numpy as np
from electrodrive.core.pinn import pinn_train_eval

class MiniLogger:
    def info(self, *a, **k): pass
    def error(self, *a, **k): pass
    def warning(self, *a, **k): pass

def test_pinn_train_eval_one_epoch_cpu():
    # Minimal spec: grounded plane, single point charge above it
    spec = types.SimpleNamespace(
        conductors=[{"type":"plane"}],
        charges=[{"type":"point", "pos":[0.0,0.0,0.5], "q":1.0}],
    )
    cfg = types.SimpleNamespace(
        fp64=False, seed=123, use_gpu=False,
        width=16, depth=3,
        n_collocation=64, n_boundary=32,
        epochs=1, early_stop_patience=0,
        learning_rate=1e-3, w_pde=1.0, w_bc=1.0
    )
    out = pinn_train_eval(spec, cfg, MiniLogger())
    assert "solution" in out and "bc_rmse" in out and np.isfinite(out["bc_rmse"])
    assert isinstance(out["boundary_samples"], list)
