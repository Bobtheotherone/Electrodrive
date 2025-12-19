# three_layer_complex checker

Run a minimal GPU checker to summarise the new layered basis candidates. The experimental basis is opt-in via `three_layer_complex` (e.g. `--basis point,axis_point,three_layer_images,three_layer_complex`). CUDA is required; there is no CPU fallback.

```bash
EDE_DEVICE=cuda EDE_DTYPE=float32 python notes/check_three_layer_complex.py
```

Outputs a JSON summary under `runs/checks/three_layer_complex/run_<timestamp>/summary.json` without overwriting prior runs.

Tiny discovery example using the experimental basis:

```bash
python -m electrodrive.tools.images_discover discover \
  --spec specs/planar_three_layer_eps2_80_sym_h04_region1.json \
  --basis point,axis_point,three_layer_images,three_layer_complex \
  --nmax 12 --reg-l1 1e-3 --out runs/demo_three_layer_complex
```
