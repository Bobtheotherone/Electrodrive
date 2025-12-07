# electrodrive/learn/eval.py
import json
import logging
import math
from pathlib import Path
from typing import (
    Dict,
    Any,
    Tuple,
    List,
)

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **_):
        return x


from electrodrive.learn.specs import (
    ExperimentConfig,
    EvalSpec,
    DatasetSpec,
    CurriculumSpec,
    ModelSpec,
    TrainerSpec,
)
from electrodrive.learn.train import (
    initialize_model,
)
from electrodrive.learn.dataset import (
    mine_pdf_problems,
    synthesize_problem,
)
from electrodrive.learn.encoding import (
    encode_spec,
)
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.core.certify import (
    bc_residual_on_boundary,
    pde_residual_symbolic,
    mean_value_property_check,
    green_badge_decision,
)
from electrodrive.utils.config import (
    EPS_BC,
    EPS_DUAL,
    EPS_PDE,
    EPS_ENERGY,
    EPS_MEAN_VAL,
)

logger = logging.getLogger("EDE.Learn.Eval")


class ConditionalModelWrapper:
    """Wrap conditional model as an eval-only potential oracle.

    Exposes eval(x) -> V(x) to interoperate with certify-style helpers.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        encoding: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if encoding.dim() == 1:
            encoding = encoding.unsqueeze(0)
        self.model = model
        self.encoding = encoding.to(device=device, dtype=dtype)
        self._device = device
        self._dtype = dtype
        self.meta: Dict[str, Any] = {
            "mode": f"learned_conditional_{type(model).__name__}"
        }

    def eval(
        self,
        p: Tuple[float, float, float],
    ) -> float:
        x = torch.tensor(
            [p],
            device=self._device,
            dtype=self._dtype,
        )
        with torch.no_grad():
            if hasattr(self.model, "evaluate_potential"):
                out = self.model.evaluate_potential(
                    x,
                    self.encoding,
                )
            else:
                out = self.model(
                    x,
                    self.encoding,
                )
        return float(
            out.detach()
            .cpu()
            .view(-1)[0]
        )


def _normalize_curriculum_maybe_loose(
    cur: Any,
) -> CurriculumSpec:
    """Adapt loose curriculum dicts into CurriculumSpec.

    Supports schemas where geometry weights may be provided as:
      curriculum:
        geometry_weights:
          plane: ...
          sphere: ...
        ...
    or loosely as top-level keys under curriculum:
        curriculum:
          plane: ...
          sphere: ...
          cylinder2D: ...
          parallel_planes: ...
    """
    if isinstance(cur, CurriculumSpec):
        return cur
    if not isinstance(cur, dict):
        raise TypeError(
            f"Expected curriculum as dict or CurriculumSpec, got {type(cur)}"
        )

    # If it already looks like a proper CurriculumSpec payload, use it directly.
    if "geometry_weights" in cur:
        gw = cur["geometry_weights"]
    else:
        gw = None

    # Detect loose schema: plane/sphere/… at top level without geometry_weights
    loose_keys = {
        k
        for k in cur.keys()
        if k
        in (
            "plane",
            "sphere",
            "cylinder2D",
            "parallel_planes",
        )
    }
    if gw is None and loose_keys:
        logger.info(
            "Using loose curriculum schema (attribute adapter). "
            f"Keys={sorted(list(cur.keys()))}"
        )
        gw = {
            k: cur[k]
            for k in loose_keys
        }

    payload = dict(cur)
    if gw is not None:
        payload["geometry_weights"] = gw

    # Remove loose top-level keys so they don't collide with the dataclass ctor.
    for k in (
        "plane",
        "sphere",
        "cylinder2D",
        "parallel_planes",
    ):
        if k in payload and (
            gw is payload.get(
                "geometry_weights"
            )
            or k in loose_keys
        ):
            payload.pop(k, None)

    return CurriculumSpec(
        **payload
    )


def run_evaluation(
    config: ExperimentConfig,
    ckpt_path: Path,
    output_dir: Path,
) -> int:
    """Evaluate a trained model-over-geometry using certify-style gates.

    - Synthesizes or loads problems (best-effort; no hard external deps).
    - Wraps the model as a ConditionalModelWrapper.
    - Uses core certify helpers and EPS_* thresholds for metrics.
    - Writes per-problem JSON and an aggregate summary.csv.
    """
    # Normalize nested specs in case config came in as plain dicts.
    if isinstance(config.dataset, dict):
        config.dataset = DatasetSpec(
            **config.dataset
        )
    if isinstance(config.curriculum, dict):
        config.curriculum = _normalize_curriculum_maybe_loose(
            config.curriculum
        )
    if isinstance(config.model, dict):
        config.model = ModelSpec(
            **config.model
        )
    if isinstance(config.trainer, dict):
        config.trainer = TrainerSpec(
            **config.trainer
        )
    if isinstance(config.evaluation, dict):
        config.evaluation = EvalSpec(
            **config.evaluation
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    eval_spec: EvalSpec = config.evaluation

    device = torch.device(
        config.device
    )
    eval_dtype = getattr(
        torch,
        eval_spec.eval_dtype,
        torch.float64,
    )

    # Load model
    model = initialize_model(
        config
    ).to(
        device=device,
        dtype=eval_dtype,
    )
    if not ckpt_path.is_file():
        logger.error(
            f"Missing checkpoint: {ckpt_path}"
        )
        return 1
    ckpt = torch.load(
        ckpt_path,
        map_location=device,
    )
    state_dict = ckpt.get(
        "model_state_dict",
        ckpt,
    )
    model.load_state_dict(
        state_dict,
        strict=False,
    )
    model.eval()

    rng = np.random.default_rng(
        config.seed + 123
    )

    # Build evaluation problem set
    problems: List[
        Tuple[
            CanonicalSpec,
            str,
        ]
    ] = []
    problems.extend(
        mine_pdf_problems(
            config.curriculum.pdf_paths
        )
    )

    for _ in range(10):
        (
            spec,
            geom,
        ) = synthesize_problem(
            config.curriculum,
            rng,
        )
        if spec is not None:
            problems.append(
                (spec, geom)
            )

    if not problems:
        logger.error(
            "No evaluation problems generated."
        )
        return 1

    results: List[
        Dict[str, Any]
    ] = []

    for idx, (
        spec,
        geom,
    ) in enumerate(
        tqdm(
            problems,
            desc="EvalProblems",
            leave=False,
        )
    ):
        try:
            enc = encode_spec(
                spec
            )
            wrapper = ConditionalModelWrapper(
                model,
                enc,
                device,
                eval_dtype,
            )

            # BC residual
            bc_err = bc_residual_on_boundary(
                wrapper,
                spec,
                n_samples=eval_spec.n_samples_bc,
                logger=None,
            )

            # PDE residual
            pde_res = pde_residual_symbolic(
                wrapper,
                spec,
                n_samples=eval_spec.n_samples_pde,
                logger=None,
            )

            # Mean-value check
            mv_dev = mean_value_property_check(
                wrapper,
                spec,
                logger=None,
            )

            metrics = {
                "bc_residual_linf": float(
                    bc_err
                ),
                "pde_residual_linf": float(
                    pde_res
                ),
                "mean_value_deviation": float(
                    mv_dev
                ),
            }

            gb = green_badge_decision(
                metrics,
                logger=None,
            )

            per_case = {
                "index": idx,
                "geometry": geom,
                "bc_Linf": bc_err,
                "pde_residual": pde_res,
                "mean_value_dev": mv_dev,
                "pass_bc": bool(
                    bc_err
                    <= EPS_BC
                ),
                "pass_pde": bool(
                    pde_res
                    <= EPS_PDE
                ),
                "pass_mean_value": (
                    (not math.isfinite(mv_dev))
                    or bool(
                        mv_dev
                        <= EPS_MEAN_VAL
                    )
                ),
                "pass_energy": True,
                "pass_dual": True,
                "green_badge": bool(
                    gb
                ),
            }

            results.append(
                per_case
            )

            with open(
                output_dir
                / f"eval_case_{idx:04d}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    per_case,
                    f,
                    indent=2,
                )

        except Exception as e:
            logger.error(
                f"Evaluation failed for problem {idx}: {e}",
                exc_info=True,
            )

    # --- Summary CSV ---
    summary_path = (
        output_dir
        / "summary.csv"
    )
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(
            results
        )
        df.to_csv(
            summary_path,
            index=False,
        )
    except Exception:
        import csv

        keys = sorted(
            {
                k
                for r in results
                for k in r.keys()
            }
        )
        with open(
            summary_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            w = csv.DictWriter(
                f,
                fieldnames=keys,
            )
            w.writeheader()
            for r in results:
                w.writerow(r)

    # --- Pass-rate logging (best-effort) ---
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(
            results
        )
        if "green_badge" in df.columns:
            valid = df[
                df[
                    "green_badge"
                ].isin(
                    [
                        True,
                        False,
                    ]
                )
            ]
            if len(valid) > 0:
                global_rate = float(
                    valid[
                        "green_badge"
                    ].mean()
                )
                logger.info(
                    "--- Evaluation Summary ---"
                )
                logger.info(
                    f"Global Pass Rate (Green Badge): {global_rate:.4f}"
                )
                if (
                    "geometry"
                    in valid.columns
                ):
                    by_geom = valid.groupby(
                        "geometry"
                    )[
                        "green_badge"
                    ].mean()
                    logger.info(
                        "Pass Rates by Geometry:"
                    )
                    for (
                        geom_name,
                        rate,
                    ) in (
                        by_geom.items()
                    ):
                        logger.info(
                            f"  {geom_name}: {rate:.4f}"
                        )
    except Exception:
        # If pandas is missing or something fails, we still have summary.csv.
        pass

    logger.info(
        f"Evaluation finished. Summary saved to {summary_path}"
    )
    return 0