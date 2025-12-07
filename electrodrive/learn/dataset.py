import logging
import math
from typing import (
    List,
    Dict,
    Any,
    Tuple,
    Optional,
    Union,
)

import numpy as np
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    get_worker_info,
)

from electrodrive.learn.specs import (
    CurriculumSpec,
    ExperimentConfig,
)
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.core.images import AnalyticSolution
from electrodrive.learn.encoding import (
    encode_spec,
    ENCODING_DIM,
)
from electrodrive.learn.collocation import (
    make_collocation_batch_for_spec,
    get_oracle_solution,
    BEM_AVAILABLE,
)

try:
    # BEMSolution is imported only for type hints / back-compat.
    from electrodrive.core.bem import BEMSolution  # type: ignore
except Exception:  # pragma: no cover - best-effort fallback
    from typing import Any as AnyType

    BEMSolution = AnyType  # type: ignore

logger = logging.getLogger("EDE.Learn.Data")


class _CollocationWorkerInit:
    def __init__(self, base_seed: int):
        self.base_seed = int(base_seed)

    def __call__(self, worker_id: int) -> None:
        info = get_worker_info()
        seed = self.base_seed + int(worker_id)

        np.random.seed(seed)
        torch.manual_seed(seed)

        if info is not None and hasattr(
            info.dataset, "rng"
        ):
            try:
                info.dataset.rng = np.random.default_rng(
                    seed
                )
            except Exception:
                # Best-effort; do not fail worker startup.
                pass


def _collocation_worker_init_fn(
    base_seed: int,
) -> _CollocationWorkerInit:
    """
    Initialize per-worker RNGs for deterministic multi-worker sampling.
    """
    return _CollocationWorkerInit(base_seed)


def _collate_collocation_batches(
    batch_list: List[
        Dict[str, torch.Tensor]
    ],
) -> Dict[str, torch.Tensor]:
    """Collate batches while skipping any empty entries."""
    valid = [
        b
        for b in batch_list
        if b
        and b["X"].numel()
        > 0
    ]
    if not valid:
        return {}
    out: Dict[
        str, torch.Tensor
    ] = {}
    for k in valid[0].keys():
        out[k] = torch.cat(
            [
                b[k]
                for b in valid
            ],
            dim=0,
        )
    return out


def synthesize_problem(
    curriculum: CurriculumSpec,
    rng: np.random.Generator,
) -> Tuple[Optional[CanonicalSpec], str]:
    geometries = list(
        curriculum.geometry_weights.keys()
    )
    weights = np.array(
        list(
            curriculum.geometry_weights.values()
        ),
        dtype=np.float64,
    )
    if weights.sum() <= 1e-9:
        return None, "none"
    geom_type = rng.choice(
        geometries,
        p=weights / weights.sum(),
    )

    q = float(
        rng.uniform(
            1e-9, 1e-8
        )
        * rng.choice([-1, 1])
    )
    spec_dict: Dict[str, Any] = {
        "domain": "R3",
        "BCs": "dirichlet",
        "conductors": [],
        "charges": [],
    }

    if geom_type == "plane":
        spec_dict["conductors"].append(
            {
                "type": "plane",
                "z": 0.0,
                "potential": 0.0,
            }
        )
        z0 = float(
            rng.uniform(
                0.1, 5.0
            )
        )
        x0 = float(
            rng.uniform(
                -5.0, 5.0
            )
        )
        y0 = float(
            rng.uniform(
                -5.0, 5.0
            )
        )
        spec_dict["charges"].append(
            {
                "type": "point",
                "q": q,
                "pos": [x0, y0, z0],
            }
        )

    elif geom_type == "sphere":
        r = float(
            rng.uniform(
                0.5, 2.0
            )
        )
        spec_dict[
            "conductors"
        ].append(
            {
                "type": "sphere",
                "radius": r,
                "potential": 0.0,
                "center": [0, 0, 0],
            }
        )
        dist = float(
            rng.uniform(
                r * 1.1,
                r * 5.0,
            )
        )
        vec = rng.standard_normal(3)
        vec /= (
            np.linalg.norm(vec)
            + 1e-9
        )
        pos = (vec * dist).tolist()
        spec_dict["charges"].append(
            {
                "type": "point",
                "q": q,
                "pos": pos,
            }
        )

    elif geom_type == "cylinder2D":
        a = float(
            rng.uniform(
                0.5, 1.5
            )
        )
        spec_dict[
            "conductors"
        ].append(
            {
                "type": "cylinder",
                "radius": a,
                "potential": 0.0,
            }
        )
        lambda_c = q
        dist = float(
            rng.uniform(
                a * 1.1,
                a * 5.0,
            )
        )
        phi = float(
            rng.uniform(
                0,
                2 * math.pi,
            )
        )
        pos_2d = [
            dist * math.cos(phi),
            dist * math.sin(phi),
        ]
        spec_dict["charges"].append(
            {
                "type": "line_charge",
                "lambda": lambda_c,
                "pos_2d": pos_2d,
            }
        )

    elif geom_type == "parallel_planes":
        d = float(
            rng.uniform(
                0.5, 2.0
            )
        )
        spec_dict[
            "conductors"
        ].append(
            {
                "type": "plane",
                "z": d,
                "potential": 0.0,
                "id": 1,
            }
        )
        spec_dict[
            "conductors"
        ].append(
            {
                "type": "plane",
                "z": -d,
                "potential": 0.0,
                "id": 2,
            }
        )
        z0 = float(
            rng.uniform(
                -0.8 * d,
                0.8 * d,
            )
        )
        x0 = float(
            rng.uniform(
                -2.0, 2.0
            )
        )
        y0 = float(
            rng.uniform(
                -2.0, 2.0
            )
        )
        spec_dict["charges"].append(
            {
                "type": "point",
                "q": q,
                "pos": [x0, y0, z0],
            }
        )

    else:
        return None, geom_type

    try:
        return (
            CanonicalSpec.from_json(
                spec_dict
            ),
            geom_type,
        )
    except ValueError:
        return None, geom_type


def mine_pdf_problems(
    pdf_paths: List[str],
) -> List[Tuple[CanonicalSpec, str]]:
    """Best-effort mining of example problems.

    Must not fail if PDFs are absent; returns a small canned set when 'hw7.pdf'
    is referenced, otherwise empty.
    """
    problems: List[
        Tuple[CanonicalSpec, str]
    ] = []
    if any(
        "hw7.pdf" in str(p)
        for p in pdf_paths
    ):
        spec_p1 = (
            CanonicalSpec.from_json(
                {
                    "domain": "R3",
                    "BCs": "dirichlet",
                    "conductors": [
                        {
                            "type": "plane",
                            "z": 0.0,
                            "potential": 0.0,
                        }
                    ],
                    "charges": [
                        {
                            "type": "point",
                            "q": 1e-9,
                            "pos": [0, 0, 1.0],
                        }
                    ],
                }
            )
        )
        problems.append(
            (spec_p1, "plane_hw7")
        )
    return problems


class ElectrostaticsJITDataset(Dataset):
    """On-the-fly problem generator for conditional electrostatics models."""

    def __init__(
        self,
        config: ExperimentConfig,
        split: str,
    ):
        self.config = config
        self.dataset_spec = (
            config.dataset
        )
        self.curriculum_spec = (
            config.curriculum
        )
        self.device = (
            config.device
        )
        self.dtype = getattr(
            torch,
            config.train_dtype,
        )
        self.rng = np.random.default_rng(
            config.seed
            + (0 if split == "train" else 1)
        )

        if split == "train":
            n_problems = (
                self.dataset_spec.n_train_problems
            )
        elif split == "val":
            n_problems = (
                self.dataset_spec.n_val_problems
            )
        else:
            n_problems = 0

        # Each entry: (CanonicalSpec, geom_type_str, encoding_tensor)
        self.problems: List[
            Tuple[CanonicalSpec, str, torch.Tensor]
        ] = []
        # Deprecated but kept for back-compat; no longer used now that the
        # collocation helper manages oracle caching internally.
        self.solutions: Dict[
            int,
            Union[AnalyticSolution, BEMSolution]
        ] = {}

        self._generate_problem_pool(
            n_problems
        )

    def _generate_problem_pool(
        self, n_target: int
    ) -> None:
        logger.info(
            f"Generating problem pool (target={n_target})..."
        )

        if (
            self.curriculum_spec.include_pdf_sources
        ):
            for (
                spec,
                geom_type,
            ) in mine_pdf_problems(
                self.curriculum_spec.pdf_paths
            ):
                self.problems.append(
                    (
                        spec,
                        geom_type,
                        encode_spec(spec),
                    )
                )

        attempts = 0
        max_attempts = max(
            1, n_target * 5
        )
        while (
            len(self.problems)
            < n_target
            and attempts < max_attempts
        ):
            attempts += 1
            (
                spec,
                geom_type,
            ) = synthesize_problem(
                self.curriculum_spec,
                self.rng,
            )
            if spec is not None:
                self.problems.append(
                    (
                        spec,
                        geom_type,
                        encode_spec(spec),
                    )
                )

        logger.info(
            f"Problem pool generated: {len(self.problems)} instances."
        )

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(
        self, idx: int
    ) -> Dict[
        str, torch.Tensor
    ]:
        (
            spec,
            geom_type,
            encoding,
        ) = self.problems[idx]

        # Delegate sampling + oracle evaluation to the shared helper.
        batch = make_collocation_batch_for_spec(
            spec=spec,
            n_points=self.dataset_spec.samples_per_problem_jit,
            ratio_boundary=self.dataset_spec.ratio_boundary,
            supervision_mode=self.dataset_spec.supervision_mode,
            device=self.device,
            dtype=self.dtype,
            rng=self.rng,
            bem_oracle_config=self.dataset_spec.bem_oracle_config,
            encoding=encoding,
            geom_type=geom_type,
        )


        # Keep behaviour identical to historical code: if no oracle could be
        # constructed (e.g. analytic path missing and BEM unavailable), we
        # return an explicit empty batch with the right shapes/dtypes.
        if batch["X"].numel() == 0:
            return self._empty_batch()

        return batch

    def _empty_batch(
        self,
    ) -> Dict[str, torch.Tensor]:
        return {
            "X": torch.zeros(
                0, 3, dtype=self.dtype
            ),
            "V_gt": torch.zeros(
                0, dtype=self.dtype
            ),
            "is_boundary": torch.zeros(
                0,
                dtype=torch.bool,
            ),
            "mask_finite": torch.zeros(
                0,
                dtype=torch.bool,
            ),
            "encoding": torch.zeros(
                0,
                ENCODING_DIM,
                dtype=self.dtype,
            ),
            "bbox_center": torch.zeros(
                0,
                3,
                dtype=self.dtype,
            ),
            "bbox_extent": torch.zeros(
                0,
                dtype=self.dtype,
            ),
        }


def build_dataloaders(
    config: ExperimentConfig,
) -> Tuple[
    DataLoader, DataLoader
]:
    trainer = config.trainer
    train_ds = (
        ElectrostaticsJITDataset(
            config,
            split="train",
        )
    )
    val_ds = (
        ElectrostaticsJITDataset(
            config,
            split="val",
        )
    )

    spp = (
        config.dataset.samples_per_problem_jit
    )
    problems_per_batch = max(
        1,
        config.trainer.batch_size
        // spp,
    )

    common = {
        "batch_size": problems_per_batch,
        "collate_fn": _collate_collocation_batches,
        "pin_memory": bool(
            getattr(
                trainer,
                "pin_memory",
                False,
            )
        ),
        "num_workers": int(
            getattr(
                trainer,
                "num_workers",
                0,
            )
        ),
        "persistent_workers": int(
            getattr(
                trainer,
                "num_workers",
                0,
            )
        )
        > 0,
        "worker_init_fn": _collocation_worker_init_fn(
            config.seed
        ),
    }

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **common,
    )
    return train_loader, val_loader
