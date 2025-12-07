# electrodrive/learn/specs.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Union, Any as AnyType
from pathlib import Path

import torch
import yaml

TorchDtype = str
Device = str


@dataclass
class DatasetSpec:
    n_train_problems: int = 1000
    n_val_problems: int = 100
    samples_per_problem_jit: int = 4096
    ratio_boundary: float = 0.4
    ratio_interior: float = 0.6
    supervision_mode: str = "analytic"  # analytic, bem, auto, neural
    bem_oracle_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "initial_h": 0.2,
            "fp64": True,
            # For oracle use in the learning stack we prefer a conservative,
            # high-accuracy configuration: CPU + fp64 + near-field quadrature
            # at evaluation points. Callers can override any of these.
            "use_gpu": False,
            "use_near_quadrature": True,
            # Keep matvec near-quad disabled by default; evaluation near-quad
            # is sufficient for analytic-vs-BEM consistency and cheaper.
            "use_near_quadrature_matvec": False,
        }
    )


@dataclass
class CurriculumSpec:
    geometry_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "plane": 0.25,
            "sphere": 0.25,
            "cylinder2D": 0.25,
            "parallel_planes": 0.25,
        }
    )
    include_pdf_sources: bool = True
    pdf_paths: List[str] = field(
        default_factory=lambda: ["hw7.pdf"]
    )


@dataclass
class ModelSpec:
    model_type: str  # 'pinn_harmonic' or 'moi_symbolic'
    params: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class TrainerSpec:
    max_epochs: int = 100
    batch_size: int = 65536
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    warmup_frac: float = 0.0
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "bc_dirichlet": 50.0,
            "pde_residual": 1.0,
            "sparsity": 1e-4,
        }
    )
    log_every_n_steps: int = 100
    val_every_n_epochs: int = 1
    ckpt_every_n_epochs: int = 5
    early_stopping_patience: int = 20
    # ---- Added knobs for microbatching/AMP/compile & DL behavior ----
    # Gradient accumulation (number of microbatches per optimizer step).
    accum_steps: int = 1
    # Explicit microbatch size for the expensive Laplacian path.
    # 0 means "autotune" (train.py will choose a safe value).
    points_per_step: int = 0
    # Target fraction of microbatch samples to draw from boundary points when available.
    boundary_fraction: float = 0.1
    # AMP selection accepted by train.py: "bf16" | "fp16" | bool | None.
    # If None or not a string, train.py auto-selects based on device.
    amp: Union[bool, str, None] = None
    # Optional torch.compile flags (train.py will gracefully fallback).
    compile: Union[bool, str] = False
    compile_mode: str = "reduce-overhead"
    # Dataloader QoL (used by build_dataloaders if supported).
    num_workers: int = 2
    pin_memory: bool = True
    # Back-compat flag (ignored by train.py if 'amp' is set).
    use_amp: bool = True


@dataclass
class EvalSpec:
    eval_dtype: TorchDtype = "float64"
    n_samples_bc: int = 1024
    n_samples_pde: int = 512
    tolerances_override: Dict[str, float] = field(
        default_factory=dict
    )
    split: str = "val"


@dataclass
class ExperimentConfig:
    exp_name: str
    seed: int = 42
    device: Device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    train_dtype: TorchDtype = "float32"
    curriculum: CurriculumSpec = field(
        default_factory=CurriculumSpec
    )
    dataset: DatasetSpec = field(
        default_factory=DatasetSpec
    )
    model: ModelSpec = field(
        default_factory=lambda: ModelSpec(
            model_type="pinn_harmonic"
        )
    )
    trainer: TrainerSpec = field(
        default_factory=TrainerSpec
    )
    evaluation: EvalSpec = field(
        default_factory=EvalSpec
    )

    @staticmethod
    def from_yaml(
        path: Union[str, Path]
    ) -> "ExperimentConfig":
        with open(
            path, "r", encoding="utf-8"
        ) as f:
            config_dict = (
                yaml.safe_load(f) or {}
            )

        def instantiate(
            cls: AnyType, data: AnyType
        ):
            if data is None:
                return cls()
            if not hasattr(
                cls,
                "__dataclass_fields__",
            ):
                return data
            kwargs: Dict[
                str, Any
            ] = {}
            for (
                name,
                fdef,
            ) in cls.__dataclass_fields__.items():  # type: ignore[attr-defined]
                if name in data:
                    val = data[name]
                    f_type = fdef.type
                    origin = getattr(
                        f_type,
                        "__origin__",
                        None,
                    )
                    args = getattr(
                        f_type,
                        "__args__",
                        (),
                    )
                    if (
                        origin is Union
                        and type(None)
                        in args
                    ):
                        actual = next(
                            (
                                t
                                for t in args
                                if t
                                is not type(
                                    None
                                )
                            ),
                            Any,
                        )
                    else:
                        actual = f_type
                    if hasattr(
                        actual,
                        "__dataclass_fields__",
                    ):
                        kwargs[
                            name
                        ] = instantiate(
                            actual,
                            val,
                        )
                    else:
                        kwargs[name] = val
            return cls(**kwargs)

        return instantiate(
            ExperimentConfig, config_dict
        )

    def to_yaml(
        self, path: Union[str, Path]
    ) -> None:
        with open(
            path, "w", encoding="utf-8"
        ) as f:
            yaml.dump(
                asdict(self),
                f,
                default_flow_style=False,
                sort_keys=False,
            )
