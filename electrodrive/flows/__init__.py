"""Flow-based parameter sampling core."""

from electrodrive.flows.device_guard import ensure_cuda, resolve_device, resolve_dtype
from electrodrive.flows.flow_matching import FlowMatchingConfig, FlowMatchingTrainer
from electrodrive.flows.models import ConditionBatch, ParamFlowNet
from electrodrive.flows.ode_solvers import euler_step, heun_step, rk4_step
from electrodrive.flows.rectified_flow import RectifiedFlowConfig, RectifiedFlowTrainer, integrate, rectify_coupling
from electrodrive.flows.sampler import ParamFlowSampler
from electrodrive.flows.schemas import (
    CanonicalSpecView,
    ParamSchema,
    ParamSchemaRegistry,
    REGISTRY,
    SCHEMA_AXIS_POINT,
    SCHEMA_COMPLEX_DEPTH,
    SCHEMA_POLE,
    SCHEMA_REAL_POINT,
    get_schema_by_id,
    get_schema_by_name,
)
from electrodrive.flows.types import FlowConfig, ParamPayload, ParamSampler, ProgramBatch

__all__ = [
    "CanonicalSpecView",
    "ConditionBatch",
    "FlowConfig",
    "FlowMatchingConfig",
    "FlowMatchingTrainer",
    "ParamFlowNet",
    "ParamFlowSampler",
    "ParamPayload",
    "ParamSampler",
    "ProgramBatch",
    "ParamSchema",
    "ParamSchemaRegistry",
    "REGISTRY",
    "RectifiedFlowConfig",
    "RectifiedFlowTrainer",
    "SCHEMA_AXIS_POINT",
    "SCHEMA_COMPLEX_DEPTH",
    "SCHEMA_POLE",
    "SCHEMA_REAL_POINT",
    "ensure_cuda",
    "resolve_device",
    "resolve_dtype",
    "euler_step",
    "heun_step",
    "rk4_step",
    "get_schema_by_id",
    "get_schema_by_name",
    "integrate",
    "rectify_coupling",
]
