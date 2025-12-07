from __future__ import annotations
from typing import Tuple
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.logging import JsonlLogger

def _has_dielectrics(spec: CanonicalSpec) -> bool:
    try:
        return bool(getattr(spec, "dielectrics", []) or (isinstance(spec.dielectrics, (list, tuple)) and len(spec.dielectrics)>0))
    except Exception:
        return False

def choose_mode(spec: CanonicalSpec, requested: str, logger: JsonlLogger) -> Tuple[str, str]:
    # 1) explicit user choice
    if requested != "auto":
        logger.info("Planner honoring user-requested mode.", rationale="user_override", selected=requested, executed=requested)
        return requested, "user_override"
    # 2) simple heuristics (minimal)
    if _has_dielectrics(spec):
        selected = executed = "bem"
        logger.info("Planner selected mode.", rationale="dielectrics_present", selected=selected, executed=executed)
        return executed, "dielectrics_present"
    # 3) default to BEM for visibility
    selected = executed = "bem"
    logger.info("Planner selected mode.", rationale="default_bem", selected=selected, executed=executed)
    return executed, "default_bem"

def choose_mode_executed(spec: CanonicalSpec, requested: str, logger: JsonlLogger) -> str:
    executed, _ = choose_mode(spec, requested, logger)
    return executed