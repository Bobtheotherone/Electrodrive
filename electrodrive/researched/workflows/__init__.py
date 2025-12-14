from __future__ import annotations

"""
Workflow registry for ResearchED.

Design Doc mapping:
- FR-1: workflow discovery & launch (UI selects template; backend builds exact CLI command)
- §3.2 RunManager: launch subprocess commands with explicit env vars

This module is intentionally stdlib-only and self-contained.
"""

from pathlib import Path
from typing import Dict, Protocol, runtime_checkable

__all__ = [
    "Workflow",
    "WORKFLOWS",
    "get_workflow",
]


@runtime_checkable
class Workflow(Protocol):
    name: str
    supports_controls: bool

    def build_command(self, request: dict, out_dir: Path) -> list[str]:
        ...

    def build_env(self, request: dict, out_dir: Path, run_id: str) -> dict[str, str]:
        ...

    def describe(self) -> dict[str, object]:
        ...

    def validate_request(self, request: dict) -> None:
        ...


# Register built-in workflows.
from .solve import SolveWorkflow  # noqa: E402
from .images_discover import ImagesDiscoverWorkflow  # noqa: E402
from .learn_train import LearnTrainWorkflow  # noqa: E402
from .fmm_suite import FmmSuiteWorkflow  # noqa: E402

_WORKFLOW_INSTANCES = [
    SolveWorkflow(),
    ImagesDiscoverWorkflow(),
    LearnTrainWorkflow(),
    FmmSuiteWorkflow(),
]

WORKFLOWS: Dict[str, Workflow] = {w.name: w for w in _WORKFLOW_INSTANCES}


def get_workflow(name: str) -> Workflow:
    try:
        return WORKFLOWS[str(name)]
    except KeyError as exc:
        raise KeyError(f"Unknown workflow: {name}. Known: {sorted(WORKFLOWS)}") from exc
