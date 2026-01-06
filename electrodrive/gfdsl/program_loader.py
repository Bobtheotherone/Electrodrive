"""Load GFDSL programs from disk and lower to basis elements."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import torch

from electrodrive.gfdsl.compile import CompileContext, lower_program, validate_program
from electrodrive.gfdsl.compile.legacy_adapter import linear_contribution_to_legacy_basis
from electrodrive.gfdsl.io import deserialize_program


def _iter_program_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.glob("*.json") if p.is_file()])


def load_gfdsl_programs(
    program_dir: Path,
    *,
    spec: object,
    device: torch.device,
    dtype: torch.dtype,
    eval_backend: str = "operator",
    logger: Optional[object] = None,
    limit: Optional[int] = None,
) -> List[object]:
    files = _iter_program_files(program_dir)
    if limit is not None:
        files = files[: max(0, int(limit))]

    elements: List[object] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            program = deserialize_program(payload)
            ctx = CompileContext(spec=spec, device=device, dtype=dtype, eval_backend=eval_backend)
            validate_program(program, ctx)
            contrib = lower_program(program, ctx)
            elems = linear_contribution_to_legacy_basis(contrib)
            program_id = path.stem
            for elem in elems:
                elem.params["program_id"] = program_id
            elements.extend(elems)
            if logger is not None:
                try:
                    logger.info("Loaded GFDSL program.", path=str(path), columns=len(elems))
                except Exception:
                    pass
        except Exception as exc:
            if logger is not None:
                try:
                    logger.warning("Failed to load GFDSL program.", path=str(path), error=str(exc))
                except Exception:
                    pass
            continue
    return elements
