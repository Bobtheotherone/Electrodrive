from __future__ import annotations

"""ResearchED: local GUI backend for Electrodrive.

This package intentionally keeps GUI dependencies optional (FastAPI/Uvicorn).
Core design document goals are implemented in submodules:

- app.py: FastAPI wiring (REST + WebSockets + optional static UI).
- api.py: REST endpoints (runs, artifacts, controls, presets, compare).
- ws.py: WebSocket endpoints (live events, raw logs, frames).
- run_manager.py: Run lifecycle manager (queue, subprocess spawn, cancel/controls).
- workflows/: workflow templates (solve, images_discover, learn_train, fmm_suite).

Versioning:
- __version__ is used by the API/UI for display and manifest stamping.
"""

__all__ = ["__version__"]

# Bump when you make user-visible changes to the ResearchED backend contract.
__version__ = "0.1.0"
