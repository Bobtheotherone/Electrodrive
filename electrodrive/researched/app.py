from __future__ import annotations

"""
FastAPI application wiring for ResearchED.

Design Doc references:

- Design Doc §3.1–§3.2: local web UI + Python backend service.
- FR-5: live monitor (WebSocket streams for logs/frames).
- Dependency policy: this package is optional; FastAPI imports are lazy.

This module must have minimal import-time side effects.
"""

import os
import threading
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from . import __version__

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import FastAPI  # noqa: F401


_ENV_RUNS_ROOT = "RESEARCHED_RUNS_ROOT"
_ENV_DB_PATH = "RESEARCHED_DB_PATH"
_ENV_STATIC_DIR = "RESEARCHED_STATIC_DIR"


def _fastapi_install_error(exc: BaseException | None = None) -> ImportError:
    msg = (
        "ResearchED backend requires FastAPI (optional extra).\n"
        "Install with:\n"
        "  pip install fastapi uvicorn\n"
        "or (recommended):\n"
        "  pip install 'fastapi>=0.100' 'uvicorn[standard]>=0.23'\n"
    )
    err = ImportError(msg)
    if exc is not None:
        err.__cause__ = exc  # type: ignore[attr-defined]
    return err


def create_app(
    runs_root: Path | str | None = None,
    db_path: Path | str | None = None,
    static_dir: Path | str | None = None,
) -> "FastAPI":
    """
    Create the ResearchED FastAPI application.

    - Stores configuration in app.state (runs_root, db_path, static_dir).
    - Wires REST API under /api/v1 and WebSocket endpoints under /ws.
    - Mounts static UI at / when available (SPA fallback to index.html).

    Additionally, initializes an in-memory process registry used by the
    RunManager-lite endpoints (Design Doc FR-1/FR-3).
    """
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from starlette.responses import FileResponse, Response
        from starlette.staticfiles import StaticFiles
        from starlette.types import Scope
    except Exception as exc:  # pragma: no cover
        raise _fastapi_install_error(exc)

    # Default path resolution (env-aware for QC tests/imports).
    runs_root_val = runs_root if runs_root is not None else os.getenv(_ENV_RUNS_ROOT, "runs")
    runs_root_p = Path(runs_root_val).expanduser()

    db_default = db_path if db_path is not None else os.getenv(_ENV_DB_PATH, "")
    if not db_default:
        db_default = Path(runs_root_p) / "researched.sqlite"
    db_path_p = Path(db_default).expanduser()

    # Treat empty-string static_dir as None (common when CLI passes "").
    static_p: Optional[Path]
    if static_dir is None:
        static_p = None
    elif isinstance(static_dir, str) and not static_dir.strip():
        static_p = None
    else:
        static_p = Path(static_dir).expanduser()

    # Best-effort make paths absolute.
    try:
        runs_root_p = runs_root_p.resolve()
    except Exception:
        pass
    try:
        db_path_p = db_path_p.resolve()
    except Exception:
        pass
    if static_p is not None:
        try:
            static_p = static_p.resolve()
        except Exception:
            pass

    # Ensure roots exist (safe defaults for direct imports/tests).
    try:
        runs_root_p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        db_path_p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    app = FastAPI(title="ResearchED", version=__version__)

    # Store configuration for downstream routers (Design Doc §3.2: backend subsystems).
    app.state.runs_root = runs_root_p
    app.state.db_path = db_path_p
    app.state.static_dir = static_p

    # Run lifecycle manager (Design Doc §3.2 RunManager).
    # This is stdlib-only and safe to construct at app startup.
    # It enables queued launches + cancel/control semantics across the REST API.
    if getattr(app.state, "run_manager", None) is None:
        try:
            from .run_manager import RunManager  # stdlib-only

            max_parallel_raw = os.getenv("RESEARCHED_MAX_PARALLEL", "").strip()
            try:
                max_parallel = int(max_parallel_raw) if max_parallel_raw else 1
            except Exception:
                max_parallel = 1
            max_parallel = max(1, max_parallel)

            app.state.run_manager = RunManager(runs_root_p, max_parallel=max_parallel)
        except Exception:
            # Defensive: server should still start even if RunManager init fails;
            # API endpoints will raise clearer errors when used.
            app.state.run_manager = None

    # In-memory subprocess registry (RunManager-lite).
    # (API also lazily initializes these, but setting them here makes behavior deterministic.)
    app.state._researched_proc_registry = {}
    app.state._researched_proc_lock = threading.Lock()

    # Conservative CORS defaults: allow localhost only (typical local dev UI).
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Routers (FastAPI imports happen inside get_*_router too).
    from .api import get_api_router
    from .ws import get_ws_router

    api_router_v1 = get_api_router()
    app.include_router(api_router_v1, prefix="/api/v1")
    # Compatibility prefix for UI clients that assume /api rather than /api/v1.
    app.include_router(get_api_router(), prefix="/api")

    ws_router = get_ws_router()
    app.include_router(ws_router, prefix="/ws")
    # Compatibility prefixes so UI WebSocket fallbacks connect without extra retries.
    app.include_router(get_ws_router(), prefix="/api/ws")
    app.include_router(get_ws_router(), prefix="/api/v1/ws")

    # Root behavior: static UI if present; otherwise a small JSON hello.
    ui_dir = static_p
    index_path = (ui_dir / "index.html") if ui_dir is not None else None
    have_ui = bool(index_path and index_path.is_file())

    if have_ui:

        class SPAStaticFiles(StaticFiles):
            async def get_response(self, path: str, scope: Scope) -> Response:  # type: ignore[override]
                resp = await super().get_response(path, scope)
                if resp.status_code == 404 and index_path and index_path.is_file():
                    return FileResponse(str(index_path))
                return resp

        # Mount last so /api/v1 and /ws take precedence.
        app.mount("/", SPAStaticFiles(directory=str(ui_dir), html=True), name="ui")
    else:

        @app.get("/")  # type: ignore[misc]
        def _hello() -> Any:
            return {
                "name": "ResearchED",
                "ok": True,
                "version": __version__,
                "ui": "not mounted (static assets not found)",
                "hint": "Provide --static-dir pointing at a built UI directory containing index.html.",
                "api_base": "/api/v1",
                "ws_base": "/ws",
                "endpoints": {
                    "runs": "/api/v1/runs",
                    "launch": "POST /api/v1/runs",
                    "control_schema": "/api/v1/control/schema",
                },
            }

    # Compatibility endpoints for QC tests and legacy clients.
    @app.get("/health")
    def _health() -> Any:
        return {
            "ok": True,
            "runs_root": str(app.state.runs_root) if getattr(app.state, "runs_root", None) is not None else None,
            "db_path": str(app.state.db_path) if getattr(app.state, "db_path", None) is not None else None,
            "version": __version__,
        }

    @app.get("/api/health")
    def _health_api() -> Any:
        return _health()

    @app.get("/runs")
    def _runs_list() -> Any:
        try:
            from .api import _index_runs  # type: ignore
        except Exception:
            return []
        return _index_runs(Path(getattr(app.state, "runs_root", "runs")))

    @app.get("/api/runs")
    def _runs_list_api() -> Any:
        return _runs_list()

    return app


def create_app_from_env() -> "FastAPI":
    """
    Uvicorn reload-friendly factory.

    CLI sets env vars when --reload is used; this function reads them and delegates
    to create_app().
    """
    runs_root = os.getenv(_ENV_RUNS_ROOT, "runs")
    db_path = os.getenv(_ENV_DB_PATH, str(Path(runs_root) / "researched.sqlite"))
    static_dir = os.getenv(_ENV_STATIC_DIR, "")
    static_val: Optional[str] = static_dir.strip() or None
    return create_app(runs_root=runs_root, db_path=db_path, static_dir=static_val)


# Export a ready-to-use app for TestClient / uvicorn entrypoints.
try:  # pragma: no cover - best-effort import convenience
    app = create_app_from_env()
except Exception:
    app = None  # type: ignore[assignment]
