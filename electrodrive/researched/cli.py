from __future__ import annotations

"""
ResearchED server CLI.

Design Doc references:

- Design Doc §3.1–§3.2: local web UI + Python backend service.
- FR-3: runs root directory contract (server points at a runs root).
- Dependency policy: GUI is optional extra; FastAPI/Uvicorn are imported lazily.

Usage:
  python -m electrodrive.researched
  python -m electrodrive.researched.cli
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence


def _find_repo_root(start: Path) -> Optional[Path]:
    """Walk upward to locate repo root (identified by .git). Returns None if not found."""
    try:
        start = start.resolve()
    except Exception:
        pass
    for p in [start, *start.parents]:
        try:
            if (p / ".git").exists():
                return p
        except Exception:
            continue
    return None


def _default_runs_root() -> Path:
    """
    Default runs root:
      - <repo_root>/runs if repo root is detectable
      - otherwise ./runs (cwd-relative)
    """
    here = Path(__file__).resolve()
    repo_root = _find_repo_root(here)
    return (repo_root / "runs") if repo_root is not None else Path("runs")


def _install_hint() -> str:
    return (
        "ResearchED backend is an optional extra and requires FastAPI + Uvicorn.\n\n"
        "Install (minimal):\n"
        "  pip install fastapi uvicorn\n\n"
        "Install (recommended, with standard extras):\n"
        "  pip install 'fastapi>=0.100' 'uvicorn[standard]>=0.23'\n"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the ResearchED local web backend.")
    p.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    p.add_argument(
        "--runs-root",
        type=str,
        default=str(_default_runs_root()),
        help="Root directory containing run folders (default: <repo_root>/runs if detectable; else ./runs).",
    )
    p.add_argument(
        "--db-path",
        type=str,
        default="",
        help="Path to ResearchED SQLite DB (default: <runs-root>/researched.sqlite).",
    )
    p.add_argument(
        "--static-dir",
        type=str,
        default="",
        help="Optional path to UI build assets directory to mount at '/'.",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="Uvicorn log level (default: info).",
    )
    p.add_argument(
        "--reload",
        action="store_true",
        help="Enable dev auto-reload (requires import-string app factory).",
    )
    return p


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Optional[Path]]:
    runs_root = Path(args.runs_root).expanduser()
    if not runs_root.is_absolute():
        try:
            runs_root = runs_root.resolve()
        except Exception:
            pass

    db_path_raw = str(getattr(args, "db_path", "") or "").strip()
    db_path = Path(db_path_raw).expanduser() if db_path_raw else (runs_root / "researched.sqlite")
    if not db_path.is_absolute():
        try:
            db_path = db_path.resolve()
        except Exception:
            pass

    static_raw = str(getattr(args, "static_dir", "") or "").strip()
    static_dir = Path(static_raw).expanduser() if static_raw else None
    if static_dir is not None and not static_dir.is_absolute():
        try:
            static_dir = static_dir.resolve()
        except Exception:
            pass

    return runs_root, db_path, static_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    runs_root, db_path, static_dir = _resolve_paths(args)

    # Ensure runs root exists (FR-3: run directory contract starts at runs_root).
    try:
        runs_root.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"[researched] ERROR: failed to create runs root: {runs_root} ({exc})", file=sys.stderr)
        return 2

    # Lazy dependency imports (optional extra policy).
    try:
        import uvicorn  # type: ignore
    except Exception:
        print(_install_hint(), file=sys.stderr)
        return 2

    # Build app (FastAPI import is inside create_app).
    try:
        if args.reload:
            # Reload requires an import-string + factory; pass config via env.
            os.environ["RESEARCHED_RUNS_ROOT"] = str(runs_root)
            os.environ["RESEARCHED_DB_PATH"] = str(db_path)
            if static_dir is not None:
                os.environ["RESEARCHED_STATIC_DIR"] = str(static_dir)
            else:
                os.environ.pop("RESEARCHED_STATIC_DIR", None)

            uvicorn.run(
                "electrodrive.researched.app:create_app_from_env",
                host=str(args.host),
                port=int(args.port),
                log_level=str(args.log_level),
                reload=True,
                factory=True,
            )
        else:
            from .app import create_app

            app = create_app(runs_root=runs_root, db_path=db_path, static_dir=static_dir)
            uvicorn.run(
                app,
                host=str(args.host),
                port=int(args.port),
                log_level=str(args.log_level),
                reload=False,
            )
    except ImportError as exc:
        # create_app likely raised due to missing FastAPI.
        print(_install_hint(), file=sys.stderr)
        print(f"Details: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[researched] ERROR: server failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
