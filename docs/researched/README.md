# ResearchED QC Repro Guide

## Backend setup (fresh virtualenv)
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -e ".[researched,researched-test]"
pytest -q tests/test_researched_*.py
```

## Frontend setup
```bash
cd researched_ui
npm ci
npm run typecheck
npm run build
```

Helper scripts from repo root:
```bash
bash scripts/researched_ui_typecheck.sh
bash scripts/researched_ui_build.sh
```

Notes:
- API base/WS paths default to `/api/v1` and `/ws` (frontend falls back to `/api` and `/api/ws` if needed).
- The backend app is available via `electrodrive.researched.app:create_app()`; static UI can be served by pointing `RESEARCHED_STATIC_DIR` at `researched_ui/dist`.
