from __future__ import annotations

import json
from typing import Any, Dict

import pytest

pytest.importorskip(
    "fastapi",
    reason="FastAPI optional extra missing. Install with: pip install -e .[researched,researched-test]",
)
pytest.importorskip(
    "httpx",
    reason="httpx optional extra missing. Install with: pip install -e .[researched,researched-test]",
)

from fastapi.testclient import TestClient


def _get_app():
    """
    Adapt this import to your actual backend shape.
    Acceptable patterns:
      - electrodrive.researched.app:create_app()
      - electrodrive.researched.app:create_app_from_env()
      - electrodrive.researched.app:app (FastAPI instance)
    """
    from electrodrive.researched import app as researched_app

    if hasattr(researched_app, "create_app"):
        try:
            return researched_app.create_app()
        except TypeError:
            # Older signature requires explicit args.
            pass
    if hasattr(researched_app, "create_app_from_env"):
        try:
            return researched_app.create_app_from_env()
        except Exception:
            pass
    if hasattr(researched_app, "app"):
        return researched_app.app
    raise AssertionError("ResearchED backend must expose create_app() or app")


@pytest.fixture()
def client():
    app = _get_app()
    return TestClient(app)


def test_health_endpoint(client: TestClient):
    # Allow either /health or /api/health; prefer /api/health
    for path in ("/api/health", "/health"):
        r = client.get(path)
        if r.status_code == 200:
            data = r.json() if "application/json" in r.headers.get("content-type", "") else {}
            assert r.status_code == 200
            return
    raise AssertionError("Expected a health endpoint at /api/health or /health")


def test_runs_list_endpoint(client: TestClient):
    # Allow either /api/runs or /runs; prefer /api/runs
    for path in ("/api/runs", "/runs"):
        r = client.get(path)
        if r.status_code == 200:
            payload = r.json()
            assert isinstance(payload, (list, dict))
            # If dict, expect items/rows field; if list, items are run summaries
            return
    raise AssertionError("Expected a runs list endpoint at /api/runs or /runs")


def test_openapi_available(client: TestClient):
    # ensures app boots with valid schema (FastAPI)
    r = client.get("/openapi.json")
    assert r.status_code == 200
    data = r.json()
    assert "paths" in data
