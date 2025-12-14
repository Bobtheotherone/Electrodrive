import pytest


def test_get_api_router_optional_dependency():
    import electrodrive.researched.api as api

    try:
        import fastapi  # type: ignore  # noqa: F401

        have_fastapi = True
    except Exception:
        have_fastapi = False

    if have_fastapi:
        from fastapi import APIRouter  # type: ignore

        router = api.get_api_router()
        assert isinstance(router, APIRouter)
    else:
        with pytest.raises(ImportError) as excinfo:
            api.get_api_router()
        msg = str(excinfo.value).lower()
        assert "fastapi" in msg
        assert "install" in msg
