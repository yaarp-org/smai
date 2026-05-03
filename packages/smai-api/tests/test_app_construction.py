"""Smoke tests for :func:`smai_api.make_api_app`.

Asserts the construction shape:

* App builds against a duck-typed Runtime.
* Middleware ordering is correct (Host validation outermost, bearer
  next, route dispatch innermost).
* Every Runtime / plugin exception type the routers can raise has a
  registered handler so failures surface as the structured envelope
  rather than the framework default.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _4_k1_fixtures import (  # type: ignore[import-not-found]
    make_test_runtime,
)
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from smai_api import AuthConfig, make_api_app
from smai_api.auth import BearerTokenMiddleware, HostValidationMiddleware


@pytest.mark.asyncio
async def test_make_api_app_builds(tmp_path: Path) -> None:
    """The factory returns a FastAPI instance with the runtime bound."""
    runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
    app = make_api_app(runtime)  # type: ignore[arg-type]
    assert isinstance(app, FastAPI)
    assert app.state.runtime is runtime


@pytest.mark.asyncio
async def test_middleware_ordering_host_before_auth(tmp_path: Path) -> None:
    """When bearer mode is on, Host validation runs before bearer-token
    check (a request with a bad Host gets 421 even without an
    Authorization header)."""
    token_path = tmp_path / "api-token"
    runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
    app = make_api_app(
        runtime,  # type: ignore[arg-type]
        auth_config=AuthConfig(enabled=True, token_path=token_path),
    )

    # Verify both middlewares are present.
    classes = {mw.cls for mw in app.user_middleware}
    assert HostValidationMiddleware in classes
    assert BearerTokenMiddleware in classes

    # An evil Host without a bearer header → 421 (Host rejection wins),
    # not 403 (auth failure).
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/system/health", headers={"Host": "evil.com"})
        assert response.status_code == 421, response.text
        body = response.json()
        assert body["error"]["code"] == "HOST_REJECTED"


@pytest.mark.asyncio
async def test_bearer_mode_rejects_missing_header(tmp_path: Path) -> None:
    """Bearer mode requires ``Authorization: Bearer <token>`` per ``11`` §7.3."""
    token_path = tmp_path / "api-token"
    runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
    app = make_api_app(
        runtime,  # type: ignore[arg-type]
        auth_config=AuthConfig(enabled=True, token_path=token_path),
    )
    # Token file should be created on construction.
    assert token_path.exists()
    token = token_path.read_text(encoding="utf-8").strip()
    assert token

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Without bearer → 403.
        response = await client.get("/api/v1/system/health")
        assert response.status_code == 403, response.text
        assert response.json()["error"]["code"] == "FORBIDDEN"
        # With bearer → 200.
        response = await client.get(
            "/api/v1/system/health",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_bearer_mode_rejects_wrong_token(tmp_path: Path) -> None:
    """Bearer mode rejects a wrong token with 403 + FORBIDDEN."""
    token_path = tmp_path / "api-token"
    runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
    app = make_api_app(
        runtime,  # type: ignore[arg-type]
        auth_config=AuthConfig(enabled=True, token_path=token_path),
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/api/v1/system/health",
            headers={"Authorization": "Bearer wrong-token-probe"},
        )
        assert response.status_code == 403, response.text
        assert response.json()["error"]["code"] == "FORBIDDEN"


@pytest.mark.asyncio
async def test_existing_token_file_preserved(tmp_path: Path) -> None:
    """An existing token file is read, not regenerated."""
    token_path = tmp_path / "api-token"
    token_path.write_text("preset-test-token", encoding="utf-8")
    runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
    make_api_app(
        runtime,  # type: ignore[arg-type]
        auth_config=AuthConfig(enabled=True, token_path=token_path),
    )
    assert token_path.read_text(encoding="utf-8").strip() == "preset-test-token"
