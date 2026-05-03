"""Edge-case tests for the system router (config redaction, version shape)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from _4_k1_fixtures import (  # type: ignore[import-not-found]
    make_test_runtime,
)
from httpx import ASGITransport, AsyncClient
from smai_api import make_api_app


def _walk(value: object) -> Any:
    """Yield every (key, leaf) pair in a nested dict."""
    if isinstance(value, dict):
        for key, sub in cast("dict[str, object]", value).items():
            yield key, sub
            yield from _walk(sub)
    elif isinstance(value, list):
        for item in cast("list[object]", value):
            yield from _walk(item)


@pytest.mark.asyncio
async def test_config_redacts_secret_shaped_keys(tmp_path: Path) -> None:
    """Keys ending in ``_token`` / ``_password`` / ``_secret`` / ``_key``
    surface as ``"<redacted>"`` per ``11`` §4.6.

    Drives the redaction by stuffing a synthetic secret-shaped field
    onto ``llm_provider_config`` (a free-form dict). The authoritative
    test lives in the conformance suite (shape-only); this asserts the
    redaction *behavior* directly.
    """
    runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
    runtime.config.plugins.llm_provider_config["api_key"] = "secret-value-do-not-leak"
    runtime.config.plugins.llm_provider_config["bearer_token"] = "another-secret"
    app = make_api_app(runtime)  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/system/config")
        assert response.status_code == 200, response.text
        body = response.json()
        # The leaked secret string must NOT appear anywhere in the
        # serialized response body.
        assert "secret-value-do-not-leak" not in response.text
        assert "another-secret" not in response.text
        # And the redacted placeholder MUST appear at the right keys.
        config = body["config"]
        plugins = config["plugins"]
        assert plugins["llm_provider_config"]["api_key"] == "<redacted>"
        assert plugins["llm_provider_config"]["bearer_token"] == "<redacted>"


@pytest.mark.asyncio
async def test_config_does_not_redact_methodology_keys(tmp_path: Path) -> None:
    """``*_artifact_key`` / ``*_hash`` field names are allowlisted (they
    look like secrets to the suffix heuristic but aren't)."""
    runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
    app = make_api_app(runtime)  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/system/config")
        assert response.status_code == 200, response.text
        # No false-positive redactions on the standard config shape.
        assert "<redacted>" not in response.text


@pytest.mark.asyncio
async def test_version_includes_smai_api_spec(tmp_path: Path) -> None:
    """``SystemVersionResponse.smai_api_spec`` is populated per
    ``11-api.md`` §13 OQ6 RESOLVED 2026-05-03."""
    runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
    app = make_api_app(runtime)  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/system/version")
        assert response.status_code == 200, response.text
        body = response.json()
        assert "smai_api_spec" in body
        assert body["smai_api_spec"]


@pytest.mark.asyncio
async def test_health_endpoint_status_ok(tmp_path: Path) -> None:
    """The health probe returns ``{"status": "ok"}`` and touches no plugins."""
    runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
    app = make_api_app(runtime)  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/system/health")
        assert response.status_code == 200, response.text
        assert response.json() == {"status": "ok"}
