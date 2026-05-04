"""Smoke test for ``smai ui`` end-to-end (Task 4.L1).

Boots :func:`smai_api.make_api_app` over a real :class:`Runtime` (per
the verb's composition surface in `12-ui-process.md` §4) and asserts
the public health endpoint returns 200. Uses
:class:`fastapi.testclient.TestClient` so we don't have to manage a
real uvicorn process — the verb's uvicorn-on-port wrapper is a thin
adapter; everything below it is what these tests exercise.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _4_l1_fakes import make_dev_runtime  # type: ignore[import-not-found]
from fastapi.testclient import TestClient
from smai_api import AuthConfig, make_api_app
from smai_api_spec.paths import SYSTEM_HEALTH


@pytest.mark.asyncio
async def test_smai_ui_health_returns_200(tmp_path: Path) -> None:
    """``GET /api/v1/system/health`` returns 200 against a clean Runtime.

    Mirrors the ``smai ui`` verb's wiring: it constructs ``Runtime``,
    calls ``make_api_app(runtime)``, and serves it under uvicorn. Here
    we substitute :class:`TestClient` for uvicorn — same FastAPI app,
    no port-binding overhead.
    """
    async with make_dev_runtime(tmp_path) as runtime:
        app = make_api_app(runtime)
        client = TestClient(app)
        response = client.get(SYSTEM_HEALTH)
        assert response.status_code == 200, response.text
        body = response.json()
        # The health endpoint per `11` §4.6 returns at minimum a status
        # tag we can assert; tighter shape lives in the conformance
        # suite. Here we just verify the endpoint is alive.
        assert "status" in body or "ok" in body or body == {}


@pytest.mark.asyncio
async def test_smai_ui_health_works_when_auth_disabled(tmp_path: Path) -> None:
    """No bearer required when ``api.auth.enabled`` is False (default)."""
    async with make_dev_runtime(tmp_path) as runtime:
        app = make_api_app(runtime, auth_config=AuthConfig(enabled=False))
        client = TestClient(app)
        # No Authorization header.
        response = client.get(SYSTEM_HEALTH)
        assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_smai_ui_runtime_event_broker_is_present(tmp_path: Path) -> None:
    """Per K2: ``Runtime.start_in_band`` always constructs a broker.

    The L1 verb relies on this — ``make_api_app(runtime)`` falls back
    to ``runtime.event_broker`` when no explicit broker is passed
    (per K1 / K2). This test pins that contract so a future refactor
    of the in-band Runtime can't silently drop the broker.
    """
    async with make_dev_runtime(tmp_path) as runtime:
        assert runtime.event_broker is not None
        # The broker exposes publish + subscribe; we don't drive them
        # here (4.M5 SSE integration owns that), but the attribute
        # check guarantees the L1 verb's auto-wire path stays valid.
