"""Bearer-token mode end-to-end (Task 4.L1 Part 4 / `11` §7.3).

Verifies the bearer-token opt-in path:

* When ``api.auth.enabled=True`` and the token file doesn't exist, it
  is auto-generated via :func:`secrets.token_urlsafe(32)` and written
  mode ``0o600``.
* Existing tokens are preserved across calls (browser tabs keep
  working).
* Requests without an ``Authorization: Bearer <token>`` header → 403.
* Requests with the correct token → 200.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from _4_l1_fakes import make_dev_runtime  # type: ignore[import-not-found]
from fastapi.testclient import TestClient
from smai_api import AuthConfig, make_api_app
from smai_api.auth import _read_or_create_token_file
from smai_api_spec.paths import SYSTEM_HEALTH


@pytest.mark.asyncio
async def test_bearer_mode_blocks_unauthenticated_requests(tmp_path: Path) -> None:
    """``api.auth.enabled=True`` rejects requests without the bearer header."""
    token_path = tmp_path / "api-token"
    async with make_dev_runtime(tmp_path) as runtime:
        app = make_api_app(
            runtime,
            auth_config=AuthConfig(enabled=True, token_path=token_path),
        )
        client = TestClient(app)
        response = client.get(SYSTEM_HEALTH)
        assert response.status_code == 403, response.text


@pytest.mark.asyncio
async def test_bearer_mode_accepts_correct_token(tmp_path: Path) -> None:
    """A request with the matching bearer header succeeds."""
    token_path = tmp_path / "api-token"
    async with make_dev_runtime(tmp_path) as runtime:
        app = make_api_app(
            runtime,
            auth_config=AuthConfig(enabled=True, token_path=token_path),
        )
        # The token file is auto-generated when make_api_app runs the
        # middleware constructor — read it and use it.
        token = token_path.read_text(encoding="utf-8").strip()
        assert token, "expected the bearer token to be auto-generated"

        client = TestClient(app)
        response = client.get(
            SYSTEM_HEALTH,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_bearer_token_file_is_mode_0600(tmp_path: Path) -> None:
    """Auto-generated token file is mode ``0o600`` per `11` §7.3."""
    token_path = tmp_path / "api-token"
    _read_or_create_token_file(token_path)
    if os.name == "posix":
        mode = stat.S_IMODE(token_path.stat().st_mode)
        assert mode == 0o600, f"expected 0o600, got {oct(mode)}"


@pytest.mark.asyncio
async def test_bearer_token_preserved_across_construction(tmp_path: Path) -> None:
    """Existing tokens survive a second ``make_api_app`` call.

    The browser-tab UX in `11` §7.3 hinges on this — restarting
    ``smai ui`` doesn't invalidate existing sessions.
    """
    token_path = tmp_path / "api-token"
    first = _read_or_create_token_file(token_path)
    second = _read_or_create_token_file(token_path)
    assert first == second
