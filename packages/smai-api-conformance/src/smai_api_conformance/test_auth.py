"""Conformance tests for the auth posture per ``designs/smai/11-api.md`` §7.

Two modes:

* ``"disabled"`` (default) — no authentication tokens. Requests
  without ``Authorization`` succeed. Host header validation is the
  loopback-bind defense (per §7.1 / §7.2). The contract surface for
  Host validation is implementation-discretionary in test environments
  (the test client typically uses ``http://test`` as base URL — the
  implementation legitimately accepts that), so the Host-rejection
  test ships as best-effort.
* ``"bearer"`` (opt-in) — ``Authorization: Bearer <token>`` required.
  Requests without it return 403 + ``ErrorEnvelope`` with
  ``code: "FORBIDDEN"``. CSRF is structurally not a concern (per §7.4
  — bearer-header auth is not vulnerable to CSRF).

Subclasses with ``auth_mode == "bearer"`` MUST also override the
``bearer_token`` fixture to supply a valid token.
"""

from __future__ import annotations

from typing import Literal

import pytest
from httpx import AsyncClient
from smai_api_spec.paths import SYSTEM_HEALTH

from smai_api_conformance._4_j2_fixtures import assert_error_envelope


class AuthConformanceTests:
    """Mixin: auth posture conformance tests."""

    auth_mode: Literal["disabled", "bearer"]

    # ---- auth_mode == "disabled" ------------------------------------------

    async def test_health_succeeds_without_auth_when_disabled(self, client: AsyncClient) -> None:
        """Default posture: GET /system/health succeeds without an
        ``Authorization`` header."""
        if self.auth_mode != "disabled":
            pytest.skip("auth_mode != 'disabled' — see bearer-mode tests")
        response = await client.get(SYSTEM_HEALTH)
        assert response.status_code == 200, response.text

    # ---- auth_mode == "bearer" --------------------------------------------

    async def test_request_without_bearer_returns_403_when_enabled(
        self, client: AsyncClient
    ) -> None:
        """Bearer mode: requests without ``Authorization`` get 403
        + ErrorEnvelope (code: FORBIDDEN)."""
        if self.auth_mode != "bearer":
            pytest.skip("auth_mode != 'bearer'")
        response = await client.get(SYSTEM_HEALTH)
        assert response.status_code == 403, response.text
        assert_error_envelope(response, expected_code="FORBIDDEN")

    async def test_request_with_bearer_succeeds_when_enabled(
        self, client: AsyncClient, bearer_token: str
    ) -> None:
        """Bearer mode: requests with valid bearer token succeed."""
        if self.auth_mode != "bearer":
            pytest.skip("auth_mode != 'bearer'")
        response = await client.get(
            SYSTEM_HEALTH, headers={"Authorization": f"Bearer {bearer_token}"}
        )
        assert response.status_code == 200, response.text

    async def test_request_with_invalid_bearer_returns_403_when_enabled(
        self, client: AsyncClient
    ) -> None:
        """Bearer mode: requests with an invalid token get 403."""
        if self.auth_mode != "bearer":
            pytest.skip("auth_mode != 'bearer'")
        response = await client.get(
            SYSTEM_HEALTH,
            headers={"Authorization": "Bearer invalid-token-conformance-probe"},
        )
        assert response.status_code == 403, response.text
        assert_error_envelope(response, expected_code="FORBIDDEN")

    # ---- Host header validation (best-effort) -----------------------------

    async def test_host_rejection_returns_421_when_implementation_supports(
        self, client: AsyncClient
    ) -> None:
        """Host header validation rejects mismatched hosts with 421.

        Per ``11-api.md`` §7.1: implementations enforce Host header
        validation against an allowlist (``127.0.0.1``, ``localhost``,
        ``[::1]``, plus any host:port variants). A request with
        ``Host: evil.com`` should be rejected with 421 +
        ``code: "HOST_REJECTED"``.

        This test ships as best-effort — implementations may
        legitimately disable Host validation in test environments
        (the ASGI client often uses ``http://test`` as base URL,
        which is itself "non-standard"). When the implementation
        accepts the request, the test passes; when it rejects with
        421, the envelope shape is asserted.
        """
        response = await client.get(SYSTEM_HEALTH, headers={"Host": "evil.com"})
        if response.status_code == 421:
            assert_error_envelope(response, expected_code="HOST_REJECTED")
        elif response.status_code == 200:
            # Implementation chose not to enforce Host validation
            # against this synthetic Host header. Spec-permissive.
            pass
        else:
            # Any other status is unexpected — the contract is 200 (no
            # enforcement) or 421 (enforcement triggered).
            pytest.fail(
                f"unexpected status {response.status_code} on Host: evil.com "
                f"probe; expected 200 (no enforcement) or 421 (HOST_REJECTED)"
            )


__all__ = ["AuthConformanceTests"]
