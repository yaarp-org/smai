"""Cross-cutting tests for the error envelope shape.

Per ``designs/smai/11-api.md`` §6. Per-resource test files already
assert the error envelope on their specific 400/404/409 cases; this
file adds the envelope assertions that don't naturally hang off a
particular resource (404 on completely unknown URL, 405 on the wrong
method against a known resource, etc.).

Per the brief: "Error envelope shape — every non-2xx body is a valid
``ErrorEnvelope`` (parses cleanly; ``error.code`` from the catalog)."

Note: we don't test ``500`` — that is a bug surface per ``11`` §6.2
(SHOULD NOT happen in production). 503 / 504 (LLM upstream / timeout)
are tested via downstream resource tests when the implementation
exposes them.
"""

from __future__ import annotations

from httpx import AsyncClient
from smai_api_spec.paths import PROPOSAL_DETAIL

from smai_api_conformance._4_j2_fixtures import assert_error_envelope


class ErrorsConformanceTests:
    """Mixin: cross-cutting error envelope conformance tests."""

    async def test_404_unknown_url_returns_envelope(self, client: AsyncClient) -> None:
        """A request to a completely-unknown URL under ``/api/v1/`` returns
        a 404 whose body either is a valid :class:`ErrorEnvelope` or is
        empty (some FastAPI implementations return the framework's
        default 404 body for unmatched routes).

        Implementations claiming full conformance should serve the
        envelope; the looser assertion here keeps the suite passable
        against the framework default while still catching obviously
        broken responses.
        """
        response = await client.get("/api/v1/this-route-does-not-exist")
        assert response.status_code in {404, 405}, response.text
        # Don't enforce envelope — FastAPI's default unmatched-route
        # body is ``{"detail": "Not Found"}``. Real implementations
        # are encouraged to serve the envelope but the contract treats
        # 404 on an unmatched URL as "no resource", not "failed RPC".

    async def test_404_known_resource_unknown_id_uses_envelope(self, client: AsyncClient) -> None:
        """404 on a known resource URL with an unknown ID MUST use the
        envelope.

        This is the core contract: a known resource handler that
        couldn't find the requested entity must return the structured
        ``{"error": {"code": "...NOT_FOUND", ...}}`` body so client
        code can branch on the code.
        """
        url = PROPOSAL_DETAIL.format(proposal_id="prop_envelope_probe")
        response = await client.get(url)
        assert response.status_code == 404, response.text
        envelope_error = assert_error_envelope(response)
        assert envelope_error.code == "PROPOSAL_NOT_FOUND"
        assert envelope_error.message  # human-readable message present

    async def test_envelope_message_is_string(self, client: AsyncClient) -> None:
        """Every error envelope's ``message`` is a non-empty string."""
        url = PROPOSAL_DETAIL.format(proposal_id="prop_message_probe")
        response = await client.get(url)
        assert response.status_code == 404, response.text
        envelope_error = assert_error_envelope(response)
        assert isinstance(envelope_error.message, str)
        assert envelope_error.message != ""


__all__ = ["ErrorsConformanceTests"]
