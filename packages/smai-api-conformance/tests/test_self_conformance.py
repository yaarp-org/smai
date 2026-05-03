"""Self-test: run the conformance suite against an in-process mock.

Per Task 4.J2 acceptance: subclass :class:`APIConformanceBase` with a
fixture that returns an :class:`httpx.AsyncClient` backed by
:class:`httpx.MockTransport`. The mock returns spec-conformant canned
responses for every endpoint the suite tests. If the suite passes
against the mock, the suite is internally coherent — the test methods
are well-formed and can run end-to-end without a real implementation.

Real implementations (smai-api at Task 4.K1; Yaarp v2 backend) do
not import the mock. They subclass :class:`APIConformanceBase` with
their own ``client`` fixture pointing at the actual app.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient, MockTransport
from smai_api_conformance import APIConformanceBase
from smai_api_conformance._4_j2_mock_responses import (
    SELF_TEST_CG_ID,
    SELF_TEST_ENTRY_ID,
    SELF_TEST_RUN_ID,
    mock_handler,
)

# Silence pyright's unused-import on ASGITransport — kept here so the
# conformance README's forward-looking subclass example (which uses
# ASGITransport) lines up with what's importable from httpx.
_ = ASGITransport


class TestSelfConformance(APIConformanceBase):
    """Run the inherited contract suite against the mock handler."""

    @pytest.fixture
    async def client(self) -> AsyncIterator[AsyncClient]:
        """Provide an :class:`AsyncClient` backed by the mock handler."""
        transport = MockTransport(mock_handler)
        async with AsyncClient(transport=transport, base_url="http://self-test") as c:
            yield c

    # The mock recognizes these IDs (per ``_4_j2_mock_responses.py``) and
    # returns the matching detail responses. Real implementations
    # override these with IDs from their own seeding path.

    @pytest.fixture
    def existing_cg_id(self) -> str:
        return SELF_TEST_CG_ID

    @pytest.fixture
    def existing_entry_id(self) -> str:
        return SELF_TEST_ENTRY_ID

    @pytest.fixture
    def existing_run_id(self) -> str:
        return SELF_TEST_RUN_ID
