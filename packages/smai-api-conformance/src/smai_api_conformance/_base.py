""":class:`APIConformanceBase` — the parameterizable test base.

Per ``designs/smai/11-api.md`` §10 / DEC-037: any HTTP API claiming to
implement the ``smai-api-spec`` contract proves it by subclassing this
base, overriding the ``client`` fixture to point at the implementation,
and running pytest. The inherited contract methods exercise every
endpoint in ``11-api.md`` §4 plus the cross-cutting concerns (errors,
pagination, auth, SSE).

Usage::

    # In packages/smai-api/tests/test_conformance.py:
    import pytest
    from httpx import ASGITransport, AsyncClient
    from smai_api import build_api_app
    from smai_api_conformance import APIConformanceBase
    from smai_cli.runtime import Runtime

    class TestSmaiApiConformance(APIConformanceBase):
        @pytest.fixture
        async def client(self) -> AsyncClient:
            runtime = await Runtime.start_in_band(test_config(), run_worker=False)
            app = build_api_app(runtime)
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as c:
                yield c

The base inherits from per-resource mixin classes — one per endpoint
group in ``11-api.md`` §4, plus mixins for the four cross-cutting
concerns (errors / pagination / auth / SSE). Multiple inheritance keeps
each mixin focused on one slice of the contract while a single
subclass picks up everything via MRO.

The ``client`` fixture in the base raises :class:`NotImplementedError`
— pytest renders that as a clear failure for every collected test
method when no subclass override is supplied. (We do not skip; a
subclass that forgets to override ``client`` is a configuration bug,
not a contract gap.)

Configuration knobs (override as class attributes):

* :attr:`sse_event_timeout_seconds` — bounded delay for the SSE
  state-change tests (per ``11-api.md`` §10.2 / §13 OQ12). Default 5s.
* :attr:`auth_mode` — ``"disabled"`` (default) or ``"bearer"``. When
  ``"bearer"`` the suite expects a ``bearer_token`` fixture override
  and runs the additional bearer-mode test branches.
"""

from __future__ import annotations

from typing import Literal

import pytest
from httpx import AsyncClient

from smai_api_conformance.test_auth import AuthConformanceTests
from smai_api_conformance.test_comparison_groups import ComparisonGroupsConformanceTests
from smai_api_conformance.test_errors import ErrorsConformanceTests
from smai_api_conformance.test_events import EventsConformanceTests
from smai_api_conformance.test_experiments import ExperimentsConformanceTests
from smai_api_conformance.test_pagination import PaginationConformanceTests
from smai_api_conformance.test_papers import PapersConformanceTests
from smai_api_conformance.test_proposals import ProposalsConformanceTests
from smai_api_conformance.test_runs import RunsConformanceTests
from smai_api_conformance.test_system import SystemConformanceTests


class APIConformanceBase(
    ProposalsConformanceTests,
    PapersConformanceTests,
    ExperimentsConformanceTests,
    ComparisonGroupsConformanceTests,
    RunsConformanceTests,
    SystemConformanceTests,
    EventsConformanceTests,
    ErrorsConformanceTests,
    PaginationConformanceTests,
    AuthConformanceTests,
):
    """Universal contract suite for any ``smai-api-spec`` HTTP API.

    Subclass and provide a ``client`` fixture pointing at the
    implementation. The inherited ``test_*`` methods then run against
    that client.

    Lifecycle / state-machine correctness testing is **out of scope** —
    those tests live in implementation-specific suites
    (``packages/smai-api/tests/integration/`` for SMAI's OSS API; the
    Yaarp v2 backend's own integration suite for the closed side). This
    base asserts the **shape and protocol semantics** of the contract:
    status codes, response-body shapes, error envelope, pagination
    round-trip, RPC verb idempotency, SSE wire-format, auth posture.
    """

    # === Configuration knobs ================================================
    #
    # Subclasses override these as class attributes:
    #
    #     class TestMyApi(APIConformanceBase):
    #         sse_event_timeout_seconds = 10.0
    #         auth_mode = "bearer"
    #         ...

    sse_event_timeout_seconds: float = 5.0
    """Bounded delay for the SSE state-change test (per ``11`` §10.2)."""

    auth_mode: Literal["disabled", "bearer"] = "disabled"
    """``"disabled"`` (default) or ``"bearer"`` per ``11`` §7."""

    # === The fixture every subclass MUST override ===========================

    @pytest.fixture
    async def client(self) -> AsyncClient:
        """Return an :class:`httpx.AsyncClient` pointed at the implementation.

        Subclasses MUST override. The fixture may either ``return`` the
        client (caller is responsible for cleanup) or use ``yield``
        inside an ``async with`` block (preferred).

        For an ASGI implementation like FastAPI::

            @pytest.fixture
            async def client(self) -> AsyncClient:
                async with AsyncClient(
                    transport=ASGITransport(app=self.app),
                    base_url="http://test",
                ) as c:
                    yield c

        For an over-the-wire implementation, point ``base_url`` at the
        running server.
        """
        raise NotImplementedError(
            "subclasses of APIConformanceBase must override the `client` fixture"
        )

    # === Optional fixture overrides for richer subclass-side coverage =======

    @pytest.fixture
    def bearer_token(self) -> str:
        """Override when ``auth_mode == "bearer"`` to supply the token.

        Default raises :class:`pytest.skip.Exception` so tests requiring
        the token skip cleanly when the subclass forgot to override.
        """
        pytest.skip(
            "bearer_token fixture not provided; override in your subclass when "
            "auth_mode == 'bearer'"
        )
        raise AssertionError("unreachable")  # pragma: no cover


__all__ = ["APIConformanceBase"]
