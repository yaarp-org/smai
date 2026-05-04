"""Subclass of :class:`APIConformanceBase` for the smai-api FastAPI app.

Per ``designs/smai/11-api.md`` §10 / DEC-037: any HTTP API claiming to
implement the ``smai-api-spec`` contract proves it by subclassing
:class:`APIConformanceBase`, overriding the ``client`` fixture (and a
small set of seeded-state fixtures), and running pytest. This file is
the smai-api side of that proof.

Task 4.K2 landed the SSE events endpoint + the in-process
``EventBroker``. The conformance suite's two SSE tests now run against
this implementation.

Why this fixture spins up a real uvicorn server (instead of using
``ASGITransport``):
``httpx``'s ``ASGITransport`` does not propagate ``http.disconnect``
to a streaming-response body generator until the body completes. An
SSE stream that stays open indefinitely (the production shape per
`11` §8.1) deadlocks the conformance tests under ASGITransport because
the body never finishes. Real HTTP cancels the body generator on TCP
close, so we run the app under ``uvicorn.Server`` on an ephemeral
loopback port and point ``httpx.AsyncClient`` at it. A background
heartbeat ticker publishes a ``WorkerHeartbeatEvent`` every 100ms so
the bounded-delay test sees an event without needing a real worker
loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _4_k1_fixtures import (  # type: ignore[import-not-found]
    EXPERIMENT_YAML,
    SeededState,
    SmaiApiTestRuntime,
    make_test_runtime,
    seed_state,
)
from _4_k2_uvicorn import serve_app  # type: ignore[import-not-found]
from httpx import AsyncClient
from smai_api import make_api_app
from smai_api_conformance import APIConformanceBase
from smai_api_spec.events import WorkerHeartbeatEvent
from smai_events import EventBroker


class TestSmaiApiConformance(APIConformanceBase):
    """Run the parameterizable contract suite against
    :func:`smai_api.make_api_app`.

    Per the per-task carry-forward the seeded-state fixtures supply
    real CG / entry / run / paper / proposal IDs (created via the
    ``MetadataStore`` Protocol surface, no agent dispatch fired); the
    conformance suite's contract assertions exercise the read +
    mutation paths against that pre-populated state.
    """

    auth_mode = "disabled"

    @pytest.fixture
    async def _seeded(self, tmp_path: Path) -> _SeededHandle:
        """One-shot fixture: build the runtime, seed it, return the IDs.

        Cached at function scope per the conformance suite's per-test
        isolation contract — each test gets a fresh in-memory store +
        artifact root, so shared state across tests doesn't leak.

        Per Task 4.K2: a fresh :class:`EventBroker` is constructed per
        test and passed to ``make_api_app`` via the ``client`` fixture
        so the SSE conformance tests have a real broker to subscribe to.
        """
        runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
        seeded = await seed_state(runtime)
        return _SeededHandle(
            seeded=seeded,
            runtime=runtime,
            event_broker=EventBroker(),
        )

    @pytest.fixture
    async def client(self, _seeded: _SeededHandle) -> AsyncIterator[AsyncClient]:
        """Override per ``11`` §10.3: an ``httpx.AsyncClient`` against
        the constructed FastAPI app, served by uvicorn.

        Per Task 4.K2: a background heartbeat ticker fires a
        ``WorkerHeartbeatEvent`` every 100ms while the test runs so the
        bounded-delay events test sees an event without needing the
        real worker loop. The ticker is cancelled on fixture teardown.
        """
        app = make_api_app(
            _seeded.runtime,  # type: ignore[arg-type]
            event_broker=_seeded.event_broker,
        )
        ticker = asyncio.create_task(_publish_heartbeats(_seeded.event_broker))
        try:
            async with serve_app(app) as base_url:
                async with AsyncClient(base_url=base_url, timeout=30.0) as c:
                    yield c
        finally:
            ticker.cancel()
            try:
                await ticker
            except asyncio.CancelledError:
                pass

    @pytest.fixture
    def existing_cg_id(self, _seeded: _SeededHandle) -> str:
        return _seeded.seeded.cg_id

    @pytest.fixture
    def existing_entry_id(self, _seeded: _SeededHandle) -> str:
        return _seeded.seeded.entry_id

    @pytest.fixture
    def existing_run_id(self, _seeded: _SeededHandle) -> str:
        return _seeded.seeded.run_id

    @pytest.fixture
    def experiment_definition_text(self) -> str:
        """Override the conformance suite's placeholder with a real
        compilable DSL document so ``POST /experiments/compile`` and
        ``POST /experiments`` can return 200/202 against this
        implementation."""
        return EXPERIMENT_YAML


async def _publish_heartbeats(broker: EventBroker) -> None:
    """Publish a worker-heartbeat into ``broker`` every 100ms.

    Per the module docstring: the conformance suite's bounded-delay
    test expects an SSE event within
    ``sse_event_timeout_seconds`` (default 5s) of opening the stream.
    The smai-api test runtime is duck-typed and does not run a worker
    loop, so we simulate the worker heartbeat here.
    """
    cycle = 0
    try:
        while True:
            cycle += 1
            broker.publish(
                WorkerHeartbeatEvent(
                    cycle_id=cycle,
                    cycles_processed=cycle,
                    ts=datetime.now(UTC),
                )
            )
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        raise


class _SeededHandle:
    """Tuple-shaped wrapper carrying both the seeded IDs and the runtime
    that owns them — lets the per-test fixtures share one runtime
    construction across the dependent fixtures (client / existing_*).

    Per Task 4.K2 the handle also carries the in-process
    :class:`EventBroker` the conformance suite's events tests subscribe
    to via ``make_api_app(..., event_broker=...)``.
    """

    def __init__(
        self,
        *,
        seeded: SeededState,
        runtime: SmaiApiTestRuntime,
        event_broker: EventBroker,
    ) -> None:
        self.seeded = seeded
        self.runtime = runtime
        self.event_broker = event_broker
