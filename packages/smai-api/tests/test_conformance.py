"""Subclass of :class:`APIConformanceBase` for the smai-api FastAPI app.

Per ``designs/smai/11-api.md`` §10 / DEC-037: any HTTP API claiming to
implement the ``smai-api-spec`` contract proves it by subclassing
:class:`APIConformanceBase`, overriding the ``client`` fixture (and a
small set of seeded-state fixtures), and running pytest. This file is
the smai-api side of that proof.

The events tests are explicitly skipped — Task 4.K2 implements the SSE
events endpoint and removes those skip overrides.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from _4_k1_fixtures import (  # type: ignore[import-not-found]
    EXPERIMENT_YAML,
    SeededState,
    SmaiApiTestRuntime,
    make_test_runtime,
    seed_state,
)
from httpx import ASGITransport, AsyncClient
from smai_api import make_api_app
from smai_api_conformance import APIConformanceBase


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
        """
        runtime = await make_test_runtime(artifact_root=tmp_path / "artifacts")
        seeded = await seed_state(runtime)
        return _SeededHandle(seeded=seeded, runtime=runtime)

    @pytest.fixture
    async def client(self, _seeded: _SeededHandle) -> AsyncIterator[AsyncClient]:
        """Override per ``11`` §10.3: an ``httpx.AsyncClient`` against the
        constructed FastAPI app via ``ASGITransport``."""
        app = make_api_app(_seeded.runtime)  # type: ignore[arg-type]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

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

    # === Skipped pending Task 4.K2 — SSE events ============================

    async def test_events_endpoint_responds_with_event_stream(self, client: AsyncClient) -> None:
        del client
        pytest.skip("Task 4.K2 implements the events endpoint")

    async def test_state_change_event_delivered_within_timeout(self, client: AsyncClient) -> None:
        del client
        pytest.skip("Task 4.K2 implements the events endpoint")


class _SeededHandle:
    """Tuple-shaped wrapper carrying both the seeded IDs and the runtime
    that owns them — lets the per-test fixtures share one runtime
    construction across the dependent fixtures (client / existing_*).
    """

    def __init__(self, *, seeded: SeededState, runtime: SmaiApiTestRuntime) -> None:
        self.seeded = seeded
        self.runtime = runtime
