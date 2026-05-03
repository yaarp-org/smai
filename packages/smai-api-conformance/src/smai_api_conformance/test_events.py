"""Conformance tests for the SSE channel at ``GET /api/v1/events``.

Per ``designs/smai/11-api.md`` §4.7 / §8.

The SSE delivery test is shape-only: it asserts that within
:attr:`APIConformanceBase.sse_event_timeout_seconds` (default 5s, per
``11`` §10.2 / §13 OQ12) the channel emits at least one event whose
JSON payload parses cleanly into one of the documented event types
(:class:`StateChangeEvent` or :class:`WorkerHeartbeatEvent`).

The test does NOT assert that a specific POST caused a specific event
— that is lifecycle coupling (worker behavior → engine fires
transitions → API drains the channel) and lives in implementation
suites. Per the brief carry-forward: 4.K1's smai-api will exercise
the worker→API channel mechanics in its own integration tests.
"""

from __future__ import annotations

import asyncio

import pytest
from httpx import AsyncClient
from pydantic import ValidationError
from smai_api_spec import StateChangeEvent, WorkerHeartbeatEvent
from smai_api_spec.paths import EVENTS, PROPOSALS

from smai_api_conformance._4_j2_fixtures import (
    read_one_sse_event,
    sample_proposal_request_body,
)


class EventsConformanceTests:
    """Mixin: SSE channel conformance tests."""

    # The SSE event timeout is configurable on APIConformanceBase. Real
    # implementations whose worker→API channel takes longer than 5s in
    # test (e.g. Postgres LISTEN/NOTIFY in CI) should override
    # ``sse_event_timeout_seconds`` on their subclass.

    sse_event_timeout_seconds: float

    async def test_events_endpoint_responds_with_event_stream(self, client: AsyncClient) -> None:
        """Connecting to /events returns an event-stream Content-Type.

        Stream is opened, the response headers are inspected, and the
        connection is closed without reading any events. This isolates
        the "endpoint exists and advertises SSE" assertion from the
        downstream "events arrive within bounded delay" assertion.
        """
        async with client.stream("GET", EVENTS) as response:
            assert response.status_code == 200, response.text
            content_type = response.headers.get("content-type", "")
            # Per ``11`` §8.1: the Content-Type is ``text/event-stream``;
            # implementations may legitimately add a charset suffix.
            assert content_type.startswith("text/event-stream"), (
                f"expected Content-Type to start with text/event-stream, got {content_type!r}"
            )

    async def test_state_change_event_delivered_within_timeout(self, client: AsyncClient) -> None:
        """A state-changing POST results in an event on /events within
        :attr:`sse_event_timeout_seconds`.

        The test:

        1. Opens the SSE stream.
        2. POSTs a fresh proposal (a state-creating event).
        3. Reads up to one event with ``asyncio.wait_for`` bounded by
           ``sse_event_timeout_seconds``.
        4. Asserts the parsed event is one of the documented payload
           types (:class:`StateChangeEvent` or
           :class:`WorkerHeartbeatEvent`).

        The test does NOT assert that the read event corresponds to
        the just-POSTed proposal — heartbeats, prior state changes
        from concurrent activity, and the worker's own emissions all
        legitimately arrive first. Shape-only.
        """
        async with client.stream("GET", EVENTS) as response:
            assert response.status_code == 200, response.text

            # Trigger a state change so a real implementation has
            # something to emit. The mock implementation may emit
            # canned events without needing this trigger; both modes
            # are spec-conformant.
            submit = await client.post(PROPOSALS, json=sample_proposal_request_body())
            assert submit.status_code == 202, submit.text

            try:
                event = await asyncio.wait_for(
                    read_one_sse_event(response.aiter_lines()),
                    timeout=self.sse_event_timeout_seconds,
                )
            except TimeoutError:
                pytest.fail(
                    f"no SSE event received within "
                    f"{self.sse_event_timeout_seconds}s — implementation may "
                    f"not be wiring state transitions to the SSE channel"
                )
            assert event is not None, "stream closed before delivering an event"
            payload = event.parse_data()
            # The payload must parse into one of the documented event
            # types. ``StateChangeEvent`` accepts the JSON aliases
            # ("from"/"to") via ``populate_by_name=True``.
            try:
                StateChangeEvent.model_validate(payload)
            except ValidationError:
                # Fall back to WorkerHeartbeatEvent — heartbeats are
                # also legitimate first events on a quiet channel.
                WorkerHeartbeatEvent.model_validate(payload)


__all__ = ["EventsConformanceTests"]
