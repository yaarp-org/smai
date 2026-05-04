"""``/api/v1/events`` SSE router per ``designs/smai/11-api.md`` §8.

The route opens a long-lived ``text/event-stream`` connection. The
handler subscribes to the in-process :class:`EventBroker` (Task 4.K2 /
Case A per ``12-ui-process.md`` §6.2) and pushes events as they arrive
from the worker engine.

Wire format per `11` §8.1:

* Each event has ``id: <int>``, ``event: <state_change|worker_heartbeat>``,
  ``data: <JSON of the spec payload>``.
* ``:keepalive`` comment line every 15s defeats idle-timeout proxies
  (`11` §8.1).
* On overflow (slow client whose buffer is full), the broker pushes a
  sentinel that this handler turns into ``event: refetch_all`` with
  empty data per `11` §8.3 — the SPA invalidates its entire cache.
* ``X-Accel-Buffering: no`` header per `11` §8.1 defeats nginx
  buffering for proxied deployments.

Reconnection per `11` §8.3:

* The browser's ``EventSource`` automatically sends ``Last-Event-ID``
  on reconnect.
* The optional ``?last_event_id=<int>`` query param is for non-
  ``EventSource`` clients (curl, custom integrations).
* The ring-buffer-resident events with id > the supplied value are
  replayed before live events resume; if the requested id is older
  than the oldest buffered event, the overflow sentinel fires.

Cancellation: Starlette's :class:`StreamingResponse` wraps the body
generator in a task group with a disconnect listener; on client
disconnect the generator is cancelled and our ``finally`` block
detaches the broker subscriber.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from smai_api_spec.events import StateChangeEvent
from smai_api_spec.paths import EVENTS
from smai_events import OVERFLOW_SENTINEL, EnvelopedEvent, EventBroker, SubscriberItem

router = APIRouter()
_log = logging.getLogger(__name__)

_KEEPALIVE_SECONDS = 15.0
"""SSE keepalive cadence per `11` §8.1."""


@router.get(EVENTS)
async def events_stream(
    request: Request,
    last_event_id_query: int | None = Query(default=None, alias="last_event_id"),
) -> StreamingResponse:
    """Open the SSE event stream.

    Reads the in-process :class:`EventBroker` from
    ``request.app.state.event_broker`` (set by
    :func:`smai_api.make_api_app` when an ``event_broker`` is supplied).
    If no broker is bound, returns 503 + ``code: "EVENTS_DISABLED"`` so
    a misconfigured deployment surfaces a clean error rather than a
    confusing 404.
    """
    broker: EventBroker | None = getattr(request.app.state, "event_broker", None)
    if broker is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "EVENTS_DISABLED",
                    "message": (
                        "events stream is not available — make_api_app(...) was "
                        "constructed without an event_broker (the smai dev / smai "
                        "ui --with-worker process binds one automatically; smai "
                        "ui --no-worker against a remote backend wires Task 4.K3's "
                        "PgNotifyEventChannel)"
                    ),
                }
            },
        )

    # Resolve the Last-Event-ID from header OR query param. Per `11`
    # §8.3 the browser-native ``EventSource`` puts it in the header on
    # reconnect; the query-param form is for non-``EventSource`` clients.
    last_event_id: int | None = last_event_id_query
    header_value = request.headers.get("last-event-id")
    if header_value is not None:
        try:
            last_event_id = int(header_value)
        except ValueError:
            # Malformed header — treat as no replay request.
            last_event_id = None

    return StreamingResponse(
        _sse_event_stream(broker=broker, last_event_id=last_event_id),
        media_type="text/event-stream",
        headers={
            # Defeat nginx buffering per `11` §8.1.
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
        },
    )


async def _sse_event_stream(
    *,
    broker: EventBroker,
    last_event_id: int | None,
) -> AsyncIterator[bytes]:
    """Generate the SSE wire bytes for one client connection.

    Cancellation: Starlette's :class:`StreamingResponse` cancels this
    generator when the client disconnects; the ``finally`` block in
    :meth:`EventBroker.subscribe`'s ``async with`` detaches the
    subscriber so the next :meth:`EventBroker.publish` does not try to
    fan out to a closed stream.
    """
    async with broker.subscribe(last_event_id=last_event_id) as subscriber:
        sub_iter = subscriber.__aiter__()

        # Emit an immediate keepalive so the test client sees the
        # response start (status + headers) without waiting for the
        # first real event. This also flushes any HTTP/1.1 idle-timeout
        # proxies sooner.
        yield b": keepalive\n\n"

        next_item_task: asyncio.Task[SubscriberItem | None] | None = None
        try:
            while True:
                if next_item_task is None:
                    next_item_task = asyncio.create_task(_next_or_none(sub_iter))

                try:
                    item = await asyncio.wait_for(
                        asyncio.shield(next_item_task),
                        timeout=_KEEPALIVE_SECONDS,
                    )
                except TimeoutError:
                    # Idle keepalive — emit a comment line and resume.
                    yield b": keepalive\n\n"
                    continue

                next_item_task = None
                if item is None:
                    # Subscriber iterator exhausted — broker closed us
                    # (overflow handled below) or shutdown.
                    return
                if item is OVERFLOW_SENTINEL:
                    yield _format_refetch_all()
                    return
                chunk = _format_sse_chunk(item)
                if chunk is not None:
                    yield chunk
        finally:
            if next_item_task is not None and not next_item_task.done():
                next_item_task.cancel()
                try:
                    await next_item_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass


async def _next_or_none(
    iterator: AsyncIterator[SubscriberItem],
) -> SubscriberItem | None:
    """Read one item or surface ``None`` on ``StopAsyncIteration``."""
    try:
        return await iterator.__anext__()
    except StopAsyncIteration:
        return None


def _format_sse_chunk(item: SubscriberItem) -> bytes | None:
    """Render one broker item as the SSE wire bytes.

    Returns ``None`` for unrecognized item types (defensive). The
    overflow sentinel is handled by the caller; here we only render
    real :class:`EnvelopedEvent` payloads.
    """
    if not isinstance(item, EnvelopedEvent):
        _log.warning("unknown subscriber item type %s; dropping", type(item).__name__)
        return None
    event = item.event
    event_kind = "state_change" if isinstance(event, StateChangeEvent) else "worker_heartbeat"
    payload_json = event.model_dump_json(by_alias=True)
    return (f"id: {item.id}\nevent: {event_kind}\ndata: {payload_json}\n\n").encode()


def _format_refetch_all() -> bytes:
    """The overflow sentinel's wire signal per `11` §8.3."""
    return b"event: refetch_all\ndata: \n\n"


__all__ = ["router"]
