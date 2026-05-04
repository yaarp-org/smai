"""``EventBroker`` — process-local pub/sub for state-change + worker-
heartbeat events.

Per ``designs/smai/12-ui-process.md`` §6.1 (the in-process abstraction
both Case A and Case B feed into) and ``designs/smai/11-api.md`` §8.3
(reconnection via Last-Event-ID over an in-memory ring buffer).

Shape:

* :meth:`EventBroker.publish` — synchronous fan-out. Assigns the next
  monotonic event id, appends to the ring buffer, and pushes onto each
  live subscriber's anyio memory-object stream. Slow subscribers whose
  bounded buffer overflows are detached; their async iterator yields
  the :data:`OVERFLOW_SENTINEL` as the final item so the SSE handler
  can emit ``event: refetch_all`` per `11` §8.3.
* :meth:`EventBroker.subscribe` — async context manager that returns
  an async iterator the SSE handler drains. Subscribers may pass
  ``last_event_id`` to receive ring-buffer replay before live events.
* :meth:`EventBroker.replay_since` — pure helper that reads the
  buffered tail without subscribing; used by tests + by
  :meth:`subscribe` internally.

Ring buffer:

* Bounded (default 100 events; tunable via ``ring_buffer_size``).
* Stores ``EnvelopedEvent`` items in append order.
* Replays the suffix with id > ``last_event_id`` in id order.
* Restart counts as overflow per `11` §8.3 — ids are reset to 1 on
  ``EventBroker.__init__``; SPAs reconnecting after a restart see
  ``id`` numbers below their cached ``last_event_id`` and trigger a
  full refetch via the sentinel handling.

Concurrency model:

* :meth:`publish` is not async (so the engine's fire-on-transition
  wrappers can call it without an extra ``await`` when adapting non-
  async hooks). The actual stream send is via
  :meth:`anyio.streams.memory.MemoryObjectSendStream.send_nowait`,
  which is non-blocking — full buffers raise :class:`anyio.WouldBlock`
  and the broker handles that as overflow.
* The ``id`` counter, ``ring`` buffer, and subscriber list are guarded
  by a per-broker :class:`threading.Lock`. The lock is held for
  microseconds; it is not a contention concern.
"""

from __future__ import annotations

import contextlib
import threading
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import anyio
from smai_api_spec.events import StateChangeEvent, WorkerHeartbeatEvent

if TYPE_CHECKING:
    from anyio.streams.memory import (
        MemoryObjectReceiveStream,
        MemoryObjectSendStream,
    )

# Either of the two payload types the API spec exposes. The broker is
# transport-agnostic — these are the only two payloads the SSE channel
# emits in v1 per ``11`` §8.2.
BrokerEvent = StateChangeEvent | WorkerHeartbeatEvent


@dataclass(frozen=True)
class EnvelopedEvent:
    """One broker payload tagged with its monotonic event id.

    Subscribers receive these; the SSE handler reads ``id`` to populate
    the ``id:`` line on the wire and ``event`` to choose between the
    ``state_change`` and ``worker_heartbeat`` ``event:`` types.
    """

    id: int
    event: BrokerEvent


class _OverflowSentinel:
    """Marker payload pushed to a subscriber whose buffer overflowed.

    The SSE handler intercepts this and emits the ``event: refetch_all``
    wire signal (per `11` §8.3). Distinct sentinel type so a subscriber
    can match on ``isinstance(item, _OverflowSentinel)`` without
    confusing it with a real :class:`EnvelopedEvent`.
    """


OVERFLOW_SENTINEL: Final = _OverflowSentinel()
"""Singleton instance dispatched into a subscriber's stream on overflow."""


_DEFAULT_RING_BUFFER_SIZE = 100
"""Default in-memory ring buffer size per `11` §8.3 ("last ~100 events")."""

_DEFAULT_PER_SUBSCRIBER_BUFFER = 100
"""Default per-subscriber backpressure buffer.

Sized at the same magnitude as the ring buffer: a subscriber whose
SSE socket has stalled for the time it takes to deliver 100 events is
already too far behind for replay to help — better to drop the buffer
and let the SPA refetch via the sentinel."""


# Subscribers receive either a real event or the overflow sentinel.
SubscriberItem = EnvelopedEvent | _OverflowSentinel


@dataclass
class _Subscriber:
    """Internal handle joining a stream to its overflow-state flag.

    The flag lets :meth:`EventBroker.publish` mark the subscriber as
    overflowed even when its bounded stream buffer is full; the
    matching :meth:`EventBroker._iter_receive` then emits the
    :data:`OVERFLOW_SENTINEL` after the stream's buffered events are
    drained — guaranteed regardless of buffer pressure.
    """

    send_stream: MemoryObjectSendStream[EnvelopedEvent]
    overflowed: bool = field(default=False)


class EventBroker:
    """Process-local pub/sub for the live-updates channel."""

    def __init__(
        self,
        *,
        ring_buffer_size: int = _DEFAULT_RING_BUFFER_SIZE,
        per_subscriber_buffer: int = _DEFAULT_PER_SUBSCRIBER_BUFFER,
    ) -> None:
        if ring_buffer_size <= 0:
            raise ValueError(f"ring_buffer_size must be > 0; got {ring_buffer_size}")
        if per_subscriber_buffer <= 0:
            raise ValueError(f"per_subscriber_buffer must be > 0; got {per_subscriber_buffer}")
        self._ring_buffer_size = ring_buffer_size
        self._per_subscriber_buffer = per_subscriber_buffer
        self._lock = threading.Lock()
        self._next_id: int = 1
        self._ring: deque[EnvelopedEvent] = deque(maxlen=ring_buffer_size)
        self._subscribers: list[_Subscriber] = []

    # === Publishing =========================================================

    def publish(self, event: BrokerEvent) -> EnvelopedEvent:
        """Fan ``event`` out to every live subscriber + the ring buffer.

        Returns the :class:`EnvelopedEvent` (with the assigned id) so
        callers can log / introspect what was published. The event id
        is monotonic per-broker (resets to 1 on construction; restart-
        as-overflow per `11` §8.3).

        Slow subscribers whose buffer is full are marked as overflowed
        and detached; the matching subscriber iterator yields
        :data:`OVERFLOW_SENTINEL` as its final item before terminating.
        """
        with self._lock:
            envelope = EnvelopedEvent(id=self._next_id, event=event)
            self._next_id += 1
            self._ring.append(envelope)
            # Snapshot subscribers so we don't hold the lock while
            # sending (anyio send_nowait is non-blocking, but the snapshot
            # also lets us mutate the list inside the lock for detaches).
            subs_snapshot = list(self._subscribers)

        overflowed: list[_Subscriber] = []
        for sub in subs_snapshot:
            if sub.overflowed:
                continue
            try:
                sub.send_stream.send_nowait(envelope)
            except anyio.WouldBlock:
                # Buffer full — mark overflowed and detach. The
                # subscriber's iterator yields OVERFLOW_SENTINEL as
                # its final item before terminating.
                sub.overflowed = True
                overflowed.append(sub)
            except anyio.BrokenResourceError:
                # Subscriber already closed its receive side (client
                # disconnected). Detach silently — no sentinel.
                overflowed.append(sub)

        if overflowed:
            with self._lock:
                for sub in overflowed:
                    if sub in self._subscribers:
                        self._subscribers.remove(sub)
            for sub in overflowed:
                with contextlib.suppress(Exception):
                    sub.send_stream.close()

        return envelope

    # === Subscribing =======================================================

    @contextlib.asynccontextmanager
    async def subscribe(
        self,
        *,
        last_event_id: int | None = None,
    ) -> AsyncGenerator[AsyncIterator[SubscriberItem]]:
        """Async-context-managed subscription.

        Yields an async iterator the caller drains for events; the
        send-side stream is automatically detached on context exit so a
        client disconnect is observable to the next :meth:`publish`.

        ``last_event_id`` enables ring-buffer replay per `11` §8.3:
        events with id > ``last_event_id`` already in the ring buffer
        are emitted before the live stream begins. If the requested id
        is older than the oldest buffered event (overflow), the
        :data:`OVERFLOW_SENTINEL` is emitted first so the SSE handler
        can issue ``event: refetch_all`` and the SPA invalidates its
        cache.
        """
        send_stream: MemoryObjectSendStream[EnvelopedEvent]
        receive_stream: MemoryObjectReceiveStream[EnvelopedEvent]
        send_stream, receive_stream = anyio.create_memory_object_stream[EnvelopedEvent](
            max_buffer_size=self._per_subscriber_buffer,
        )
        subscriber = _Subscriber(send_stream=send_stream)

        # Build the replay list under the lock so a concurrent publish
        # cannot interleave a "new" event between replay and live.
        replay_payloads: list[SubscriberItem]
        replay_overflowed = False
        with self._lock:
            if last_event_id is not None:
                replay_payloads = list(self._replay_locked(last_event_id))
                # If the replay starts with the sentinel, surface it
                # via the iterator (without relying on the bounded
                # stream buffer, which may not have room).
                if replay_payloads and isinstance(replay_payloads[0], _OverflowSentinel):
                    replay_overflowed = True
                    replay_payloads = replay_payloads[1:]
            else:
                replay_payloads = []
            self._subscribers.append(subscriber)

        try:
            async with receive_stream:
                yield self._iter_receive(
                    receive_stream=receive_stream,
                    subscriber=subscriber,
                    replay=replay_payloads,
                    replay_overflowed=replay_overflowed,
                )
        finally:
            with self._lock:
                if subscriber in self._subscribers:
                    self._subscribers.remove(subscriber)
            with contextlib.suppress(Exception):
                send_stream.close()

    @staticmethod
    async def _iter_receive(
        *,
        receive_stream: MemoryObjectReceiveStream[EnvelopedEvent],
        subscriber: _Subscriber,
        replay: list[SubscriberItem],
        replay_overflowed: bool,
    ) -> AsyncIterator[SubscriberItem]:
        if replay_overflowed:
            yield OVERFLOW_SENTINEL
            return
        for item in replay:
            yield item
        async for envelope in receive_stream:
            yield envelope
        # Stream closed (broker detach or client side aclose). If the
        # broker marked us overflowed, surface the sentinel so the SSE
        # handler can emit ``event: refetch_all`` per `11` §8.3.
        if subscriber.overflowed:
            yield OVERFLOW_SENTINEL

    def _replay_locked(self, last_event_id: int) -> list[SubscriberItem]:
        """Build the replay list under the broker lock (`11` §8.3).

        The lock is held by :meth:`subscribe`; this is a pure helper
        that reads ``self._ring`` and returns either:

        * A list whose first element is the overflow sentinel (when the
          requested id is older than the oldest buffered event — the
          SPA missed too many events), and the rest is the buffered
          suffix. Callers handle the sentinel-first case as "drop the
          rest, the SPA will refetch."
        * An empty list (when ``last_event_id`` is ahead of every
          buffered event — process restart with the SPA's cached id
          higher than the new broker's id counter; we have nothing to
          replay).
        * The buffered suffix with id > ``last_event_id`` in id order.
        """
        if not self._ring:
            return []
        oldest_id = self._ring[0].id
        if last_event_id < oldest_id - 1:
            # The SPA's cached id refers to an event we no longer have.
            return [OVERFLOW_SENTINEL]
        return [item for item in self._ring if item.id > last_event_id]

    def replay_since(self, last_event_id: int) -> list[SubscriberItem]:
        """Public read-only replay helper (testing + tooling)."""
        with self._lock:
            return self._replay_locked(last_event_id)

    # === Introspection (for tests + tooling) ===============================

    @property
    def ring_buffer_size(self) -> int:
        return self._ring_buffer_size

    @property
    def per_subscriber_buffer(self) -> int:
        return self._per_subscriber_buffer

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def buffered_count(self) -> int:
        with self._lock:
            return len(self._ring)


__all__ = [
    "BrokerEvent",
    "EnvelopedEvent",
    "EventBroker",
    "OVERFLOW_SENTINEL",
    "SubscriberItem",
]
