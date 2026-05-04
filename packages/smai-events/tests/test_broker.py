"""Tests for :class:`smai_events.EventBroker`.

Per ``designs/smai/12-ui-process.md`` §6.1 + ``11-api.md`` §8.3.
Covers: monotonic ids, ring-buffer overflow + replay, fan-out to
multiple subscribers, slow-subscriber overflow with sentinel, the
async-context-managed subscribe/unsubscribe lifecycle, and the
restart-as-overflow case (`11` §8.3).
"""

from __future__ import annotations

import anyio
import pytest
from _4_k2_fixtures import make_heartbeat, make_state_change  # type: ignore[import-not-found]
from smai_events import (
    OVERFLOW_SENTINEL,
    EnvelopedEvent,
    EventBroker,
    SubscriberItem,
)


async def _drain(
    iterator: object,  # AsyncIterator[SubscriberItem] but typed loosely so test reads cleanly
    *,
    n: int,
    timeout: float = 1.0,
) -> list[SubscriberItem]:
    """Drain up to ``n`` items off an SSE-shaped subscriber iterator."""
    out: list[SubscriberItem] = []

    async def _read() -> None:
        async for item in iterator:  # type: ignore[attr-defined]
            out.append(item)
            if len(out) >= n:
                break

    with anyio.fail_after(timeout):
        await _read()
    return out


def test_publish_assigns_monotonic_ids() -> None:
    broker = EventBroker()
    e1 = broker.publish(make_state_change(id="cg-1"))
    e2 = broker.publish(make_state_change(id="cg-2"))
    e3 = broker.publish(make_heartbeat(cycle_id=42))
    assert (e1.id, e2.id, e3.id) == (1, 2, 3)


def test_publish_records_in_ring_buffer() -> None:
    broker = EventBroker(ring_buffer_size=3)
    broker.publish(make_state_change(id="cg-1"))
    broker.publish(make_state_change(id="cg-2"))
    broker.publish(make_state_change(id="cg-3"))
    broker.publish(make_state_change(id="cg-4"))  # evicts id=1
    assert broker.buffered_count() == 3
    # Replay since id=1 means events with id > 1; the suffix is 2/3/4.
    replay = broker.replay_since(1)
    assert all(isinstance(item, EnvelopedEvent) for item in replay)
    ids = [item.id for item in replay if isinstance(item, EnvelopedEvent)]
    assert ids == [2, 3, 4]


def test_replay_since_overflow_returns_sentinel() -> None:
    broker = EventBroker(ring_buffer_size=3)
    broker.publish(make_state_change(id="cg-1"))
    broker.publish(make_state_change(id="cg-2"))
    broker.publish(make_state_change(id="cg-3"))
    broker.publish(make_state_change(id="cg-4"))  # evicts id=1, oldest is now id=2
    # Subscriber's last_event_id refers to an event we no longer have.
    replay = broker.replay_since(0)
    assert replay == [OVERFLOW_SENTINEL]


def test_replay_since_ahead_of_buffer_returns_empty() -> None:
    """Restart case (`11` §8.3): SPA's cached id is higher than ours.

    A fresh broker has next_id=1; if the SPA presents
    last_event_id=999 (cached from a prior process), the buffer holds
    nothing newer — return empty so the SPA proceeds with the live
    stream and triggers refetch on its own when the next id < 999
    arrives.
    """
    broker = EventBroker()
    assert broker.replay_since(999) == []


async def test_subscribe_receives_published_events() -> None:
    broker = EventBroker()
    async with broker.subscribe() as subscriber:
        event = make_state_change(id="cg-async")
        broker.publish(event)
        items = await _drain(subscriber, n=1)
    assert len(items) == 1
    item = items[0]
    assert isinstance(item, EnvelopedEvent)
    assert item.event == event


async def test_subscribe_replays_buffer_on_attach() -> None:
    broker = EventBroker(ring_buffer_size=10)
    e1 = broker.publish(make_state_change(id="cg-1"))
    e2 = broker.publish(make_state_change(id="cg-2"))
    async with broker.subscribe(last_event_id=e1.id) as subscriber:
        items = await _drain(subscriber, n=1)
    # Only id > 1 replays; e2.
    assert len(items) == 1
    item = items[0]
    assert isinstance(item, EnvelopedEvent)
    assert item.id == e2.id


async def test_multiple_subscribers_each_receive_each_event() -> None:
    broker = EventBroker()
    async with (
        broker.subscribe() as sub_a,
        broker.subscribe() as sub_b,
    ):
        broker.publish(make_state_change(id="cg-fanout"))
        items_a = await _drain(sub_a, n=1)
        items_b = await _drain(sub_b, n=1)
    assert len(items_a) == 1
    assert len(items_b) == 1
    assert items_a[0] == items_b[0]


async def test_subscriber_detached_on_context_exit() -> None:
    broker = EventBroker()
    async with broker.subscribe():
        assert broker.subscriber_count() == 1
    assert broker.subscriber_count() == 0


async def test_slow_subscriber_overflow_emits_sentinel_and_detaches() -> None:
    """Per `11` §8.3: a subscriber whose buffer is full is detached
    with a refetch_all sentinel so the SPA invalidates its cache."""
    broker = EventBroker(per_subscriber_buffer=2)
    async with broker.subscribe() as subscriber:
        # Fill the per-subscriber buffer to capacity but DO NOT drain.
        broker.publish(make_state_change(id="cg-1"))
        broker.publish(make_state_change(id="cg-2"))
        # The next publish overflows — the broker pushes the sentinel
        # into the subscriber's stream and detaches.
        broker.publish(make_state_change(id="cg-3"))
        # After the publish that overflowed, the broker should have
        # detached the slow subscriber.
        assert broker.subscriber_count() == 0
        items = await _drain(subscriber, n=3)
    # The subscriber drained the two buffered events plus the sentinel.
    kinds = [type(item).__name__ for item in items]
    assert kinds[-1] == "_OverflowSentinel"


async def test_publish_after_subscribe_close_does_not_raise() -> None:
    broker = EventBroker()
    async with broker.subscribe():
        pass
    # Should not raise even though the subscriber is gone.
    broker.publish(make_state_change(id="cg-after"))


def test_invalid_buffer_sizes_raise() -> None:
    with pytest.raises(ValueError, match="ring_buffer_size"):
        EventBroker(ring_buffer_size=0)
    with pytest.raises(ValueError, match="per_subscriber_buffer"):
        EventBroker(per_subscriber_buffer=0)


def test_publish_returns_envelope() -> None:
    broker = EventBroker()
    event = make_state_change(id="cg-envelope")
    envelope = broker.publish(event)
    assert envelope.event == event
    assert envelope.id == 1
