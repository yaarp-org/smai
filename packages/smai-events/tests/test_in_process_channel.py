"""Tests for :class:`smai_events.InProcessEventChannel`.

The channel is a thin adapter over :class:`EventBroker` — these tests
just confirm the wire-shape payloads land in the broker correctly.
"""

from __future__ import annotations

from smai_api_spec.events import StateChangeEvent, WorkerHeartbeatEvent
from smai_events import EnvelopedEvent, EventBroker, InProcessEventChannel


async def test_fire_transition_publishes_state_change_event() -> None:
    broker = EventBroker()
    channel = InProcessEventChannel(broker)
    await channel.fire_transition(
        kind="comparison_group",
        id="cg-fire",
        from_state="draft",
        to_state="implementing",
    )
    replay = broker.replay_since(0)
    assert len(replay) == 1
    item = replay[0]
    assert isinstance(item, EnvelopedEvent)
    assert isinstance(item.event, StateChangeEvent)
    assert item.event.kind == "comparison_group"
    assert item.event.id == "cg-fire"
    assert item.event.from_state == "draft"
    assert item.event.to_state == "implementing"


async def test_fire_heartbeat_publishes_worker_heartbeat_event() -> None:
    broker = EventBroker()
    channel = InProcessEventChannel(broker)
    await channel.fire_heartbeat(cycle_id=7, cycles_processed=42)
    replay = broker.replay_since(0)
    assert len(replay) == 1
    item = replay[0]
    assert isinstance(item, EnvelopedEvent)
    assert isinstance(item.event, WorkerHeartbeatEvent)
    assert item.event.cycle_id == 7
    assert item.event.cycles_processed == 42


def test_broker_property_exposes_underlying_broker() -> None:
    broker = EventBroker()
    channel = InProcessEventChannel(broker)
    assert channel.broker is broker
