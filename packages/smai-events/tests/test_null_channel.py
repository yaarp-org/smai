"""Tests for :class:`smai_events.NullEventChannel`.

Smoke-only — confirms the no-op default raises nothing on either fire
method and is structurally an :class:`EventChannel` (Protocol check).
"""

from __future__ import annotations

from smai_events import EventChannel, NullEventChannel


async def test_fire_transition_is_no_op() -> None:
    channel = NullEventChannel()
    await channel.fire_transition(
        kind="proposal",
        id="prop-noop",
        from_state="proposal_submitted",
        to_state="designing",
    )


async def test_fire_heartbeat_is_no_op() -> None:
    channel = NullEventChannel()
    await channel.fire_heartbeat(cycle_id=1, cycles_processed=1)


def test_null_channel_satisfies_event_channel_protocol() -> None:
    channel: EventChannel = NullEventChannel()
    assert isinstance(channel, EventChannel)
