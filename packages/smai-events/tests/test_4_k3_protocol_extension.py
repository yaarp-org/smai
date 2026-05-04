"""Task 4.K3 EventChannel Protocol-extension regression tests.

Pin the additive shape changes:

* :attr:`EventChannel.wants_transactional_fire` exists, defaults
  ``False`` on K2 channels (Null + InProcess), is meant to be ``True``
  on cross-process channels (PgNotifyEventChannel — tested in the
  Postgres plugin's credentialed suite).
* :meth:`EventChannel.fire_transition` accepts the new optional
  ``connection`` kwarg without raising; K2 channels ignore it.
"""

from __future__ import annotations

from smai_events import EventBroker, InProcessEventChannel, NullEventChannel


def test_null_channel_advertises_no_transactional_fire() -> None:
    """Null channels never need a transactional context — there's
    nothing to fire."""
    assert NullEventChannel.wants_transactional_fire is False


def test_in_process_channel_advertises_no_transactional_fire() -> None:
    """InProcess publishes to a broker; pre-commit publish would let
    SSE fan out events for transitions that subsequently rolled back.
    Must stay on the K2 post-commit fire path."""
    assert InProcessEventChannel.wants_transactional_fire is False


async def test_null_channel_accepts_connection_kwarg() -> None:
    """The K3 additive ``connection`` kwarg must be accepted (and
    ignored) by every K2 channel."""
    channel = NullEventChannel()
    await channel.fire_transition(
        kind="comparison_group",
        id="cg-k3",
        from_state="draft",
        to_state="implementing",
        connection="anything-opaque",
    )


async def test_in_process_channel_accepts_connection_kwarg() -> None:
    """InProcess discards the ``connection`` kwarg; broker still
    receives the publish."""
    broker = EventBroker()
    channel = InProcessEventChannel(broker)
    await channel.fire_transition(
        kind="comparison_group",
        id="cg-k3",
        from_state="draft",
        to_state="implementing",
        connection=object(),  # opaque sentinel; channel ignores
    )
    replay = broker.replay_since(0)
    assert len(replay) == 1
