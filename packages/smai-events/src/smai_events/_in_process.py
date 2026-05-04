"""``InProcessEventChannel`` — engine-side ``EventChannel`` adapter that
writes through to a process-local :class:`EventBroker`.

Per ``designs/smai/12-ui-process.md`` §6.2 (Case A): the in-band
Runtime constructs an :class:`EventBroker`, wraps it in this channel,
threads the channel into the engine's :class:`EngineConfig`, and
exposes the broker so the in-process API can subscribe.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

from smai_api_spec.events import StateChangeEvent, WorkerHeartbeatEvent

from smai_events._broker import EventBroker
from smai_events._channel import EventEntityKind


class InProcessEventChannel:
    """Adapter from :class:`EventChannel` calls to broker publishes.

    Constructs the spec-shape :class:`StateChangeEvent` /
    :class:`WorkerHeartbeatEvent` payloads with a wall-clock ``ts``
    captured at fire time and dispatches via :meth:`EventBroker.publish`.

    :attr:`wants_transactional_fire` is ``False`` — Task 4.K3
    introduced the flag for ``PgNotifyEventChannel``'s pre-commit fire
    path; the in-process broker neither needs nor wants it. Publishing
    pre-commit would let the SSE handler fan out events for transitions
    that subsequently rolled back; the K2 post-commit fire path
    (engine calls :meth:`MetadataStore.transition_*_state` first, then
    invokes :meth:`fire_transition` on success) preserves
    state-vs-event consistency.
    """

    wants_transactional_fire: ClassVar[bool] = False

    def __init__(self, broker: EventBroker) -> None:
        self._broker = broker

    @property
    def broker(self) -> EventBroker:
        """The wrapped broker; exposed for the API process to subscribe."""
        return self._broker

    async def fire_transition(
        self,
        *,
        kind: EventEntityKind,
        id: str,
        from_state: str,
        to_state: str,
        connection: object | None = None,
    ) -> None:
        del connection  # unused — see class docstring
        event = StateChangeEvent.model_validate(
            {
                "kind": kind,
                "id": id,
                "from": from_state,
                "to": to_state,
                "ts": datetime.now(UTC),
            }
        )
        self._broker.publish(event)

    async def fire_heartbeat(
        self,
        *,
        cycle_id: int,
        cycles_processed: int,
    ) -> None:
        event = WorkerHeartbeatEvent(
            cycle_id=cycle_id,
            cycles_processed=cycles_processed,
            ts=datetime.now(UTC),
        )
        self._broker.publish(event)


__all__ = ["InProcessEventChannel"]
