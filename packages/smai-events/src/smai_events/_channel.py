"""``EventChannel`` Protocol + the no-op ``NullEventChannel`` default.

Per ``designs/smai/12-ui-process.md`` §6.4 (resolved 2026-05-03 as the
``smai-events`` sibling package OQ1, engine-wraps OQ4): the engine
calls into one of these implementations after every successful state-
machine transition. The Protocol exposes exactly two methods so the
surface is small enough for K3's ``PgNotifyEventChannel`` (cross-
process Postgres LISTEN/NOTIFY) to implement against without further
shape changes.

Both methods are ``async`` for forward compatibility — the in-process
implementation is synchronous in spirit (anyio memory streams accept
non-blocking sends), but K3's Postgres-side ``pg_notify`` runs inside
the active asyncpg transaction and is async by construction. Picking
``async`` for v1 keeps the call sites uniform across implementations
and avoids a method-shape break later.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

# The five entity kinds match :data:`smai_api_spec._common.EntityKind`
# used by :class:`StateChangeEvent`. Re-declared as a local Literal so
# this module's type signature does not pull the spec package types
# into the engine call sites' Pyright namespace — those sites already
# import the spec package, but keeping the Protocol surface declared
# locally documents that the same five kinds are the contract here.
EventEntityKind = Literal["proposal", "paper", "comparison_group", "entry", "run"]


@runtime_checkable
class EventChannel(Protocol):
    """Where the engine fires state-change + worker-heartbeat events.

    Implementations:

    * :class:`smai_events.NullEventChannel` — no-op default; engine
      configurations that don't care about live updates use this.
    * :class:`smai_events.InProcessEventChannel` — writes to a local
      :class:`smai_events.EventBroker`; the in-band Runtime
      (``smai dev`` / ``smai ui --with-worker``) uses this so the
      same-process API can subscribe.
    * ``PgNotifyEventChannel`` (Task 4.K3, ``plugins/smai-store-postgres``)
      — issues ``pg_notify('smai_events', payload)`` inside the active
      asyncpg transaction; cross-process consumers (``smai ui --no-worker``
      against a remote Postgres-backed worker) ``LISTEN`` for it.

    Contract notes for K3 implementers:

    * ``fire_transition`` is invoked by the engine layer **after** the
      ``MetadataStore.transition_*_state`` CAS succeeds (per `12` §6.4
      "engine-wraps" resolution). Implementations may run synchronous
      or asynchronous work; for the Postgres case the implementation
      should ensure ``pg_notify`` runs **inside the same transaction**
      as the underlying UPDATE so a ROLLBACK suppresses the wire signal
      (per `12` §6.5). The in-process implementation here has no such
      coupling — by the time the engine wrapper calls ``fire_transition``
      the row write has already committed (the SQL plugin commits per
      ``transition_*_state`` call).
    * ``fire_heartbeat`` is invoked by the worker loop's
      ``on_cycle_complete`` callback after each cycle finishes. There
      is no transactional context here; implementations fire-and-forget
      (errors are logged-and-swallowed at the call site).
    * Both methods MUST be safe to call from an async task; neither
      should block. Implementations that need to do blocking work
      (an HTTP POST, a file write) should defer to a background task.
    """

    async def fire_transition(
        self,
        *,
        kind: EventEntityKind,
        id: str,
        from_state: str,
        to_state: str,
    ) -> None:
        """Fire a ``state_change`` event for a successful transition."""
        ...

    async def fire_heartbeat(
        self,
        *,
        cycle_id: int,
        cycles_processed: int,
    ) -> None:
        """Fire a ``worker_heartbeat`` event for a completed worker cycle."""
        ...


class NullEventChannel:
    """No-op :class:`EventChannel`.

    Default for :attr:`smai_orchestrator.engine.config.EngineConfig.event_channel`
    — keeps the live-updates path entirely dormant for deployments
    that have no API consumer (``smai dev`` headless, ``smai start``
    standalone production worker without a paired ``smai ui --no-worker``).
    The fire-on-transition wrappers in the engine layer become no-op
    function calls (one ``await`` and a ``return``), so existing engine
    tests that don't care about events stay shape-clean.
    """

    async def fire_transition(
        self,
        *,
        kind: EventEntityKind,
        id: str,
        from_state: str,
        to_state: str,
    ) -> None:
        del kind, id, from_state, to_state

    async def fire_heartbeat(
        self,
        *,
        cycle_id: int,
        cycles_processed: int,
    ) -> None:
        del cycle_id, cycles_processed


__all__ = ["EventChannel", "EventEntityKind", "NullEventChannel"]
