""":class:`PgNotifyEventChannel` — Postgres LISTEN/NOTIFY EventChannel.

Task 4.K3 / ``designs/smai/12-ui-process.md`` §6.3 + §6.5 + §12 OQ2.

Cross-process counterpart to :class:`smai_events.InProcessEventChannel`:
production deployments where the worker process (``smai start`` against
Postgres) and the API process (``smai ui --no-worker``) live on
different machines feed SSE updates over Postgres's native
``LISTEN``/``NOTIFY`` substrate.

How it bolts onto the engine:

* :attr:`PgNotifyEventChannel.wants_transactional_fire` is ``True``,
  so the engine fire site
  (:func:`smai_orchestrator.engine._metadata_ops.transition_state`)
  opens :meth:`MetadataStore.transaction`, runs the CAS UPDATE on the
  transaction's connection, then invokes
  :meth:`fire_transition` with that connection threaded through.
  ``pg_notify('smai_events', payload)`` runs against that connection,
  so Postgres buffers the payload until COMMIT and discards it on
  ROLLBACK (per `12` §6.5: the wire signal is therefore aligned with
  the persisted state change — no "transition fired then rolled back"
  lies).
* :meth:`fire_heartbeat` is invoked from the worker loop's per-cycle
  hook with no transactional context. It acquires a short-lived
  connection from the paired :class:`PostgresStore`'s pool and runs
  ``pg_notify`` autocommit-style. Heartbeats are never load-bearing
  for state-vs-event consistency; the SPA's "last cycle at" indicator
  is approximate by design.

Wire-payload shape — JSON, single ``type``-tagged envelope so the API-
side listener (in :mod:`smai_api._pg_listener`) can dispatch into the
right :mod:`smai_api_spec.events` model::

    {"type": "state_change",
     "data": {"kind": "comparison_group", "id": "cg_x",
              "from": "draft", "to": "implementing",
              "ts": "2026-05-04T17:23:11.123456+00:00"}}

    {"type": "worker_heartbeat",
     "data": {"cycle_id": 42, "cycles_processed": 100,
              "ts": "2026-05-04T17:23:14.789012+00:00"}}

The ``data`` block is the spec model's ``model_dump_json(by_alias=True)``
output (so ``from_state``/``to_state`` serialize as ``"from"``/``"to"``
matching the SSE wire format from `11` §8.1). The envelope is opaque
to the engine and the consumer.

Postgres caps NOTIFY payloads at 8000 bytes (default
``max_notify_queue_pages`` — `documentation
<https://www.postgresql.org/docs/current/sql-notify.html>`_). Our
payloads at v1 scale are ~150–250 bytes (state-change events carry
short ids + state names + a timestamp). The class checks the cap
defensively and raises :class:`PayloadTooLargeError` if exceeded so a
future schema-bloat regression surfaces clearly instead of silently
losing the notify.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar

from smai_events import EventEntityKind
from sqlalchemy import bindparam, text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection

    from smai_store_postgres._store import PostgresStore


# Channel name shared with the API-side listener
# (:mod:`smai_api._pg_listener`). The single channel suffices for v1 —
# topic-per-resource is post-M5 polish (per the K3 brief). Listeners
# fan-out: every Postgres connection ``LISTEN smai_events`` receives
# every payload regardless of how many listeners are attached.
NOTIFY_CHANNEL: str = "smai_events"

# Postgres's built-in NOTIFY payload cap is 8000 bytes (the default
# value of the server's ``NOTIFY`` length limit). Cross-checked at fire
# time so a future regression that bloats the payload surfaces here
# rather than silently failing somewhere in the asyncpg wire layer.
_PG_NOTIFY_PAYLOAD_MAX_BYTES: int = 7900
"""Conservative cap leaving headroom under the documented 8000-byte
Postgres NOTIFY payload limit."""


class PayloadTooLargeError(RuntimeError):
    """Raised when a NOTIFY payload would exceed
    :data:`_PG_NOTIFY_PAYLOAD_MAX_BYTES`.

    Defensive: v1 payloads are ~150–250 bytes. If this fires the
    schema has bloated and either the payload or the channel design
    needs revisiting (e.g., split into per-resource channels).
    """


class PgNotifyEventChannel:
    """Postgres-side :class:`smai_events.EventChannel` (Task 4.K3).

    Constructed with the :class:`PostgresStore` it pairs with (so
    :meth:`fire_heartbeat` can acquire a fresh connection from the
    same pool). The store is referenced by attribute access only —
    no inheritance, no monkey-patching — so the channel can be used
    against any plugin that exposes a SQLAlchemy ``AsyncEngine``-shaped
    ``_engine`` attribute, though Postgres is the only one that ships
    with ``pg_notify`` support.

    Lifecycle: stateless beyond the store reference. Construct once
    at engine-config time (the ``smai start`` boot path); the same
    instance is invoked from every transition fire site. Concurrent
    invocations are safe — each call uses the connection passed in
    by the engine fire site (transitions) or acquires its own short-
    lived connection (heartbeats).
    """

    wants_transactional_fire: ClassVar[bool] = True
    """``True``: the engine fire site opens
    :meth:`MetadataStore.transaction` and threads the active connection
    through ``connection`` so ``pg_notify`` is bundled with the CAS
    UPDATE's COMMIT (per `12` §6.5).
    """

    def __init__(self, store: PostgresStore) -> None:
        """``store``: the paired :class:`PostgresStore` whose pool
        :meth:`fire_heartbeat` borrows from for its short-lived
        connection. Avoids opening a second engine for the heartbeat
        path; the existing pool is sized for production load."""
        self._store = store

    async def fire_transition(
        self,
        *,
        kind: EventEntityKind,
        id: str,
        from_state: str,
        to_state: str,
        connection: object | None = None,
    ) -> None:
        """Issue ``pg_notify('smai_events', payload)`` for a successful
        state-machine transition (``12-ui-process.md`` §6.5).

        The engine fire site (:func:`smai_orchestrator.engine.
        _metadata_ops.transition_state` K3 path) hands us its active
        :class:`Transaction.connection`; we narrow it to the SQLAlchemy
        ``AsyncConnection`` we expect from the paired
        :class:`PostgresStore` and run ``SELECT pg_notify(...)`` on it.

        Postgres semantics: the payload is buffered server-side until
        the enclosing transaction COMMITs; on ROLLBACK the payload is
        silently discarded. The engine relies on this exact behavior
        — see :func:`transition_state`'s docstring for how a raise
        from this method causes the engine to retry on the next poll
        cycle.

        Raises:
            ValueError: if ``connection`` is ``None`` (the engine fire
                site MUST supply one when invoking a channel that
                advertises ``wants_transactional_fire = True``).
            PayloadTooLargeError: if the JSON payload would exceed
                :data:`_PG_NOTIFY_PAYLOAD_MAX_BYTES` (defensive; v1
                payloads are well under).
        """
        if connection is None:
            raise ValueError(
                "PgNotifyEventChannel.fire_transition requires a transactional "
                "connection (engine fire site must run inside MetadataStore.transaction "
                "when channel.wants_transactional_fire is True)"
            )
        # Narrow opaque ``object`` → ``AsyncConnection``. The Protocol
        # types this loosely so smai-core (which owns the Transaction
        # Protocol) doesn't depend on SQLAlchemy; the channel is paired
        # with the Postgres plugin so we know what we're getting.
        from sqlalchemy.ext.asyncio import AsyncConnection

        if not isinstance(connection, AsyncConnection):
            raise TypeError(
                f"PgNotifyEventChannel.fire_transition expected an AsyncConnection "
                f"from the engine's transaction context; got {type(connection).__name__}. "
                "Only the Postgres plugin can be paired with this channel."
            )

        payload = _serialize_state_change(
            kind=kind,
            id=id,
            from_state=from_state,
            to_state=to_state,
            ts=datetime.now(UTC),
        )
        await _execute_pg_notify(connection, payload)

    async def fire_heartbeat(
        self,
        *,
        cycle_id: int,
        cycles_processed: int,
    ) -> None:
        """Issue a heartbeat ``pg_notify`` (autocommit, fresh connection).

        Heartbeats fire from the worker loop's per-cycle hook; there
        is no transactional context to bundle into. We acquire a
        short-lived connection from the paired store's pool and run
        ``pg_notify`` in its own (auto-committed) transaction.
        Heartbeats are never load-bearing for state-vs-event
        consistency — they drive the SPA's "last cycle at" indicator,
        which is approximate by design.

        Failures here surface to the worker loop's heartbeat call
        site, which logs-and-swallows per the existing K2 contract
        (broken event channel cannot break the worker loop). We do
        not catch here so the call site's structured log keeps its
        single source of truth.
        """
        payload = _serialize_heartbeat(
            cycle_id=cycle_id,
            cycles_processed=cycles_processed,
            ts=datetime.now(UTC),
        )
        async with self._store._engine.begin() as conn:  # pyright: ignore[reportPrivateUsage]
            await _execute_pg_notify(conn, payload)


# ---- Wire-payload helpers ---------------------------------------------------


def _serialize_state_change(
    *,
    kind: EventEntityKind,
    id: str,
    from_state: str,
    to_state: str,
    ts: datetime,
) -> str:
    """Build the type-tagged JSON envelope for a state-change event.

    The ``data`` block matches :class:`smai_api_spec.events.StateChangeEvent`'s
    by-alias JSON output (``from_state`` → ``from``; ``to_state`` →
    ``to``) so the API-side listener can ``model_validate`` directly.
    Building it inline (rather than constructing the Pydantic model
    just to dump it) keeps fire latency in the microsecond range.
    """
    envelope = {
        "type": "state_change",
        "data": {
            "kind": kind,
            "id": id,
            "from": from_state,
            "to": to_state,
            "ts": ts.isoformat(),
        },
    }
    return _check_payload_size(json.dumps(envelope, separators=(",", ":")))


def _serialize_heartbeat(
    *,
    cycle_id: int,
    cycles_processed: int,
    ts: datetime,
) -> str:
    """Build the type-tagged JSON envelope for a worker-heartbeat event."""
    envelope = {
        "type": "worker_heartbeat",
        "data": {
            "cycle_id": cycle_id,
            "cycles_processed": cycles_processed,
            "ts": ts.isoformat(),
        },
    }
    return _check_payload_size(json.dumps(envelope, separators=(",", ":")))


def _check_payload_size(payload: str) -> str:
    """Defensive cap check; raise if exceeded."""
    encoded_len = len(payload.encode("utf-8"))
    if encoded_len > _PG_NOTIFY_PAYLOAD_MAX_BYTES:
        raise PayloadTooLargeError(
            f"NOTIFY payload is {encoded_len} bytes; max "
            f"{_PG_NOTIFY_PAYLOAD_MAX_BYTES} (Postgres caps NOTIFY at 8000 "
            f"bytes; we leave headroom)"
        )
    return payload


async def _execute_pg_notify(conn: AsyncConnection, payload: str) -> None:
    """Run ``SELECT pg_notify(:channel, :payload)`` against ``conn``.

    Uses parameterized binding so payload escaping is the driver's
    responsibility — no manual quoting, no SQL-injection risk from a
    malformed entity id (the payload is internally generated JSON
    anyway, but the parameterized form is the right discipline).
    """
    stmt = text("SELECT pg_notify(:channel, :payload)").bindparams(
        bindparam("channel", value=NOTIFY_CHANNEL),
        bindparam("payload", value=payload),
    )
    await conn.execute(stmt)


__all__ = ["NOTIFY_CHANNEL", "PayloadTooLargeError", "PgNotifyEventChannel"]
