"""Postgres ``LISTEN smai_events`` task that feeds the in-process
:class:`smai_events.EventBroker` from the cross-process wire signal.

Task 4.K3 / ``designs/smai/12-ui-process.md`` §6.3.

Sibling of :class:`smai_store_postgres.PgNotifyEventChannel`: the
publisher runs on the worker side (a ``smai start`` process holding
its own asyncpg-via-SQLAlchemy connection pool); this listener runs on
the API side (a ``smai ui --no-worker`` process). They are coupled
only by the ``smai_events`` channel name (see
:data:`smai_store_postgres.NOTIFY_CHANNEL`) and the type-tagged JSON
envelope shape — no shared in-memory state.

How it bolts onto the API process:

* :func:`make_api_app`'s lifespan checks
  :attr:`MetadataStoreCapabilities.supports_listen_notify` (per `12`
  §6.3 capability detection). When ``True``, the lifespan extracts the
  DSN from the store (a SQLAlchemy ``AsyncEngine`` URL) and starts
  :func:`pg_listener_task` as a background ``asyncio.Task`` for the
  duration of the app's lifetime. The lifespan also constructs an
  :class:`smai_events.EventBroker` (since :class:`Runtime` doesn't
  carry one in the no-worker case — there's no in-process publisher
  to feed it) and wires it onto ``app.state.event_broker`` so the SSE
  route handler subscribes against the same broker the listener feeds.
* When the bit is ``False`` (e.g., SQLite — ``smai dev`` /
  ``smai ui --with-worker`` against the SQLite default), the listener
  task is NOT spawned and :class:`Runtime.event_broker` (fed by the
  in-band :class:`smai_events.InProcessEventChannel`) is the broker.

Failure mode: if the dedicated asyncpg connection drops after startup
(network blip, Postgres restart), :func:`pg_listener_task` logs the
exception and exits. The supervisor (systemd / kubelet / pm2 wrapping
the ``smai ui`` process) restarts the API process, which re-spawns
the listener task on next boot. v1 does not auto-reconnect inside the
task — that is a backlog item per the K3 brief's "Out of scope"
section. Surfaced in :data:`AUTORECONNECT_BACKLOG_NOTE` so a future
operator-noticed regression has somewhere to land.

Connection budget: one extra Postgres connection per
``smai ui --no-worker`` process, regardless of SSE subscriber count
(per `12` §6.3 last paragraph). Sized into the deployment's
connection pool budget; the cost is documented in
``packages/smai-cli/OPERATIONS.md``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Final, cast

import asyncpg  # pyright: ignore[reportMissingTypeStubs]
from smai_api_spec.events import StateChangeEvent, WorkerHeartbeatEvent
from smai_events import EventBroker

_log = logging.getLogger(__name__)

NOTIFY_CHANNEL: Final[str] = "smai_events"
"""Mirrors :data:`smai_store_postgres.NOTIFY_CHANNEL`. Re-declared here
(rather than imported from the plugin) because :mod:`smai_api` is the
*consumer* — it knows the wire-protocol channel name as a constant,
not as a plugin-provided value. If the channel name ever needs to
change, both ends must update in lockstep."""

AUTORECONNECT_BACKLOG_NOTE: Final[str] = (
    "v1 listener task exits on connection drop; supervisor restart "
    "re-spawns. Auto-reconnect is post-M5 polish (see Task 4.K3 brief)."
)


async def pg_listener_task(
    *,
    dsn: str,
    broker: EventBroker,
    shutdown: asyncio.Event,
    ready: asyncio.Event | None = None,
) -> None:
    """Hold a dedicated asyncpg connection on ``LISTEN smai_events``;
    feed broker.

    ``dsn``: A pure ``postgresql://...`` DSN — NOT a SQLAlchemy URL.
    Callers convert ``postgresql+asyncpg://...`` to ``postgresql://...``
    via :func:`sqlalchemy_url_to_asyncpg_dsn` before passing.

    ``broker``: The :class:`EventBroker` the SSE handler subscribes
    against. Each received NOTIFY payload is parsed into a
    :class:`StateChangeEvent` or :class:`WorkerHeartbeatEvent` and
    published.

    ``shutdown``: An asyncio event the task waits on. The lifespan
    sets this on app shutdown so the task can clean up (remove the
    listener; close the connection) before the process exits.

    ``ready`` (optional): An asyncio event the task sets once the
    ``LISTEN`` is registered and the dedicated connection is live.
    Tests use this to know when it's safe to issue a publisher-side
    ``NOTIFY``; production code can leave it as ``None``.

    Lifecycle: holds a single dedicated asyncpg connection for the
    duration of the API process. ``LISTEN`` requires a connection
    parked on the channel — the SQLAlchemy pool can't satisfy this,
    which is why the task acquires its own connection rather than
    borrowing.
    """
    try:
        # asyncpg ships no type stubs; cast the Connection out to ``Any``
        # at the boundary so the rest of the function reads cleanly.
        conn: Any = cast(Any, await asyncpg.connect(dsn))  # pyright: ignore[reportUnknownMemberType]
    except Exception:
        _log.exception("pg_listener_task: initial asyncpg.connect(%s) failed", _redact_dsn(dsn))
        return

    listener = _make_notify_listener(broker)
    try:
        await conn.add_listener(NOTIFY_CHANNEL, listener)
        _log.info(
            "pg_listener_task: listening on channel %r (dsn=%s)", NOTIFY_CHANNEL, _redact_dsn(dsn)
        )
        if ready is not None:
            ready.set()
        await shutdown.wait()
    except Exception:
        _log.exception(
            "pg_listener_task: connection error; exiting (%s)", AUTORECONNECT_BACKLOG_NOTE
        )
    finally:
        try:
            await conn.remove_listener(NOTIFY_CHANNEL, listener)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            _log.warning("pg_listener_task: remove_listener cleanup failed", exc_info=True)
        try:
            await conn.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            _log.warning("pg_listener_task: connection close cleanup failed", exc_info=True)


def _make_notify_listener(
    broker: EventBroker,
) -> Any:
    """Build the asyncpg listener callback bound to ``broker``.

    asyncpg's ``add_listener`` callback signature is
    ``(connection, pid, channel, payload)``; we ignore the connection
    and pid and dispatch on payload. The callback runs synchronously
    on asyncpg's event-loop task — broker.publish is also synchronous
    (anyio memory-stream send_nowait), so this is one fast hop with
    no extra awaits.
    """

    def _on_notify(
        _connection: Any,
        _pid: int,
        channel: str,
        payload: str,
    ) -> None:
        if channel != NOTIFY_CHANNEL:
            # Defensive — we only listen on the one channel, but
            # asyncpg's callback contract delivers the channel name
            # so the assertion is cheap.
            _log.warning("pg_listener_task: unexpected channel %r; dropping", channel)
            return
        try:
            envelope = _parse_envelope(payload)
        except (ValueError, TypeError, json.JSONDecodeError):
            _log.exception("pg_listener_task: malformed payload; dropping (payload=%r)", payload)
            return
        try:
            broker.publish(envelope)
        except Exception:  # noqa: BLE001 - broker errors must not kill the listener
            _log.exception("pg_listener_task: broker.publish raised; continuing")

    return _on_notify


def _parse_envelope(payload: str) -> StateChangeEvent | WorkerHeartbeatEvent:
    """Parse the type-tagged JSON envelope from
    :func:`smai_store_postgres._event_channel._serialize_state_change`
    / ``_serialize_heartbeat`` into the spec-shape Pydantic model.

    Wire shape::

        {"type": "state_change",
         "data": {"kind": ..., "id": ..., "from": ..., "to": ..., "ts": ...}}

        {"type": "worker_heartbeat",
         "data": {"cycle_id": ..., "cycles_processed": ..., "ts": ...}}

    Raises :class:`ValueError` on unknown ``type`` or missing fields;
    :class:`json.JSONDecodeError` on malformed JSON. The caller
    (:func:`_make_notify_listener`'s ``_on_notify``) catches both and
    drops the payload with a structured log entry.
    """
    parsed_any: Any = json.loads(payload)
    if not isinstance(parsed_any, dict):
        raise ValueError(f"envelope is not a JSON object: {parsed_any!r}")
    parsed = cast("dict[str, Any]", parsed_any)
    kind = parsed.get("type")
    data = parsed.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"envelope missing 'data' object: {parsed!r}")
    typed_data = cast("dict[str, Any]", data)
    if kind == "state_change":
        return StateChangeEvent.model_validate(_with_parsed_ts(typed_data))
    if kind == "worker_heartbeat":
        return WorkerHeartbeatEvent.model_validate(_with_parsed_ts(typed_data))
    raise ValueError(f"unknown envelope type: {kind!r}")


def _with_parsed_ts(data: dict[str, Any]) -> dict[str, Any]:
    """Coerce a string ``ts`` into :class:`datetime` for Pydantic.

    Pydantic accepts ISO-8601 strings on its own, but going through
    :func:`datetime.fromisoformat` lets us surface a clearer error
    (and centralizes the parse) than letting Pydantic raise the
    generic schema error inside the dispatch.
    """
    ts = data.get("ts")
    if isinstance(ts, str):
        data = {**data, "ts": datetime.fromisoformat(ts)}
    return data


def sqlalchemy_url_to_asyncpg_dsn(url: str) -> str:
    """Strip the ``+asyncpg`` driver suffix from a SQLAlchemy URL.

    asyncpg's :func:`asyncpg.connect` does NOT accept the
    ``postgresql+asyncpg://...`` form — only ``postgresql://...``.
    The PostgresStore plugin uses the SQLAlchemy form because
    SQLAlchemy needs the driver hint to pick asyncpg over psycopg2;
    the listener task talks asyncpg directly so we drop the hint.

    Idempotent on already-stripped URLs.
    """
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _redact_dsn(dsn: str) -> str:
    """Best-effort password redaction for log lines.

    The DSN may contain a password — we don't want it in production
    logs. asyncpg accepts the standard URL form
    ``scheme://user:password@host:port/db``; we mask ``:password@``.
    """
    try:
        scheme, rest = dsn.split("://", 1)
    except ValueError:
        return dsn
    if "@" not in rest:
        return dsn
    creds, host = rest.split("@", 1)
    if ":" in creds:
        user, _ = creds.split(":", 1)
        creds = f"{user}:***"
    return f"{scheme}://{creds}@{host}"


__all__ = [
    "AUTORECONNECT_BACKLOG_NOTE",
    "NOTIFY_CHANNEL",
    "pg_listener_task",
    "sqlalchemy_url_to_asyncpg_dsn",
]
