"""Task 4.K3 PgNotifyEventChannel + asyncpg listener integration tests.

Per ``designs/smai/12-ui-process.md`` §6.3 + §6.5 + §12 OQ2: this
suite exercises the cross-process event path against a real Postgres
fixture. Uses the existing ``postgres_url`` fixture (see
``conftest.py``) which auto-skips when no Postgres is reachable —
matching the in-repo Postgres test pattern (3.F1).

Coverage:

* :class:`PgNotifyEventChannel` implements
  :class:`smai_events.EventChannel` (Protocol check).
* :attr:`PgNotifyEventChannel.wants_transactional_fire` is ``True``.
* ``capabilities.supports_listen_notify`` is ``True``.
* End-to-end: engine fires transition → ``pg_notify`` (in-tx) → a
  separate ``LISTEN`` connection receives → spec-shape payload.
* ROLLBACK semantics: the engine's transactional-fire path rolls
  back the transaction when ``fire_transition`` raises (verified at
  the engine layer in
  ``packages/smai-orchestrator/tests/engine/test_4_k3_transactional_fire.py``);
  here we exercise the lower-level ROLLBACK-suppresses-NOTIFY
  Postgres semantic by manually opening a transaction, firing, then
  rolling back.
* Multi-listener fan-out: two ``LISTEN`` connections both receive
  the same ``NOTIFY``.
* Heartbeat path: ``fire_heartbeat`` (no transactional context)
  delivers a parsed :class:`WorkerHeartbeatEvent` to the listener.

Marked credentialed; skips cleanly without ``SMAI_POSTGRES_TEST_URL``
or a reachable docker compose fixture.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import asyncpg
import pytest
import pytest_asyncio
from smai_api._pg_listener import pg_listener_task, sqlalchemy_url_to_asyncpg_dsn
from smai_api_spec.events import StateChangeEvent, WorkerHeartbeatEvent
from smai_events import EnvelopedEvent, EventBroker, EventChannel
from smai_orchestrator.engine._metadata_ops import transition_state
from smai_orchestrator.entities.tracking import ComparisonGroupRecord
from smai_store_postgres import (
    NOTIFY_CHANNEL,
    PayloadTooLargeError,
    PgNotifyEventChannel,
    PostgresStore,
)
from smai_store_postgres._event_channel import _serialize_state_change
from sqlalchemy import text

pytestmark = pytest.mark.credentialed


# ---- Fixtures ---------------------------------------------------------------


@pytest_asyncio.fixture
async def fresh_store_pg(postgres_url: str) -> AsyncIterator[PostgresStore]:
    """Like ``fresh_store`` (in conftest.py) but local — keeps the K3
    suite hermetic from changes to the shared fixture."""
    store = PostgresStore(uri=postgres_url)
    await store.drop_all()
    await store.migrate()
    try:
        yield store
    finally:
        await store.drop_all()
        await store.dispose()


@pytest_asyncio.fixture
async def asyncpg_dsn(postgres_url: str) -> str:
    """Convert the SQLAlchemy URL the conftest yields into a plain
    ``postgresql://...`` DSN suitable for :func:`asyncpg.connect`."""
    return sqlalchemy_url_to_asyncpg_dsn(postgres_url)


# ---- Capability + Protocol shape -------------------------------------------


def test_pg_notify_channel_satisfies_event_channel_protocol() -> None:
    """Avoid constructing a real store — Protocol check is structural."""
    instance = PgNotifyEventChannel.__new__(PgNotifyEventChannel)
    assert isinstance(instance, EventChannel)


def test_pg_notify_channel_advertises_transactional_fire() -> None:
    assert PgNotifyEventChannel.wants_transactional_fire is True


async def test_postgres_store_advertises_listen_notify_capability(
    fresh_store_pg: PostgresStore,
) -> None:
    assert fresh_store_pg.capabilities.supports_listen_notify is True


# ---- Payload size guard -----------------------------------------------------


def test_serialize_raises_when_payload_exceeds_pg_notify_cap() -> None:
    """Defensive: v1 payloads are ~150–250 bytes, but a future schema
    bloat must surface here rather than silently failing in asyncpg."""
    long_id = "x" * 9000
    with pytest.raises(PayloadTooLargeError):
        _serialize_state_change(
            kind="comparison_group",
            id=long_id,
            from_state="draft",
            to_state="implementing",
            ts=datetime(2026, 1, 1, tzinfo=UTC),
        )


# ---- End-to-end: engine fires → LISTEN receives ----------------------------


async def test_engine_transition_delivers_notify_to_listener(
    fresh_store_pg: PostgresStore,
    asyncpg_dsn: str,
) -> None:
    """The full K3 happy path: engine restructure opens
    :meth:`MetadataStore.transaction`, runs the CAS UPDATE, fires
    ``pg_notify`` against the same connection; a separately-connected
    ``LISTEN smai_events`` callback receives the spec-shape payload."""
    record = await fresh_store_pg.create_cg(
        ComparisonGroupRecord(
            id="cg_k3_e2e",
            proposal_id="prop_k3_e2e",
            experiment_definition_id="exp_k3_e2e",
            state="draft",
            version=1,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )

    broker = EventBroker()
    shutdown = asyncio.Event()
    ready = asyncio.Event()
    listener = asyncio.create_task(
        pg_listener_task(dsn=asyncpg_dsn, broker=broker, shutdown=shutdown, ready=ready)
    )
    try:
        await asyncio.wait_for(ready.wait(), timeout=5.0)
        channel = PgNotifyEventChannel(fresh_store_pg)

        await transition_state(
            fresh_store_pg,
            "cg",
            record.id,
            record.version,
            "implementing",
            {},
            event_channel=channel,
            from_state="draft",
        )

        # asyncpg dispatches NOTIFY to listeners on the connection's
        # event loop; give it a moment to land.
        await _wait_for_broker(broker, expected=1, timeout=5.0)

        items = broker.replay_since(0)
        assert len(items) == 1
        item = items[0]
        assert isinstance(item, EnvelopedEvent)
        assert isinstance(item.event, StateChangeEvent)
        assert item.event.kind == "comparison_group"
        assert item.event.id == "cg_k3_e2e"
        assert item.event.from_state == "draft"
        assert item.event.to_state == "implementing"
    finally:
        shutdown.set()
        await asyncio.wait_for(listener, timeout=5.0)


async def test_rollback_suppresses_notify(
    fresh_store_pg: PostgresStore,
    asyncpg_dsn: str,
) -> None:
    """Postgres NOTIFY semantics: payloads are buffered until COMMIT
    and discarded on ROLLBACK. We open a transaction, fire pg_notify,
    then rollback; the listener must receive nothing."""
    broker = EventBroker()
    shutdown = asyncio.Event()
    ready = asyncio.Event()
    listener = asyncio.create_task(
        pg_listener_task(dsn=asyncpg_dsn, broker=broker, shutdown=shutdown, ready=ready)
    )
    try:
        await asyncio.wait_for(ready.wait(), timeout=5.0)
        channel = PgNotifyEventChannel(fresh_store_pg)

        async with fresh_store_pg._engine.begin() as conn:
            await channel.fire_transition(
                kind="comparison_group",
                id="cg_rollback",
                from_state="draft",
                to_state="implementing",
                connection=conn,
            )
            # Force rollback by raising out of the transaction.
            raise _AbortFixture()
    except _AbortFixture:
        pass

    # Give asyncpg time to deliver if it were going to (it shouldn't).
    await asyncio.sleep(0.5)
    assert broker.replay_since(0) == []

    shutdown.set()
    await asyncio.wait_for(listener, timeout=5.0)


async def test_multi_listener_fan_out(
    fresh_store_pg: PostgresStore,
    asyncpg_dsn: str,
) -> None:
    """Two ``LISTEN`` connections both receive the same ``NOTIFY``
    (Postgres native fan-out). Confirms the deployment pattern of
    multiple ``smai ui --no-worker`` processes attaching to one
    Postgres-backed worker (`12` §5.4)."""
    broker_a = EventBroker()
    broker_b = EventBroker()
    shutdown = asyncio.Event()
    ready_a = asyncio.Event()
    ready_b = asyncio.Event()
    listener_a = asyncio.create_task(
        pg_listener_task(dsn=asyncpg_dsn, broker=broker_a, shutdown=shutdown, ready=ready_a)
    )
    listener_b = asyncio.create_task(
        pg_listener_task(dsn=asyncpg_dsn, broker=broker_b, shutdown=shutdown, ready=ready_b)
    )
    try:
        await asyncio.wait_for(asyncio.gather(ready_a.wait(), ready_b.wait()), timeout=5.0)
        channel = PgNotifyEventChannel(fresh_store_pg)

        async with fresh_store_pg._engine.begin() as conn:
            await channel.fire_transition(
                kind="proposal",
                id="prop_fanout",
                from_state="proposal_submitted",
                to_state="designing",
                connection=conn,
            )

        await _wait_for_broker(broker_a, expected=1, timeout=5.0)
        await _wait_for_broker(broker_b, expected=1, timeout=5.0)
        assert len(broker_a.replay_since(0)) == 1
        assert len(broker_b.replay_since(0)) == 1
    finally:
        shutdown.set()
        await asyncio.gather(
            asyncio.wait_for(listener_a, timeout=5.0),
            asyncio.wait_for(listener_b, timeout=5.0),
        )


async def test_heartbeat_delivers_worker_heartbeat_event(
    fresh_store_pg: PostgresStore,
    asyncpg_dsn: str,
) -> None:
    """Heartbeats fire from the worker loop with no transactional
    context; the channel uses its own short-lived connection."""
    broker = EventBroker()
    shutdown = asyncio.Event()
    ready = asyncio.Event()
    listener = asyncio.create_task(
        pg_listener_task(dsn=asyncpg_dsn, broker=broker, shutdown=shutdown, ready=ready)
    )
    try:
        await asyncio.wait_for(ready.wait(), timeout=5.0)
        channel = PgNotifyEventChannel(fresh_store_pg)
        await channel.fire_heartbeat(cycle_id=99, cycles_processed=10000)

        await _wait_for_broker(broker, expected=1, timeout=5.0)
        items = broker.replay_since(0)
        assert len(items) == 1
        item = items[0]
        assert isinstance(item, EnvelopedEvent)
        assert isinstance(item.event, WorkerHeartbeatEvent)
        assert item.event.cycle_id == 99
        assert item.event.cycles_processed == 10000
    finally:
        shutdown.set()
        await asyncio.wait_for(listener, timeout=5.0)


async def test_fire_transition_rejects_missing_connection(
    fresh_store_pg: PostgresStore,
) -> None:
    """The channel raises ``ValueError`` when called with
    ``connection=None`` — a misuse signal: the engine fire site MUST
    pass an active transaction connection."""
    channel = PgNotifyEventChannel(fresh_store_pg)
    with pytest.raises(ValueError, match="requires a transactional connection"):
        await channel.fire_transition(
            kind="comparison_group",
            id="cg_misuse",
            from_state="draft",
            to_state="implementing",
            connection=None,
        )


async def test_fire_transition_payload_shape_via_raw_listen(
    fresh_store_pg: PostgresStore,
    asyncpg_dsn: str,
) -> None:
    """Sanity-check the on-the-wire JSON envelope shape via raw
    asyncpg (not through the broker) so a future envelope-format
    drift would surface here independent of the parser."""
    received: list[str] = []
    raw_conn = await asyncpg.connect(asyncpg_dsn)

    def _on_notify(_conn, _pid, channel, payload):  # type: ignore[no-untyped-def]
        if channel == NOTIFY_CHANNEL:
            received.append(payload)

    await raw_conn.add_listener(NOTIFY_CHANNEL, _on_notify)
    try:
        channel = PgNotifyEventChannel(fresh_store_pg)
        async with fresh_store_pg._engine.begin() as conn:
            await channel.fire_transition(
                kind="entry",
                id="entry_wire",
                from_state="pending",
                to_state="implementing",
                connection=conn,
            )
        # Wait for asyncpg to dispatch the notify.
        for _ in range(50):
            if received:
                break
            await asyncio.sleep(0.1)
        assert received, "no payload received within 5s"
        envelope = json.loads(received[0])
        assert envelope["type"] == "state_change"
        # Wire form uses "from"/"to" aliases per `11` §8.1.
        data = envelope["data"]
        assert data["kind"] == "entry"
        assert data["id"] == "entry_wire"
        assert data["from"] == "pending"
        assert data["to"] == "implementing"
        assert "ts" in data
    finally:
        await raw_conn.remove_listener(NOTIFY_CHANNEL, _on_notify)
        await raw_conn.close()


# ---- Helpers ---------------------------------------------------------------


class _AbortFixture(Exception):
    """Sentinel used to force a transaction rollback in
    ``test_rollback_suppresses_notify``."""


async def _wait_for_broker(broker: EventBroker, *, expected: int, timeout: float) -> None:
    """Poll the broker's ring buffer until ``expected`` items land or
    we time out. asyncpg's NOTIFY dispatch happens on the listener's
    event loop, so a tight ``await asyncio.sleep(0)`` hand-off is
    usually enough — but we give a generous timeout for slow CI."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if len(broker.replay_since(0)) >= expected:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(
        f"broker did not receive {expected} item(s) within {timeout}s "
        f"(saw {len(broker.replay_since(0))})"
    )


# Smoke check that ``text`` is importable from sqlalchemy at the top
# of this module (so a failed import surfaces here, not deep in the
# event-channel module on first transition fire).
def test_sqlalchemy_text_import() -> None:
    assert callable(text)
