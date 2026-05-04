"""Task 4.K3 unit tests for the asyncpg listener task internals.

These tests don't require Postgres — they exercise the payload parser
and the broker-feeding callback directly. The end-to-end wire path
(worker fires → ``pg_notify`` → ``LISTEN`` → broker → SSE) is covered
by the credentialed test in
``plugins/smai-store-postgres/tests/test_4_k3_pg_event_channel.py``.
"""

from __future__ import annotations

import json

import pytest
from smai_api._pg_listener import (
    NOTIFY_CHANNEL,
    _make_notify_listener,
    _parse_envelope,
    sqlalchemy_url_to_asyncpg_dsn,
)
from smai_api_spec.events import StateChangeEvent, WorkerHeartbeatEvent
from smai_events import EnvelopedEvent, EventBroker

# ---- _parse_envelope -------------------------------------------------------


def test_parse_envelope_state_change() -> None:
    payload = json.dumps(
        {
            "type": "state_change",
            "data": {
                "kind": "comparison_group",
                "id": "cg_x",
                "from": "draft",
                "to": "implementing",
                "ts": "2026-05-04T17:23:11.123456+00:00",
            },
        }
    )
    event = _parse_envelope(payload)
    assert isinstance(event, StateChangeEvent)
    assert event.kind == "comparison_group"
    assert event.from_state == "draft"
    assert event.to_state == "implementing"


def test_parse_envelope_worker_heartbeat() -> None:
    payload = json.dumps(
        {
            "type": "worker_heartbeat",
            "data": {
                "cycle_id": 42,
                "cycles_processed": 100,
                "ts": "2026-05-04T17:23:14.789012+00:00",
            },
        }
    )
    event = _parse_envelope(payload)
    assert isinstance(event, WorkerHeartbeatEvent)
    assert event.cycle_id == 42
    assert event.cycles_processed == 100


def test_parse_envelope_unknown_type_raises() -> None:
    payload = json.dumps({"type": "novel_type", "data": {}})
    with pytest.raises(ValueError, match="unknown envelope type"):
        _parse_envelope(payload)


def test_parse_envelope_missing_data_raises() -> None:
    payload = json.dumps({"type": "state_change"})
    with pytest.raises(ValueError, match="missing 'data' object"):
        _parse_envelope(payload)


def test_parse_envelope_non_object_raises() -> None:
    payload = json.dumps([1, 2, 3])
    with pytest.raises(ValueError, match="not a JSON object"):
        _parse_envelope(payload)


# ---- _make_notify_listener -------------------------------------------------


def test_listener_publishes_state_change_to_broker() -> None:
    broker = EventBroker()
    listener = _make_notify_listener(broker)
    payload = json.dumps(
        {
            "type": "state_change",
            "data": {
                "kind": "run",
                "id": "run_y",
                "from": "running",
                "to": "succeeded",
                "ts": "2026-05-04T17:23:11+00:00",
            },
        }
    )
    listener(None, 0, NOTIFY_CHANNEL, payload)
    items = broker.replay_since(0)
    assert len(items) == 1
    item = items[0]
    assert isinstance(item, EnvelopedEvent)
    assert isinstance(item.event, StateChangeEvent)
    assert item.event.id == "run_y"


def test_listener_drops_unexpected_channel_silently() -> None:
    """asyncpg's callback signature delivers the channel name; we
    defensively drop notifies on any other channel."""
    broker = EventBroker()
    listener = _make_notify_listener(broker)
    listener(None, 0, "wrong_channel", "{}")
    assert broker.replay_since(0) == []


def test_listener_drops_malformed_payload() -> None:
    broker = EventBroker()
    listener = _make_notify_listener(broker)
    listener(None, 0, NOTIFY_CHANNEL, "not-json-{")
    listener(None, 0, NOTIFY_CHANNEL, json.dumps({"type": "unknown", "data": {}}))
    assert broker.replay_since(0) == []


# ---- sqlalchemy_url_to_asyncpg_dsn -----------------------------------------


def test_sqlalchemy_url_strip_asyncpg_driver() -> None:
    assert (
        sqlalchemy_url_to_asyncpg_dsn("postgresql+asyncpg://u:p@host:5432/db")
        == "postgresql://u:p@host:5432/db"
    )


def test_sqlalchemy_url_idempotent_on_bare_postgres() -> None:
    assert (
        sqlalchemy_url_to_asyncpg_dsn("postgresql://u:p@host:5432/db")
        == "postgresql://u:p@host:5432/db"
    )
