"""Task 4.K3 engine-side transactional-fire tests.

Per ``designs/smai/12-ui-process.md`` §6.5: when an
:class:`smai_events.EventChannel` advertises
:attr:`wants_transactional_fire = True`, the engine fire site
(:func:`smai_orchestrator.engine._metadata_ops.transition_state`)
opens :meth:`MetadataStore.transaction`, runs the CAS UPDATE inside
that transaction, and invokes the channel's :meth:`fire_transition`
pre-commit with :attr:`Transaction.connection` threaded through.

These tests use an in-memory :class:`SqliteStore` plus a fake
EventChannel — they exercise the engine restructure without needing
Postgres. The plugin-side credentialed test
(``plugins/smai-store-postgres/tests/test_4_k3_pg_event_channel.py``)
covers the actual ``pg_notify`` wire path against a real database.

Acceptance shape:

* ``wants_transactional_fire = True`` channels receive a non-``None``
  ``connection`` from the engine.
* If :meth:`fire_transition` raises, the transaction rolls back —
  the row write is also lost; the engine's next poll would re-attempt
  (verified by reading the row back: state unchanged, version
  unchanged).
* ``wants_transactional_fire = False`` channels (the K2 default) keep
  the existing post-commit fire path unchanged.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from typing import ClassVar

import pytest
from smai_core.plugins import ConflictError
from smai_events import EventEntityKind
from smai_orchestrator.engine._metadata_ops import transition_state
from smai_orchestrator.entities.tracking import ComparisonGroupRecord


@dataclasses.dataclass
class _FireRecord:
    kind: EventEntityKind
    id: str
    from_state: str
    to_state: str
    connection: object | None


class _FakeTransactionalChannel:
    """Channel that captures every fire and reports the connection it
    was given — used to assert the engine threads
    :attr:`Transaction.connection` through correctly."""

    wants_transactional_fire: ClassVar[bool] = True

    def __init__(self, *, raise_on_fire: bool = False) -> None:
        self._raise_on_fire = raise_on_fire
        self.fires: list[_FireRecord] = []
        self.heartbeats: list[tuple[int, int]] = []

    async def fire_transition(
        self,
        *,
        kind: EventEntityKind,
        id: str,
        from_state: str,
        to_state: str,
        connection: object | None = None,
    ) -> None:
        self.fires.append(
            _FireRecord(
                kind=kind, id=id, from_state=from_state, to_state=to_state, connection=connection
            )
        )
        if self._raise_on_fire:
            raise RuntimeError("fire failure (test)")

    async def fire_heartbeat(self, *, cycle_id: int, cycles_processed: int) -> None:
        self.heartbeats.append((cycle_id, cycles_processed))


def _seed_record() -> ComparisonGroupRecord:
    return ComparisonGroupRecord(
        id="cg_k3_test",
        proposal_id="prop_k3",
        experiment_definition_id="exp_k3",
        state="draft",
        version=1,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


async def test_transactional_channel_receives_active_connection(sqlite_store) -> None:  # type: ignore[no-untyped-def]
    """Engine threads :attr:`Transaction.connection` through to a
    transactional channel's ``fire_transition``."""
    record = await sqlite_store.create_cg(_seed_record())
    channel = _FakeTransactionalChannel()

    new_record = await transition_state(
        sqlite_store,
        "cg",
        record.id,
        record.version,
        "implementing",
        {},
        event_channel=channel,
        from_state="draft",
    )

    assert new_record.state == "implementing"
    assert len(channel.fires) == 1
    fire = channel.fires[0]
    assert fire.kind == "comparison_group"
    assert fire.from_state == "draft"
    assert fire.to_state == "implementing"
    # Connection is opaque to the engine but MUST be supplied to a
    # ``wants_transactional_fire = True`` channel.
    assert fire.connection is not None


async def test_transactional_fire_raise_rolls_back_row_write(sqlite_store) -> None:  # type: ignore[no-untyped-def]
    """A raise from :meth:`fire_transition` rolls back the transaction
    so the CAS UPDATE is also reverted (per `12` §6.5 — the wire
    signal and the row write are atomic)."""
    record = await sqlite_store.create_cg(_seed_record())
    channel = _FakeTransactionalChannel(raise_on_fire=True)

    with pytest.raises(RuntimeError, match="fire failure"):
        await transition_state(
            sqlite_store,
            "cg",
            record.id,
            record.version,
            "implementing",
            {},
            event_channel=channel,
            from_state="draft",
        )

    # Row write must have been rolled back: state + version unchanged.
    fresh = await sqlite_store.get_cg(record.id)
    assert fresh is not None
    assert fresh.state == "draft"
    assert fresh.version == 1


async def test_transactional_channel_skips_fire_on_same_state_write(sqlite_store) -> None:  # type: ignore[no-untyped-def]
    """Same-state writes (``from_state == target_state`` — handle-only
    writes that re-target the in-progress state) are not transitions
    and produce no event, even on the K3 path. Mirrors K2 behavior."""
    record = await sqlite_store.create_cg(_seed_record())
    channel = _FakeTransactionalChannel()

    await transition_state(
        sqlite_store,
        "cg",
        record.id,
        record.version,
        "draft",  # same-state write
        {},
        event_channel=channel,
        from_state="draft",
    )

    assert channel.fires == []


async def test_transactional_channel_no_event_on_conflict_error(sqlite_store) -> None:  # type: ignore[no-untyped-def]
    """A failed CAS (``ConflictError``) from inside the transaction
    must not produce a fire — the transition didn't happen."""
    record = await sqlite_store.create_cg(_seed_record())
    channel = _FakeTransactionalChannel()

    with pytest.raises(ConflictError):
        await transition_state(
            sqlite_store,
            "cg",
            record.id,
            999,  # wrong expected_version
            "implementing",
            {},
            event_channel=channel,
            from_state="draft",
        )
    assert channel.fires == []
