"""Fair-scheduling tests for :class:`PostgresStore`.

Per ``07-plugin-interfaces.md`` §5.5 / §5.6.8 the OSS Postgres reference
plugin reports ``is_tenant_aware=False`` and returns scheduling-query
results in FIFO order by ``created_at, id``. Tenant-fair ordering
(``ROW_NUMBER() OVER (PARTITION BY tenant_bucket ORDER BY created_at)``
interleaving across tenants) is documented as a seam for the closed
``AuroraStore`` plugin to subclass / share — the OSS schema has no
``tenant_id`` column.

These tests pin the OSS contract:

1. Multi-page FIFO ordering across the proposal scheduling query (the
   one Task 3.G2 will exercise hardest — proposal pipeline is the
   primary input path per DEC-032).
2. Cursor stability under interleaved writes.

The tenant-aware behavior advertised in the brief (two synthetic
tenants with backlogged work; round-robin interleaving) is not
exercised here — see the Task 3.F1 status note for the spec
adjudication. Carry-forward: when AuroraStore lands its tenant-aware
override of ``_paginate_predicate``, its own conformance subclass adds
the round-robin assertion.

Skipped cleanly when no Postgres test database is reachable (see
``conftest.py``).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from smai_core.plugins.conformance._common import make_record
from smai_orchestrator.entities.tracking import ProposalRecord
from smai_store_postgres import PostgresStore


async def test_fifo_ordering_by_created_at_proposals(fresh_store: PostgresStore) -> None:
    """Per §5.5: single-tenant plugins return FIFO by ``created_at``.

    Seeds 5 proposal_submitted rows with explicit, monotonic
    ``created_at`` timestamps in reverse insertion order; the scheduling
    query MUST return them in ``created_at`` order (not insertion
    order), demonstrating the ordering is anchored on the column, not
    on physical row insertion.
    """
    base = datetime(2026, 1, 1, tzinfo=UTC)
    # Insert in reverse-time order so insertion order != created_at order.
    for i in reversed(range(5)):
        await fresh_store.create_proposal(
            make_record(
                ProposalRecord,
                id=f"prop_fifo_{i}",
                state="proposal_submitted",
                version=1,
                created_at=base + timedelta(seconds=i),
                updated_at=base + timedelta(seconds=i),
            )
        )

    page = await fresh_store.get_ready_for_proposal_design(limit=100, cursor=None)
    returned_ids = [p.id for p in page.items]
    expected = [f"prop_fifo_{i}" for i in range(5)]
    assert returned_ids == expected, (
        f"FIFO ordering broken: expected {expected}, got {returned_ids}"
    )


async def test_pagination_visits_each_row_exactly_once(fresh_store: PostgresStore) -> None:
    """Cursor-paginated iteration over a fixed dataset visits each row
    exactly once and terminates with ``next_cursor=None``.

    Per §5.6.10's pagination clause. Uses a 7-row dataset paginated at
    limit=2 so the iteration yields {2, 2, 2, 1} and a terminal
    ``next_cursor=None``.
    """
    base = datetime(2026, 1, 1, tzinfo=UTC)
    expected_ids: list[str] = []
    for i in range(7):
        prop_id = f"prop_page_{i:02d}"
        expected_ids.append(prop_id)
        await fresh_store.create_proposal(
            make_record(
                ProposalRecord,
                id=prop_id,
                state="proposal_submitted",
                version=1,
                created_at=base + timedelta(seconds=i),
                updated_at=base + timedelta(seconds=i),
            )
        )

    seen: list[str] = []
    cursor: str | None = None
    pages = 0
    while True:
        page = await fresh_store.get_ready_for_proposal_design(limit=2, cursor=cursor)
        seen.extend(p.id for p in page.items)
        pages += 1
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
        assert pages < 100, "pagination did not terminate"

    assert seen == expected_ids, f"pagination order incorrect: expected {expected_ids}, got {seen}"
    assert len(set(seen)) == len(seen), f"row visited more than once: {seen}"


async def test_cursor_stability_under_interleaved_writes(fresh_store: PostgresStore) -> None:
    """During pagination, inserting a new earlier-created row does not
    cause skip/duplicate of previously-returned rows.

    Per §5.6.10's "ordering stability under writes" clause: cursor-
    anchored iteration is stable for entities present at iteration
    start. The new entity may or may not appear depending on its
    position relative to the cursor — this is OK; the contract is
    no-skip/no-duplicate for entities present when iteration began.

    Uses ``created_at`` ordering: insert {0, 2, 4, 6}, page through
    {0, 2}, then insert a row at created_at=1, then resume pagination.
    Existing rows {4, 6} must still appear; the inserted row at 1 may
    or may not.
    """
    base = datetime(2026, 1, 1, tzinfo=UTC)
    initial_ids: list[str] = []
    for i in (0, 2, 4, 6):
        prop_id = f"prop_stab_{i}"
        initial_ids.append(prop_id)
        await fresh_store.create_proposal(
            make_record(
                ProposalRecord,
                id=prop_id,
                state="proposal_submitted",
                version=1,
                created_at=base + timedelta(seconds=i),
                updated_at=base + timedelta(seconds=i),
            )
        )

    page1 = await fresh_store.get_ready_for_proposal_design(limit=2, cursor=None)
    page1_ids = [p.id for p in page1.items]
    assert page1_ids == ["prop_stab_0", "prop_stab_2"]
    assert page1.next_cursor is not None

    # Insert a new row with created_at between 0 and 2 — earlier than
    # the cursor anchor, so it should NOT appear in the next page.
    await fresh_store.create_proposal(
        make_record(
            ProposalRecord,
            id="prop_stab_inserted",
            state="proposal_submitted",
            version=1,
            created_at=base + timedelta(seconds=1),
            updated_at=base + timedelta(seconds=1),
        )
    )

    page2 = await fresh_store.get_ready_for_proposal_design(limit=10, cursor=page1.next_cursor)
    page2_ids = [p.id for p in page2.items]
    # Existing rows at created_at=4, 6 must still appear on page 2.
    assert "prop_stab_4" in page2_ids
    assert "prop_stab_6" in page2_ids
    # The inserted earlier row must NOT skip the cursor.
    assert "prop_stab_inserted" not in page2_ids, (
        "cursor-anchored pagination returned a row whose created_at "
        "predates the cursor — cursor stability contract violated"
    )


async def test_capability_flag_reports_single_tenant(fresh_store: PostgresStore) -> None:
    """Per §5.5 / §5.6.8: OSS PostgresStore reports
    ``is_tenant_aware=False``.

    The closed AuroraStore is the tenant-aware sibling. This test pins
    the OSS contract — flipping this flag implies a tenant_id column
    in the schema and a window-function-derived ordering override (see
    the "Tenant-fair scheduling carry-forward" comment in
    ``_store.py``).
    """
    assert fresh_store.capabilities.is_tenant_aware is False
    assert fresh_store.capabilities.supports_transactions is True
