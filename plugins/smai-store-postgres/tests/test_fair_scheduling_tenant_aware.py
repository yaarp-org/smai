"""Tenant-aware fair-scheduling tests for :class:`PostgresStore` (Task 3.G2).

Per ``07-plugin-interfaces.md`` §5.5 / §5.6.8: the OSS PostgresStore opt-in
``tenant_aware=True`` mode flips ``capabilities.is_tenant_aware`` and
routes scheduling-query pagination through a window-function ordering
shape that interleaves (``fair_scheduling="round_robin"``) or
weight-modulates (``fair_scheduling="weighted"``) candidates across
tenants. Default OSS deployments leave ``tenant_aware=False`` and pay
zero cost — the existing ``test_fair_scheduling.py`` covers that path.

These tests pin the opt-in contract:

1. **Capability flip.** ``tenant_aware=True`` reports
   ``capabilities.is_tenant_aware=True``; ``tenant_aware=False`` reports
   ``False`` (regression guard against the constructor-flag wiring).
2. **``round_robin`` interleaved ordering.** Two synthetic tenants with
   3 backlogged ``proposal_submitted`` rows each return interleaved as
   ``[a1, b1, a2, b2, a3, b3]`` (not FIFO ``[a1, a2, a3, b1, b2, b3]``).
3. **``weighted`` ordering.** Same fixture with weights
   ``{tenant_a: 2.0, tenant_b: 1.0}`` returns
   ``[a1, a2, b1, a3, b2, b3]`` per the rank/weight math documented in
   ``PostgresStore._tenant_fair_rank_expr``.
4. **``off`` mode falls back to FIFO** even when ``tenant_aware=True``
   is set — the schema flag is decoupled from the policy switch.
5. **Cursor stability under interleaved writes** (mirror
   ``test_fair_scheduling.py::test_cursor_stability_under_interleaved_writes``)
   in tenant_aware=True / fair_scheduling=off mode — the FIFO cursor
   contract holds with the tenant-aware schema present.

Each test constructs its own :class:`PostgresStore` with the right
constructor kwargs (rather than reusing the ``fresh_store`` fixture
which constructs without ``tenant_aware=True``). The store's
``migrate()`` runs the ``tenant_aware`` Alembic branch at construction
time so the schema carries ``tenant_id`` columns + indexes.

Skipped cleanly when no Postgres test database is reachable (see
``conftest.py``).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from typing import Literal

import pytest
from smai_core.plugins.conformance._common import make_record
from smai_orchestrator.entities.tracking import ProposalRecord
from smai_store_postgres import PostgresStore
from sqlalchemy import text


@pytest.fixture
async def tenant_aware_store(postgres_url: str) -> AsyncGenerator[PostgresStore, None]:
    """:class:`PostgresStore` with ``tenant_aware=True`` and FIFO mode.

    The default policy is ``fair_scheduling="off"`` — individual tests
    that exercise round_robin / weighted reconstruct the store with
    those overrides since the policy is constructor-baked and can't be
    flipped post-init without a re-migrate.
    """
    store = PostgresStore(uri=postgres_url, tenant_aware=True)
    await store.drop_all()
    await store.migrate()
    try:
        yield store
    finally:
        await store.drop_all()
        await store.dispose()


async def _seed_tenant_proposals(
    store: PostgresStore,
    *,
    tenant: str,
    id_prefix: str,
    base: datetime,
    count: int,
    second_offsets: tuple[int, ...] | None = None,
) -> list[str]:
    """Create ``count`` ``proposal_submitted`` rows for ``tenant``.

    Returns the inserted ids in created_at order. Two-pass shape:
    ``store.create_proposal`` to get the canonical OSS row in (the
    Pydantic :class:`ProposalRecord` carries no ``tenant_id`` field by
    design — it's a tenant_aware-mode-only SQL column), then a raw
    ``UPDATE proposals SET tenant_id = :t`` to stamp the tenancy. The
    column is added by the Task 3.G2 / 0002 Alembic revision; the
    shared ``proposals_table`` SQLAlchemy declaration does not include
    it (lifting tenant_id into the shared schema would defeat the
    "OSS pays zero cost" contract).
    """
    ids: list[str] = []
    offsets = second_offsets if second_offsets is not None else tuple(range(count))
    assert len(offsets) == count
    for i, off in enumerate(offsets):
        prop_id = f"{id_prefix}_{i}"
        ids.append(prop_id)
        record = make_record(
            ProposalRecord,
            id=prop_id,
            state="proposal_submitted",
            version=1,
            created_at=base + timedelta(seconds=off),
            updated_at=base + timedelta(seconds=off),
        )
        await store.create_proposal(record)
        async with store._engine.begin() as conn:  # noqa: SLF001 — test helper
            await conn.execute(
                text("UPDATE proposals SET tenant_id = :t WHERE id = :i").bindparams(
                    t=tenant, i=prop_id
                )
            )
    return ids


# === Capability pin ==========================================================


async def test_capability_flag_default_off(postgres_url: str) -> None:
    """Default ``tenant_aware=False`` → ``is_tenant_aware=False``.

    The pre-3.G2 capability surface — single-tenant deployments report
    no tenant awareness; their scheduling queries are FIFO regardless
    of ``fair_scheduling``.
    """
    store = PostgresStore(uri=postgres_url)
    try:
        assert store.capabilities.is_tenant_aware is False
        assert store.capabilities.supports_transactions is True
    finally:
        await store.dispose()


async def test_capability_flag_tenant_aware_on(postgres_url: str) -> None:
    """``tenant_aware=True`` → ``is_tenant_aware=True``.

    The opt-in shape — flipping the constructor kwarg flips the
    capability flag the conformance suite (`07` §5.8) keys on.
    """
    store = PostgresStore(uri=postgres_url, tenant_aware=True)
    try:
        assert store.capabilities.is_tenant_aware is True
        assert store.capabilities.supports_transactions is True
    finally:
        await store.dispose()


async def test_tenant_aware_migration_creates_tenant_id_column(
    tenant_aware_store: PostgresStore,
) -> None:
    """The opt-in migration adds a nullable ``tenant_id VARCHAR(64)``
    column to every pipeline-tracking table per Task 3.G2's 0002
    revision.
    """
    expected_tables = ("cgs", "entries", "runs", "proposals", "papers")
    async with tenant_aware_store._engine.connect() as conn:  # noqa: SLF001 — test
        for table_name in expected_tables:
            result = await conn.execute(
                text(
                    "SELECT data_type, is_nullable FROM information_schema.columns "
                    "WHERE table_name = :t AND column_name = 'tenant_id'"
                ).bindparams(t=table_name)
            )
            row = result.first()
            assert row is not None, f"tenant_id column missing on {table_name!r}"
            data_type, is_nullable = row
            assert data_type == "character varying", (
                f"tenant_id on {table_name!r} has unexpected type {data_type!r}"
            )
            assert is_nullable == "YES", f"tenant_id on {table_name!r} is unexpectedly NOT NULL"


# === Round-robin ordering ====================================================


async def test_round_robin_interleaves_tenants(postgres_url: str) -> None:
    """Two tenants × 3 backlogged proposals returns interleaved per
    `07` §5.5 round-robin contract.

    Without tenant-fair ordering the result would be FIFO ``[a0, a1, a2,
    b0, b1, b2]`` (a's earlier created_at rows come first because we
    insert them first). Tenant-fair round-robin produces ``[a0, b0, a1,
    b1, a2, b2]`` — the ``ROW_NUMBER() OVER (PARTITION BY tenant_id
    ORDER BY created_at, id)`` ranks rank-1 of every tenant before
    rank-2 of any tenant.
    """
    store = PostgresStore(
        uri=postgres_url,
        tenant_aware=True,
        fair_scheduling="round_robin",
    )
    try:
        await store.drop_all()
        await store.migrate()
        base = datetime(2026, 1, 1, tzinfo=UTC)
        # Stamp tenant_a rows at seconds 0,1,2 then tenant_b at 10,11,12
        # so FIFO would visibly NOT match tenant-fair (FIFO drains all
        # of tenant_a first).
        a_ids = await _seed_tenant_proposals(
            store,
            tenant="tenant_a",
            id_prefix="prop_a",
            base=base,
            count=3,
            second_offsets=(0, 1, 2),
        )
        b_ids = await _seed_tenant_proposals(
            store,
            tenant="tenant_b",
            id_prefix="prop_b",
            base=base,
            count=3,
            second_offsets=(10, 11, 12),
        )

        page = await store.get_ready_for_proposal_design(limit=100, cursor=None)
        returned_ids = [p.id for p in page.items]
        # Round-robin: rank-1 of each tenant interleaved by tenant_id ASC.
        expected = [a_ids[0], b_ids[0], a_ids[1], b_ids[1], a_ids[2], b_ids[2]]
        assert returned_ids == expected, (
            f"round_robin ordering broken: expected {expected}, got {returned_ids}"
        )
    finally:
        await store.drop_all()
        await store.dispose()


# === Weighted ordering =======================================================


async def test_weighted_modulates_rank_by_tenant_weight(postgres_url: str) -> None:
    """``fair_scheduling="weighted"`` with weights ``{a: 2.0, b: 1.0}``
    schedules tenant_a's rows twice as densely as tenant_b's.

    Per :meth:`PostgresStore._tenant_fair_rank_expr`'s docstring the
    effective ranks are::

        a0=0.5  a1=1.0  a2=1.5
        b0=1.0  b1=2.0  b2=3.0

    Sorted by (effective_rank ASC, tenant_id ASC, created_at ASC, id
    ASC) the order is::

        a0 (0.5), a1 (1.0, tenant_a < tenant_b breaks the tie), b0
        (1.0), a2 (1.5), b1 (2.0), b2 (3.0)
    """
    store = PostgresStore(
        uri=postgres_url,
        tenant_aware=True,
        fair_scheduling="weighted",
        fair_scheduling_weights={"tenant_a": 2.0, "tenant_b": 1.0},
    )
    try:
        await store.drop_all()
        await store.migrate()
        base = datetime(2026, 1, 1, tzinfo=UTC)
        a_ids = await _seed_tenant_proposals(
            store,
            tenant="tenant_a",
            id_prefix="prop_a",
            base=base,
            count=3,
            second_offsets=(0, 1, 2),
        )
        b_ids = await _seed_tenant_proposals(
            store,
            tenant="tenant_b",
            id_prefix="prop_b",
            base=base,
            count=3,
            second_offsets=(10, 11, 12),
        )

        page = await store.get_ready_for_proposal_design(limit=100, cursor=None)
        returned_ids = [p.id for p in page.items]
        expected = [a_ids[0], a_ids[1], b_ids[0], a_ids[2], b_ids[1], b_ids[2]]
        assert returned_ids == expected, (
            f"weighted ordering broken: expected {expected}, got {returned_ids}"
        )
    finally:
        await store.drop_all()
        await store.dispose()


# === Off mode falls back to FIFO ============================================


async def test_off_mode_with_tenant_aware_falls_back_to_fifo(postgres_url: str) -> None:
    """``fair_scheduling="off"`` with ``tenant_aware=True`` returns FIFO.

    The tenant_aware schema flag is decoupled from the policy switch:
    operators may run with the tenant_id column present but choose FIFO
    ordering — the plugin silently takes the FIFO path. This is the
    "single-tenant deployments pay zero cost" contract from the brief
    inverted: tenant-aware-schema deployments without the fair-
    scheduling policy still pay zero CTE cost.
    """
    store = PostgresStore(
        uri=postgres_url,
        tenant_aware=True,
        fair_scheduling="off",
    )
    try:
        await store.drop_all()
        await store.migrate()
        base = datetime(2026, 1, 1, tzinfo=UTC)
        a_ids = await _seed_tenant_proposals(
            store,
            tenant="tenant_a",
            id_prefix="prop_a",
            base=base,
            count=3,
            second_offsets=(0, 1, 2),
        )
        b_ids = await _seed_tenant_proposals(
            store,
            tenant="tenant_b",
            id_prefix="prop_b",
            base=base,
            count=3,
            second_offsets=(10, 11, 12),
        )

        page = await store.get_ready_for_proposal_design(limit=100, cursor=None)
        returned_ids = [p.id for p in page.items]
        # FIFO: tenant_a rows (created_at 0,1,2) come before tenant_b
        # rows (created_at 10,11,12).
        expected = [*a_ids, *b_ids]
        assert returned_ids == expected, (
            f"off-mode FIFO ordering broken: expected {expected}, got {returned_ids}"
        )
    finally:
        await store.drop_all()
        await store.dispose()


# === Cursor stability ========================================================


async def test_cursor_stability_under_interleaved_writes_tenant_aware(
    postgres_url: str,
) -> None:
    """Cursor-anchored pagination is stable under interleaved earlier
    inserts when ``tenant_aware=True`` / ``fair_scheduling="off"``.

    Mirrors ``test_fair_scheduling.py::test_cursor_stability_under_interleaved_writes``
    against the tenant-aware schema (the FIFO cursor contract per `07`
    §5.6.10 holds with ``tenant_id`` column present). The fair-mode
    cursor contract is exercised single-page in the round_robin /
    weighted tests above; cross-page tenant-fair cursor stability is
    inherently best-effort under steady-state writes (a row inserted
    mid-iteration may shift effective ranks) and is not part of v1's
    contract.
    """
    store = PostgresStore(
        uri=postgres_url,
        tenant_aware=True,
        fair_scheduling="off",
    )
    try:
        await store.drop_all()
        await store.migrate()
        base = datetime(2026, 1, 1, tzinfo=UTC)
        for i in (0, 2, 4, 6):
            await _seed_tenant_proposals(
                store,
                tenant="tenant_a",
                id_prefix=f"prop_stab_{i}",
                base=base,
                count=1,
                second_offsets=(i,),
            )

        page1 = await store.get_ready_for_proposal_design(limit=2, cursor=None)
        page1_ids = [p.id for p in page1.items]
        assert page1_ids == ["prop_stab_0_0", "prop_stab_2_0"]
        assert page1.next_cursor is not None

        # Insert a new row at created_at=1 — earlier than the cursor
        # anchor (created_at=2) — so it should NOT appear on page 2.
        await _seed_tenant_proposals(
            store,
            tenant="tenant_a",
            id_prefix="prop_stab_inserted",
            base=base,
            count=1,
            second_offsets=(1,),
        )

        page2 = await store.get_ready_for_proposal_design(limit=10, cursor=page1.next_cursor)
        page2_ids = [p.id for p in page2.items]
        assert "prop_stab_4_0" in page2_ids
        assert "prop_stab_6_0" in page2_ids
        assert "prop_stab_inserted_0" not in page2_ids, (
            "cursor-anchored pagination returned a row whose created_at "
            "predates the cursor — cursor stability contract violated"
        )
    finally:
        await store.drop_all()
        await store.dispose()


# === Type annotation drag-along =============================================
#
# Reference the Literal type so pyright doesn't flag the import.
_POLICY_LITERAL: Literal["off", "round_robin", "weighted"] = "off"
