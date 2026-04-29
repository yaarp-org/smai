"""Multi-worker lease-contention tests for :class:`PostgresStore`.

Verifies the advisory-lock fast path on :meth:`PostgresStore.acquire_lease`
under simulated concurrent workers (per ``07-plugin-interfaces.md``
§5.6.7). Carry-forward to Task 3.G1 (multi-worker leasing engine
wrapper): these tests exercise the plugin's lease primitives; 3.G1
exercises the engine's dispatch wrapper that consumes them.

Skipped cleanly when no Postgres test database is reachable (see
``conftest.py``).
"""

from __future__ import annotations

import asyncio

import pytest
from smai_core.plugins.conformance._common import make_record
from smai_orchestrator.entities.tracking import ComparisonGroupRecord
from smai_store_postgres import PostgresStore
from smai_store_postgres._store import _advisory_lock_key
from sqlalchemy import text


async def test_two_workers_race_acquire_only_one_wins(fresh_store: PostgresStore) -> None:
    """Two simulated workers race to ``acquire_lease`` on the same CG.

    Exactly one acquires; the other gets ``None``. Per §5.6.7 / DEC-035 #2:
    the loser does NOT raise — lease-acquire failure is normal in the poll
    loop.

    Mechanically: each ``acquire_lease`` opens its own engine.begin()
    transaction. The advisory-lock fast path serializes the two contenders;
    the row UPDATE inside the winning transaction sets the nonce, so the
    losing transaction's UPDATE matches 0 rows (or, more commonly, never
    runs because ``pg_try_advisory_xact_lock`` returned False).
    """
    cg = make_record(ComparisonGroupRecord, id="cg_race", state="draft", version=1)
    await fresh_store.create_cg(cg)

    results = await asyncio.gather(
        fresh_store.acquire_lease("cg", "cg_race", 60, "worker-a"),
        fresh_store.acquire_lease("cg", "cg_race", 60, "worker-b"),
    )
    winners = [r for r in results if r is not None]
    losers = [r for r in results if r is None]
    assert len(winners) == 1, f"expected exactly one winner, got {winners}"
    assert len(losers) == 1, f"expected exactly one loser, got {losers}"


async def test_advisory_lock_fast_path_is_engaged(fresh_store: PostgresStore) -> None:
    """Verify the advisory-lock fast path is actually engaged.

    Holds an advisory lock for ``(cg, cg_lock_test)`` outside the store,
    then issues ``acquire_lease`` from the store. With the fast path
    active, the store's acquire returns ``None`` (the advisory lock
    already taken); without it, the row UPDATE alone would succeed.

    This is the pin: it documents that ``use_advisory_locks=True`` is
    not a no-op — the Protocol-level ``CAS-via-nonce`` semantics still
    hold, but the advisory-lock layer does serialize contenders before
    the UPDATE.
    """
    cg = make_record(ComparisonGroupRecord, id="cg_lock_test", state="draft", version=1)
    await fresh_store.create_cg(cg)
    lock_key = _advisory_lock_key("cg", "cg_lock_test")

    engine = fresh_store._engine  # type: ignore[reportPrivateUsage]
    async with engine.connect() as conn:
        # Hold the advisory lock at session scope so we can query
        # pg_locks in the same backend without releasing the lock.
        await conn.execute(text("SELECT pg_advisory_lock(:k)").bindparams(k=lock_key))
        try:
            # The store's acquire should bounce off the advisory lock.
            token = await fresh_store.acquire_lease("cg", "cg_lock_test", 60, "worker-x")
            assert token is None, (
                "acquire_lease succeeded despite a held advisory lock — "
                "the use_advisory_locks fast path is not engaged"
            )
            # Verify the lock is visible in pg_locks (sanity check).
            locks = await conn.execute(
                text(
                    "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND objid = :objid"
                ).bindparams(objid=lock_key & 0xFFFFFFFF)
            )
            count = locks.scalar_one()
            assert count >= 1, "advisory lock not visible in pg_locks"
        finally:
            await conn.execute(text("SELECT pg_advisory_unlock(:k)").bindparams(k=lock_key))


async def test_fallback_path_without_advisory_locks(postgres_url: str) -> None:
    """When ``use_advisory_locks=False``, the row UPDATE alone suffices.

    Same single-CG contention scenario, but with the fast path
    disabled — verifies the fallback path works (mirrors the SQLite
    plugin's behavior). Useful as a regression pin against the
    advisory-lock seam being the *only* correctness mechanism — the
    row-CAS-via-nonce predicate must hold on its own per §5.6.7's
    "the contract surface is the same regardless".
    """
    store = PostgresStore(uri=postgres_url, use_advisory_locks=False)
    await store.drop_all()
    await store.migrate()
    try:
        cg = make_record(ComparisonGroupRecord, id="cg_no_advlock", state="draft", version=1)
        await store.create_cg(cg)

        results = await asyncio.gather(
            store.acquire_lease("cg", "cg_no_advlock", 60, "worker-a"),
            store.acquire_lease("cg", "cg_no_advlock", 60, "worker-b"),
        )
        winners = [r for r in results if r is not None]
        losers = [r for r in results if r is None]
        assert len(winners) == 1
        assert len(losers) == 1
    finally:
        await store.drop_all()
        await store.dispose()


@pytest.mark.parametrize("worker_count", [4, 8])
async def test_n_workers_race_only_one_wins(fresh_store: PostgresStore, worker_count: int) -> None:
    """Stress-shaped: N workers race; exactly one wins.

    Same contract as the 2-worker test, scaled up. Catches subtle bugs
    in the contention layer (e.g., a missed FOR-UPDATE / advisory-lock
    interaction that allows two simultaneous acquires under heavier
    fanout). Cap of 8 workers — the conformance gate is correctness,
    not throughput; a wider race can be added in the perf lane (§5.6.9).
    """
    cg = make_record(ComparisonGroupRecord, id=f"cg_n_{worker_count}", state="draft", version=1)
    await fresh_store.create_cg(cg)

    results = await asyncio.gather(
        *[
            fresh_store.acquire_lease("cg", f"cg_n_{worker_count}", 60, f"worker-{i}")
            for i in range(worker_count)
        ]
    )
    winners = [r for r in results if r is not None]
    assert len(winners) == 1, (
        f"N={worker_count} workers raced; expected 1 winner, got {len(winners)}: "
        f"{[w.lease_holder_id for w in winners]}"
    )
