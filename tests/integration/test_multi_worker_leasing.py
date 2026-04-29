"""Multi-worker leasing integration test (Task 3.G1).

Pins the engine-level acceptance criteria from
``designs/smai/implementation_plan.md`` §3.4 Task 3.G1:

* **Two workers driving the same MetadataStore do not duplicate
  dispatches under contention.** Both workers run phase-2 + phase-3
  cycles in parallel against the same :class:`SqliteStore`; the
  per-entity lease wrapper inside :func:`drive_entity_phase3` (Task
  3.G1) serializes the dispatch handler invocations. Tracker observes
  each entity's handler firing **exactly once** across both workers.

* **A worker crash mid-dispatch is recovered by the next worker's
  ``acquire_lease`` after ``lease_seconds``.** Simulated by manually
  stamping an entity with an expired lease (the "crashed" worker never
  released its hold); the next worker cycle's ``acquire_lease`` returns
  a fresh token (DEC-035 #2 implicit reclamation) and the dispatch
  handler fires.

The plugin-level lease primitives (``acquire_lease`` /
``release_lease`` / ``extend_lease``) under multi-worker contention are
covered by ``plugins/smai-store-postgres/tests/test_lease_contention.py``
(Task 3.F1's reference Postgres plugin) — this file sits one layer
above and exercises the engine wrapper that consumes them. SQLite is
sufficient for the engine-level pin: the wrapper is plugin-agnostic
(per `07` §5.6.7's "the contract surface is the same regardless").
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _g1_fakes import (
    CallTracker,
    FakeArtifactStore,
    FakeCompute,
    make_g1_spec,
    make_seed_cg,
)
from smai_orchestrator.engine import EngineConfig
from smai_orchestrator.worker.loop import run_worker_cycle
from smai_store_sqlite import SqliteStore


@pytest.mark.asyncio
async def test_two_workers_no_duplicate_dispatches(tmp_path: Path) -> None:
    """Two workers in parallel against the same store; each entity's
    dispatch handler fires exactly once.

    Mechanics:

    * 5 CGs seeded in ``draft``.
    * Worker A and worker B run :func:`run_worker_cycle` concurrently
      with distinct ``worker_id``s. Both phase-2 queries return the
      same draft set; both attempt phase-3 dispatch on each entity.
    * The phase-3 lease wrapper (Task 3.G1) serializes per-entity:
      one worker's ``acquire_lease`` succeeds, the other gets ``None``
      and surfaces ``status="lease_held"``. Independently, the CAS
      state transition in step 1 ensures correctness even without the
      lease — but the lease is the canonical mechanism per `05` §3.5.

    Asserts each entity ID appears exactly once in the tracker's call
    list across both workers.
    """
    db_path = tmp_path / "multi_worker.db"
    store = SqliteStore(f"sqlite+aiosqlite:///{db_path}")
    await store.migrate()
    try:
        cg_ids = [f"cg_g1_{i}" for i in range(5)]
        for cg_id in cg_ids:
            await store.create_cg(make_seed_cg(cg_id))

        tracker = CallTracker()
        # ``hold_seconds=0.05`` widens the contention window so phase-2
        # query results overlap across the two workers — a vanishingly
        # small handler would let worker A complete its full cycle
        # before worker B even reads phase-2.
        spec = make_g1_spec(tracker, hold_seconds=0.05)
        compute = FakeCompute()
        artifacts = FakeArtifactStore()

        async def _drive(worker_id: str) -> None:
            await run_worker_cycle(
                spec=spec,
                metadata_store=store,
                artifact_store=artifacts,  # type: ignore[arg-type]
                compute=compute,  # type: ignore[arg-type]
                llm_providers=None,
                config=EngineConfig(),
                worker_id=worker_id,
            )

        await asyncio.gather(_drive("worker-a"), _drive("worker-b"))

        # Each entity dispatched exactly once across both workers.
        dispatched_ids = [entity_id for _, entity_id in tracker.calls]
        assert sorted(dispatched_ids) == sorted(cg_ids), (
            f"expected every CG to dispatch exactly once across both workers; got {tracker.calls}"
        )

        # Every CG advanced to ``implemented`` (terminal).
        for cg_id in cg_ids:
            row = await store.get_cg(cg_id)
            assert row is not None
            assert row.state == "implemented", (
                f"CG {cg_id!r} should have advanced to implemented; got {row.state}"
            )
            # Lease released cleanly — no stale leased_by / nonce.
            assert row.leased_by is None
            assert row.lease_nonce is None
    finally:
        await store.dispose()


@pytest.mark.asyncio
async def test_crashed_worker_lease_reclaimed_after_expiry(tmp_path: Path) -> None:
    """Worker A "crashes" mid-dispatch (lease never released); after
    ``lease_seconds`` elapses, worker B's cycle reclaims the entity
    via ``acquire_lease`` and dispatches it (DEC-035 #2 implicit
    reclamation).

    Mechanics:

    * Seed one CG in ``draft``.
    * Manually stamp it with an expired lease (10 seconds in the past),
      simulating a crashed worker that never released. No separate
      sweeper task per DEC-035 #2 — implicit reclamation via
      ``acquire_lease``'s ``WHERE lease_expires_at < now()`` predicate.
    * Worker B drives one cycle. Its ``acquire_lease`` succeeds (the
      stale lease is silently overwritten); the dispatch handler fires
      against the entity and advances it.
    """
    db_path = tmp_path / "crashed_worker.db"
    store = SqliteStore(f"sqlite+aiosqlite:///{db_path}")
    await store.migrate()
    try:
        cg = make_seed_cg("cg_crashed")
        await store.create_cg(cg)

        # Stamp an expired lease (simulating the crashed worker).
        from smai_store_sqlite._store import ENTITY_TABLE
        from sqlalchemy import update

        cg_table = ENTITY_TABLE["cg"]
        expired_at = datetime.now(UTC) - timedelta(seconds=10)
        async with store._engine.begin() as conn:  # type: ignore[attr-defined]
            await conn.execute(
                update(cg_table)
                .where(cg_table.c.id == cg.id)
                .values(
                    leased_by="worker-crashed",
                    lease_expires_at=expired_at,
                    lease_nonce="stale-nonce",
                )
            )

        # Sanity: pre-cycle the row is in ``draft`` with the stale lease.
        pre = await store.get_cg(cg.id)
        assert pre is not None
        assert pre.state == "draft"
        assert pre.leased_by == "worker-crashed"

        tracker = CallTracker()
        spec = make_g1_spec(tracker)
        compute = FakeCompute()
        artifacts = FakeArtifactStore()

        await run_worker_cycle(
            spec=spec,
            metadata_store=store,
            artifact_store=artifacts,  # type: ignore[arg-type]
            compute=compute,  # type: ignore[arg-type]
            llm_providers=None,
            config=EngineConfig(),
            worker_id="worker-recovery",
        )

        # The recovery worker dispatched the CG exactly once.
        assert len(tracker.calls) == 1
        assert tracker.calls[0][1] == cg.id

        # The CG advanced + lease released.
        post = await store.get_cg(cg.id)
        assert post is not None
        assert post.state == "implemented"
        assert post.leased_by is None
        assert post.lease_nonce is None
    finally:
        await store.dispose()


@pytest.mark.asyncio
async def test_distinct_worker_ids_recorded_via_lease_holder(tmp_path: Path) -> None:
    """Each worker's ``worker_id`` flows through into
    ``LeaseToken.lease_holder_id`` (visible on the entity row's
    ``leased_by`` column during the dispatch).

    The synthetic spec's handler reads ``cg.leased_by`` and records it
    into the tracker; with a single worker, every dispatched entity
    carries that worker's id.

    This pins the worker-identity threading: ``run_worker_cycle`` →
    ``_drive_phase3_for_records`` → ``drive_entity_phase3`` →
    ``run_dispatch_with_lease`` → ``acquire_lease(... lease_holder_id=
    worker_id)``.
    """
    db_path = tmp_path / "worker_id_thread.db"
    store = SqliteStore(f"sqlite+aiosqlite:///{db_path}")
    await store.migrate()
    try:
        for i in range(3):
            await store.create_cg(make_seed_cg(f"cg_id_{i}"))

        tracker = CallTracker()
        spec = make_g1_spec(tracker)
        compute = FakeCompute()
        artifacts = FakeArtifactStore()

        await run_worker_cycle(
            spec=spec,
            metadata_store=store,
            artifact_store=artifacts,  # type: ignore[arg-type]
            compute=compute,  # type: ignore[arg-type]
            llm_providers=None,
            config=EngineConfig(),
            worker_id="worker-id-thread-test",
        )

        # Every recorded call carries the worker_id we passed in.
        worker_ids = {wid for wid, _ in tracker.calls}
        assert worker_ids == {"worker-id-thread-test"}, (
            f"worker_id did not propagate to lease_holder_id; got {worker_ids}"
        )
    finally:
        await store.dispose()
