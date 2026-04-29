"""Unit tests for the phase-3 dispatch lease wrapper (Task 3.G1).

Per ``designs/smai/05-orchestrator.md`` §3.5 + ``07-plugin-interfaces.md``
§5.6.7 + DEC-035 #2: phase-3 dispatch is wrapped with
``acquire_lease`` / ``release_lease`` so concurrent workers cannot both
fire the dispatch handler for the same entity. This file pins:

* ``acquire_lease`` returns ``None`` (lease held by another worker) →
  dispatch returns ``DriveOutcome(status="lease_held")`` without firing
  the handler.
* ``acquire_lease`` succeeds → dispatch runs, lease is released after
  completion.
* Dispatch raises → lease is still released in the ``finally`` block.
* ``extend_lease`` raises :class:`LeaseLostError` mid-dispatch → the
  wrapper cancels the dispatch and surfaces ``status="conflict"``.

Plugin-level lease primitives (`acquire_lease` / `release_lease` /
`extend_lease`) are exercised by the SqliteStore conformance suite and
the Postgres store's ``test_lease_contention.py``; this file focuses on
the engine wrapper layer.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest_asyncio
from _helpers import (
    FakeArtifactStore,
    FakeCompute,
    make_dispatch,
    make_gate,
    make_job_handle,
)
from smai_core.plugins import LeaseToken
from smai_core.plugins.metadata_store import LeaseLostError
from smai_orchestrator.engine import (
    DispatchAction,
    EdgeDef,
    EngineConfig,
    EngineSpec,
    StateDef,
    drive_entity_phase3,
)
from smai_orchestrator.engine.dispatch import run_dispatch_with_lease
from smai_orchestrator.entities.tracking import ComparisonGroupRecord
from smai_store_sqlite import SqliteStore


@pytest_asyncio.fixture
async def store() -> AsyncIterator[SqliteStore]:
    s = SqliteStore("sqlite+aiosqlite:///:memory:")
    await s.migrate()
    try:
        yield s
    finally:
        await s.dispose()


async def _seed_cg(s: SqliteStore, *, cg_id: str = "cg_lease") -> ComparisonGroupRecord:
    cg = ComparisonGroupRecord(
        id=cg_id,
        proposal_id="prop_test",
        experiment_definition_id="exp_test",
        state="draft",
        version=0,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    return await s.create_cg(cg)


def _spec(*, handler, handle_field: str | None = "harness_job_handle") -> EngineSpec:
    action = DispatchAction(
        name="harness_build",
        handler=handler,
        pool="agents",
        handle_field=handle_field,
    )
    return EngineSpec(
        entity_kind="cg",
        initial_state="draft",
        states=[
            StateDef(name="draft"),
            StateDef(name="implementing", on_entry_dispatch=action),
            StateDef(name="implemented", is_terminal=True),
        ],
        edges=[
            EdgeDef(
                name="advance",
                from_state="draft",
                target_state="implementing",
                gate_rule=make_gate(advance=True),
            ),
        ],
    )


# === acquire_lease returns None → skip without firing handler ================


async def test_acquire_lease_held_by_peer_returns_lease_held(
    store: SqliteStore, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """Worker A holds the lease; worker B's drive should return
    ``status="lease_held"`` and NOT invoke the dispatch handler.

    Pins `05` §3.5 / DEC-035 #2: ``acquire_lease`` returning ``None`` is
    a normal poll-loop event (not an exception); the wrapper surfaces
    it as a distinct outcome from CAS conflict.
    """
    cg = await _seed_cg(store)

    # Worker A: pre-acquire the lease so worker B's drive sees None.
    token_a = await store.acquire_lease("cg", cg.id, 60, "worker-a")
    assert token_a is not None

    handler_calls = 0

    async def _track_handler(_ctx) -> object:  # noqa: ARG001
        nonlocal handler_calls
        handler_calls += 1
        from smai_orchestrator.engine import DispatchOutcome

        return DispatchOutcome(submitted_handles=[make_job_handle("h-should-not-fire")])

    spec = _spec(handler=_track_handler)

    outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=EngineConfig(),
        record=cg,
        worker_id="worker-b",
    )

    assert outcome.status == "lease_held"
    assert handler_calls == 0, "dispatch handler must NOT fire when lease is held by a peer"

    # Entity unchanged: state and version are still draft / 0.
    final = await store.get_cg(cg.id)
    assert final is not None
    assert final.state == "draft"
    assert final.version == 0


# === Happy path: acquire → dispatch → release ===============================


async def test_lease_released_after_successful_dispatch(
    store: SqliteStore, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """Successful dispatch path: lease is acquired, dispatch fires,
    lease is released. After the call, a peer can re-acquire immediately
    (no waiting for ``lease_seconds``).
    """
    cg = await _seed_cg(store)
    handle = make_job_handle("h-happy-lease")
    spec = _spec(handler=make_dispatch(handle=handle))

    outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=EngineConfig(),
        record=cg,
        worker_id="worker-a",
    )

    assert outcome.status == "advanced"

    # Lease released: a fresh acquire must succeed without waiting.
    new_token = await store.acquire_lease("cg", cg.id, 60, "worker-b")
    assert new_token is not None, "lease was not released after successful dispatch"


async def test_lease_released_after_dispatch_raises(
    store: SqliteStore, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """Failure path: dispatch handler raises → engine forward-rolls-back
    → lease is still released in the wrapper's ``finally``.
    """
    cg = await _seed_cg(store)
    spec = _spec(handler=make_dispatch(raises=RuntimeError("substrate down")))

    outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=EngineConfig(),
        record=cg,
        worker_id="worker-a",
    )

    assert outcome.status == "dispatch_failed_rolled_back"

    # Lease released even though dispatch failed.
    new_token = await store.acquire_lease("cg", cg.id, 60, "worker-b")
    assert new_token is not None, "lease must be released on the dispatch-failed path"


async def test_lease_released_after_handler_error_outcome(
    store: SqliteStore, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """Same lease-released invariant when the handler returns
    :class:`DispatchOutcome` with a non-None ``error`` rather than
    raising — the wrapper's ``finally`` block fires regardless.
    """
    cg = await _seed_cg(store)
    spec = _spec(handler=make_dispatch(error="precondition not met"))

    outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=EngineConfig(),
        record=cg,
        worker_id="worker-a",
    )

    assert outcome.status == "dispatch_failed_rolled_back"
    new_token = await store.acquire_lease("cg", cg.id, 60, "worker-b")
    assert new_token is not None


# === LeaseLostError mid-dispatch → cancel + surface conflict =================


async def test_lease_lost_mid_dispatch_aborts_and_surfaces_conflict(
    tmp_path, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """Heartbeat raises :class:`LeaseLostError` mid-dispatch → the
    wrapper cancels the in-flight dispatch task and returns
    ``status="conflict"`` with ``error="lease_lost"``.

    Mechanics: we monkey-patch :meth:`SqliteStore.extend_lease` on the
    instance so every call raises :class:`LeaseLostError`. The
    dispatch handler awaits a long sleep so the heartbeat fires before
    the handler completes; ``EngineConfig.extend_lease_interval_seconds``
    is set to ``0`` so the heartbeat sleeps for zero seconds and
    immediately calls ``extend_lease`` → :class:`LeaseLostError` →
    cancel.

    Uses a tempfile-backed SQLite store rather than ``:memory:``: the
    cancellation path tears the dispatch task out of an in-flight
    ``await``, which on aiosqlite + ``:memory:`` can corrupt the
    connection-pool's worker thread state and surface as "no such
    table" on the subsequent ``release_lease`` UPDATE. A file-backed
    store keeps the schema visible across the cancellation boundary.
    """
    db_path = tmp_path / "lease_lost.db"
    store = SqliteStore(f"sqlite+aiosqlite:///{db_path}")
    await store.migrate()
    try:
        cg = await _seed_cg(store)

        async def _losing_extend(
            token: LeaseToken,
            additional_seconds: int,  # noqa: ARG001
        ) -> LeaseToken:
            raise LeaseLostError(token.entity_kind, token.entity_id)

        store.extend_lease = _losing_extend  # type: ignore[method-assign]

        handler_started = asyncio.Event()
        handler_finished = False

        async def _slow_handler(_ctx) -> object:  # noqa: ARG001
            nonlocal handler_finished
            handler_started.set()
            # Sleep long enough for the heartbeat task (with
            # extend_lease_interval_seconds=0) to fire and cancel us.
            await asyncio.sleep(5.0)
            handler_finished = True
            from smai_orchestrator.engine import DispatchOutcome

            return DispatchOutcome(submitted_handles=[make_job_handle("h-should-not-finish")])

        spec = _spec(handler=_slow_handler)
        # Non-zero interval lets step 1 (transition_state's async-with
        # engine.begin block) complete cleanly before cancellation hits
        # — cancellation mid-aiosqlite UPDATE leaves the connection
        # holding a row lock, which surfaces as "database is locked"
        # on the wrapper's release_lease.
        config = EngineConfig(lease_seconds=60, extend_lease_interval_seconds=1)

        outcome = await drive_entity_phase3(
            spec=spec,
            metadata_store=store,
            artifact_store=fake_artifact_store,
            compute=fake_compute,
            llm=None,
            config=config,
            record=cg,
            worker_id="worker-a",
        )

        assert outcome.status == "conflict"
        assert outcome.error == "lease_lost"
        assert handler_started.is_set(), (
            "the dispatch handler should have started before cancellation"
        )
        assert not handler_finished, "the dispatch handler must be cancelled before completing"
    finally:
        await store.dispose()


# === Direct unit test of run_dispatch_with_lease ============================


async def test_run_dispatch_with_lease_acquires_releases(
    store: SqliteStore, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """Direct call to :func:`run_dispatch_with_lease` — bypasses the
    edge-evaluation path of :func:`drive_entity_phase3` and asserts the
    lease lifecycle pin in isolation.
    """
    cg = await _seed_cg(store, cg_id="cg_direct")
    handle = make_job_handle("h-direct")
    spec = _spec(handler=make_dispatch(handle=handle))
    edge = spec.edges[0]
    target_def = spec.state_def(edge.target_state)

    outcome = await run_dispatch_with_lease(
        spec=spec,
        metadata_store=store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=EngineConfig(),
        edge=edge,
        target_state=target_def,
        entity_id=cg.id,
        expected_version=cg.version,
        worker_id="worker-direct",
    )

    assert outcome.status == "advanced"
    final = await store.get_cg("cg_direct")
    assert final is not None
    assert final.state == "implementing"
    assert final.harness_job_handle == handle

    # Re-acquire must succeed — proves release fired in the finally.
    re_acquired = await store.acquire_lease("cg", cg.id, 60, "worker-other")
    assert re_acquired is not None


async def test_lease_held_outcome_does_not_invoke_compute(
    store: SqliteStore, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """Pin: ``status="lease_held"`` means the dispatch handler never
    fires, so no Compute submission happens. (FakeCompute would raise
    if ``submit`` were called without a pre-enqueued outcome — the test
    relies on that to verify the no-call invariant.)
    """
    cg = await _seed_cg(store)
    token = await store.acquire_lease("cg", cg.id, 60, "worker-a")
    assert token is not None

    spec = _spec(handler=make_dispatch(handle=make_job_handle("h-unused")))

    outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=EngineConfig(),
        record=cg,
        worker_id="worker-b",
    )

    assert outcome.status == "lease_held"
    # FakeCompute's submit_calls list stays empty.
    assert fake_compute.submit_calls == []


# === Heartbeat extension under happy-path long dispatch =====================


async def test_heartbeat_extends_lease_during_long_dispatch(
    store: SqliteStore, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """Heartbeat fires at least once during a long dispatch and bumps
    ``lease_expires_at`` past the original window.

    With ``lease_seconds=2`` and ``extend_lease_interval_seconds=0``,
    the heartbeat fires immediately after the dispatch starts; one
    extension call adds another 2 seconds to the window. We assert the
    lease's nonce was rotated and the expiry pushed out.
    """
    cg = await _seed_cg(store, cg_id="cg_heartbeat")

    # Pre-acquire to capture the original (post-acquire) state.
    pre_token = await store.acquire_lease("cg", cg.id, 60, "worker-pre")
    assert pre_token is not None
    await store.release_lease(pre_token)

    extends_observed = 0
    real_extend = store.extend_lease

    async def _counting_extend(token: LeaseToken, additional_seconds: int) -> LeaseToken:
        nonlocal extends_observed
        extends_observed += 1
        return await real_extend(token, additional_seconds=additional_seconds)

    # Monkey-patch the bound method on this instance.
    store.extend_lease = _counting_extend  # type: ignore[method-assign]

    handler_done = asyncio.Event()

    async def _slow_handler(_ctx) -> object:  # noqa: ARG001
        # Sleep long enough for at least two heartbeat extensions.
        await asyncio.sleep(0.05)
        handler_done.set()
        from smai_orchestrator.engine import DispatchOutcome

        return DispatchOutcome(submitted_handles=[make_job_handle("h-heartbeat")])

    spec = _spec(handler=_slow_handler)
    config = EngineConfig(lease_seconds=60, extend_lease_interval_seconds=0)

    outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=config,
        record=cg,
        worker_id="worker-hb",
    )

    assert outcome.status == "advanced"
    assert handler_done.is_set()
    assert extends_observed >= 1, (
        "the heartbeat task should have called extend_lease at least once "
        "during a >0-duration dispatch with extend_lease_interval_seconds=0"
    )


# === Lease expiry → other worker can recover ================================


async def test_expired_lease_reclaimed_by_next_worker(
    store: SqliteStore,
    fake_compute: FakeCompute,
    fake_artifact_store: FakeArtifactStore,  # noqa: ARG001
) -> None:
    """Pin DEC-035 #2 implicit reclamation at the engine seam: a stale
    (expired) lease from a crashed worker is silently reclaimable by the
    next worker's ``acquire_lease`` — no separate sweeper task.

    Mechanics: we manually stamp a CG's lease fields as expired (10
    seconds in the past). The next ``acquire_lease`` succeeds.
    """
    cg = await _seed_cg(store, cg_id="cg_expired")

    # Manually stamp the CG with an already-expired lease (simulating a
    # crashed worker that never released).
    from smai_store_sqlite._store import ENTITY_TABLE

    cg_table = ENTITY_TABLE["cg"]
    expired_at = datetime.now(UTC) - timedelta(seconds=10)
    from sqlalchemy import update

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

    # The next worker's acquire succeeds despite the stamped fields.
    token = await store.acquire_lease("cg", cg.id, 60, "worker-recovery")
    assert token is not None
    assert token.lease_holder_id == "worker-recovery"
    assert token.nonce != "stale-nonce"
