"""Three-phase worker poll-cycle tests per ``05-orchestrator.md`` §3.

Covers:

* :func:`run_worker_cycle` — phase 1 → phase 2 → phase 3 against a
  real :class:`SqliteStore` and synthetic specs.
* :func:`run_worker_loop` — N-iteration loop driven by a fake clock,
  ``shutdown_event`` halts the loop cleanly.
* Pool-slot accounting — entities skipped when their pool is full.
* Per-record exception handling — a single bad record doesn't stall
  the cycle.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from _helpers import (  # type: ignore[import-not-found]
    FakeArtifactStore,
    FakeCompute,
    FakeMonotonic,
    make_dispatch,
    make_gate,
    make_job_handle,
    make_job_status,
)
from smai_orchestrator.engine import (
    ConcurrencyPool,
    DispatchAction,
    EdgeDef,
    EngineConfig,
    EngineSpec,
    SchedulingQueryRef,
    StateDef,
)
from smai_orchestrator.entities.tracking import ComparisonGroupRecord
from smai_orchestrator.worker import compute_pool_slots
from smai_orchestrator.worker.loop import (
    WorkerCycleStats,
    run_worker_cycle,
    run_worker_loop,
)
from smai_store_sqlite import SqliteStore

# ===== Fixtures / helpers =====================================================


def _make_cg(
    *,
    cg_id: str,
    state: str,
    version: int = 0,
    handle: object = None,
) -> ComparisonGroupRecord:
    return ComparisonGroupRecord(
        id=cg_id,
        proposal_id=f"prop_{cg_id}",
        experiment_definition_id=f"exp_{cg_id}",
        state=state,  # type: ignore[arg-type]
        version=version,
        harness_job_handle=handle,  # type: ignore[arg-type]
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


async def _seed_cg(
    store: SqliteStore,
    *,
    cg_id: str,
    state: str,
    handle: object = None,
) -> ComparisonGroupRecord:
    cg = _make_cg(cg_id=cg_id, state=state, handle=handle)
    return await store.create_cg(cg)


def _basic_spec(
    *,
    dispatch_handler,
    pool_name: str = "agents",
    pool_limit: int = 10,
):
    """Synthetic 4-state CG spec: draft → implementing →
    {implemented, implementation_failed} (terminal).

    Phase-2 query binds to :meth:`MetadataStore.get_ready_for_harness_build`;
    phase-1 query binds to :meth:`MetadataStore.get_in_flight_harness_build`.
    """
    in_progress = StateDef(
        name="implementing",
        on_entry_dispatch=DispatchAction(
            name="harness_build",
            handler=dispatch_handler,
            pool=pool_name,
            handle_field="harness_job_handle",
        ),
    )
    return EngineSpec(
        entity_kind="cg",
        initial_state="draft",
        states=[
            StateDef(name="draft"),
            in_progress,
            StateDef(name="implemented", is_terminal=True),
            StateDef(name="implementation_failed", is_terminal=True),
        ],
        edges=[
            EdgeDef(
                name="advance",
                from_state="draft",
                target_state="implementing",
                gate_rule=make_gate(advance=True),
            ),
            EdgeDef(
                name="job-succeeded",
                from_state="implementing",
                target_state="implemented",
                gate_rule=make_gate(advance=True),
                fires_on="job_succeeded",
            ),
            EdgeDef(
                name="job-failed",
                from_state="implementing",
                target_state="implementation_failed",
                gate_rule=make_gate(advance=True),
                fires_on="job_failed",
            ),
        ],
        pools=[ConcurrencyPool(name=pool_name, limit=pool_limit)],
        phase1_queries={
            "implementing": SchedulingQueryRef(
                name="get_in_flight_harness_build",
                fn=_in_flight_harness_build,
            ),
        },
        phase2_queries={
            "draft": SchedulingQueryRef(
                name="get_ready_for_harness_build",
                fn=_ready_for_harness_build,
            ),
        },
    )


async def _in_flight_harness_build(store):  # type: ignore[no-untyped-def]
    """Adapter: drain the cursor page into the list shape that
    :class:`SchedulingQueryRef.fn` expects."""
    page = await store.get_in_flight_harness_build(limit=100, cursor=None)
    return list(page.items)


async def _ready_for_harness_build(store):  # type: ignore[no-untyped-def]
    page = await store.get_ready_for_harness_build(limit=100, cursor=None)
    return list(page.items)


# ===== Phase-1 / phase-2 / phase-3 round-trip ================================


async def test_run_worker_cycle_advances_draft_to_implementing(
    sqlite_store: SqliteStore,
) -> None:
    """A CG in ``draft`` is discovered (phase 2) and dispatched (phase 3)
    to ``implementing`` with a recorded handle."""
    handle = make_job_handle("h-1")
    fake_compute = FakeCompute()
    fake_artifacts = FakeArtifactStore()
    spec = _basic_spec(dispatch_handler=make_dispatch(handle=handle))

    await _seed_cg(sqlite_store, cg_id="cg_A", state="draft")

    stats = await run_worker_cycle(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifacts,  # type: ignore[arg-type]
        compute=fake_compute,  # type: ignore[arg-type]
        llm_providers=None,
        config=EngineConfig(),
    )

    assert stats.phase2_candidates == 1
    assert stats.phase3_advanced == 1
    assert stats.phase3_no_match == 0

    after = await sqlite_store.get_cg("cg_A")
    assert after is not None
    assert after.state == "implementing"
    assert after.harness_job_handle == handle


async def test_run_worker_cycle_phase1_drives_in_flight_to_terminal(
    sqlite_store: SqliteStore,
) -> None:
    """A CG already in ``implementing`` with a live handle gets driven
    to ``implemented`` once :class:`Compute` reports ``succeeded``."""
    handle = make_job_handle("h-flight")
    fake_compute = FakeCompute()
    fake_compute.set_status("h-flight", make_job_status("succeeded", exit_code=0))
    fake_artifacts = FakeArtifactStore()

    spec = _basic_spec(dispatch_handler=make_dispatch(handle=handle))

    # Seed the CG mid-flight: state=implementing, handle set.
    cg = _make_cg(cg_id="cg_B", state="implementing", version=1, handle=handle)
    await sqlite_store.create_cg(cg)

    stats = await run_worker_cycle(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifacts,  # type: ignore[arg-type]
        compute=fake_compute,  # type: ignore[arg-type]
        llm_providers=None,
        config=EngineConfig(),
    )

    assert stats.phase1_inspected == 1
    assert stats.phase1_advanced == 1

    final = await sqlite_store.get_cg("cg_B")
    assert final is not None
    assert final.state == "implemented"
    assert final.harness_job_handle is None


async def test_run_worker_cycle_full_lifecycle_two_cycles(
    sqlite_store: SqliteStore,
) -> None:
    """Two cycles: cycle 1 dispatches ``draft → implementing`` (compute
    job submitted); cycle 2 phase-1 polls and drives to terminal."""
    handle = make_job_handle("h-life")
    fake_compute = FakeCompute()
    fake_artifacts = FakeArtifactStore()
    spec = _basic_spec(dispatch_handler=make_dispatch(handle=handle))

    await _seed_cg(sqlite_store, cg_id="cg_C", state="draft")

    # Cycle 1: dispatch fires, job submitted, status not yet succeeded.
    fake_compute.set_status("h-life", make_job_status("running"))
    stats_1 = await run_worker_cycle(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifacts,  # type: ignore[arg-type]
        compute=fake_compute,  # type: ignore[arg-type]
        llm_providers=None,
        config=EngineConfig(),
    )
    assert stats_1.phase3_advanced == 1
    mid = await sqlite_store.get_cg("cg_C")
    assert mid is not None
    assert mid.state == "implementing"

    # Cycle 2: job has succeeded; phase-1 picks it up.
    fake_compute.set_status("h-life", make_job_status("succeeded", exit_code=0))
    stats_2 = await run_worker_cycle(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifacts,  # type: ignore[arg-type]
        compute=fake_compute,  # type: ignore[arg-type]
        llm_providers=None,
        config=EngineConfig(),
    )
    assert stats_2.phase1_advanced == 1
    final = await sqlite_store.get_cg("cg_C")
    assert final is not None
    assert final.state == "implemented"


# ===== Pool-slot accounting ==================================================


async def test_run_worker_cycle_skips_when_pool_full(
    sqlite_store: SqliteStore,
) -> None:
    """Two CGs in ``draft`` but pool limit is 1; only one advances per
    cycle (the other counted as ``phase3_skipped_pool_full``)."""
    handle = make_job_handle("h-pool")
    fake_compute = FakeCompute()
    fake_artifacts = FakeArtifactStore()

    # Pre-seed an in-flight CG so the pool's slot is consumed before
    # the cycle starts.
    in_flight_handle = make_job_handle("h-in-flight")
    cg_inflight = _make_cg(
        cg_id="cg_already",
        state="implementing",
        version=1,
        handle=in_flight_handle,
    )
    await sqlite_store.create_cg(cg_inflight)
    fake_compute.set_status("h-in-flight", make_job_status("running"))

    # Pool capacity = 1; the in-flight CG already consumes the slot.
    spec = _basic_spec(dispatch_handler=make_dispatch(handle=handle), pool_limit=1)

    # Two new draft candidates; neither should advance this cycle.
    await _seed_cg(sqlite_store, cg_id="cg_new1", state="draft")
    await _seed_cg(sqlite_store, cg_id="cg_new2", state="draft")

    stats = await run_worker_cycle(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifacts,  # type: ignore[arg-type]
        compute=fake_compute,  # type: ignore[arg-type]
        llm_providers=None,
        config=EngineConfig(),
    )

    assert stats.phase2_candidates == 2
    assert stats.phase3_advanced == 0
    assert stats.phase3_skipped_pool_full == 2
    # Both CGs remain in draft.
    for cid in ("cg_new1", "cg_new2"):
        cg = await sqlite_store.get_cg(cid)
        assert cg is not None
        assert cg.state == "draft"


async def test_run_worker_cycle_pool_drains_in_priority_order(
    sqlite_store: SqliteStore,
) -> None:
    """Pool ordering is descending priority — verified via the slot
    list in ``stats.pool_slots`` (the actual high-priority-drains-
    first behavior is dispatched in priority order)."""
    handle = make_job_handle("h-prio")
    fake_compute = FakeCompute()
    fake_artifacts = FakeArtifactStore()
    spec = _basic_spec(dispatch_handler=make_dispatch(handle=handle))
    # Override pool with priority bumps.
    spec.pools = [
        ConcurrencyPool(name="agents", limit=10, priority=5),
        ConcurrencyPool(name="runs", limit=10, priority=20),
    ]

    await _seed_cg(sqlite_store, cg_id="cg_X", state="draft")
    stats = await run_worker_cycle(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifacts,  # type: ignore[arg-type]
        compute=fake_compute,  # type: ignore[arg-type]
        llm_providers=None,
        config=EngineConfig(),
    )
    pool_names = [p.name for p in stats.pool_slots]
    assert pool_names == ["runs", "agents"]


# ===== Per-record exception handling ========================================


async def test_run_worker_cycle_continues_after_single_record_dispatch_error(
    sqlite_store: SqliteStore,
) -> None:
    """A handler that raises rolls back the failing CG but the cycle
    keeps processing the next one."""
    handle = make_job_handle("h-ok")
    fake_compute = FakeCompute()
    fake_artifacts = FakeArtifactStore()

    # First handler call raises; second returns the OK handle.
    call_state = {"call": 0}

    async def _flaky_handler(ctx):  # type: ignore[no-untyped-def]
        call_state["call"] += 1
        if call_state["call"] == 1:
            raise RuntimeError("first call fails")
        from smai_orchestrator.engine import DispatchOutcome

        out = DispatchOutcome()
        out.submitted_handles.append(handle)
        return out

    spec = _basic_spec(dispatch_handler=_flaky_handler)

    await _seed_cg(sqlite_store, cg_id="cg_fail", state="draft")
    await _seed_cg(sqlite_store, cg_id="cg_ok", state="draft")

    stats = await run_worker_cycle(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifacts,  # type: ignore[arg-type]
        compute=fake_compute,  # type: ignore[arg-type]
        llm_providers=None,
        config=EngineConfig(),
    )

    # Either of the two could come first depending on FIFO ordering; both
    # were inspected.
    assert stats.phase2_candidates == 2
    # Exactly one advanced and exactly one rolled back.
    assert stats.phase3_advanced == 1
    assert stats.phase3_dispatch_failed == 1


# ===== run_worker_loop semantics ============================================


async def test_run_worker_loop_runs_n_iterations_and_halts(
    sqlite_store: SqliteStore,
) -> None:
    """The loop runs cycles until ``shutdown_event`` is set; each cycle
    advances one CG. Verifies no entity leakage across N iterations."""
    handle_a = make_job_handle("h-a")
    handle_b = make_job_handle("h-b")
    fake_compute = FakeCompute()
    fake_compute.set_status("h-a", make_job_status("succeeded", exit_code=0))
    fake_compute.set_status("h-b", make_job_status("succeeded", exit_code=0))
    fake_artifacts = FakeArtifactStore()

    handles = iter([handle_a, handle_b])

    async def _handler(ctx):  # type: ignore[no-untyped-def]
        from smai_orchestrator.engine import DispatchOutcome

        out = DispatchOutcome()
        out.submitted_handles.append(next(handles))
        return out

    spec = _basic_spec(dispatch_handler=_handler)

    await _seed_cg(sqlite_store, cg_id="cg_a", state="draft")
    await _seed_cg(sqlite_store, cg_id="cg_b", state="draft")

    # Fake monotonic so the loop's sleep is instant; we drive cycle
    # cadence via the shutdown event.
    fake_clock = FakeMonotonic(start=0.0)
    config = EngineConfig(
        time_provider=fake_clock,
        poll_interval_seconds=0,  # sleep -> asyncio.sleep(0)
    )

    shutdown = asyncio.Event()
    cycle_count = 0
    completed_stats: list[WorkerCycleStats] = []

    async def _on_cycle(stats: WorkerCycleStats) -> None:
        nonlocal cycle_count
        cycle_count += 1
        completed_stats.append(stats)
        # Stop after a few cycles — gives phase-1 a chance to drive the
        # second CG to terminal after phase-3 dispatched it.
        if cycle_count >= 4:
            shutdown.set()

    await run_worker_loop(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifacts,  # type: ignore[arg-type]
        compute=fake_compute,  # type: ignore[arg-type]
        llm_providers=None,
        config=config,
        shutdown_event=shutdown,
        on_cycle_complete=_on_cycle,
    )

    assert cycle_count >= 4
    # Both CGs reach the terminal state.
    final_a = await sqlite_store.get_cg("cg_a")
    final_b = await sqlite_store.get_cg("cg_b")
    assert final_a is not None and final_a.state == "implemented"
    assert final_b is not None and final_b.state == "implemented"


async def test_run_worker_loop_respects_shutdown_immediately(
    sqlite_store: SqliteStore,
) -> None:
    """An already-set shutdown event short-circuits the loop on entry."""
    fake_compute = FakeCompute()
    fake_artifacts = FakeArtifactStore()
    spec = _basic_spec(
        dispatch_handler=make_dispatch(handle=make_job_handle("h-never"))
    )

    shutdown = asyncio.Event()
    shutdown.set()  # set BEFORE entry

    cycles_seen = 0

    async def _on_cycle(_: WorkerCycleStats) -> None:
        nonlocal cycles_seen
        cycles_seen += 1

    await run_worker_loop(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifacts,  # type: ignore[arg-type]
        compute=fake_compute,  # type: ignore[arg-type]
        llm_providers=None,
        config=EngineConfig(poll_interval_seconds=0),
        shutdown_event=shutdown,
        on_cycle_complete=_on_cycle,
    )
    assert cycles_seen == 0


async def test_run_worker_loop_no_leak_on_empty_store(
    sqlite_store: SqliteStore,
) -> None:
    """Many cycles against an empty store: the worker handles "no work"
    cleanly without raising, and stats reflect zero entities."""
    fake_compute = FakeCompute()
    fake_artifacts = FakeArtifactStore()
    spec = _basic_spec(dispatch_handler=make_dispatch(handle=make_job_handle("h")))

    config = EngineConfig(time_provider=FakeMonotonic(start=0.0), poll_interval_seconds=0)
    shutdown = asyncio.Event()
    n = 0

    async def _on_cycle(stats: WorkerCycleStats) -> None:
        nonlocal n
        n += 1
        assert stats.phase2_candidates == 0
        assert stats.phase1_inspected == 0
        if n >= 5:
            shutdown.set()

    await run_worker_loop(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifacts,  # type: ignore[arg-type]
        compute=fake_compute,  # type: ignore[arg-type]
        llm_providers=None,
        config=config,
        shutdown_event=shutdown,
        on_cycle_complete=_on_cycle,
    )
    assert n >= 5


# ===== Spec-author bug detection =============================================


async def test_phase2_query_pointing_at_terminal_state_raises(
    sqlite_store: SqliteStore,
) -> None:
    fake_compute = FakeCompute()
    fake_artifacts = FakeArtifactStore()
    bad_spec = _basic_spec(dispatch_handler=make_dispatch(handle=make_job_handle("h")))
    # Misconfigure: phase-2 query keyed on a terminal state.
    bad_spec.phase2_queries = {
        "implemented": SchedulingQueryRef(
            name="bogus", fn=_ready_for_harness_build
        ),
    }
    with pytest.raises(ValueError, match="terminal"):
        await run_worker_cycle(
            spec=bad_spec,
            metadata_store=sqlite_store,
            artifact_store=fake_artifacts,  # type: ignore[arg-type]
            compute=fake_compute,  # type: ignore[arg-type]
            llm_providers=None,
            config=EngineConfig(),
        )


async def test_phase1_query_unregistered_state_raises(
    sqlite_store: SqliteStore,
) -> None:
    fake_compute = FakeCompute()
    fake_artifacts = FakeArtifactStore()
    bad_spec = _basic_spec(dispatch_handler=make_dispatch(handle=make_job_handle("h")))
    bad_spec.phase1_queries = {
        "no_such_state": SchedulingQueryRef(
            name="bogus", fn=_in_flight_harness_build
        ),
    }
    with pytest.raises(LookupError, match="no_such_state"):
        await run_worker_cycle(
            spec=bad_spec,
            metadata_store=sqlite_store,
            artifact_store=fake_artifacts,  # type: ignore[arg-type]
            compute=fake_compute,  # type: ignore[arg-type]
            llm_providers=None,
            config=EngineConfig(),
        )


# ===== Slot-compute integration check =======================================


async def test_pool_slots_call_count_with_in_flight_jobs(
    sqlite_store: SqliteStore,
) -> None:
    """Smoke-test the integration with :meth:`MetadataStore.count_with_in_flight_jobs`
    against the real :class:`SqliteStore` plugin."""
    spec = _basic_spec(dispatch_handler=make_dispatch(handle=make_job_handle("h")))

    # Seed one in-flight CG.
    cg = _make_cg(
        cg_id="cg_inflight",
        state="implementing",
        version=1,
        handle=make_job_handle("h-x"),
    )
    await sqlite_store.create_cg(cg)

    slots = await compute_pool_slots(
        pools=spec.pools,
        state_to_pool=spec.state_to_pool(),
        entity_kind=spec.entity_kind,
        metadata_store=sqlite_store,
        config=EngineConfig(),
    )
    assert len(slots) == 1
    assert slots[0].name == "agents"
    assert slots[0].in_flight == 1
    assert slots[0].available == 9
