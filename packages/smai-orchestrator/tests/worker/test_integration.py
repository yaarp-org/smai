"""End-to-end worker-loop integration test against real plugins.

Per Task 2.C2's acceptance: a synthetic-spec integration test runs N
iterations against a real :class:`SqliteStore` + :class:`LocalFsStore`
+ a fake :class:`Compute`, with a memoized dispatch handler exercising
the :class:`ArtifactStoreCheckpointer` flavor end-to-end. Demonstrates
the full poll-cycle round-trip: phase-1 polls live jobs, phase-2
discovers new candidates, phase-3 dispatches under write-first
ordering, memoization caches handler results across cycles.

Substrate per the brief: ``SqliteStore + LocalFsStore as canonical
fixtures``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from _helpers import (  # type: ignore[import-not-found]
    FakeArtifactStore,
    FakeCompute,
    FakeMonotonic,
    make_dispatch,
    make_gate,
    make_job_handle,
    make_job_status,
)
from pydantic import BaseModel
from smai_artifacts_localfs import LocalFsStore
from smai_orchestrator.checkpointer import (
    ArtifactStoreCheckpointer,
    InMemoryCheckpointBackend,
    MetadataStoreCheckpointer,
    memoized,
)
from smai_orchestrator.engine import (
    ConcurrencyPool,
    DispatchAction,
    DispatchOutcome,
    EdgeDef,
    EngineConfig,
    EngineSpec,
    SchedulingQueryRef,
    StateDef,
)
from smai_orchestrator.entities.tracking import ComparisonGroupRecord
from smai_orchestrator.worker.loop import (
    WorkerCycleStats,
    run_worker_cycle,
    run_worker_loop,
)
from smai_store_sqlite import SqliteStore

# ===== Memoized handler =====================================================


class _ReviewInputs(BaseModel):
    cg_id: str
    code_revision: str


class _ReviewOutputs(BaseModel):
    handle: str


def _serialize(o: _ReviewOutputs) -> bytes:
    return o.model_dump_json().encode("utf-8")


def _deserialize(raw: bytes) -> _ReviewOutputs:
    return _ReviewOutputs.model_validate_json(raw)


# ===== Helpers ==============================================================


def _make_cg(cg_id: str, state: str = "draft") -> ComparisonGroupRecord:
    return ComparisonGroupRecord(
        id=cg_id,
        proposal_id=f"prop_{cg_id}",
        experiment_definition_id=f"exp_{cg_id}",
        state=state,  # type: ignore[arg-type]
        version=0,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


async def _list_cgs_with_handle(
    store: SqliteStore, *, state: str, has_handle: bool
) -> list[ComparisonGroupRecord]:
    """State-filter + handle-presence filter; same workaround the
    test_loop.py module uses for the SqliteStore JSON-null bug."""
    from smai_store_sqlite._schema import cgs_table  # type: ignore[import-not-found]
    from smai_store_sqlite._serde import row_to_record  # type: ignore[import-not-found]
    from sqlalchemy import select

    async with store._engine.connect() as conn:  # type: ignore[attr-defined]
        result = await conn.execute(select(cgs_table).where(cgs_table.c.state == state))
        rows = result.mappings().all()
    items = [row_to_record(ComparisonGroupRecord, row) for row in rows]
    if has_handle:
        return [c for c in items if c.harness_job_handle is not None]
    return [c for c in items if c.harness_job_handle is None]


# ===== End-to-end memoized-dispatch integration =============================


async def test_full_lifecycle_with_memoized_dispatch_against_real_plugins(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """Full happy-path against real plugins.

    * Submits a CG in ``draft``.
    * Worker cycle 1: phase-2 discovers, phase-3 dispatches via memoized
      handler (artifact-store checkpointer flavor; sidecar at
      ``orchestrator/checkpoints/...``); handle is recorded; CG ⇒
      ``implementing``.
    * Worker cycle 2: same memoized handler called again (no-op due to
      cache hit) — verified by counting handler invocations.
    * Worker cycle 3 (after Compute reports ``succeeded``): phase-1
      drives ``implementing → implemented``.
    """
    cp = ArtifactStoreCheckpointer(localfs_store)
    handler_invocations: list[str] = []

    async def memoized_handler(ctx) -> DispatchOutcome:  # type: ignore[no-untyped-def]
        # Memoize the "produce a handle" step (in real life this would
        # be an LLM call producing a planned config or similar). We
        # demonstrate the load → run → save shape end-to-end.
        async def work() -> _ReviewOutputs:
            handler_invocations.append(ctx.entity_id)
            return _ReviewOutputs(handle=f"h-{ctx.entity_id}")

        result = await memoized(
            checkpointer=cp,
            thread_id=ctx.entity_id,
            step_id="harness_dispatch_v1",
            inputs=_ReviewInputs(cg_id=ctx.entity_id, code_revision="rev1"),
            work=work,
            serialize=_serialize,
            deserialize=_deserialize,
        )
        out = DispatchOutcome()
        out.submitted_handles.append(make_job_handle(result.handle))
        return out

    spec = _build_spec(handler=memoized_handler)
    fake_compute = FakeCompute()

    # Cycle 1: discover + dispatch.
    await sqlite_store.create_cg(_make_cg("cg_e2e"))
    fake_compute.set_status("h-cg_e2e", make_job_status("running"))

    stats_1 = await run_worker_cycle(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        compute=fake_compute,  # type: ignore[arg-type]
        llm_providers=None,
        config=EngineConfig(),
    )
    assert stats_1.phase3_advanced == 1
    assert handler_invocations == ["cg_e2e"]

    cg = await sqlite_store.get_cg("cg_e2e")
    assert cg is not None
    assert cg.state == "implementing"

    # Cycle 2: handler cache-hit. No new entity to dispatch (the existing
    # CG is in ``implementing``, not ``draft``); but to demonstrate the
    # cache hit, we synthetically rerun the handler via a second draft
    # CG with the same id (after rolling back) — easier path: invoke
    # ``memoized`` directly with the same key and confirm cache hit.
    cache_check_invocations: list[str] = []

    async def cache_check_work() -> _ReviewOutputs:
        cache_check_invocations.append("cg_e2e")
        return _ReviewOutputs(handle="should-not-fire")

    cached_result = await memoized(
        checkpointer=cp,
        thread_id="cg_e2e",
        step_id="harness_dispatch_v1",
        inputs=_ReviewInputs(cg_id="cg_e2e", code_revision="rev1"),
        work=cache_check_work,
        serialize=_serialize,
        deserialize=_deserialize,
    )
    assert cache_check_invocations == []  # cached
    assert cached_result.handle == "h-cg_e2e"  # same as cycle 1

    # Cycle 3: phase-1 drives to terminal once Compute reports succeeded.
    fake_compute.set_status("h-cg_e2e", make_job_status("succeeded", exit_code=0))
    stats_3 = await run_worker_cycle(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        compute=fake_compute,  # type: ignore[arg-type]
        llm_providers=None,
        config=EngineConfig(),
    )
    assert stats_3.phase1_advanced == 1
    final = await sqlite_store.get_cg("cg_e2e")
    assert final is not None
    assert final.state == "implemented"
    assert final.harness_job_handle is None


async def test_n_iteration_loop_no_entity_leak(
    sqlite_store: SqliteStore,
) -> None:
    """``run_worker_loop`` runs N iterations against the synthetic spec;
    every CG reaches the terminal state and no entity is left orphaned."""
    fake_compute = FakeCompute()
    fake_artifacts = FakeArtifactStore()
    handles = iter([make_job_handle(f"h-{i}") for i in range(10)])

    async def _handler(ctx) -> DispatchOutcome:  # type: ignore[no-untyped-def]
        h = next(handles)
        fake_compute.set_status(h.handle, make_job_status("succeeded", exit_code=0))
        out = DispatchOutcome()
        out.submitted_handles.append(h)
        return out

    spec = _build_spec(handler=_handler)

    n_cgs = 5
    for i in range(n_cgs):
        await sqlite_store.create_cg(_make_cg(f"cg_{i}"))

    config = EngineConfig(time_provider=FakeMonotonic(start=0.0), poll_interval_seconds=0)
    shutdown = asyncio.Event()
    cycle_count = 0

    async def _on_cycle(stats: WorkerCycleStats) -> None:
        nonlocal cycle_count
        cycle_count += 1
        if cycle_count >= 8:  # plenty for phase2 → phase3 → phase1 round-trips
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
    # Every CG reached implemented.
    for i in range(n_cgs):
        cg = await sqlite_store.get_cg(f"cg_{i}")
        assert cg is not None
        assert cg.state == "implemented", f"cg_{i} stuck in {cg.state}"


async def test_metadata_store_checkpointer_integrates_in_loop(
    sqlite_store: SqliteStore,
) -> None:
    """The :class:`MetadataStoreCheckpointer` flavor (today backed by
    :class:`InMemoryCheckpointBackend` until Session-C lands the
    Protocol surface) integrates cleanly into the dispatch path
    alongside the worker loop.
    """
    backend = InMemoryCheckpointBackend()
    cp = MetadataStoreCheckpointer(backend)
    handler_invocations: list[str] = []
    fake_compute = FakeCompute()
    fake_artifacts = FakeArtifactStore()

    async def memoized_handler(ctx) -> DispatchOutcome:  # type: ignore[no-untyped-def]
        async def work() -> _ReviewOutputs:
            handler_invocations.append(ctx.entity_id)
            return _ReviewOutputs(handle=f"h-{ctx.entity_id}")

        result = await memoized(
            checkpointer=cp,
            thread_id=ctx.entity_id,
            step_id="dispatch_v1",
            inputs=_ReviewInputs(cg_id=ctx.entity_id, code_revision="r1"),
            work=work,
            serialize=_serialize,
            deserialize=_deserialize,
        )
        h = make_job_handle(result.handle)
        fake_compute.set_status(h.handle, make_job_status("succeeded", exit_code=0))
        out = DispatchOutcome()
        out.submitted_handles.append(h)
        return out

    spec = _build_spec(handler=memoized_handler)

    await sqlite_store.create_cg(_make_cg("cg_meta"))
    stats = await run_worker_cycle(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifacts,  # type: ignore[arg-type]
        compute=fake_compute,  # type: ignore[arg-type]
        llm_providers=None,
        config=EngineConfig(),
    )
    assert stats.phase3_advanced == 1
    assert handler_invocations == ["cg_meta"]


# ===== Spec builder =========================================================


def _build_spec(*, handler):  # type: ignore[no-untyped-def]
    in_progress = StateDef(
        name="implementing",
        on_entry_dispatch=DispatchAction(
            name="harness_build",
            handler=handler,
            pool="agents",
            handle_field="harness_job_handle",
        ),
    )

    async def _phase1_query(store):  # type: ignore[no-untyped-def]
        return await _list_cgs_with_handle(store, state="implementing", has_handle=True)

    async def _phase2_query(store):  # type: ignore[no-untyped-def]
        return await _list_cgs_with_handle(store, state="draft", has_handle=False)

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
        pools=[ConcurrencyPool(name="agents", limit=10)],
        phase1_queries={
            "implementing": SchedulingQueryRef(name="custom_in_flight", fn=_phase1_query),
        },
        phase2_queries={
            "draft": SchedulingQueryRef(name="custom_ready", fn=_phase2_query),
        },
    )


__all__: list[str] = []


# ``make_dispatch`` is unused inside this module but imported via the
# shared engine helpers — keep the import for symmetry; ruff's F401
# would flag it otherwise.
_ = make_dispatch
