"""End-to-end integration test — Task 2.C3 acceptance criterion.

Per ``implementation_plan.md`` §3.3 Task 2.C3:

    A `RuntimeConfig` constructed in code instantiates all four plugins,
    registers a synthetic spec, and runs to completion through the
    engine.

The synthetic spec is a 3-state CG pipeline: ``draft → implementing →
implemented`` with a ``job_failed`` failure-edge to
``implementation_failed``. The dispatch handler returns a fake job
handle; the fake :class:`Compute` reports ``succeeded`` so phase-1
drives the entity to its terminal.

Plugin instantiation:

* :class:`MetadataStore` — real :class:`SqliteStore` (in-memory).
* :class:`ArtifactStore` — real :class:`LocalFsStore` rooted at
  ``tmp_path``.
* :class:`Compute` — fake (avoid Docker dep in CI).
* :class:`LlmProvider` — fake (avoid Bedrock dep in CI).

The first two go through entry-point discovery (full RuntimeConfig
flow); the last two go through :class:`PluginOverrides` (no Docker /
Bedrock available in test envs).
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from smai_core.plugins import EntityKind
from smai_orchestrator.engine import (
    ConcurrencyPool,
    DispatchAction,
    DispatchOutcome,
    EdgeDef,
    EngineConfig,
    SchedulingQueryRef,
    StateDef,
)
from smai_orchestrator.entities.tracking import ComparisonGroupRecord
from smai_orchestrator.runtime import (
    DEFAULT_TASK_ROLES,
    PipelineSpec,
    PluginOverrides,
    PluginSelection,
    RuntimeConfig,
    get_pipeline_spec,
    instantiate_plugins,
    register_pipeline_spec,
)
from smai_orchestrator.worker.loop import (
    WorkerCycleStats,
    run_worker_loop,
)

_ENGINE_TESTS_DIR = Path(__file__).resolve().parent.parent / "engine"
sys.path.insert(0, str(_ENGINE_TESTS_DIR))

from _helpers import (  # type: ignore[import-not-found] # noqa: E402
    FakeCompute,
    FakeMonotonic,
    make_gate,
    make_job_handle,
    make_job_status,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runtime_fakes import FakeLlmProvider  # type: ignore[import-not-found] # noqa: E402


def _build_synthetic_spec(*, name: str, dispatch_handler) -> PipelineSpec:  # type: ignore[no-untyped-def]
    in_progress = StateDef(
        name="implementing",
        on_entry_dispatch=DispatchAction(
            name="harness_build",
            handler=dispatch_handler,
            pool="agents",
            handle_field="harness_job_handle",
        ),
    )

    async def _phase2_query(store):  # type: ignore[no-untyped-def]
        page = await store.get_ready_for_harness_build(limit=100, cursor=None)
        return list(page.items)

    async def _phase1_query(store):  # type: ignore[no-untyped-def]
        page = await store.get_in_flight_harness_build(limit=100, cursor=None)
        return list(page.items)

    return PipelineSpec(
        name=name,
        entity_kind=cast(EntityKind, "cg"),
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
        pools=[ConcurrencyPool(name="agents", limit=4)],
        scheduling_queries={
            "draft": SchedulingQueryRef(name="ready_harness", fn=_phase2_query),
            "implementing": SchedulingQueryRef(name="in_flight_harness", fn=_phase1_query),
        },
    )


def _make_cg(cg_id: str) -> ComparisonGroupRecord:
    return ComparisonGroupRecord(
        id=cg_id,
        proposal_id=f"prop_{cg_id}",
        experiment_definition_id=f"exp_{cg_id}",
        state="draft",
        version=0,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


async def test_runtime_config_drives_synthetic_spec_to_completion(
    tmp_path: Path,
) -> None:
    """The full Task 2.C3 acceptance — construct RuntimeConfig, register
    a synthetic spec, instantiate the four plugins, run the worker loop
    against real SqliteStore + LocalFsStore (discovery flow) plus fake
    Compute + LLM (override flow), and verify the entity reaches
    terminal.
    """
    # ----- Build the synthetic spec and register it -----
    fake_compute = FakeCompute()
    submitted_handles: list[str] = []

    async def _handler(ctx) -> DispatchOutcome:  # type: ignore[no-untyped-def]
        handle = make_job_handle(f"h-{ctx.entity_id}")
        submitted_handles.append(handle.handle)
        # Pre-set the status the fake compute will report so phase-1
        # drives the entity to terminal on the next cycle.
        fake_compute.set_status(handle.handle, make_job_status("succeeded", exit_code=0))
        out = DispatchOutcome()
        out.submitted_handles.append(handle)
        return out

    spec = _build_synthetic_spec(name="acceptance_spec", dispatch_handler=_handler)
    register_pipeline_spec(spec)
    assert get_pipeline_spec("acceptance_spec") is spec

    # ----- Construct RuntimeConfig -----
    config = RuntimeConfig(
        engine=EngineConfig(
            time_provider=FakeMonotonic(start=0.0),
            poll_interval_seconds=0,
        ),
        plugins=PluginSelection(
            llm_provider="bedrock",  # not consumed — overridden below
            metadata_store="sqlite",
            artifact_store="localfs",
            compute="localgpu",  # not consumed — overridden below
            metadata_store_config={"uri": "sqlite+aiosqlite:///:memory:"},
            artifact_store_config={"root": str(tmp_path / "artifacts")},
        ),
        pipelines=["acceptance_spec"],
    )

    # ----- Plugin instantiation: real for storage, fake for compute / LLM -----
    fake_llm = FakeLlmProvider()
    overrides = PluginOverrides(
        compute=fake_compute,  # type: ignore[arg-type]
        llm_providers=dict.fromkeys(DEFAULT_TASK_ROLES, fake_llm),  # type: ignore[arg-type]
    )

    async with instantiate_plugins(config.plugins, overrides=overrides) as plugins:
        # Acceptance: plugins.metadata_store is the discovered SqliteStore.
        from smai_artifacts_localfs import LocalFsStore
        from smai_store_sqlite import SqliteStore

        assert isinstance(plugins.metadata_store, SqliteStore)
        assert isinstance(plugins.artifact_store, LocalFsStore)
        assert plugins.compute is fake_compute
        for role in DEFAULT_TASK_ROLES:
            assert plugins.llm_providers[role] is fake_llm

        # ----- Seed two CGs and drive the spec to completion -----
        n_cgs = 2
        for i in range(n_cgs):
            await plugins.metadata_store.create_cg(_make_cg(f"cg_{i}"))

        shutdown = asyncio.Event()
        cycle_count = 0

        async def _on_cycle(stats: WorkerCycleStats) -> None:
            nonlocal cycle_count
            cycle_count += 1
            # Phase-2 dispatches in cycle 1; phase-1 advances in cycle 2.
            # Add slack for ordering.
            if cycle_count >= 4:
                shutdown.set()

        engine_spec = get_pipeline_spec("acceptance_spec").engine_spec()
        await run_worker_loop(
            spec=engine_spec,
            metadata_store=plugins.metadata_store,
            artifact_store=plugins.artifact_store,
            compute=plugins.compute,
            llm_providers=plugins.llm_providers,
            config=config.engine,
            shutdown_event=shutdown,
            on_cycle_complete=_on_cycle,
        )

        # ----- Assert: every CG reached the terminal state -----
        for i in range(n_cgs):
            cg = await plugins.metadata_store.get_cg(f"cg_{i}")
            assert cg is not None, f"cg_{i} missing"
            assert cg.state == "implemented", (
                f"cg_{i} stuck in state {cg.state} after {cycle_count} cycles"
            )
            assert cg.harness_job_handle is None  # cleared on phase-1 advance

        # Both handlers fired at least once each.
        assert {h.split("-")[1] for h in submitted_handles} == {"cg_0", "cg_1"}


async def test_runtime_config_with_overrides_for_all_four_plugins(
    tmp_path: Path,
) -> None:
    """End-to-end with all four plugins override-supplied — exercises the
    "discovery is bypassed" path. Useful as a Tier B integrator's escape
    hatch.
    """
    fake_compute = FakeCompute()
    fake_llm = FakeLlmProvider()

    async def _handler(ctx) -> DispatchOutcome:  # type: ignore[no-untyped-def]
        handle = make_job_handle(f"h-{ctx.entity_id}")
        fake_compute.set_status(handle.handle, make_job_status("succeeded", exit_code=0))
        out = DispatchOutcome()
        out.submitted_handles.append(handle)
        return out

    spec = _build_synthetic_spec(name="all_override_spec", dispatch_handler=_handler)
    register_pipeline_spec(spec)

    # Use real SqliteStore + LocalFsStore via overrides (constructed
    # outside instantiate_plugins) — the test owns lifecycle.
    from smai_artifacts_localfs import LocalFsStore
    from smai_store_sqlite import SqliteStore

    store = SqliteStore("sqlite+aiosqlite:///:memory:")
    await store.migrate()
    artifacts = LocalFsStore(tmp_path / "artifacts")

    selection = PluginSelection(
        llm_provider="ghost-llm",
        metadata_store="ghost-store",
        artifact_store="ghost-arts",
        compute="ghost-compute",
    )
    overrides = PluginOverrides(
        metadata_store=store,
        artifact_store=artifacts,  # type: ignore[arg-type]
        compute=fake_compute,  # type: ignore[arg-type]
        llm_providers=dict.fromkeys(DEFAULT_TASK_ROLES, fake_llm),  # type: ignore[arg-type]
    )

    try:
        async with instantiate_plugins(selection, overrides=overrides) as plugins:
            await plugins.metadata_store.create_cg(_make_cg("cg_x"))
            shutdown = asyncio.Event()
            cycles = 0

            async def _on_cycle(stats: WorkerCycleStats) -> None:
                nonlocal cycles
                cycles += 1
                if cycles >= 4:
                    shutdown.set()

            await run_worker_loop(
                spec=get_pipeline_spec("all_override_spec").engine_spec(),
                metadata_store=plugins.metadata_store,
                artifact_store=plugins.artifact_store,
                compute=plugins.compute,
                llm_providers=plugins.llm_providers,
                config=EngineConfig(
                    time_provider=FakeMonotonic(start=0.0),
                    poll_interval_seconds=0,
                ),
                shutdown_event=shutdown,
                on_cycle_complete=_on_cycle,
            )

            cg = await plugins.metadata_store.get_cg("cg_x")
            assert cg is not None
            assert cg.state == "implemented"
    finally:
        await store.dispose()
