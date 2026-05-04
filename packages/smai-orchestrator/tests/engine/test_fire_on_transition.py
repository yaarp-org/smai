"""Tests for the Task 4.K2 fire-on-transition hook.

Per ``designs/smai/12-ui-process.md`` §6.4 RESOLVED 2026-05-03 (engine-
wraps): the engine layer wraps every successful ``transition_*_state``
call with an :meth:`EventChannel.fire_transition` invocation. Tests
exercise the four sites:

* phase-3 happy-path advance (``run_dispatch`` step 1)
* phase-3 forward-rollback on handler failure (``_forward_rollback``)
* phase-1 advance on job termination (``phase1_step``)
* phase-1 orphan reset (``reset_orphan``)

Plus the worker-loop heartbeat fire and a smoke test confirming that
the default :class:`NullEventChannel` produces no observable side
effects.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from _helpers import (  # type: ignore[import-not-found]
    FakeArtifactStore,
    FakeCompute,
    make_dispatch,
    make_gate,
    make_job_handle,
    make_job_status,
)
from smai_api_spec.events import StateChangeEvent, WorkerHeartbeatEvent
from smai_events import EnvelopedEvent, EventBroker, InProcessEventChannel
from smai_orchestrator.engine import (
    DispatchAction,
    EdgeDef,
    EngineConfig,
    EngineSpec,
    StateDef,
    drive_entity_phase3,
)
from smai_orchestrator.engine.phase1 import phase1_step
from smai_orchestrator.entities.tracking import ComparisonGroupRecord
from smai_orchestrator.worker.loop import run_worker_loop


def _seed_record(*, version: int = 0) -> ComparisonGroupRecord:
    return ComparisonGroupRecord(
        id="cg_fire_test",
        proposal_id="prop_fire",
        experiment_definition_id="exp_fire",
        state="draft",
        version=version,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _two_state_external_spec(*, handler) -> EngineSpec:  # type: ignore[no-untyped-def]
    action = DispatchAction(
        name="harness_build",
        handler=handler,
        pool="agents",
        handle_field="harness_job_handle",
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


def _state_change_events(broker: EventBroker) -> list[StateChangeEvent]:
    """Drain the ring buffer; return only :class:`StateChangeEvent`s."""
    items = broker.replay_since(0)
    return [
        item.event
        for item in items
        if isinstance(item, EnvelopedEvent) and isinstance(item.event, StateChangeEvent)
    ]


# ---- Phase-3 advance: fires on draft → implementing -----------------------


async def test_phase3_advance_fires_state_change_event(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    record = await sqlite_store.create_cg(_seed_record())
    handle = make_job_handle("h-fire")
    spec = _two_state_external_spec(handler=make_dispatch(handle=handle))

    broker = EventBroker()
    config = EngineConfig(event_channel=InProcessEventChannel(broker))

    outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=config,
        record=record,
    )

    assert outcome.status == "advanced"
    events = _state_change_events(broker)
    # Exactly one event — step 3 (handle write at same state) must not
    # emit a second event.
    assert len(events) == 1
    event = events[0]
    assert event.kind == "comparison_group"  # "cg" → "comparison_group"
    assert event.id == "cg_fire_test"
    assert event.from_state == "draft"
    assert event.to_state == "implementing"


# ---- Phase-3 rollback: fires on implementing → draft ----------------------


async def test_phase3_rollback_fires_event_for_state_reset(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    record = await sqlite_store.create_cg(_seed_record())
    spec = _two_state_external_spec(handler=make_dispatch(raises=RuntimeError("substrate down")))

    broker = EventBroker()
    config = EngineConfig(event_channel=InProcessEventChannel(broker))

    outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=config,
        record=record,
    )

    assert outcome.status == "dispatch_failed_rolled_back"
    events = _state_change_events(broker)
    # Two events: draft→implementing (CAS step 1), then
    # implementing→draft (forward-rollback after handler failure).
    assert len(events) == 2
    assert (events[0].from_state, events[0].to_state) == ("draft", "implementing")
    assert (events[1].from_state, events[1].to_state) == ("implementing", "draft")


# ---- Phase-1 advance: fires on implementing → implemented ----------------


async def test_phase1_advance_fires_event(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """A succeeded job triggers a ``job_succeeded`` edge transition."""
    handle = make_job_handle("h-phase1")
    record = await sqlite_store.create_cg(
        ComparisonGroupRecord(
            id="cg_phase1",
            proposal_id="prop_phase1",
            experiment_definition_id="exp_phase1",
            state="implementing",
            version=0,
            harness_job_handle=handle,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )

    fake_compute.set_status(handle.handle, make_job_status("succeeded", exit_code=0))

    spec = EngineSpec(
        entity_kind="cg",
        initial_state="draft",
        states=[
            StateDef(name="draft"),
            StateDef(
                name="implementing",
                on_entry_dispatch=DispatchAction(
                    name="harness_build",
                    handler=make_dispatch(),
                    pool="agents",
                    handle_field="harness_job_handle",
                ),
            ),
            StateDef(name="implemented", is_terminal=True),
        ],
        edges=[
            EdgeDef(
                name="job_done",
                from_state="implementing",
                target_state="implemented",
                gate_rule=make_gate(advance=True),
                fires_on="job_succeeded",
            ),
        ],
    )

    broker = EventBroker()
    config = EngineConfig(event_channel=InProcessEventChannel(broker))

    outcome = await phase1_step(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        config=config,
        record=record,
    )

    assert outcome.status == "advanced"
    events = _state_change_events(broker)
    assert len(events) == 1
    event = events[0]
    assert event.kind == "comparison_group"
    assert event.from_state == "implementing"
    assert event.to_state == "implemented"


# ---- Phase-1 orphan reset: fires implementing → draft --------------------


async def test_phase1_orphan_reset_fires_event(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """An orphaned in-progress entity (null handle past grace) fires a
    state_change event for the rollback (``implementing → draft``)."""
    seed_time = datetime(2026, 1, 1, tzinfo=UTC)
    record = await sqlite_store.create_cg(
        ComparisonGroupRecord(
            id="cg_orphan",
            proposal_id="prop_orphan",
            experiment_definition_id="exp_orphan",
            state="implementing",
            version=0,
            harness_job_handle=None,
            created_at=seed_time,
            updated_at=seed_time,
        )
    )

    spec = EngineSpec(
        entity_kind="cg",
        initial_state="draft",
        states=[
            StateDef(name="draft"),
            StateDef(
                name="implementing",
                on_entry_dispatch=DispatchAction(
                    name="harness_build",
                    handler=make_dispatch(),
                    pool="agents",
                    handle_field="harness_job_handle",
                ),
            ),
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

    broker = EventBroker()
    config = EngineConfig(
        # Grace = 1s; wall_clock pinned past the grace window.
        orphan_grace_seconds=1,
        wall_clock=lambda: seed_time + timedelta(seconds=60),
        event_channel=InProcessEventChannel(broker),
    )

    outcome = await phase1_step(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        config=config,
        record=record,
    )

    assert outcome.status == "orphan_reset"
    events = _state_change_events(broker)
    assert len(events) == 1
    event = events[0]
    assert event.kind == "comparison_group"
    assert event.from_state == "implementing"
    assert event.to_state == "draft"


# ---- Worker-loop heartbeat ------------------------------------------------


async def test_worker_loop_fires_heartbeat_per_cycle(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """The worker loop's per-cycle heartbeat (`12` §6.4) reaches the
    broker. Two cycles → two heartbeats with monotonic ``cycle_id``."""
    spec = EngineSpec(
        entity_kind="cg",
        initial_state="draft",
        states=[StateDef(name="draft", is_terminal=True)],
        edges=[],
    )
    broker = EventBroker()
    shutdown = asyncio.Event()
    cycles_done = 0

    async def _on_cycle(stats) -> None:  # type: ignore[no-untyped-def]
        del stats
        nonlocal cycles_done
        cycles_done += 1
        if cycles_done >= 2:
            shutdown.set()

    config = EngineConfig(
        poll_interval_seconds=0,
        event_channel=InProcessEventChannel(broker),
    )
    await run_worker_loop(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm_providers=None,
        config=config,
        shutdown_event=shutdown,
        on_cycle_complete=_on_cycle,
    )

    items = broker.replay_since(0)
    heartbeats = [
        item.event
        for item in items
        if isinstance(item, EnvelopedEvent) and isinstance(item.event, WorkerHeartbeatEvent)
    ]
    assert len(heartbeats) >= 2
    assert [h.cycle_id for h in heartbeats[:2]] == [1, 2]
    assert [h.cycles_processed for h in heartbeats[:2]] == [1, 2]


# ---- Default channel: silent --------------------------------------------


async def test_default_event_channel_is_silent(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """The default :class:`NullEventChannel` produces no observable
    side effects — the existing engine tests are the bulk acceptance,
    this is a smoke test that an :class:`EngineConfig()` without
    overrides is event-silent."""
    record = await sqlite_store.create_cg(_seed_record())
    spec = _two_state_external_spec(handler=make_dispatch(handle=make_job_handle("h-x")))

    outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=EngineConfig(),
        record=record,
    )
    assert outcome.status == "advanced"
