"""Tests for phase-1 wait-for-job + orphan detection.

Per Task 2.C1's acceptance: ``Compute.status(handle)`` returning
SUCCEEDED/FAILED triggers correct ``fires_on`` edges; ``job_outcome`` is
populated in :class:`GateContext`; cost-record hook fires on
completion if registered; orphan detection past grace window.
"""

from __future__ import annotations

from datetime import UTC, datetime

from _helpers import (
    FakeArtifactStore,
    FakeClock,
    FakeCompute,
    make_dispatch,
    make_gate,
    make_job_handle,
    make_job_status,
)
from smai_core.plugins import JobHandle, JobStatus
from smai_orchestrator.engine import (
    CostRecordContext,
    DispatchAction,
    EdgeDef,
    EngineConfig,
    EngineSpec,
    GateContext,
    GateOutcome,
    StateDef,
    phase1_step,
)
from smai_orchestrator.entities.tracking import ComparisonGroupRecord


def _three_state_phase1_spec(
    *,
    succ_gate=None,
    fail_gate=None,
) -> EngineSpec:
    """Spec with a single in-progress state and two phase-1 edges out."""
    succ_gate = succ_gate or make_gate(advance=True)
    fail_gate = fail_gate or make_gate(advance=True)
    in_progress = StateDef(
        name="implementing",
        on_entry_dispatch=DispatchAction(
            name="harness_build",
            handler=make_dispatch(handle=make_job_handle()),
            pool="agents",
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
                name="enter",
                from_state="draft",
                target_state="implementing",
                gate_rule=make_gate(advance=True),
            ),
            EdgeDef(
                name="job-succeeded",
                from_state="implementing",
                target_state="implemented",
                gate_rule=succ_gate,
                fires_on="job_succeeded",
            ),
            EdgeDef(
                name="job-failed",
                from_state="implementing",
                target_state="implementation_failed",
                gate_rule=fail_gate,
                fires_on="job_failed",
            ),
        ],
    )


async def _seed_in_progress_cg(
    store, *, handle: JobHandle | None = None, version: int = 1
) -> ComparisonGroupRecord:
    """Seed a CG already in the in-progress state for phase-1 tests."""
    cg = ComparisonGroupRecord(
        id="cg_phase1",
        proposal_id="prop_test",
        experiment_definition_id="exp_test",
        state="implementing",
        version=version,
        harness_job_handle=handle,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    return await store.create_cg(cg)


async def test_running_status_leaves_entity_in_place(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    handle = make_job_handle("h-running")
    fake_compute.set_status("h-running", make_job_status("running"))
    cg = await _seed_in_progress_cg(sqlite_store, handle=handle)
    spec = _three_state_phase1_spec()

    outcome = await phase1_step(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        config=EngineConfig(),
        record=cg,
    )

    assert outcome.status == "running"
    final = await sqlite_store.get_cg("cg_phase1")
    assert final is not None
    assert final.state == "implementing"
    assert final.version == cg.version  # no transition


async def test_succeeded_status_fires_success_edge(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    handle = make_job_handle("h-succ")
    fake_compute.set_status("h-succ", make_job_status("succeeded", exit_code=0))
    cg = await _seed_in_progress_cg(sqlite_store, handle=handle)

    captured: list[GateContext] = []

    async def succ_gate(ctx: GateContext) -> GateOutcome:
        captured.append(ctx)
        return GateOutcome(advance=True, reason="contract satisfied")

    spec = _three_state_phase1_spec(succ_gate=succ_gate)

    outcome = await phase1_step(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        config=EngineConfig(),
        record=cg,
    )

    assert outcome.status == "advanced"
    assert outcome.fired_edge is not None
    assert outcome.fired_edge.name == "job-succeeded"

    final = await sqlite_store.get_cg("cg_phase1")
    assert final is not None
    assert final.state == "implemented"
    assert final.harness_job_handle is None  # cleared as part of transition

    # `job_outcome` populated in the gate context (`05` §9 #9).
    assert len(captured) == 1
    assert captured[0].job_outcome is not None
    assert captured[0].job_outcome.state == "succeeded"
    assert captured[0].job_outcome.exit_code == 0


async def test_failed_status_fires_failure_edge(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    handle = make_job_handle("h-fail")
    fake_compute.set_status(
        "h-fail",
        make_job_status("failed", exit_code=137, failure_reason="OOM"),
    )
    cg = await _seed_in_progress_cg(sqlite_store, handle=handle)

    captured_outcomes: list[JobStatus] = []

    async def fail_gate(ctx: GateContext) -> GateOutcome:
        if ctx.job_outcome is not None:
            captured_outcomes.append(ctx.job_outcome)
        return GateOutcome(advance=True, reason="retry budget exhausted")

    spec = _three_state_phase1_spec(fail_gate=fail_gate)

    outcome = await phase1_step(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        config=EngineConfig(),
        record=cg,
    )

    assert outcome.status == "advanced"
    assert outcome.fired_edge is not None
    assert outcome.fired_edge.name == "job-failed"

    final = await sqlite_store.get_cg("cg_phase1")
    assert final is not None
    assert final.state == "implementation_failed"

    assert len(captured_outcomes) == 1
    assert captured_outcomes[0].state == "failed"
    assert captured_outcomes[0].exit_code == 137
    assert captured_outcomes[0].failure_reason == "OOM"


async def test_cancelled_and_timeout_route_through_failure_path(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """Per :data:`smai_core.plugins.JobState`: ``cancelled`` and
    ``timeout`` route through ``fires_on=job_failed`` edges (`05` §3.1)."""
    for terminal_state, handle_id in (
        ("cancelled", "h-cancel"),
        ("timeout", "h-timeout"),
    ):
        # Re-seed each iteration with a fresh entity (PK collision).
        cg = ComparisonGroupRecord(
            id=f"cg_{handle_id}",
            proposal_id="prop_test",
            experiment_definition_id="exp_test",
            state="implementing",
            version=1,
            harness_job_handle=make_job_handle(handle_id),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        cg = await sqlite_store.create_cg(cg)
        fake_compute.set_status(handle_id, make_job_status(terminal_state))  # type: ignore[arg-type]
        spec = _three_state_phase1_spec()
        outcome = await phase1_step(
            spec=spec,
            metadata_store=sqlite_store,
            artifact_store=fake_artifact_store,
            compute=fake_compute,
            config=EngineConfig(),
            record=cg,
        )
        assert outcome.status == "advanced"
        assert outcome.fired_edge is not None
        assert outcome.fired_edge.name == "job-failed", terminal_state


async def test_terminated_no_match_when_gates_block(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """If every fires_on gate returns ``advance=False`` (e.g., retry
    budget says 'wait, attempt counter not yet rolled forward'), the
    engine surfaces ``terminated_no_match`` and the entity stays put."""
    handle = make_job_handle("h-blocked")
    fake_compute.set_status("h-blocked", make_job_status("succeeded"))
    cg = await _seed_in_progress_cg(sqlite_store, handle=handle)
    spec = _three_state_phase1_spec(succ_gate=make_gate(advance=False))

    outcome = await phase1_step(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        config=EngineConfig(),
        record=cg,
    )

    assert outcome.status == "terminated_no_match"
    final = await sqlite_store.get_cg("cg_phase1")
    assert final is not None
    assert final.state == "implementing"  # not advanced


async def test_orphan_detection_resets_after_grace(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """Orphan path: handle is null AND elapsed past grace → reset to
    prior state via forward CAS (`05` §3.1 #1)."""
    cg = await _seed_in_progress_cg(sqlite_store, handle=None)  # null handle
    spec = _three_state_phase1_spec()

    # Wall-clock has advanced 1000s (past 600s default grace).
    clock = FakeClock(start=datetime(2026, 1, 1, 0, 16, 40, tzinfo=UTC))
    cfg = EngineConfig(wall_clock=clock, orphan_grace_seconds=600)

    outcome = await phase1_step(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        config=cfg,
        record=cg,
    )

    assert outcome.status == "orphan_reset"
    final = await sqlite_store.get_cg("cg_phase1")
    assert final is not None
    assert final.state == "draft"  # rolled back to ``from_state``
    # Version monotonic: started 1, forward CAS to draft → 2.
    assert final.version == 2


async def test_orphan_no_grace_leaves_entity(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """Orphan path: handle null but grace not yet elapsed → no reset."""
    cg = await _seed_in_progress_cg(sqlite_store, handle=None)
    spec = _three_state_phase1_spec()

    # Wall-clock has advanced only 30 seconds.
    clock = FakeClock(start=datetime(2026, 1, 1, 0, 0, 30, tzinfo=UTC))
    cfg = EngineConfig(wall_clock=clock, orphan_grace_seconds=600)

    outcome = await phase1_step(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        config=cfg,
        record=cg,
    )

    assert outcome.status == "orphan_no_grace"
    final = await sqlite_store.get_cg("cg_phase1")
    assert final is not None
    assert final.state == "implementing"  # still in progress
    assert final.version == cg.version  # no transition


async def test_cost_handler_fires_on_completion(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """Per `05` §3.1 #3: spec-attached cost-record handler fires after
    a phase-1 transition completes."""
    handle = make_job_handle("h-cost")
    fake_compute.set_status("h-cost", make_job_status("succeeded", exit_code=0))
    cg = await _seed_in_progress_cg(sqlite_store, handle=handle)
    spec = _three_state_phase1_spec()

    captured: list[CostRecordContext] = []

    async def cost_handler(ctx: CostRecordContext) -> None:
        captured.append(ctx)

    outcome = await phase1_step(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        config=EngineConfig(),
        record=cg,
        cost_handler=cost_handler,
    )

    assert outcome.status == "advanced"
    assert len(captured) == 1
    assert captured[0].entity_id == "cg_phase1"
    assert captured[0].job_handle.handle == "h-cost"
    assert captured[0].job_status.state == "succeeded"


async def test_phase1_skips_states_with_no_dispatch(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """Entities sitting in non-in-progress states (e.g., manually
    placed in ``draft``) are a no-op for phase-1; status returns
    ``running`` (a quiet "leave it" signal)."""
    cg = ComparisonGroupRecord(
        id="cg_no_dispatch",
        proposal_id="prop_test",
        experiment_definition_id="exp_test",
        state="draft",
        version=0,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    cg = await sqlite_store.create_cg(cg)
    spec = _three_state_phase1_spec()

    outcome = await phase1_step(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        config=EngineConfig(),
        record=cg,
    )
    assert outcome.status == "running"
