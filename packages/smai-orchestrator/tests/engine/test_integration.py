"""End-to-end integration test against the real :class:`SqliteStore`.

Per Task 2.C1's acceptance: synthetic-pipeline-spec test runs through the
engine end-to-end against a real :class:`SqliteStore` (Task 2.A2's
canonical fixture). Demonstrates: phase-3 dispatch through write-first
ordering with a transition_log entry written; phase-1 polling driving a
live job to a terminal verdict; orphan detection past grace; rollback on
:class:`Compute.submit` failure preserves version monotonicity.
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
from smai_orchestrator.engine import (
    DispatchAction,
    EdgeDef,
    EngineConfig,
    EngineSpec,
    StateDef,
    drive_entity_phase3,
    phase1_step,
)
from smai_orchestrator.entities.tracking import ComparisonGroupRecord


def _full_e2e_spec(*, dispatch_handler) -> EngineSpec:
    """Spec covering: draft → implementing → implemented (success path)
    plus implementing → implementation_failed (failure path) plus the
    rollback edge from the same dispatch.
    """
    in_progress = StateDef(
        name="implementing",
        on_entry_dispatch=DispatchAction(
            name="harness_build",
            handler=dispatch_handler,
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
    )


async def _create_initial_cg(store, *, cg_id: str = "cg_e2e") -> ComparisonGroupRecord:
    cg = ComparisonGroupRecord(
        id=cg_id,
        proposal_id="prop_e2e",
        experiment_definition_id="exp_e2e",
        state="draft",
        version=0,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    return await store.create_cg(cg)


async def test_end_to_end_happy_path(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """Full draft → implementing → implemented path against real SqliteStore."""
    handle = make_job_handle("h-e2e-happy")
    fake_compute.set_status("h-e2e-happy", make_job_status("succeeded", exit_code=0))

    spec = _full_e2e_spec(dispatch_handler=make_dispatch(handle=handle))
    cfg = EngineConfig()

    cg = await _create_initial_cg(sqlite_store)

    # ---- Phase-3: drive draft → implementing (with dispatch) ------------
    p3_outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=cfg,
        record=cg,
    )
    assert p3_outcome.status == "advanced"

    after_p3 = await sqlite_store.get_cg("cg_e2e")
    assert after_p3 is not None
    assert after_p3.state == "implementing"
    assert after_p3.harness_job_handle == handle
    assert after_p3.version == 2  # state CAS + handle CAS

    # ---- Phase-1: drive implementing → implemented (job succeeded) -----
    p1_outcome = await phase1_step(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        config=cfg,
        record=after_p3,
    )
    assert p1_outcome.status == "advanced"
    assert p1_outcome.fired_edge is not None
    assert p1_outcome.fired_edge.name == "job-succeeded"

    final = await sqlite_store.get_cg("cg_e2e")
    assert final is not None
    assert final.state == "implemented"
    assert final.harness_job_handle is None
    assert final.version == 3  # state CAS + handle CAS + final CAS


async def test_end_to_end_compute_submit_failure_rolls_back(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """Substrate failure during Compute.submit → rollback path against
    real SqliteStore (`05` §1.4). Verifies version monotonicity."""
    spec = _full_e2e_spec(
        dispatch_handler=make_dispatch(raises=RuntimeError("compute substrate down"))
    )
    cfg = EngineConfig()

    cg = await _create_initial_cg(sqlite_store, cg_id="cg_e2e_fail")

    outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=cfg,
        record=cg,
    )

    assert outcome.status == "dispatch_failed_rolled_back"

    final = await sqlite_store.get_cg("cg_e2e_fail")
    assert final is not None
    # Rolled back; version monotonic.
    assert final.state == "draft"
    assert final.version == 2  # forward CAS to implementing + forward CAS back to draft


async def test_end_to_end_orphan_detection_past_grace(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """Orphan-grace path against real SqliteStore (`05` §3.1).

    Setup: an entity is forward-CAS'd to ``implementing`` but the worker
    dies before recording the handle (simulated by directly seeding an
    ``implementing`` row with ``harness_job_handle=None``). Phase-1 with
    a fake-clock advanced past the grace window resets it back to
    ``draft``.
    """
    # Seed the orphan directly.
    orphan = ComparisonGroupRecord(
        id="cg_orphan",
        proposal_id="prop_orphan",
        experiment_definition_id="exp_orphan",
        state="implementing",
        version=1,
        harness_job_handle=None,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    orphan = await sqlite_store.create_cg(orphan)

    spec = _full_e2e_spec(
        dispatch_handler=make_dispatch(handle=make_job_handle("h-orphan-recovery"))
    )

    # Wall-clock has advanced past grace.
    clock = FakeClock(start=datetime(2026, 1, 1, 1, 0, 0, tzinfo=UTC))
    cfg = EngineConfig(wall_clock=clock, orphan_grace_seconds=600)

    outcome = await phase1_step(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        config=cfg,
        record=orphan,
    )
    assert outcome.status == "orphan_reset"

    final = await sqlite_store.get_cg("cg_orphan")
    assert final is not None
    assert final.state == "draft"
    # Forward CAS bumped version 1 → 2.
    assert final.version == 2


async def test_end_to_end_full_lifecycle_with_orphan_recovery(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """Combined scenario: dispatch fails → rollback → re-dispatch
    succeeds → phase-1 success.

    This validates the full v1-style "recover from a transient substrate
    failure" workflow against the real SqliteStore. Doesn't exercise the
    transition_log table directly (Session-C-pending CRUD per Task 2.A2's
    status note); the transition shape is verified through the post-CAS
    record state.
    """
    cfg = EngineConfig()
    handle = make_job_handle("h-e2e-recover")

    cg = await _create_initial_cg(sqlite_store, cg_id="cg_e2e_recover")

    # Attempt 1: handler raises; entity rolls back.
    spec_fail = _full_e2e_spec(
        dispatch_handler=make_dispatch(raises=RuntimeError("transient outage"))
    )
    o1 = await drive_entity_phase3(
        spec=spec_fail,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=cfg,
        record=cg,
    )
    assert o1.status == "dispatch_failed_rolled_back"
    cg_v2 = await sqlite_store.get_cg("cg_e2e_recover")
    assert cg_v2 is not None
    assert cg_v2.state == "draft"
    assert cg_v2.version == 2

    # Attempt 2: handler succeeds; entity advances.
    fake_compute.set_status("h-e2e-recover", make_job_status("succeeded", exit_code=0))
    spec_ok = _full_e2e_spec(dispatch_handler=make_dispatch(handle=handle))
    o2 = await drive_entity_phase3(
        spec=spec_ok,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=cfg,
        record=cg_v2,
    )
    assert o2.status == "advanced"
    cg_v4 = await sqlite_store.get_cg("cg_e2e_recover")
    assert cg_v4 is not None
    assert cg_v4.state == "implementing"
    assert cg_v4.harness_job_handle == handle
    assert cg_v4.version == 4

    # Phase-1 drives to terminal.
    o3 = await phase1_step(
        spec=spec_ok,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        config=cfg,
        record=cg_v4,
    )
    assert o3.status == "advanced"
    final = await sqlite_store.get_cg("cg_e2e_recover")
    assert final is not None
    assert final.state == "implemented"
    assert final.version == 5


def _spec_with_gate_reasons(*, dispatch_handler) -> EngineSpec:
    """Same shape as :func:`_full_e2e_spec` but the gates carry a
    ``reason`` so we can assert it lands in the ``transition_log`` row."""
    in_progress = StateDef(
        name="implementing",
        on_entry_dispatch=DispatchAction(
            name="harness_build",
            handler=dispatch_handler,
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
                name="advance",
                from_state="draft",
                target_state="implementing",
                gate_rule=make_gate(advance=True, reason="ready to build"),
            ),
            EdgeDef(
                name="job-failed",
                from_state="implementing",
                target_state="implementation_failed",
                gate_rule=make_gate(advance=True, reason="harness build failed"),
                fires_on="job_failed",
            ),
        ],
    )


async def _read_transition_log(store, *, entity_id: str):  # type: ignore[no-untyped-def]
    from typing import Any as _Any  # noqa: PLC0415

    from sqlalchemy import text  # noqa: PLC0415

    async with await store.transaction() as tx:
        conn: _Any = tx.connection
        result = await conn.execute(
            text(
                "SELECT from_state, to_state, edge_name, worker_id, gate_outcome_reason "
                "FROM transition_log WHERE entity_kind = 'cg' AND entity_id = :i "
                "ORDER BY occurred_at, rowid"
            ),
            {"i": entity_id},
        )
        return result.mappings().all()


async def test_phase3_transition_log_records_gate_reason(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """A phase-3 dispatch transition records the firing gate's
    :attr:`GateOutcome.reason` in the ``transition_log`` row (item 3 of
    the observability follow-up — the engine threads ``gate_outcome_reason``
    through ``transition_state``)."""
    handle = make_job_handle("h-translog-reason")
    spec = _spec_with_gate_reasons(dispatch_handler=make_dispatch(handle=handle))
    cfg = EngineConfig()
    cg = await _create_initial_cg(sqlite_store, cg_id="cg_translog_reason")

    outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=cfg,
        record=cg,
        worker_id="worker-translog",
    )
    assert outcome.status == "advanced"

    rows = await _read_transition_log(sqlite_store, entity_id="cg_translog_reason")
    # Step-1 (draft → implementing) carries the gate reason; the step-3
    # handle-recording write is a same-state field-only update and is NOT
    # logged as a transition.
    step1 = next(r for r in rows if r["to_state"] == "implementing")
    assert step1["from_state"] == "draft"
    assert step1["edge_name"] == "advance"
    assert step1["worker_id"] == "worker-translog"
    assert step1["gate_outcome_reason"] == "ready to build"
    assert all(r["to_state"] != "implementing" or r["from_state"] == "draft" for r in rows)


async def test_phase1_transition_log_records_gate_reason(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """A phase-1 job-terminal transition records the firing gate's reason."""
    handle = make_job_handle("h-translog-p1")
    spec = _spec_with_gate_reasons(dispatch_handler=make_dispatch(handle=handle))
    cfg = EngineConfig()
    cg = await _create_initial_cg(sqlite_store, cg_id="cg_translog_p1")

    o1 = await drive_entity_phase3(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=cfg,
        record=cg,
    )
    assert o1.status == "advanced"
    cg_v = await sqlite_store.get_cg("cg_translog_p1")
    assert cg_v is not None and cg_v.state == "implementing"

    fake_compute.set_status("h-translog-p1", make_job_status("failed", exit_code=1))
    o2 = await phase1_step(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        config=cfg,
        record=cg_v,
    )
    assert o2.status == "advanced"

    rows = await _read_transition_log(sqlite_store, entity_id="cg_translog_p1")
    failed_row = next(r for r in rows if r["to_state"] == "implementation_failed")
    assert failed_row["from_state"] == "implementing"
    assert failed_row["edge_name"] == "job-failed"
    assert failed_row["gate_outcome_reason"] == "harness build failed"
