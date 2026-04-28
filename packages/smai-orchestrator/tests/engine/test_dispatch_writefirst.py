"""Unit + integration tests for write-first dispatch ordering.

Per Task 2.C1's acceptance: write-first ordering recovers from a
simulated `Compute.submit` failure (entity rolls back to prior state);
crash-between-steps simulation surfaces as orphan-detection in
:mod:`test_phase1`. This file focuses on the dispatch-side write-first
sequence (steps 1 / 2 / 3), exercising the real :class:`SqliteStore`
because the version-monotonicity invariant is most legible against a
real CAS substrate.
"""

from __future__ import annotations

from datetime import UTC, datetime

from _helpers import (
    FakeArtifactStore,
    FakeCompute,
    make_dispatch,
    make_gate,
    make_job_handle,
)
from smai_orchestrator.engine import (
    DispatchAction,
    EdgeDef,
    EngineConfig,
    EngineSpec,
    StateDef,
    drive_entity_phase3,
)
from smai_orchestrator.entities.tracking import ComparisonGroupRecord


async def _seed_cg(store, *, version: int = 0) -> ComparisonGroupRecord:
    """Seed a single CG into the store for write-first tests."""
    cg = ComparisonGroupRecord(
        id="cg_writefirst",
        proposal_id="prop_test",
        experiment_definition_id="exp_test",
        state="draft",
        version=version,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    return await store.create_cg(cg)


def _spec_with_external_dispatch(
    *,
    handler,
    handle_field: str = "harness_job_handle",
) -> EngineSpec:
    """Two-state spec: ``draft → implementing`` with an external dispatch."""
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


async def test_happy_path_advances_and_records_handle(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """All three steps fire: CAS state → submit → CAS handle (`05` §1.4)."""
    cg = await _seed_cg(sqlite_store)
    handle = make_job_handle("h-happy")
    spec = _spec_with_external_dispatch(handler=make_dispatch(handle=handle))

    outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=EngineConfig(),
        record=cg,
    )

    assert outcome.status == "advanced"
    # Inspect end state.
    final = await sqlite_store.get_cg("cg_writefirst")
    assert final is not None
    assert final.state == "implementing"
    # Two CAS writes (state + handle); version went 0 → 1 → 2.
    assert final.version == 2
    assert final.harness_job_handle == handle


async def test_compute_submit_failure_rolls_back_state(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """`05` §1.4 rollback path: handler raises → forward CAS reset.

    Version monotonicity preserved: the rollback is a *forward*
    increment, not a decrement, per the design's load-bearing invariant.
    """
    cg = await _seed_cg(sqlite_store)
    spec = _spec_with_external_dispatch(
        handler=make_dispatch(raises=RuntimeError("substrate down"))
    )

    outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=EngineConfig(),
        record=cg,
    )

    assert outcome.status == "dispatch_failed_rolled_back"
    assert outcome.error is not None
    assert "substrate down" in outcome.error

    final = await sqlite_store.get_cg("cg_writefirst")
    assert final is not None
    # Rolled back to ``from_state`` of the edge that fired.
    assert final.state == "draft"
    # Version monotonic: started at 0, transitioned to implementing
    # (+1 → 1), forward-rolled-back to draft (+1 → 2).
    assert final.version == 2
    # No handle recorded.
    assert final.harness_job_handle is None


async def test_handler_returning_error_rolls_back(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """Same rollback shape, but the handler returns ``DispatchOutcome`` with
    a non-None ``error`` rather than raising."""
    cg = await _seed_cg(sqlite_store)
    spec = _spec_with_external_dispatch(handler=make_dispatch(error="precondition not met"))

    outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=EngineConfig(),
        record=cg,
    )

    assert outcome.status == "dispatch_failed_rolled_back"
    assert outcome.error == "precondition not met"
    final = await sqlite_store.get_cg("cg_writefirst")
    assert final is not None
    assert final.state == "draft"
    assert final.version == 2
    assert final.harness_job_handle is None


async def test_external_handler_with_no_handle_rolls_back(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """Engine invariant: external dispatches MUST produce a handle for
    phase-1 polling. A handler returning empty ``submitted_handles``
    while declaring an external dispatch site is treated as a handler
    error and rolled back (else the entity gets stuck in the
    in-progress state with a null handle, awaiting orphan-grace)."""
    cg = await _seed_cg(sqlite_store)
    spec = _spec_with_external_dispatch(handler=make_dispatch())  # no handle

    outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=EngineConfig(),
        record=cg,
    )

    assert outcome.status == "dispatch_failed_rolled_back"
    final = await sqlite_store.get_cg("cg_writefirst")
    assert final is not None
    assert final.state == "draft"


async def test_inline_dispatch_completes_in_one_cas(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """Inline dispatches (handle_field=None) require only step 1; the
    handler runs synchronously and the transition is considered complete
    after it returns successfully (`05` §1.4)."""
    cg = await _seed_cg(sqlite_store)
    inline_action = DispatchAction(
        name="inline_review",
        handler=make_dispatch(),  # no handle
        pool="agents",
        # handle_field=None default — inline dispatch
    )
    spec = EngineSpec(
        entity_kind="cg",
        initial_state="draft",
        states=[
            StateDef(name="draft"),
            StateDef(name="implementing", on_entry_dispatch=inline_action),
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

    outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=EngineConfig(),
        record=cg,
    )

    assert outcome.status == "advanced"
    final = await sqlite_store.get_cg("cg_writefirst")
    assert final is not None
    assert final.state == "implementing"
    assert final.version == 1  # only one CAS, not two


async def test_no_match_keeps_entity_in_state(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """When no edge passes, the entity stays put (`05` §3.3)."""
    cg = await _seed_cg(sqlite_store)
    spec = EngineSpec(
        entity_kind="cg",
        initial_state="draft",
        states=[StateDef(name="draft"), StateDef(name="implementing")],
        edges=[
            EdgeDef(
                name="advance-blocked",
                from_state="draft",
                target_state="implementing",
                gate_rule=make_gate(advance=False, reason="contract not satisfied"),
            ),
        ],
    )
    outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=EngineConfig(),
        record=cg,
    )
    assert outcome.status == "no_match"
    final = await sqlite_store.get_cg("cg_writefirst")
    assert final is not None
    assert final.state == "draft"
    assert final.version == 0  # no transition, no version bump


async def test_terminal_target_advances_in_one_cas(
    sqlite_store, fake_compute: FakeCompute, fake_artifact_store: FakeArtifactStore
) -> None:
    """A target state with no on_entry_dispatch (e.g., a terminal state)
    completes in one CAS — no dispatch, no handle to record."""
    cg = await _seed_cg(sqlite_store)
    spec = EngineSpec(
        entity_kind="cg",
        initial_state="draft",
        states=[
            StateDef(name="draft"),
            StateDef(name="complete", is_terminal=True),
        ],
        edges=[
            EdgeDef(
                name="advance",
                from_state="draft",
                target_state="complete",
                gate_rule=make_gate(advance=True),
            ),
        ],
    )
    outcome = await drive_entity_phase3(
        spec=spec,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        compute=fake_compute,
        llm=None,
        config=EngineConfig(),
        record=cg,
    )
    assert outcome.status == "advanced"
    final = await sqlite_store.get_cg("cg_writefirst")
    assert final is not None
    assert final.state == "complete"
    assert final.version == 1
