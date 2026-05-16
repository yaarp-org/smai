"""Tests for the proposal pipeline-spec — Task 3.E1.

Two surfaces:

* Spec-structure tests against :func:`build_proposal_pipeline_spec`'s
  return value (no engine, no plugins).
* End-to-end driving the spec through the worker loop against a stub
  :class:`MetadataStore` (in-memory SQLite) + stub
  :class:`ArtifactStore` + stub :class:`LlmProvider`. Verifies state
  transitions, gate-rule firing, and one-to-many CG creation on
  ``registered``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _e1_fakes import (  # type: ignore[import-not-found]
    build_finalized_buffer_payload,
    make_planner_responses_for_finalize,
    make_planner_session_runner,
    make_proposal_record,
)
from _specs_fakes import StubLlmProvider  # type: ignore[import-not-found]
from smai_artifacts_localfs import LocalFsStore
from smai_orchestrator.engine.config import EngineConfig
from smai_orchestrator.specs.proposal import (
    DESIGN_PLAN_KEY_TEMPLATE,
    POOL_PROPOSAL_PIPELINE,
    POOL_PROPOSAL_PIPELINE_LIMIT,
    POOL_PROPOSAL_PIPELINE_PRIORITY,
    PROPOSAL_PIPELINE_SPEC_NAME,
    build_proposal_pipeline_spec,
)
from smai_orchestrator.worker.loop import run_worker_cycle
from smai_store_sqlite import SqliteStore


@pytest.fixture
def proposal_spec(tmp_path: Path):  # type: ignore[no-untyped-def]
    planner_llm = StubLlmProvider([])
    return build_proposal_pipeline_spec(
        workspace_root=tmp_path / "ws",
        llm_for_planner=planner_llm,  # type: ignore[arg-type]
        require_human_approval=False,  # auto-approve for the structure tests
    )


# === Structure tests ========================================================


def test_proposal_spec_name_and_entity_kind(proposal_spec) -> None:  # type: ignore[no-untyped-def]
    assert proposal_spec.name == PROPOSAL_PIPELINE_SPEC_NAME
    assert proposal_spec.entity_kind == "proposal"
    assert proposal_spec.initial_state == "proposal_submitted"


def test_proposal_spec_states_match_03_section_4_1(proposal_spec) -> None:  # type: ignore[no-untyped-def]
    """Per `03-state-machine.md` §4.1 — six states.

    proposal_submitted, designing, designed, registered, rejected,
    failed.
    """
    state_names = {s.name for s in proposal_spec.states}
    assert state_names == {
        "proposal_submitted",
        "designing",
        "designed",
        "registered",
        "rejected",
        "failed",
    }


def test_proposal_spec_terminal_states(proposal_spec) -> None:  # type: ignore[no-untyped-def]
    """`registered`, `rejected`, `failed` are terminals."""
    terminals = {s.name for s in proposal_spec.states if s.is_terminal}
    assert terminals == {"registered", "rejected", "failed"}


def test_proposal_spec_pool_per_dec_034_4(proposal_spec) -> None:  # type: ignore[no-untyped-def]
    """Single `proposal_pipeline` pool per DEC-034 #4 — limit 2, priority 30."""
    pool_names = {p.name for p in proposal_spec.pools}
    assert pool_names == {POOL_PROPOSAL_PIPELINE}
    pool = next(p for p in proposal_spec.pools if p.name == POOL_PROPOSAL_PIPELINE)
    assert pool.limit == POOL_PROPOSAL_PIPELINE_LIMIT
    assert pool.priority == POOL_PROPOSAL_PIPELINE_PRIORITY


def test_proposal_spec_edges_per_03_section_4_2(proposal_spec) -> None:  # type: ignore[no-untyped-def]
    """Edges per `03` §4.2 (table) — collapsed for the inline-planner pattern.

    The canonical `03` §4.2 lists a ``designing → designing`` retry
    edge intended for ``job_failed`` re-firing after a structural-
    soundness failure. Per Phase-3 inline-planner pattern: the retry
    happens within the planner session itself (the planner's
    ``finalize_plan`` tool returns errors so the agent self-corrects)
    rather than via an inter-state engine transition.

    Round 10: ``designing → failed`` and ``designed → failed`` (the
    registration retry exhausted terminal) are no longer manual
    ``EdgeDef``s — the engine synthesizes them from the dispatch
    actions' :class:`RetryPolicy` declarations. The ``failed`` terminal
    is reached via the synthesized edge off the dispatch state
    (``designing → failed`` from the planner dispatch's policy;
    ``registered → failed`` from the registration dispatch's policy).
    """
    by_pair = {(e.from_state, e.target_state) for e in proposal_spec.edges}
    assert ("proposal_submitted", "designing") in by_pair
    assert ("designing", "designed") in by_pair
    assert ("designed", "registered") in by_pair
    assert ("designed", "rejected") in by_pair
    # ``designing → failed`` and ``designed/registered → failed`` are
    # engine-synthesized from RetryPolicy and don't appear in the
    # manual edges list (round 10).
    designing = next(s for s in proposal_spec.states if s.name == "designing")
    assert designing.on_entry_dispatch is not None
    designing_policy = designing.on_entry_dispatch.retry_policy
    assert designing_policy is not None
    assert designing_policy.on_exhaustion_target_state == "failed"
    registered = next(s for s in proposal_spec.states if s.name == "registered")
    assert registered.on_entry_dispatch is not None
    registered_policy = registered.on_entry_dispatch.retry_policy
    assert registered_policy is not None
    assert registered_policy.on_exhaustion_target_state == "failed"


def test_proposal_spec_scheduling_queries_declared(proposal_spec) -> None:  # type: ignore[no-untyped-def]
    """Three scheduling queries — one per non-terminal state."""
    keys = set(proposal_spec.scheduling_queries.keys())
    assert keys == {"proposal_submitted", "designing", "designed"}


def test_proposal_spec_designing_has_planner_dispatch(proposal_spec) -> None:  # type: ignore[no-untyped-def]
    """Per `03` §4.3 the `designing` state's on-entry dispatch is the
    planner agent."""
    designing = next(s for s in proposal_spec.states if s.name == "designing")
    assert designing.on_entry_dispatch is not None
    assert designing.on_entry_dispatch.pool == POOL_PROPOSAL_PIPELINE
    assert designing.on_entry_dispatch.handle_field == "planner_job_handle"


def test_proposal_spec_registered_state_dispatches_registration(  # type: ignore[no-untyped-def]
    proposal_spec,
) -> None:
    """Per the proposal-spec module docstring: registration runs as the
    `registered` state's on-entry dispatch (since the engine drives
    DispatchAction from StateDef.on_entry_dispatch only)."""
    registered = next(s for s in proposal_spec.states if s.name == "registered")
    assert registered.is_terminal is True
    assert registered.on_entry_dispatch is not None
    assert "register" in registered.on_entry_dispatch.name


# === End-to-end driving =====================================================


@pytest.fixture
async def sqlite_store():  # type: ignore[no-untyped-def]
    """In-memory SqliteStore per test."""
    store = SqliteStore("sqlite+aiosqlite:///:memory:")
    await store.migrate()
    try:
        yield store
    finally:
        await store.dispose()


@pytest.fixture
def localfs(tmp_path: Path) -> LocalFsStore:
    return LocalFsStore(tmp_path / "artifacts")


@pytest.mark.asyncio
async def test_proposal_round_trip_proposal_submitted_to_registered(  # type: ignore[no-untyped-def]
    sqlite_store,
    localfs: LocalFsStore,
    tmp_path: Path,
) -> None:
    """End-to-end: proposal_submitted → designing → designed → registered.

    Exercises the spec against the stub planner that drives finalize_plan
    via canned LLM responses. Auto-approves at the human gate. Asserts
    one ComparisonGroupRecord is created per buffer CG (1:1 here; the
    one-to-many test below exercises N:1).
    """
    from smai_orchestrator.engine.types import DispatchContext  # noqa: PLC0415

    planner_responses = make_planner_responses_for_finalize(
        proposal_id="prop-rt-1",
        cg_drafts=[
            {
                "id": "cg-1",
                "factor_dimension": "augmentation",
                "factor_type": "additive",
            }
        ],
    )
    planner_llm = StubLlmProvider(planner_responses)

    spec = build_proposal_pipeline_spec(
        workspace_root=tmp_path / "ws",
        llm_for_planner=planner_llm,  # type: ignore[arg-type]
        require_human_approval=False,
    )
    config = EngineConfig(supervisor_enabled=False)

    # Submit proposal record.
    proposal = make_proposal_record(
        proposal_id="prop-rt-1",
        state="proposal_submitted",
    )
    await sqlite_store.create_proposal(proposal)

    # Drive cycles until terminal (`registered`).
    final_state: str | None = None
    for _ in range(8):
        await run_worker_cycle(
            spec=spec.engine_spec(),
            metadata_store=sqlite_store,
            artifact_store=localfs,  # type: ignore[arg-type]
            compute=_NoComputeStub(),  # type: ignore[arg-type]
            llm_providers=None,
            config=config,
        )
        rec = await sqlite_store.get_proposal("prop-rt-1")
        assert rec is not None
        if rec.state in {"registered", "rejected", "failed"}:
            final_state = rec.state
            break

    assert final_state == "registered", f"got {final_state}"

    # ≥ 1 CG was created per the buffer.
    cgs = await sqlite_store.list_cgs_for_proposal("prop-rt-1")
    assert len(cgs.items) >= 1
    cg = cgs.items[0]
    assert cg.proposal_id == "prop-rt-1"
    assert cg.state == "draft"
    # Round-9: primary id is a fresh ULID-shaped string; planner's draft
    # id is preserved on ``symbolic_id``.
    assert cg.id.startswith("cg-")
    assert len(cg.id) <= 64
    assert cg.symbolic_id == "cg-1"
    assert cg.experiment_definition_id == cg.id

    # Contract artifacts persisted to the well-known paths.
    cg_id = cg.id
    assert await localfs.exists(f"comparison-groups/{cg_id}/experiment_plan.json")
    assert await localfs.exists(f"comparison-groups/{cg_id}/harness/contract.json")
    assert await localfs.exists(f"comparison-groups/{cg_id}/validation_config.json")

    # Suppress unused-import warning.
    _ = DispatchContext


@pytest.mark.asyncio
async def test_proposal_one_to_many_cg_creation(  # type: ignore[no-untyped-def]
    sqlite_store,
    localfs: LocalFsStore,
    tmp_path: Path,
) -> None:
    """Per DEC-032 OQ2: a single proposal can produce 1..N CGs.

    Drives the planner with a buffer carrying TWO ComparisonGroups and
    confirms two ComparisonGroupRecords land in MetadataStore parented
    by the same proposal.
    """
    planner_responses = make_planner_responses_for_finalize(
        proposal_id="prop-multi",
        cg_drafts=[
            {
                "id": "cg-a",
                "factor_dimension": "augmentation",
                "factor_type": "additive",
            },
            {
                "id": "cg-b",
                "factor_dimension": "augmentation",
                "factor_type": "additive",
            },
        ],
    )
    planner_llm = StubLlmProvider(planner_responses)

    spec = build_proposal_pipeline_spec(
        workspace_root=tmp_path / "ws",
        llm_for_planner=planner_llm,  # type: ignore[arg-type]
        require_human_approval=False,
    )
    config = EngineConfig(supervisor_enabled=False)

    proposal = make_proposal_record(
        proposal_id="prop-multi",
        state="proposal_submitted",
    )
    await sqlite_store.create_proposal(proposal)

    final_state: str | None = None
    for _ in range(8):
        await run_worker_cycle(
            spec=spec.engine_spec(),
            metadata_store=sqlite_store,
            artifact_store=localfs,  # type: ignore[arg-type]
            compute=_NoComputeStub(),  # type: ignore[arg-type]
            llm_providers=None,
            config=config,
        )
        rec = await sqlite_store.get_proposal("prop-multi")
        if rec is not None and rec.state in {"registered", "rejected", "failed"}:
            final_state = rec.state
            break
    assert final_state == "registered"

    cgs = await sqlite_store.list_cgs_for_proposal("prop-multi")
    assert len(cgs.items) == 2
    # Round-9: CG primary ids are fresh ULID-shaped strings; the planner-
    # supplied draft ids live on ``symbolic_id`` instead. The 1:N
    # parentage is still verifiable via ``proposal_id`` + the symbolic
    # label.
    symbolic_ids = sorted(c.symbolic_id for c in cgs.items if c.symbolic_id is not None)
    assert symbolic_ids == ["cg-a", "cg-b"], f"got {symbolic_ids}"
    for cg in cgs.items:
        assert cg.id.startswith("cg-"), f"expected ULID-shaped id, got {cg.id!r}"
        assert len(cg.id) <= 64
        assert cg.experiment_definition_id == cg.id


@pytest.mark.asyncio
async def test_proposal_human_gate_blocks_until_approval(  # type: ignore[no-untyped-def]
    sqlite_store,
    localfs: LocalFsStore,
    tmp_path: Path,
) -> None:
    """With require_human_approval=True the proposal parks at `designed`."""
    planner_responses = make_planner_responses_for_finalize(
        proposal_id="prop-gate",
        cg_drafts=[
            {
                "id": "cg-1",
                "factor_dimension": "augmentation",
                "factor_type": "additive",
            }
        ],
    )
    planner_llm = StubLlmProvider(planner_responses)
    spec = build_proposal_pipeline_spec(
        workspace_root=tmp_path / "ws",
        llm_for_planner=planner_llm,  # type: ignore[arg-type]
        require_human_approval=True,
    )
    config = EngineConfig(supervisor_enabled=False)

    await sqlite_store.create_proposal(
        make_proposal_record(proposal_id="prop-gate", state="proposal_submitted")
    )

    # Drive cycles until the proposal lands in `designed` (parked).
    parked_state: str | None = None
    for _ in range(6):
        await run_worker_cycle(
            spec=spec.engine_spec(),
            metadata_store=sqlite_store,
            artifact_store=localfs,  # type: ignore[arg-type]
            compute=_NoComputeStub(),  # type: ignore[arg-type]
            llm_providers=None,
            config=config,
        )
        rec = await sqlite_store.get_proposal("prop-gate")
        if rec is not None and rec.state == "designed":
            parked_state = rec.state
            break
    assert parked_state == "designed"

    # Now write user_decision="approved" (simulating the API surface).
    rec = await sqlite_store.get_proposal("prop-gate")
    assert rec is not None
    from datetime import UTC, datetime  # noqa: PLC0415

    await sqlite_store.transition_proposal_state(
        "prop-gate",
        rec.version,
        "designed",
        user_decision="approved",
        user_decided_at=datetime.now(UTC),
    )

    # Drive more cycles — the gate should now fire.
    final_state: str | None = None
    for _ in range(6):
        await run_worker_cycle(
            spec=spec.engine_spec(),
            metadata_store=sqlite_store,
            artifact_store=localfs,  # type: ignore[arg-type]
            compute=_NoComputeStub(),  # type: ignore[arg-type]
            llm_providers=None,
            config=config,
        )
        rec = await sqlite_store.get_proposal("prop-gate")
        if rec is not None and rec.state in {"registered", "rejected", "failed"}:
            final_state = rec.state
            break
    assert final_state == "registered"


@pytest.mark.asyncio
async def test_proposal_rejected_terminal(  # type: ignore[no-untyped-def]
    sqlite_store,
    localfs: LocalFsStore,
    tmp_path: Path,
) -> None:
    """User rejection routes the proposal to `rejected` terminal."""
    planner_responses = make_planner_responses_for_finalize(
        proposal_id="prop-reject",
        cg_drafts=[
            {
                "id": "cg-1",
                "factor_dimension": "augmentation",
                "factor_type": "additive",
            }
        ],
    )
    planner_llm = StubLlmProvider(planner_responses)
    spec = build_proposal_pipeline_spec(
        workspace_root=tmp_path / "ws",
        llm_for_planner=planner_llm,  # type: ignore[arg-type]
        require_human_approval=True,
    )
    config = EngineConfig(supervisor_enabled=False)
    await sqlite_store.create_proposal(
        make_proposal_record(proposal_id="prop-reject", state="proposal_submitted")
    )

    # Drive to designed.
    for _ in range(6):
        await run_worker_cycle(
            spec=spec.engine_spec(),
            metadata_store=sqlite_store,
            artifact_store=localfs,  # type: ignore[arg-type]
            compute=_NoComputeStub(),  # type: ignore[arg-type]
            llm_providers=None,
            config=config,
        )
        rec = await sqlite_store.get_proposal("prop-reject")
        if rec is not None and rec.state == "designed":
            break

    # Reject.
    from datetime import UTC, datetime  # noqa: PLC0415

    rec = await sqlite_store.get_proposal("prop-reject")
    assert rec is not None
    await sqlite_store.transition_proposal_state(
        "prop-reject",
        rec.version,
        "designed",
        user_decision="rejected",
        user_decided_at=datetime.now(UTC),
    )

    # Drive more cycles — should land in `rejected`.
    final_state: str | None = None
    for _ in range(4):
        await run_worker_cycle(
            spec=spec.engine_spec(),
            metadata_store=sqlite_store,
            artifact_store=localfs,  # type: ignore[arg-type]
            compute=_NoComputeStub(),  # type: ignore[arg-type]
            llm_providers=None,
            config=config,
        )
        rec = await sqlite_store.get_proposal("prop-reject")
        if rec is not None and rec.state in {"registered", "rejected", "failed"}:
            final_state = rec.state
            break
    assert final_state == "rejected"
    # No CGs created.
    cgs = await sqlite_store.list_cgs_for_proposal("prop-reject")
    assert len(cgs.items) == 0


@pytest.mark.asyncio
async def test_design_finalized_gate_blocks_on_unfinalized_buffer(  # type: ignore[no-untyped-def]
    sqlite_store,
    localfs: LocalFsStore,
    tmp_path: Path,
) -> None:
    """The `designing → designed` gate requires the buffer to be finalized.

    Pre-stage a partial buffer (finalized=False) at the design-plan
    artifact key; drive one cycle; assert the proposal stays in
    `designing` (the gate blocks).
    """
    planner_llm = StubLlmProvider([])
    spec = build_proposal_pipeline_spec(
        workspace_root=tmp_path / "ws",
        llm_for_planner=planner_llm,  # type: ignore[arg-type]
        require_human_approval=False,
    )
    config = EngineConfig(supervisor_enabled=False)
    proposal = make_proposal_record(proposal_id="prop-block", state="designing")
    # Synthetic in-flight handle so phase-1 polling fires.
    from smai_core.plugins import JobHandle  # noqa: PLC0415

    proposal = proposal.model_copy(
        update={"planner_job_handle": JobHandle(plugin="inline", handle="inline-prop-block")}
    )
    await sqlite_store.create_proposal(proposal)
    # Pre-stage an UNFINALIZED buffer.
    plan_key = DESIGN_PLAN_KEY_TEMPLATE.format(proposal_id="prop-block")
    payload = build_finalized_buffer_payload(
        proposal_id="prop-block",
        cg_drafts=[
            {
                "id": "cg-1",
                "factor_dimension": "augmentation",
                "factor_type": "additive",
            }
        ],
        finalized=False,
    )
    await localfs.put(plan_key, json.dumps(payload).encode("utf-8"))

    # Need a Compute stub that returns "succeeded" for the inline handle.
    compute = _NoComputeStub()
    compute.set_status("inline-prop-block", "succeeded")

    await run_worker_cycle(
        spec=spec.engine_spec(),
        metadata_store=sqlite_store,
        artifact_store=localfs,  # type: ignore[arg-type]
        compute=compute,  # type: ignore[arg-type]
        llm_providers=None,
        config=config,
    )
    rec = await sqlite_store.get_proposal("prop-block")
    assert rec is not None
    assert rec.state == "designing", f"gate should block; got {rec.state}"


# === Round-10 (replaces round-8 fix B): registration retry exhaustion ======


def test_proposal_registered_dispatch_carries_retry_policy(  # type: ignore[no-untyped-def]
    proposal_spec,
) -> None:
    """Round 10 replaces the round-8 manual ``designed → failed
    (registration retry exhausted)`` EdgeDef with a declarative
    :class:`RetryPolicy` on the ``registered`` state's dispatch action.
    The engine synthesizes the retry-exhausted terminal off
    ``registered`` (not off ``designed``) when the registration handler
    fails enough times.

    Pre-round-8 the approval edge came first and a registration error
    re-fired indefinitely. Round 8 patched the bug by reordering manual
    edges. Round 10 eliminates the bug class by removing the manual
    edge entirely — the synthesized terminal is always declaration-
    first-equivalent inside ``_handle_dispatch_failure``.
    """
    registered = next(s for s in proposal_spec.states if s.name == "registered")
    assert registered.on_entry_dispatch is not None
    policy = registered.on_entry_dispatch.retry_policy
    assert policy is not None
    assert policy.attempt_counter_field == "registration_attempt"
    assert policy.on_exhaustion_target_state == "failed"
    assert "registration retry budget exhausted" in policy.on_exhaustion_reason


@pytest.mark.asyncio
async def test_proposal_registration_retry_exhaustion_lands_in_failed(  # type: ignore[no-untyped-def]
    sqlite_store,
    localfs: LocalFsStore,
    tmp_path: Path,
) -> None:
    """Repeated registration failures advance the proposal to ``failed``
    (not another retry) once ``registration_attempt >= max_registration_attempts``.

    The reproduced bug: a buffer that passes ``finalize_plan`` (rules
    1..10 checked field presence) but cannot be projected to
    ``ExperimentDocument`` (e.g. ``controlled_conditions.dataset`` as a
    bare string instead of a dict) wedged the proposal in ``designed``
    forever pre-round-8. Round 8 fixed it by reordering manual edges;
    round 10 lifts the retry bookkeeping into a :class:`RetryPolicy`
    declaration so the synthesized terminal fires inside
    :func:`_handle_dispatch_failure` directly off ``registered``
    (instead of off ``designed`` after a rollback) — one fewer round-
    trip but the same final state with ``last_error`` populated.
    """
    planner_llm = StubLlmProvider([])  # no planner runs — we pre-stage everything
    spec = build_proposal_pipeline_spec(
        workspace_root=tmp_path / "ws",
        llm_for_planner=planner_llm,  # type: ignore[arg-type]
        require_human_approval=True,
        max_registration_attempts=2,
    )
    config = EngineConfig(supervisor_enabled=False)

    # Pre-stage a finalized but un-projectable buffer.
    proposal_id = "prop-reg-fail"
    plan_key = DESIGN_PLAN_KEY_TEMPLATE.format(proposal_id=proposal_id)
    payload = build_finalized_buffer_payload(
        proposal_id=proposal_id,
        cg_drafts=[
            {
                "id": "cg-1",
                "factor_dimension": "augmentation",
                "factor_type": "additive",
                "controlled_conditions": {
                    "dataset": "MNIST",  # WRONG — bare string instead of dict
                    "optimization": {"optimizer": "sgd", "lr": 0.1},
                    "seeds": [0, 1],
                },
            }
        ],
        finalized=True,
    )
    await localfs.put(plan_key, json.dumps(payload).encode("utf-8"))

    # Seed the proposal already in ``designed`` with the user-decision
    # set to "approved" so the registration-time gates fire.
    from datetime import UTC, datetime  # noqa: PLC0415

    proposal = make_proposal_record(proposal_id=proposal_id, state="designed")
    proposal = proposal.model_copy(
        update={
            "user_decision": "approved",
            "user_decided_at": datetime.now(UTC),
        }
    )
    await sqlite_store.create_proposal(proposal)

    # Drive cycles until terminal.
    final_state: str | None = None
    for _ in range(10):
        await run_worker_cycle(
            spec=spec.engine_spec(),
            metadata_store=sqlite_store,
            artifact_store=localfs,  # type: ignore[arg-type]
            compute=_NoComputeStub(),  # type: ignore[arg-type]
            llm_providers=None,
            config=config,
        )
        rec = await sqlite_store.get_proposal(proposal_id)
        assert rec is not None
        if rec.state in {"registered", "rejected", "failed"}:
            final_state = rec.state
            break

    assert final_state == "failed", f"expected failed; got {final_state}"
    final = await sqlite_store.get_proposal(proposal_id)
    assert final is not None
    # The terminal gate fired AFTER registration_attempt reached the cap.
    assert final.registration_attempt >= 2
    # ``last_error`` reflects the registration failure (set by the
    # dispatch-failure rollback before the terminal gate fired; the
    # terminal gate's CAS preserves it).
    assert final.last_error is not None
    assert "register" in final.last_error.lower() or "project" in final.last_error.lower()

    # No CGs were created (registration kept failing).
    cgs = await sqlite_store.list_cgs_for_proposal(proposal_id)
    assert len(cgs.items) == 0

    # The terminal transition is recorded in ``transition_log`` with the
    # right edge name + gate-outcome reason.
    from sqlalchemy import text  # noqa: PLC0415

    async with await sqlite_store.transaction() as tx:
        conn: Any = tx.connection
        result = await conn.execute(
            text(
                "SELECT from_state, to_state, edge_name, gate_outcome_reason "
                "FROM transition_log WHERE entity_kind = :k AND entity_id = :i "
                "ORDER BY id"
            ),
            {"k": "proposal", "i": proposal_id},
        )
        rows = [dict(r) for r in result.mappings().all()]
    # Round 10: the synthesized terminal fires off ``registered`` (the
    # dispatch state) rather than off ``designed`` (the pre-round-10
    # manual edge fired post-rollback). Same final state with the same
    # last_error, but the transition-log row is ``registered → failed``.
    terminal_rows = [
        r for r in rows if (r["from_state"], r["to_state"]) == ("registered", "failed")
    ]
    assert len(terminal_rows) == 1, f"expected one registered→failed row; got {rows}"
    assert "synthesized retry-exhausted terminal" in terminal_rows[0]["edge_name"]
    assert "registration retry budget exhausted" in (terminal_rows[0]["gate_outcome_reason"] or "")


@pytest.mark.asyncio
async def test_proposal_quiesces_after_registration_failure_terminal(  # type: ignore[no-untyped-def]
    sqlite_store,
    localfs: LocalFsStore,
    tmp_path: Path,
) -> None:
    """Once the proposal lands in ``failed`` via the registration-exhausted
    gate, further worker cycles produce no new transitions — the terminal
    is sticky."""
    planner_llm = StubLlmProvider([])
    spec = build_proposal_pipeline_spec(
        workspace_root=tmp_path / "ws",
        llm_for_planner=planner_llm,  # type: ignore[arg-type]
        require_human_approval=True,
        max_registration_attempts=2,
    )
    config = EngineConfig(supervisor_enabled=False)

    proposal_id = "prop-reg-fail-quiesce"
    plan_key = DESIGN_PLAN_KEY_TEMPLATE.format(proposal_id=proposal_id)
    payload = build_finalized_buffer_payload(
        proposal_id=proposal_id,
        cg_drafts=[
            {
                "id": "cg-1",
                "factor_dimension": "augmentation",
                "factor_type": "additive",
                "controlled_conditions": {
                    "dataset": "MNIST",  # bare-string projection failure
                    "optimization": {"optimizer": "sgd", "lr": 0.1},
                    "seeds": [0, 1],
                },
            }
        ],
        finalized=True,
    )
    await localfs.put(plan_key, json.dumps(payload).encode("utf-8"))
    from datetime import UTC, datetime  # noqa: PLC0415

    proposal = make_proposal_record(proposal_id=proposal_id, state="designed").model_copy(
        update={
            "user_decision": "approved",
            "user_decided_at": datetime.now(UTC),
        }
    )
    await sqlite_store.create_proposal(proposal)

    # Drive to terminal.
    for _ in range(10):
        await run_worker_cycle(
            spec=spec.engine_spec(),
            metadata_store=sqlite_store,
            artifact_store=localfs,  # type: ignore[arg-type]
            compute=_NoComputeStub(),  # type: ignore[arg-type]
            llm_providers=None,
            config=config,
        )
        rec = await sqlite_store.get_proposal(proposal_id)
        if rec is not None and rec.state == "failed":
            break
    pre_quiesce = await sqlite_store.get_proposal(proposal_id)
    assert pre_quiesce is not None
    assert pre_quiesce.state == "failed"
    pre_version = pre_quiesce.version
    pre_attempt = pre_quiesce.registration_attempt

    # Drive more cycles; nothing should change.
    for _ in range(3):
        await run_worker_cycle(
            spec=spec.engine_spec(),
            metadata_store=sqlite_store,
            artifact_store=localfs,  # type: ignore[arg-type]
            compute=_NoComputeStub(),  # type: ignore[arg-type]
            llm_providers=None,
            config=config,
        )
    post = await sqlite_store.get_proposal(proposal_id)
    assert post is not None
    assert post.state == "failed"
    assert post.version == pre_version
    assert post.registration_attempt == pre_attempt


# === Round-9 fix A: ULID-shaped CG ids + symbolic_id label =================


@pytest.mark.asyncio
async def test_proposal_registers_with_long_symbolic_id(  # type: ignore[no-untyped-def]
    sqlite_store,
    localfs: LocalFsStore,
    tmp_path: Path,
) -> None:
    """Round-9 keystone: a planner-emitted long symbolic CG id no longer
    crashes the registration handler's :class:`ComparisonGroupRecord`
    construction.

    Pre-round-9, ``make_default_cg_id`` returned
    ``f"{proposal_id}--{draft_cg_id}"`` (~80 chars when both halves were
    realistic), which busted past the 64-char id-format cap and crashed
    Pydantic validation at registration time — wedging the proposal in
    ``designed`` until the round-8 registration retry-exhausted gate
    fired with no CG ever materializing.

    The fix: generate a fresh ULID-shaped CG id and preserve the planner's
    symbolic id on :attr:`ComparisonGroupRecord.symbolic_id`. This test
    drives the same long-symbolic-id case that crashed in production and
    asserts the proposal reaches ``registered`` with a materialized CG.
    """
    long_symbolic = "cg-batchnorm-vs-dropout-mlp-fashionmnist"
    proposal_id = "proposal-1778660622026-b56b8d1a"
    planner_responses = make_planner_responses_for_finalize(
        proposal_id=proposal_id,
        cg_drafts=[
            {
                "id": long_symbolic,
                "factor_dimension": "regularization",
                "factor_type": "substitutive",
            }
        ],
    )
    planner_llm = StubLlmProvider(planner_responses)

    spec = build_proposal_pipeline_spec(
        workspace_root=tmp_path / "ws",
        llm_for_planner=planner_llm,  # type: ignore[arg-type]
        require_human_approval=False,
    )
    config = EngineConfig(supervisor_enabled=False)
    await sqlite_store.create_proposal(
        make_proposal_record(proposal_id=proposal_id, state="proposal_submitted")
    )

    final_state: str | None = None
    for _ in range(8):
        await run_worker_cycle(
            spec=spec.engine_spec(),
            metadata_store=sqlite_store,
            artifact_store=localfs,  # type: ignore[arg-type]
            compute=_NoComputeStub(),  # type: ignore[arg-type]
            llm_providers=None,
            config=config,
        )
        rec = await sqlite_store.get_proposal(proposal_id)
        if rec is not None and rec.state in {"registered", "rejected", "failed"}:
            final_state = rec.state
            break
    assert final_state == "registered", f"expected registered; got {final_state}"

    cgs = await sqlite_store.list_cgs_for_proposal(proposal_id)
    assert len(cgs.items) == 1
    cg = cgs.items[0]
    # The primary id is a fresh ULID-shaped string under the 64-char cap.
    assert cg.id.startswith("cg-")
    assert len(cg.id) <= 64
    # The long symbolic id is preserved on the label field — even though
    # ``f"{proposal_id}--{long_symbolic}"`` would exceed 64 chars.
    assert cg.symbolic_id == long_symbolic
    assert len(f"{proposal_id}--{long_symbolic}") > 64, (
        "fixture invariant: composite shape would have exceeded the id cap"
    )
    # ``experiment_definition_id`` matches the ULID-shaped primary id.
    assert cg.experiment_definition_id == cg.id


@pytest.mark.asyncio
async def test_proposal_registers_with_explicit_cg_id_for_resolver(  # type: ignore[no-untyped-def]
    sqlite_store,
    localfs: LocalFsStore,
    tmp_path: Path,
) -> None:
    """The ``cg_id_for`` extension point still lets callers pin a
    deterministic id shape (used by integration tests that pre-stage
    artifacts at known CG paths)."""

    def deterministic(proposal_id: str, draft_cg_id: str) -> str:
        return f"{proposal_id}::{draft_cg_id}"

    proposal_id = "prop-pin"
    planner_responses = make_planner_responses_for_finalize(
        proposal_id=proposal_id,
        cg_drafts=[
            {
                "id": "cg-fixed",
                "factor_dimension": "augmentation",
                "factor_type": "additive",
            }
        ],
    )
    planner_llm = StubLlmProvider(planner_responses)
    spec = build_proposal_pipeline_spec(
        workspace_root=tmp_path / "ws",
        llm_for_planner=planner_llm,  # type: ignore[arg-type]
        require_human_approval=False,
        cg_id_for=deterministic,
    )
    config = EngineConfig(supervisor_enabled=False)
    await sqlite_store.create_proposal(
        make_proposal_record(proposal_id=proposal_id, state="proposal_submitted")
    )

    for _ in range(8):
        await run_worker_cycle(
            spec=spec.engine_spec(),
            metadata_store=sqlite_store,
            artifact_store=localfs,  # type: ignore[arg-type]
            compute=_NoComputeStub(),  # type: ignore[arg-type]
            llm_providers=None,
            config=config,
        )
        rec = await sqlite_store.get_proposal(proposal_id)
        if rec is not None and rec.state == "registered":
            break

    cgs = await sqlite_store.list_cgs_for_proposal(proposal_id)
    assert len(cgs.items) == 1
    assert cgs.items[0].id == "prop-pin::cg-fixed"
    assert cgs.items[0].symbolic_id == "cg-fixed"


# === Compute stub ===========================================================


class _NoComputeStub:
    """A bare Compute stub for inline-only specs.

    The proposal pipeline-spec runs the planner inline and never
    submits external Compute jobs. Phase-1 polling on `designing` may
    fire `Compute.status` for synthetic handles — this stub returns
    ``"succeeded"`` for any handle it's been told about, otherwise
    raises (which the engine treats as the job not yet complete).
    """

    name = "no-compute"

    def __init__(self) -> None:
        from smai_core.plugins import ComputeCapabilities  # noqa: PLC0415

        self.capabilities = ComputeCapabilities(
            supports_gpu=False,
            max_timeout_seconds=3600,
        )
        self._statuses: dict[str, str] = {}

    def set_status(self, handle_id: str, state: str) -> None:
        self._statuses[handle_id] = state

    async def submit(  # noqa: PLR0913
        self,
        image: str,
        command: list[str],
        env: dict[str, str],
        gpu: bool = False,
        timeout_seconds: int = 3600,
        **plugin_options: object,
    ) -> object:
        del image, command, env, gpu, timeout_seconds, plugin_options
        raise RuntimeError("_NoComputeStub.submit should not be called")

    async def status(self, handle):  # type: ignore[no-untyped-def]
        from smai_core.plugins import JobStatus  # noqa: PLC0415

        state = self._statuses.get(handle.handle, "succeeded")  # default succeeded
        return JobStatus(
            state=state,  # type: ignore[arg-type]
            exit_code=0,
            started_at=None,
            finished_at=None,
            failure_reason=None,
        )

    async def logs(self, handle):  # type: ignore[no-untyped-def]
        del handle
        return ""

    async def cancel(self, handle):  # type: ignore[no-untyped-def]
        del handle


# Suppress unused-import noise — these are referenced but may be flagged.
_ = make_planner_session_runner
