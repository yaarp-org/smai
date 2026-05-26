"""Retry-counter increments after the round-10 RetryPolicy refactor.

Pre-round-10 these counters were bumped by hand inside each dispatch
handler, and the budget gates lived as manual ``EdgeDef``s declared
alongside the happy-path edges. Round 10 lifted both into a declarative
:class:`RetryPolicy` declaration on :class:`DispatchAction`:

* The engine bumps the counter on step-1's CAS (riding the state-
  transition write) — every spec that previously had a manual
  ``transition_*_state(..., <counter>=record.<counter>+1)`` call inside
  a dispatch handler now relies on this.
* The engine synthesizes a retry-exhausted terminal edge in
  :func:`_handle_dispatch_failure` whose gate predicate is
  ``record.<counter> >= max_attempts`` — every spec that previously
  registered a manual ``*_failed (retry exhausted)`` edge now relies on
  this.

These tests confirm the new behavior is byte-equivalent to the old
manual machinery (counters still move, retry-budget terminals still
fire at the same cap). The engine-level synthesis primitives have
deeper-coverage tests in :mod:`tests.engine.test_retry_policy_synthesis`;
this module covers the spec-level wiring.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from _e1_fakes import (  # type: ignore[import-not-found]
    make_planner_responses_for_finalize,
    make_proposal_record,
)
from _helpers import (  # type: ignore[import-not-found]
    FakeCompute,
    make_job_handle,
)
from _specs_fakes import (  # type: ignore[import-not-found]
    StubLlmProvider,
    make_cg,
    make_entry,
)
from smai_artifacts_localfs import LocalFsStore
from smai_orchestrator.engine.config import EngineConfig
from smai_orchestrator.engine.types import DispatchContext
from smai_orchestrator.entities.tracking import RunRecord
from smai_orchestrator.specs.proposal import (
    build_proposal_pipeline_spec,
)
from smai_orchestrator.specs.run_record import _make_dispatch_run_compute_submit
from smai_orchestrator.worker.loop import run_worker_cycle
from smai_store_sqlite import SqliteStore


def _make_run(
    run_id: str,
    *,
    cg_id: str = "cg-1",
    entry_id: str = "entry-1",
    seed: int = 0,
    state: str = "submitted",
) -> RunRecord:
    now = datetime(2026, 4, 28, tzinfo=UTC)
    return RunRecord(
        id=run_id,
        cg_id=cg_id,
        entry_id=entry_id,
        seed=seed,
        state=state,  # type: ignore[arg-type]
        version=0,
        created_at=now,
        updated_at=now,
    )


class _InlineComputeStub:
    """Minimal Compute stub for the inline-planner proposal spec.

    ``Compute.status`` returns ``succeeded`` for any handle (the planner
    runs inline; phase-1 polling on a synthetic ``inline-...`` handle
    must observe a terminal state). ``submit`` is never called.
    """

    name = "inline-stub"

    def __init__(self) -> None:
        from smai_core.plugins import ComputeCapabilities  # noqa: PLC0415

        self.capabilities = ComputeCapabilities(supports_gpu=False, max_timeout_seconds=3600)

    async def submit(self, *args: object, **kwargs: object) -> object:  # noqa: ANN401
        del args, kwargs
        raise RuntimeError("_InlineComputeStub.submit should not be called")

    async def status(self, handle):  # type: ignore[no-untyped-def]
        from smai_core.plugins import JobStatus  # noqa: PLC0415

        del handle
        return JobStatus(
            state="succeeded",
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

    async def stage_workspace(self, local_path):  # type: ignore[no-untyped-def]
        from smai_core.plugins import WorkspaceHandle  # noqa: PLC0415

        return WorkspaceHandle(plugin=self.name, handle=str(local_path))

    async def harvest_workspace(self, handle, local_path):  # type: ignore[no-untyped-def]
        del handle, local_path


# === run_attempt: round-10 — engine bumps it on each step-1 CAS ============


async def test_run_attempt_handler_does_not_bump_itself_post_round_10(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """Direct call to the dispatch handler: round-10 moved the
    ``run_attempt`` bump out of the handler body and into the engine's
    step-1 CAS. A direct invocation of the handler (bypassing the
    engine) MUST NOT bump the counter on its own — that would
    double-count once the engine bump goes through.
    """
    cg_id = "cg-ra"
    await sqlite_store.create_cg(make_cg(cg_id=cg_id, state="running"))
    await sqlite_store.create_entry(
        make_entry("entry-ra", cg_id=cg_id, state="implemented", technique_id="tq-ra")
    )
    run = _make_run("run-ra", cg_id=cg_id, entry_id="entry-ra", seed=3, state="submitted")
    await sqlite_store.create_run(run)
    assert run.run_attempt == 0

    handler = _make_dispatch_run_compute_submit(
        gpu_image="smai-runtime:test", cpu_image="smai-runtime-cpu:test"
    )

    fake_compute = FakeCompute()
    fake_compute.enqueue_submit_handle(make_job_handle("ra-handle-1"))
    ctx = DispatchContext(
        entity_kind="run",
        entity_id="run-ra",
        entity_state="submitted",
        entity_version=run.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        compute=fake_compute,  # type: ignore[arg-type]
        llm=None,
        config=EngineConfig(),
        checkpointer=None,
    )
    outcome = await handler(ctx)
    assert outcome.error is None
    # The handler MUST NOT have bumped ``run_attempt`` itself — the
    # bump happens in the engine's step-1 CAS, which we bypassed here.
    after = await sqlite_store.get_run("run-ra")
    assert after is not None
    assert after.run_attempt == 0


# === design_attempt + registration_attempt: bump once per round trip ===


async def test_design_and_registration_attempts_bump_on_dispatch(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
    tmp_path: Path,
) -> None:
    """A clean proposal_submitted → registered round trip transitions
    into ``designing`` once (engine bumps ``design_attempt`` to 1) and
    into ``registered`` once (engine bumps ``registration_attempt`` to
    1). The handler bodies themselves are no longer responsible for the
    bump — the engine writes it on step-1's CAS.
    """
    planner_responses = make_planner_responses_for_finalize(
        proposal_id="prop-counters",
        cg_drafts=[{"id": "cg-1", "factor_dimension": "augmentation", "factor_type": "additive"}],
    )
    planner_llm = StubLlmProvider(planner_responses)
    spec = build_proposal_pipeline_spec(
        workspace_root=tmp_path / "ws",
        llm_for_planner=planner_llm,  # type: ignore[arg-type]
        require_human_approval=False,
    )
    config = EngineConfig(supervisor_enabled=False)

    proposal = make_proposal_record(proposal_id="prop-counters", state="proposal_submitted")
    await sqlite_store.create_proposal(proposal)
    assert proposal.design_attempt == 0
    assert proposal.registration_attempt == 0

    final_state: str | None = None
    for _ in range(8):
        await run_worker_cycle(
            spec=spec.engine_spec(),
            metadata_store=sqlite_store,
            artifact_store=localfs_store,  # type: ignore[arg-type]
            compute=_InlineComputeStub(),  # type: ignore[arg-type]
            llm_providers=None,
            config=config,
        )
        rec = await sqlite_store.get_proposal("prop-counters")
        assert rec is not None
        if rec.state in {"registered", "rejected", "failed"}:
            final_state = rec.state
            break
    assert final_state == "registered", f"got {final_state}"

    final = await sqlite_store.get_proposal("prop-counters")
    assert final is not None
    assert final.design_attempt == 1
    assert final.registration_attempt == 1


# === RetryPolicy declarations on the spec's DispatchActions ================


def test_proposal_spec_designing_declares_retry_policy(tmp_path: Path) -> None:
    """The ``designing`` state's :class:`DispatchAction` must carry the
    round-10 :class:`RetryPolicy` — the engine reads this to know which
    counter to bump on step-1 and which terminal edge to synthesize on
    dispatch failure."""
    planner_llm = StubLlmProvider([])
    spec = build_proposal_pipeline_spec(
        workspace_root=tmp_path / "ws",
        llm_for_planner=planner_llm,  # type: ignore[arg-type]
        require_human_approval=False,
        max_design_attempts=2,
    )
    designing = next(s for s in spec.states if s.name == "designing")
    assert designing.on_entry_dispatch is not None
    policy = designing.on_entry_dispatch.retry_policy
    assert policy is not None
    assert policy.attempt_counter_field == "design_attempt"
    assert policy.max_attempts == 2
    assert policy.on_exhaustion_target_state == "failed"
    assert "design retry budget exhausted" in policy.on_exhaustion_reason


def test_proposal_spec_registered_declares_retry_policy(tmp_path: Path) -> None:
    """The ``registered`` state's dispatch action carries a
    :class:`RetryPolicy` for the registration handler — round-10 moved
    the manual ``designed → failed (registration retry exhausted)``
    terminal into engine synthesis off ``registered``."""
    planner_llm = StubLlmProvider([])
    spec = build_proposal_pipeline_spec(
        workspace_root=tmp_path / "ws",
        llm_for_planner=planner_llm,  # type: ignore[arg-type]
        require_human_approval=False,
        max_registration_attempts=2,
    )
    registered = next(s for s in spec.states if s.name == "registered")
    assert registered.on_entry_dispatch is not None
    policy = registered.on_entry_dispatch.retry_policy
    assert policy is not None
    assert policy.attempt_counter_field == "registration_attempt"
    assert policy.max_attempts == 2
    assert policy.on_exhaustion_target_state == "failed"
    assert "registration retry budget exhausted" in policy.on_exhaustion_reason


def test_proposal_spec_has_no_manual_retry_exhausted_edges(tmp_path: Path) -> None:
    """Round 10 removed every manual ``*_failed (retry exhausted)``
    :class:`EdgeDef` — the engine synthesizes them now. A regression in
    which a spec author re-introduces a manual edge alongside the
    :class:`RetryPolicy` would double-evaluate and break the round-8
    declaration-order rule."""
    planner_llm = StubLlmProvider([])
    spec = build_proposal_pipeline_spec(
        workspace_root=tmp_path / "ws",
        llm_for_planner=planner_llm,  # type: ignore[arg-type]
        require_human_approval=False,
    )
    edge_names = [e.name for e in spec.edges]
    assert not any("retry exhausted" in n for n in edge_names), (
        f"unexpected manual retry-exhausted edges still in spec: {edge_names}"
    )
