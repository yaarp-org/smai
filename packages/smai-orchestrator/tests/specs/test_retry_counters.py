"""Retry-counter increments — observability follow-up, item 1.

Pass A wired three retry counters that the retry-budget gates read but
that nothing incremented:

* ``proposal.design_attempt`` — bumped on each (re-)entry into the
  planner dispatch (``specs/proposal.py``); the ``designing → failed``
  gate (``max_design_attempts``) reads it.
* ``proposal.registration_attempt`` — bumped on each (re-)entry into the
  registration dispatch (``specs/proposal.py``); the ``designed → failed``
  gate (``max_registration_attempts``) reads it.
* ``run.run_attempt`` — bumped on each (re-)entry into the run-compute
  submit dispatch (``specs/run_record.py``).

These tests assert the counters actually move and that the proposal
retry-budget gates fire once the count meets the cap. (There is no
``max_run_attempts`` gate in v1 — ``specs/run_record.py`` only references
a hypothetical future one — so for runs we assert the increment only.)
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
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
from smai_orchestrator.engine.types import DispatchContext, GateContext
from smai_orchestrator.entities.tracking import RunRecord
from smai_orchestrator.specs.proposal import (
    _make_gate_design_failed_terminal,
    _make_gate_registration_exhausted,
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


# === run_attempt: increment on each (re-)entry into the submit dispatch ===


async def test_run_attempt_increments_on_each_redispatch(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """``run.run_attempt`` goes 0 → 1 → 2 across two invocations of the
    ``submitted`` on-entry dispatch handler (the engine re-enters this
    handler when a run is forward-rolled-back to ``pending`` and
    re-dispatched)."""
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
    ctx1 = DispatchContext(
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
    outcome1 = await handler(ctx1)
    assert outcome1.error is None
    after1 = await sqlite_store.get_run("run-ra")
    assert after1 is not None
    assert after1.run_attempt == 1

    # Second (re-)dispatch — fresh ctx at the bumped version, fresh handle.
    fake_compute.enqueue_submit_handle(make_job_handle("ra-handle-2"))
    ctx2 = ctx1.model_copy(update={"entity_version": after1.version})
    outcome2 = await handler(ctx2)
    assert outcome2.error is None
    after2 = await sqlite_store.get_run("run-ra")
    assert after2 is not None
    assert after2.run_attempt == 2


# === design_attempt + registration_attempt: bump once per round trip ===


async def test_design_and_registration_attempts_bump_on_dispatch(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
    tmp_path: Path,
) -> None:
    """A clean proposal_submitted → registered round trip runs the
    planner dispatch once and the registration dispatch once, so
    ``design_attempt`` and ``registration_attempt`` each end at 1
    (they started at 0)."""
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


# === retry-budget gates fire once the counter meets the cap ===


def _proposal_gate_ctx(store: SqliteStore, fs: LocalFsStore) -> GateContext:
    return GateContext(
        entity_kind="proposal",
        entity_id="prop-gate",
        entity_state="designing",
        entity_version=0,
        metadata_store=store,
        artifact_store=fs,  # type: ignore[arg-type]
        config=EngineConfig(),
        job_outcome=None,
    )


@pytest.mark.parametrize(
    ("design_attempt", "max_attempts", "expect_advance"),
    [(0, 1, False), (1, 1, True), (1, 2, False), (2, 2, True), (3, 2, True)],
)
async def test_design_failed_terminal_gate_respects_budget(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
    design_attempt: int,
    max_attempts: int,
    expect_advance: bool,
) -> None:
    """``designing → failed`` fires iff ``design_attempt >= max_design_attempts``."""
    proposal = make_proposal_record(proposal_id="prop-gate", state="designing")
    proposal = proposal.model_copy(update={"design_attempt": design_attempt})
    await sqlite_store.create_proposal(proposal)
    gate = _make_gate_design_failed_terminal(max_design_attempts=max_attempts)
    outcome = await gate(_proposal_gate_ctx(sqlite_store, localfs_store))
    assert outcome.advance is expect_advance


@pytest.mark.parametrize(
    ("registration_attempt", "max_attempts", "expect_advance"),
    [(0, 2, False), (1, 2, False), (2, 2, True), (3, 2, True)],
)
async def test_registration_exhausted_gate_respects_budget(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
    registration_attempt: int,
    max_attempts: int,
    expect_advance: bool,
) -> None:
    """``designed → failed`` fires iff ``registration_attempt >= max_registration_attempts``."""
    proposal = make_proposal_record(proposal_id="prop-gate", state="designed")
    proposal = proposal.model_copy(update={"registration_attempt": registration_attempt})
    await sqlite_store.create_proposal(proposal)
    gate = _make_gate_registration_exhausted(max_registration_attempts=max_attempts)
    ctx = _proposal_gate_ctx(sqlite_store, localfs_store).model_copy(
        update={"entity_state": "designed"}
    )
    outcome = await gate(ctx)
    assert outcome.advance is expect_advance
