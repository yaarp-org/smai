"""Inline-approximation runs dispatch — `03` §3.9.

Per the Phase-2 inline approximation (Task 2.C4 brief): the
``running`` state's on-entry dispatch synchronously creates
:class:`RunRecord` rows per ``(entry × seed)`` pair, submits Compute
jobs, polls :meth:`Compute.status` until terminal, and updates the
RunRecord state with the metrics-artifact key. The brief defers the
full :class:`RunRecord` sub-spec to Task 3.E3.

These tests verify the dispatch handler creates the right RunRecord
shape and progresses the run state correctly when Compute reports
``succeeded`` (the smoke-test happy path).
"""

from __future__ import annotations

from _helpers import (  # type: ignore[import-not-found]
    FakeCompute,
    make_job_handle,
    make_job_status,
)
from _specs_fakes import (  # type: ignore[import-not-found]
    make_cg,
    make_entry,
)
from smai_artifacts_localfs import LocalFsStore
from smai_orchestrator.engine.config import EngineConfig
from smai_orchestrator.engine.types import DispatchContext
from smai_orchestrator.specs.cg_execution import _make_dispatch_runs
from smai_store_sqlite import SqliteStore


async def test_runs_dispatch_creates_runrecord_per_entry_seed(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """For each ``(entry × seed)`` pair where the entry is
    ``implemented``, the dispatch creates a :class:`RunRecord` and
    submits a Compute job."""
    cg_id = "cg-runs"
    cg = make_cg(cg_id=cg_id, state="running")
    await sqlite_store.create_cg(cg)
    entry_a = make_entry("entry-a", cg_id=cg_id, state="implemented", technique_id="tq-a")
    entry_b = make_entry("entry-b", cg_id=cg_id, state="implemented", technique_id="tq-b")
    await sqlite_store.create_entry(entry_a)
    await sqlite_store.create_entry(entry_b)

    fake_compute = FakeCompute()
    # Two entries × one seed = 2 submits, all succeed immediately.
    for i in range(2):
        h = make_job_handle(f"run-h-{i}")
        fake_compute.enqueue_submit_handle(h)
        fake_compute.set_status(h.handle, make_job_status("succeeded"))

    handler = _make_dispatch_runs(seeds=(0,))
    ctx = DispatchContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="running",
        entity_version=cg.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        compute=fake_compute,  # type: ignore[arg-type]
        llm=None,
        config=EngineConfig(),
        checkpointer=None,
    )
    outcome = await handler(ctx)
    assert outcome.error is None

    # Two RunRecords created, both terminal-succeeded.
    page_a = await sqlite_store.list_runs_for_entry("entry-a", limit=10)
    page_b = await sqlite_store.list_runs_for_entry("entry-b", limit=10)
    assert len(page_a.items) == 1
    assert len(page_b.items) == 1
    for run in (page_a.items[0], page_b.items[0]):
        assert run.state == "succeeded"
        assert run.raw_metrics_artifact_key is not None
        assert run.cg_id == cg_id
        assert run.seed == 0


async def test_runs_dispatch_routes_failed_jobs_to_failed_state(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """When Compute reports ``failed``, the run lands in ``failed``
    state with ``failure_reason`` populated."""
    cg_id = "cg-runs-fail"
    cg = make_cg(cg_id=cg_id, state="running")
    await sqlite_store.create_cg(cg)
    await sqlite_store.create_entry(
        make_entry("entry-x", cg_id=cg_id, state="implemented", technique_id="tq-x")
    )

    fake_compute = FakeCompute()
    h = make_job_handle("run-h-fail")
    fake_compute.enqueue_submit_handle(h)
    fake_compute.set_status(h.handle, make_job_status("failed"))

    handler = _make_dispatch_runs(seeds=(0,))
    ctx = DispatchContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="running",
        entity_version=cg.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        compute=fake_compute,  # type: ignore[arg-type]
        llm=None,
        config=EngineConfig(),
        checkpointer=None,
    )
    outcome = await handler(ctx)
    assert outcome.error is None

    page = await sqlite_store.list_runs_for_entry("entry-x", limit=10)
    run = page.items[0]
    assert run.state == "failed"
    assert run.failure_reason is not None


async def test_runs_dispatch_returns_error_when_no_implemented_entries(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """No implemented entries → handler returns an error so the engine
    rolls the CG back to its prior state."""
    cg_id = "cg-no-impl"
    cg = make_cg(cg_id=cg_id, state="running")
    await sqlite_store.create_cg(cg)
    # Entry exists but is failed.
    await sqlite_store.create_entry(
        make_entry(
            "entry-failed",
            cg_id=cg_id,
            state="implementation_failed",
            technique_id="tq-x",
        )
    )

    handler = _make_dispatch_runs(seeds=(0,))
    ctx = DispatchContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="running",
        entity_version=cg.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        compute=FakeCompute(),  # type: ignore[arg-type]
        llm=None,
        config=EngineConfig(),
        checkpointer=None,
    )
    outcome = await handler(ctx)
    assert outcome.error is not None
    assert "no implemented entries" in outcome.error
