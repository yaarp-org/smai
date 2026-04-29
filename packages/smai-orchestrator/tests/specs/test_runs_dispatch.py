"""CG-execution runs dispatch — peer-spec coordination per `03` §3.9.

Per Task 3.E3 / DEC-034 #3 the ``running`` state's on-entry dispatch
no longer submits Compute jobs or polls inline — it creates one
:class:`RunRecord` per ``(entry × seed)`` pair in ``pending`` state and
returns. The :class:`RunRecord` sub-spec (in :mod:`.run_record`) drives
each run to terminal via the engine's standard write-first dispatch +
phase-1 polling mechanics; see :mod:`.test_run_record_spec` for the
sub-spec's tests.

These tests verify the post-lift CG-level dispatch handler creates the
right RunRecord shape per ``(entry × seed)`` and lands them in
``pending``.
"""

from __future__ import annotations

from _helpers import (  # type: ignore[import-not-found]
    FakeCompute,
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


async def test_runs_dispatch_creates_pending_runrecord_per_entry_seed(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """For each ``(entry × seed)`` pair where the entry is
    ``implemented``, the dispatch creates one :class:`RunRecord` in
    ``pending`` state. The run sub-spec drives each from there.
    """
    cg_id = "cg-runs"
    cg = make_cg(cg_id=cg_id, state="running")
    await sqlite_store.create_cg(cg)
    entry_a = make_entry("entry-a", cg_id=cg_id, state="implemented", technique_id="tq-a")
    entry_b = make_entry("entry-b", cg_id=cg_id, state="implemented", technique_id="tq-b")
    await sqlite_store.create_entry(entry_a)
    await sqlite_store.create_entry(entry_b)

    fake_compute = FakeCompute()  # never invoked under the lifted dispatch

    handler = _make_dispatch_runs(seeds=(0, 1))
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

    # Two entries × two seeds = 4 RunRecords, all in pending.
    page_a = await sqlite_store.list_runs_for_entry("entry-a", limit=10)
    page_b = await sqlite_store.list_runs_for_entry("entry-b", limit=10)
    assert len(page_a.items) == 2
    assert len(page_b.items) == 2
    for run in (*page_a.items, *page_b.items):
        assert run.state == "pending"
        assert run.cg_id == cg_id
        assert run.compute_job_handle is None
        assert run.raw_metrics_artifact_key is None
    seeds_a = sorted(r.seed for r in page_a.items)
    seeds_b = sorted(r.seed for r in page_b.items)
    assert seeds_a == [0, 1]
    assert seeds_b == [0, 1]

    # No Compute submits happened — the run sub-spec owns Compute.submit.
    assert fake_compute.submit_calls == []


async def test_runs_dispatch_does_not_call_compute(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """Even with one entry × one seed, the CG-level dispatch never
    touches :class:`Compute`. Compute work belongs to the run sub-spec.
    """
    cg_id = "cg-no-compute"
    cg = make_cg(cg_id=cg_id, state="running")
    await sqlite_store.create_cg(cg)
    await sqlite_store.create_entry(
        make_entry("entry-x", cg_id=cg_id, state="implemented", technique_id="tq-x")
    )

    fake_compute = FakeCompute()

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
    assert fake_compute.submit_calls == []
    assert fake_compute.cancel_calls == []

    page = await sqlite_store.list_runs_for_entry("entry-x", limit=10)
    assert len(page.items) == 1
    assert page.items[0].state == "pending"


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
