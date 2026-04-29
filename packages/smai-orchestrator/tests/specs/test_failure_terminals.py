"""DEC-034 #1 — three phase-specific failure terminals.

Per `03-state-machine.md` §2.9 / DEC-034 #1 the CG-execution spec
splits v1's single ``failed`` terminal into three:
``implementation_failed``, ``running_failed``, ``evaluation_failed``.

Each terminal is reached via a distinct gate:

* ``implementation_failed`` — harness build failed (job_failed gate)
  OR the post-success gate found zero implemented entries OR
  code-review failed with no surviving entries.
* ``running_failed`` — all runs terminal but zero succeeded.
* ``evaluation_failed`` — evaluation dispatch wrote an error file
  (e.g., RawMetricsShapeError, missing ValidationConfig).

These tests drive specific failure paths through the engine to
confirm the terminal is reached.
"""

from __future__ import annotations

from _helpers import (  # type: ignore[import-not-found]
    make_job_status,
)
from _specs_fakes import (  # type: ignore[import-not-found]
    make_cg,
    make_entry,
)
from smai_artifacts_localfs import LocalFsStore
from smai_orchestrator.engine.config import EngineConfig
from smai_orchestrator.engine.types import GateContext
from smai_orchestrator.specs.cg_execution import (
    _make_gate_evaluation_complete,
    _make_gate_evaluation_failed,
    _make_gate_harness_failed_terminal,
    _make_gate_harness_succeeded_no_survivors,
    _make_gate_runs_all_terminal_zero_success,
)
from smai_store_sqlite import SqliteStore

# === implementation_failed paths ============================================


async def test_implementation_failed_via_harness_job_failed(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """The ``cg.harness_failed_terminal`` gate fires unconditionally on
    phase-1 ``job_failed``. Per `03` §3.3 there's no auto-retry on
    harness build."""
    cg_id = "cg-harness-fail"
    cg = make_cg(cg_id=cg_id, state="implementing")
    await sqlite_store.create_cg(cg)

    gate = _make_gate_harness_failed_terminal()
    ctx = GateContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="implementing",
        entity_version=cg.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        config=EngineConfig(),
        job_outcome=make_job_status("failed"),
    )
    outcome = await gate(ctx)
    assert outcome.advance is True


async def test_implementation_failed_via_zero_implemented_entries(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """The ``cg.harness_succeeded_no_survivors`` gate fires when the
    harness succeeded but every entry hit ``implementation_failed``."""
    cg_id = "cg-zero-impl"
    cg = make_cg(cg_id=cg_id, state="implementing")
    await sqlite_store.create_cg(cg)
    # All entries are terminal-failed.
    for entry_id in ("entry-a", "entry-b"):
        await sqlite_store.create_entry(
            make_entry(
                entry_id,
                cg_id=cg_id,
                state="implementation_failed",
                technique_id=f"tq-{entry_id}",
            )
        )

    gate = _make_gate_harness_succeeded_no_survivors()
    ctx = GateContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="implementing",
        entity_version=cg.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        config=EngineConfig(),
        job_outcome=make_job_status("succeeded"),
    )
    outcome = await gate(ctx)
    assert outcome.advance is True


# === running_failed path =====================================================


async def test_running_failed_via_zero_successful_runs(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """The ``cg.runs_all_terminal_zero_success`` gate fires when every
    run reached terminal but none succeeded."""
    from datetime import UTC, datetime  # noqa: PLC0415

    from smai_orchestrator.entities.tracking import RunRecord  # noqa: PLC0415

    cg_id = "cg-runs-fail"
    cg = make_cg(cg_id=cg_id, state="running")
    await sqlite_store.create_cg(cg)
    entry = make_entry("entry-1", cg_id=cg_id, state="implemented", technique_id="tq-1")
    await sqlite_store.create_entry(entry)

    now = datetime(2026, 4, 28, tzinfo=UTC)
    for seed, state in ((0, "failed"), (1, "inconclusive")):
        run = RunRecord(
            id=f"run_{cg_id}_entry-1_{seed}",
            cg_id=cg_id,
            entry_id="entry-1",
            seed=seed,
            state=state,  # type: ignore[arg-type]
            failure_reason="test failure" if state == "failed" else None,
            version=0,
            created_at=now,
            updated_at=now,
        )
        await sqlite_store.create_run(run)

    gate = _make_gate_runs_all_terminal_zero_success()
    ctx = GateContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="running",
        entity_version=cg.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        config=EngineConfig(),
    )
    outcome = await gate(ctx)
    assert outcome.advance is True


# === evaluation_failed path =================================================


async def test_evaluation_failed_gate_fires_when_error_artifact_present(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """The ``cg.evaluation_failed`` dispatch_time gate advances when
    ``comparison-groups/{cg_id}/evaluation_error.json`` exists."""
    cg_id = "cg-eval-fail"
    cg = make_cg(cg_id=cg_id, state="evaluating")
    await sqlite_store.create_cg(cg)
    # Pre-stage the error artifact (e.g., as the dispatch handler would
    # write on RawMetricsShapeError).
    await localfs_store.put(
        f"comparison-groups/{cg_id}/evaluation_error.json",
        b"RawMetricsShapeError: missing baseline",
    )

    gate = _make_gate_evaluation_failed()
    ctx = GateContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="evaluating",
        entity_version=cg.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        config=EngineConfig(),
    )
    outcome = await gate(ctx)
    assert outcome.advance is True


async def test_evaluation_complete_gate_holds_when_only_error_present(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """The ``cg.evaluation_complete`` gate does NOT fire when the
    EvaluationResult artifact is missing (even if other artifacts are
    present)."""
    cg_id = "cg-only-error"
    cg = make_cg(cg_id=cg_id, state="evaluating")
    await sqlite_store.create_cg(cg)
    await localfs_store.put(
        f"comparison-groups/{cg_id}/evaluation_error.json",
        b"some error",
    )

    gate = _make_gate_evaluation_complete()
    ctx = GateContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="evaluating",
        entity_version=cg.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        config=EngineConfig(),
    )
    outcome = await gate(ctx)
    assert outcome.advance is False
    assert "EvaluationResult missing" in (outcome.reason or "")
