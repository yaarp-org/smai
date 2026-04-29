""":class:`RunRecord` sub-state-machine pipeline-spec — Task 3.E3.

Per ``designs/smai/03-state-machine.md`` §3.9 / DEC-034 #3. The run
sub-spec lifts the per-(entry × seed) lifecycle out of the CG-execution
spec's inline ``running`` dispatch handler. These unit tests exercise
each gate / dispatch-handler factory in isolation against a real
:class:`SqliteStore` + :class:`LocalFsStore` + :class:`FakeCompute`,
mirroring the test pattern of
:mod:`tests.specs.test_implemented_to_running_gate` for consistency.

Spec ambiguities surfaced (per the run_record module docstring):

* ``submitted → running`` is unreachable under the current engine —
  phase-1 only fires terminal :class:`Compute` states. The terminal
  edges from ``submitted`` (the in-progress state where phase-1 actually
  fires) are exercised here. Mirror edges from ``running`` are declared
  for forward-compatibility and are exercised via direct gate invocation
  (the engine wouldn't drive them today, but the gate semantics are
  identical).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

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
from smai_orchestrator.engine.types import DispatchContext, GateContext
from smai_orchestrator.entities.tracking import RunRecord
from smai_orchestrator.specs.cg_execution import RUN_METRICS_KEY_TEMPLATE
from smai_orchestrator.specs.run_record import (
    RUN_RECORD_SPEC_NAME,
    _make_dispatch_run_compute_submit,
    _make_gate_pending_ready,
    _make_gate_run_failed_terminal,
    _make_gate_run_succeeded_no_metrics,
    _make_gate_run_succeeded_with_metrics,
    build_run_record_spec,
    register_run_record_spec,
)
from smai_store_sqlite import SqliteStore

# === Helpers =================================================================


def _make_run(
    run_id: str = "run-1",
    *,
    cg_id: str = "cg-1",
    entry_id: str = "entry-1",
    seed: int = 0,
    state: str = "pending",
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


async def _stage_metrics(
    *,
    artifact_store: LocalFsStore,
    cg_id: str,
    entry_id: str,
    seed: int,
    body: bytes,
) -> None:
    key = RUN_METRICS_KEY_TEMPLATE.format(cg_id=cg_id, entry_id=entry_id, seed=seed)
    await artifact_store.put(key, body)


# === Spec structure =========================================================


def test_build_run_record_spec_structure() -> None:
    """The factory returns a structurally valid :class:`PipelineSpec`."""
    spec = build_run_record_spec()
    assert spec.name == RUN_RECORD_SPEC_NAME
    assert spec.entity_kind == "run"
    assert spec.initial_state == "pending"
    state_names = {s.name for s in spec.states}
    # Six states per `01` §5.5.
    assert state_names == {"pending", "submitted", "running", "succeeded", "failed", "inconclusive"}
    terminals = {s.name for s in spec.states if s.is_terminal}
    assert terminals == {"succeeded", "failed", "inconclusive"}

    # ``submitted`` carries the on-entry dispatch (Compute.submit).
    submitted_def = spec.state_def("submitted")
    assert submitted_def.on_entry_dispatch is not None
    assert submitted_def.on_entry_dispatch.handle_field == "compute_job_handle"
    assert submitted_def.on_entry_dispatch.pool == "runs"

    edge_names = {e.name for e in spec.edges}
    assert "run.pending → submitted" in edge_names
    # Phase-1 edges from ``submitted`` (the engine-reachable terminals).
    assert "run.submitted → succeeded" in edge_names
    assert "run.submitted → inconclusive" in edge_names
    assert "run.submitted → failed" in edge_names
    # Mirror edges from ``running`` for forward-compat.
    assert "run.running → succeeded" in edge_names
    assert "run.running → inconclusive" in edge_names
    assert "run.running → failed" in edge_names

    # Pool: ``runs`` only, limit 4, priority 100 per DEC-034 #4.
    pool_names = {p.name for p in spec.pools}
    assert pool_names == {"runs"}
    runs_pool = next(p for p in spec.pools if p.name == "runs")
    assert runs_pool.limit == 4
    assert runs_pool.priority == 100

    # Phase-1 vs phase-2 partition via :meth:`engine_spec`.
    engine_spec = spec.engine_spec()
    # ``pending`` has only dispatch_time outgoing → phase-2 query.
    assert "pending" in engine_spec.phase2_queries
    assert "pending" not in engine_spec.phase1_queries
    # ``submitted`` and ``running`` have phase-1 outgoing → phase-1 queries.
    assert "submitted" in engine_spec.phase1_queries
    assert "running" in engine_spec.phase1_queries


def test_register_run_record_spec_registers_into_process_registry() -> None:
    """The helper installs the spec into the process-local registry."""
    from smai_orchestrator.runtime import (  # noqa: PLC0415
        get_pipeline_spec,
        list_registered_specs,
    )

    spec = register_run_record_spec()
    assert spec.name == RUN_RECORD_SPEC_NAME
    assert RUN_RECORD_SPEC_NAME in list_registered_specs()
    looked_up = get_pipeline_spec(RUN_RECORD_SPEC_NAME)
    assert looked_up is spec


# === Gate-rule unit tests ====================================================


async def test_pending_ready_gate_always_advances(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """``pending → submitted`` is an always-fire gate — the phase-2
    query already filtered to ``pending`` runs."""
    gate = _make_gate_pending_ready()
    ctx = GateContext(
        entity_kind="run",
        entity_id="run-1",
        entity_state="pending",
        entity_version=0,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        config=EngineConfig(),
        job_outcome=None,
    )
    outcome = await gate(ctx)
    assert outcome.advance is True


async def test_succeeded_with_metrics_gate_advances_when_metrics_parseable(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """``submitted → succeeded`` advances iff ``Compute.status==succeeded``
    AND the per-run metrics file is present and parseable."""
    cg_id = "cg-ok"
    cg = make_cg(cg_id=cg_id, state="running")
    await sqlite_store.create_cg(cg)
    await sqlite_store.create_entry(
        make_entry("entry-x", cg_id=cg_id, state="implemented", technique_id="tq-x")
    )
    run = _make_run("run-x", cg_id=cg_id, entry_id="entry-x", seed=0, state="submitted")
    await sqlite_store.create_run(run)
    await _stage_metrics(
        artifact_store=localfs_store,
        cg_id=cg_id,
        entry_id="entry-x",
        seed=0,
        body=json.dumps({"accuracy": 0.92}).encode(),
    )

    gate = _make_gate_run_succeeded_with_metrics()
    ctx = GateContext(
        entity_kind="run",
        entity_id="run-x",
        entity_state="submitted",
        entity_version=run.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        config=EngineConfig(),
        job_outcome=make_job_status("succeeded"),
    )
    outcome = await gate(ctx)
    assert outcome.advance is True


async def test_succeeded_with_metrics_gate_blocks_when_metrics_missing(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """No metrics file → the success edge does NOT advance (the
    inconclusive edge will fire instead)."""
    cg_id = "cg-no-metrics"
    cg = make_cg(cg_id=cg_id, state="running")
    await sqlite_store.create_cg(cg)
    await sqlite_store.create_entry(
        make_entry("entry-y", cg_id=cg_id, state="implemented", technique_id="tq-y")
    )
    run = _make_run("run-y", cg_id=cg_id, entry_id="entry-y", state="submitted")
    await sqlite_store.create_run(run)

    gate = _make_gate_run_succeeded_with_metrics()
    ctx = GateContext(
        entity_kind="run",
        entity_id="run-y",
        entity_state="submitted",
        entity_version=run.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        config=EngineConfig(),
        job_outcome=make_job_status("succeeded"),
    )
    outcome = await gate(ctx)
    assert outcome.advance is False
    assert outcome.reason is not None and "missing or unparseable" in outcome.reason


async def test_succeeded_with_metrics_gate_rejects_unparseable_payload(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """Non-JSON metrics body routes to ``inconclusive``, not ``succeeded``."""
    cg_id = "cg-bad-metrics"
    cg = make_cg(cg_id=cg_id, state="running")
    await sqlite_store.create_cg(cg)
    await sqlite_store.create_entry(
        make_entry("entry-z", cg_id=cg_id, state="implemented", technique_id="tq-z")
    )
    run = _make_run("run-z", cg_id=cg_id, entry_id="entry-z", state="submitted")
    await sqlite_store.create_run(run)
    await _stage_metrics(
        artifact_store=localfs_store,
        cg_id=cg_id,
        entry_id="entry-z",
        seed=0,
        body=b"this is not json",
    )

    gate = _make_gate_run_succeeded_with_metrics()
    ctx = GateContext(
        entity_kind="run",
        entity_id="run-z",
        entity_state="submitted",
        entity_version=run.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        config=EngineConfig(),
        job_outcome=make_job_status("succeeded"),
    )
    outcome = await gate(ctx)
    assert outcome.advance is False


async def test_inconclusive_gate_advances_when_metrics_missing(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """``submitted → inconclusive`` advances when job exited 0 but
    metrics file is missing — the canonical "completed but no usable
    metric" route per `01` §5.5 / `06` §1."""
    cg_id = "cg-incon"
    cg = make_cg(cg_id=cg_id, state="running")
    await sqlite_store.create_cg(cg)
    await sqlite_store.create_entry(
        make_entry("entry-i", cg_id=cg_id, state="implemented", technique_id="tq-i")
    )
    run = _make_run("run-i", cg_id=cg_id, entry_id="entry-i", state="submitted")
    await sqlite_store.create_run(run)

    gate = _make_gate_run_succeeded_no_metrics()
    ctx = GateContext(
        entity_kind="run",
        entity_id="run-i",
        entity_state="submitted",
        entity_version=run.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        config=EngineConfig(),
        job_outcome=make_job_status("succeeded"),
    )
    outcome = await gate(ctx)
    assert outcome.advance is True
    assert outcome.reason is not None and "inconclusive" in outcome.reason


async def test_inconclusive_gate_blocks_when_metrics_present(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """Metrics file present + parseable → the inconclusive edge does
    NOT fire (the succeeded edge wins per declaration order)."""
    cg_id = "cg-metrics-ok"
    cg = make_cg(cg_id=cg_id, state="running")
    await sqlite_store.create_cg(cg)
    await sqlite_store.create_entry(
        make_entry("entry-w", cg_id=cg_id, state="implemented", technique_id="tq-w")
    )
    run = _make_run("run-w", cg_id=cg_id, entry_id="entry-w", state="submitted")
    await sqlite_store.create_run(run)
    await _stage_metrics(
        artifact_store=localfs_store,
        cg_id=cg_id,
        entry_id="entry-w",
        seed=0,
        body=json.dumps({"accuracy": 0.7}).encode(),
    )

    gate = _make_gate_run_succeeded_no_metrics()
    ctx = GateContext(
        entity_kind="run",
        entity_id="run-w",
        entity_state="submitted",
        entity_version=run.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        config=EngineConfig(),
        job_outcome=make_job_status("succeeded"),
    )
    outcome = await gate(ctx)
    assert outcome.advance is False


async def test_failed_gate_always_advances(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """``submitted → failed`` is an always-fire gate on phase-1
    job_failed (Compute returned failed/cancelled/timeout)."""
    gate = _make_gate_run_failed_terminal()
    ctx = GateContext(
        entity_kind="run",
        entity_id="run-f",
        entity_state="submitted",
        entity_version=0,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        config=EngineConfig(),
        job_outcome=make_job_status("failed", failure_reason="OOM"),
    )
    outcome = await gate(ctx)
    assert outcome.advance is True


# === Dispatch handler =======================================================


async def test_dispatch_compute_submit_calls_compute_with_runtime_command(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """The ``submitted`` on-entry dispatch reads the run record, calls
    :meth:`Compute.submit` with the canonical runtime command + env,
    and returns the handle in :attr:`DispatchOutcome.submitted_handles`.
    """
    cg_id = "cg-submit"
    cg = make_cg(cg_id=cg_id, state="running")
    await sqlite_store.create_cg(cg)
    await sqlite_store.create_entry(
        make_entry("entry-s", cg_id=cg_id, state="implemented", technique_id="tq-s")
    )
    # Engine has already CASed the run from ``pending`` to ``submitted``
    # (phase-3 step 1) before invoking the handler — mirror that here.
    run = _make_run("run-s", cg_id=cg_id, entry_id="entry-s", seed=7, state="submitted")
    await sqlite_store.create_run(run)

    fake_compute = FakeCompute()
    handle = make_job_handle("submitted-job-handle")
    fake_compute.enqueue_submit_handle(handle)

    handler = _make_dispatch_run_compute_submit(image="smai-runtime:test")
    ctx = DispatchContext(
        entity_kind="run",
        entity_id="run-s",
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
    assert outcome.submitted_handles == [handle]
    assert len(fake_compute.submit_calls) == 1
    call = fake_compute.submit_calls[0]
    assert call["image"] == "smai-runtime:test"
    assert call["gpu"] is True
    expected_metrics_key = RUN_METRICS_KEY_TEMPLATE.format(cg_id=cg_id, entry_id="entry-s", seed=7)
    assert call["env"] == {
        "SMAI_CG_ID": cg_id,
        "SMAI_ENTRY_ID": "entry-s",
        "SMAI_SEED": "7",
        "SMAI_METRICS_KEY": expected_metrics_key,
    }
    assert "--metrics-key" in call["command"]
    assert expected_metrics_key in call["command"]


async def test_dispatch_compute_submit_returns_error_when_run_missing(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """If the run record disappeared between phase-3 step 1 and step 2
    (e.g., another worker cleaned it up), the handler reports an error
    and the engine forward-rolls-back."""
    fake_compute = FakeCompute()
    handler = _make_dispatch_run_compute_submit(image="smai-runtime:test")
    ctx = DispatchContext(
        entity_kind="run",
        entity_id="run-missing",
        entity_state="submitted",
        entity_version=0,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        compute=fake_compute,  # type: ignore[arg-type]
        llm=None,
        config=EngineConfig(),
        checkpointer=None,
    )
    outcome = await handler(ctx)
    assert outcome.error is not None
    assert "run-missing" in outcome.error
    assert fake_compute.submit_calls == []
