"""End-to-end CG-execution happy path — Task 2.C4 acceptance criterion.

Per the Phase-2 acceptance: the full state graph round-trips end-to-end
against a mocked-LLM fixture, terminating in ``complete`` with a valid
:class:`EvaluationResult` written to ArtifactStore.

This test bypasses the harness builder agent loop by pre-staging the
harness artifacts (manifest, validation_results, contracts) and seeding
the CG directly in ``implementing`` state with a fake harness job
handle. Phase-1 polling then sees the handle as ``succeeded``, the
gate trio runs, manifest fanout fires, code review passes, runs
dispatch creates pending :class:`RunRecord`s (per Task 3.E3 lift —
the CG-execution dispatch no longer submits Compute jobs inline),
the run sub-spec drives runs to terminal, and evaluation produces
the result + verdict.

Per Task 3.E3 the run lifecycle is owned by the :class:`RunRecord`
sub-spec; this test now drives both specs in parallel, mirroring
:func:`smai_cli.runtime.Runtime.run_one_cycle`'s multi-spec drive.
"""

from __future__ import annotations

from _helpers import (  # type: ignore[import-not-found]
    FakeCompute,
    make_job_handle,
    make_job_status,
)
from _specs_fakes import (  # type: ignore[import-not-found]
    StubLlmProvider,
    make_cg,
    make_contextual_response,
    make_entry,
    make_review_response,
    stage_harness_artifacts,
    stage_per_entry_artifacts,
    stage_validation_config,
)
from smai_artifacts_localfs import LocalFsStore
from smai_orchestrator.engine.config import EngineConfig
from smai_orchestrator.specs.cg_execution import (
    CONTEXTUAL_VERDICT_KEY_TEMPLATE,
    EVALUATION_RESULT_KEY_TEMPLATE,
    build_cg_execution_spec,
)
from smai_orchestrator.specs.run_record import build_run_record_spec
from smai_orchestrator.worker.loop import run_worker_cycle
from smai_store_sqlite import SqliteStore


async def test_full_cg_lifecycle_terminates_in_complete(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
    tmp_path,
) -> None:
    """Drive a CG through the full state graph against the engine.

    Bypasses the harness builder by pre-staging artifacts and seeding
    the CG already in ``implementing`` with a synthetic
    ``harness_job_handle`` whose Compute.status returns succeeded.
    """

    cg_id = "cg-happy"
    # Stage the artifacts the harness builder would have produced.
    staged = await stage_harness_artifacts(artifact_store=localfs_store, cg_id=cg_id)
    await stage_validation_config(
        artifact_store=localfs_store,
        cg_id=cg_id,
        baseline_entry_id="entry-baseline",
    )
    # Two entries: a baseline (additive — no code) and a treatment.
    await stage_per_entry_artifacts(
        artifact_store=localfs_store,
        cg_id=cg_id,
        entry_id="entry-baseline",
        technique_id=None,
        is_baseline=True,
        parent_harness_contract_hash=staged.harness_contract.envelope.content_hash,
    )
    await stage_per_entry_artifacts(
        artifact_store=localfs_store,
        cg_id=cg_id,
        entry_id="entry-treatment",
        technique_id="tq-1",
        is_baseline=False,
        parent_harness_contract_hash=staged.harness_contract.envelope.content_hash,
        code="def apply(): return {'train_transforms': []}\n",
    )

    # Seed CG already in ``implementing`` with a synthetic handle.
    cg = make_cg(cg_id=cg_id, state="implementing")
    cg = cg.model_copy(update={"harness_job_handle": make_job_handle("harness-h")})
    await sqlite_store.create_cg(cg)
    # Both entries already in ``implemented`` (the test bypasses the
    # entry-level dispatch path; entries are pre-implemented).
    await sqlite_store.create_entry(
        make_entry(
            "entry-baseline",
            cg_id=cg_id,
            state="implemented",
            technique_id=None,
            is_baseline=True,
        )
    )
    await sqlite_store.create_entry(
        make_entry(
            "entry-treatment",
            cg_id=cg_id,
            state="implemented",
            technique_id="tq-1",
        )
    )

    # FakeCompute reports the harness job + every run job as succeeded.
    fake_compute = FakeCompute()
    fake_compute.set_status("harness-h", make_job_status("succeeded"))
    # Pre-enqueue handles for the run sub-spec's Compute.submit (one
    # per (entry × seed) — 2 entries × 1 seed). Per Task 3.E3 the
    # run sub-spec owns Compute.submit; the CG-execution spec just
    # creates pending RunRecords.
    for i in range(2):
        h = make_job_handle(f"run-h-{i}")
        fake_compute.enqueue_submit_handle(h)
        fake_compute.set_status(h.handle, make_job_status("succeeded"))
    # Also pre-stage metrics for each (entry, seed).
    import json  # noqa: PLC0415

    for entry_id, accuracy in [("entry-baseline", 0.5), ("entry-treatment", 0.85)]:
        await localfs_store.put(
            f"comparison-groups/{cg_id}/runs/{entry_id}/0/metrics.json",
            json.dumps({"accuracy": accuracy}).encode("utf-8"),
        )

    review_llm = StubLlmProvider([make_review_response(overall_pass=True)])
    contextual_llm = StubLlmProvider([make_contextual_response()])

    cg_spec = build_cg_execution_spec(
        workspace_root=tmp_path / "workspaces",
        llm_for_code_reviewer=review_llm,  # type: ignore[arg-type]
        llm_for_contextual_evaluator=contextual_llm,  # type: ignore[arg-type]
        seeds=(0,),
    )
    run_spec = build_run_record_spec()

    config = EngineConfig()

    # Drive both specs each cycle (mirroring
    # :meth:`Runtime.run_one_cycle`'s multi-spec drive). Per Task 3.E3
    # the run sub-spec drives the per-(entry × seed) lifecycle in
    # parallel with the CG-level lifecycle; cycle ordering is
    # CG-spec-first, run-spec-second so a CG that lands in ``running``
    # gets its child runs advanced before the next cycle's CG-level
    # phase-2 ``running → evaluating`` discovery.
    final_state: str | None = None
    cycles_executed = 0
    for _ in range(8):  # generous cap; expected ≤6 cycles post-lift.
        cycles_executed += 1
        for spec_ in (cg_spec, run_spec):
            await run_worker_cycle(
                spec=spec_.engine_spec(),
                metadata_store=sqlite_store,
                artifact_store=localfs_store,  # type: ignore[arg-type]
                compute=fake_compute,  # type: ignore[arg-type]
                llm_providers=None,
                config=config,
            )
        cg_now = await sqlite_store.get_cg(cg_id)
        assert cg_now is not None
        if cg_now.state in {
            "complete",
            "implementation_failed",
            "running_failed",
            "evaluation_failed",
        }:
            final_state = cg_now.state
            break
    assert final_state == "complete", (
        f"expected terminal `complete` within 8 cycles; got {final_state} "
        f"after {cycles_executed} cycle(s)"
    )

    # Both evaluation artifacts persisted.
    assert await localfs_store.exists(EVALUATION_RESULT_KEY_TEMPLATE.format(cg_id=cg_id))
    assert await localfs_store.exists(CONTEXTUAL_VERDICT_KEY_TEMPLATE.format(cg_id=cg_id))

    # All entries got the manifest hash from the implemented-state fanout.
    for entry_id in ("entry-baseline", "entry-treatment"):
        e = await sqlite_store.get_entry(entry_id)
        assert e is not None
        assert e.harness_api_manifest_hash == staged.manifest.content_hash


async def test_full_cg_lifecycle_routes_to_evaluation_failed_on_missing_validation_config(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
    tmp_path,
) -> None:
    """A CG that reaches ``evaluating`` without a ValidationConfig in
    ArtifactStore lands in ``evaluation_failed`` (DEC-034 #1).

    Seeds the CG in ``running`` with succeeded runs so the
    ``running → evaluating`` ``dispatch_time`` edge fires + on-entry
    dispatch executes; the dispatch handler can't find the
    ValidationConfig and writes ``evaluation_error.json``; the next
    cycle's ``evaluating → evaluation_failed`` edge picks up the
    error file and routes to terminal.
    """
    from datetime import UTC, datetime  # noqa: PLC0415

    from smai_orchestrator.entities.tracking import RunRecord  # noqa: PLC0415

    cg_id = "cg-eval-fail-e2e"
    # NO validation_config.json — evaluation will write error file.
    staged = await stage_harness_artifacts(artifact_store=localfs_store, cg_id=cg_id)
    await stage_per_entry_artifacts(
        artifact_store=localfs_store,
        cg_id=cg_id,
        entry_id="entry-1",
        technique_id="tq-1",
        is_baseline=False,
        parent_harness_contract_hash=staged.harness_contract.envelope.content_hash,
        code="def apply(): return {'train_transforms': []}\n",
    )

    cg = make_cg(cg_id=cg_id, state="running")
    await sqlite_store.create_cg(cg)
    await sqlite_store.create_entry(
        make_entry("entry-1", cg_id=cg_id, state="implemented", technique_id="tq-1")
    )
    # Pre-create a succeeded run so the running→evaluating gate advances.
    now = datetime(2026, 4, 28, tzinfo=UTC)
    await sqlite_store.create_run(
        RunRecord(
            id="run-eval-fail",
            cg_id=cg_id,
            entry_id="entry-1",
            seed=0,
            state="succeeded",
            raw_metrics_artifact_key="comparison-groups/cg-eval-fail-e2e/runs/entry-1/0/metrics.json",
            version=0,
            created_at=now,
            updated_at=now,
        )
    )
    # Stage the metrics artifact (the run's raw_metrics_artifact_key
    # points at this); without it, _build_raw_metrics_for_cg encodes
    # the run as failed.
    import json as _json  # noqa: PLC0415

    await localfs_store.put(
        "comparison-groups/cg-eval-fail-e2e/runs/entry-1/0/metrics.json",
        _json.dumps({"accuracy": 0.5}).encode("utf-8"),
    )

    review_llm = StubLlmProvider([])  # never called
    contextual_llm = StubLlmProvider([])  # never called
    spec = build_cg_execution_spec(
        workspace_root=tmp_path / "workspaces",
        llm_for_code_reviewer=review_llm,  # type: ignore[arg-type]
        llm_for_contextual_evaluator=contextual_llm,  # type: ignore[arg-type]
    )

    fake_compute = FakeCompute()
    config = EngineConfig()

    final_state: str | None = None
    for _ in range(8):
        await run_worker_cycle(
            spec=spec.engine_spec(),
            metadata_store=sqlite_store,
            artifact_store=localfs_store,  # type: ignore[arg-type]
            compute=fake_compute,  # type: ignore[arg-type]
            llm_providers=None,
            config=config,
        )
        cg_now = await sqlite_store.get_cg(cg_id)
        assert cg_now is not None
        if cg_now.state in {
            "complete",
            "implementation_failed",
            "running_failed",
            "evaluation_failed",
        }:
            final_state = cg_now.state
            break
    assert final_state == "evaluation_failed"
