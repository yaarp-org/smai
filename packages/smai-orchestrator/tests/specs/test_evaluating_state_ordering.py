"""DEC-034 #2 — evaluating state's mechanical-then-contextual ordering.

Per DEC-034 #2 and ``03-state-machine.md`` §3.4, the ``evaluating``
state's on-entry dispatch handler runs the mechanical evaluator first,
then the contextual evaluator. The acceptance criterion requires an
ordering assertion in the dispatch trace.

The dispatch handler accepts an optional ``dispatch_trace: list[str]``
spy hook (passed via :func:`build_cg_execution_spec`'s
``evaluation_dispatch_trace`` kwarg). On invocation, the handler
appends ``"mechanical_evaluator"`` then ``"contextual_evaluator"`` to
the list — observable for test assertions without invasive
introspection of the LLM call sequence.
"""

from __future__ import annotations

from _helpers import FakeCompute  # type: ignore[import-not-found]
from _specs_fakes import (  # type: ignore[import-not-found]
    StubLlmProvider,
    make_cg,
    make_contextual_response,
    make_entry,
    stage_harness_artifacts,
    stage_per_entry_artifacts,
    stage_run_metrics,
    stage_validation_config,
)
from smai_artifacts_localfs import LocalFsStore
from smai_orchestrator.engine.config import EngineConfig
from smai_orchestrator.engine.types import DispatchContext
from smai_orchestrator.specs.cg_execution import (
    CONTEXTUAL_VERDICT_KEY_TEMPLATE,
    EVALUATION_RESULT_KEY_TEMPLATE,
    _make_dispatch_evaluation,
)
from smai_store_sqlite import SqliteStore


async def test_evaluation_dispatch_runs_mechanical_then_contextual(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """The evaluation dispatch records ``mechanical_evaluator`` before
    ``contextual_evaluator`` in the dispatch_trace spy."""
    cg_id = "cg-eval-order"
    # Stage harness + entry artifacts.
    staged = await stage_harness_artifacts(artifact_store=localfs_store, cg_id=cg_id)
    await stage_validation_config(
        artifact_store=localfs_store,
        cg_id=cg_id,
        baseline_entry_id="entry-baseline",
    )
    # Two entries: a baseline (id matches validation_config) + a treatment.
    cg = make_cg(cg_id=cg_id, state="evaluating")
    await sqlite_store.create_cg(cg)
    baseline = make_entry(
        "entry-baseline",
        cg_id=cg_id,
        state="implemented",
        technique_id=None,
        is_baseline=True,
    )
    treatment = make_entry(
        "entry-tq",
        cg_id=cg_id,
        state="implemented",
        technique_id="tq-1",
    )
    await sqlite_store.create_entry(baseline)
    await sqlite_store.create_entry(treatment)
    await stage_per_entry_artifacts(
        artifact_store=localfs_store,
        cg_id=cg_id,
        entry_id=baseline.id,
        technique_id=None,
        is_baseline=True,
        parent_harness_contract_hash=staged.harness_contract.envelope.content_hash,
    )
    await stage_per_entry_artifacts(
        artifact_store=localfs_store,
        cg_id=cg_id,
        entry_id=treatment.id,
        technique_id="tq-1",
        is_baseline=False,
        parent_harness_contract_hash=staged.harness_contract.envelope.content_hash,
    )

    # Pre-create + transition runs to terminal-succeeded with metrics.
    from datetime import UTC, datetime  # noqa: PLC0415

    from smai_orchestrator.entities.tracking import RunRecord  # noqa: PLC0415

    for entry_id, accuracy in [("entry-baseline", 0.5), ("entry-tq", 0.85)]:
        seed = 0
        run_key = await stage_run_metrics(
            artifact_store=localfs_store,
            cg_id=cg_id,
            entry_id=entry_id,
            seed=seed,
            accuracy=accuracy,
        )
        now = datetime(2026, 4, 28, tzinfo=UTC)
        run = RunRecord(
            id=f"run_{cg_id}_{entry_id}_{seed}",
            cg_id=cg_id,
            entry_id=entry_id,
            seed=seed,
            state="succeeded",
            raw_metrics_artifact_key=run_key,
            version=0,
            created_at=now,
            updated_at=now,
        )
        await sqlite_store.create_run(run)

    # Dispatch trace spy + canned contextual verdict response.
    trace: list[str] = []
    contextual_llm = StubLlmProvider([make_contextual_response()])

    handler = _make_dispatch_evaluation(
        llm_for_contextual_evaluator=contextual_llm,  # type: ignore[arg-type]
        dispatch_trace=trace,
    )
    ctx = DispatchContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="evaluating",
        entity_version=cg.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        compute=FakeCompute(),  # type: ignore[arg-type]
        llm=None,
        config=EngineConfig(),
        checkpointer=None,
    )
    outcome = await handler(ctx)
    assert outcome.error is None

    # Ordering: mechanical first, contextual second.
    assert trace == ["mechanical_evaluator", "contextual_evaluator"]

    # Both artifacts were written.
    assert await localfs_store.exists(EVALUATION_RESULT_KEY_TEMPLATE.format(cg_id=cg_id))
    assert await localfs_store.exists(CONTEXTUAL_VERDICT_KEY_TEMPLATE.format(cg_id=cg_id))


async def test_evaluation_dispatch_writes_error_on_missing_validation_config(
    sqlite_store: SqliteStore,
    localfs_store: LocalFsStore,
) -> None:
    """Mechanical evaluation can't proceed without a ValidationConfig
    in ArtifactStore — handler writes evaluation_error.json and
    returns DispatchOutcome() (no raise)."""
    cg_id = "cg-no-config"
    cg = make_cg(cg_id=cg_id, state="evaluating")
    await sqlite_store.create_cg(cg)

    trace: list[str] = []
    contextual_llm = StubLlmProvider([])  # never called

    handler = _make_dispatch_evaluation(
        llm_for_contextual_evaluator=contextual_llm,  # type: ignore[arg-type]
        dispatch_trace=trace,
    )
    ctx = DispatchContext(
        entity_kind="cg",
        entity_id=cg_id,
        entity_state="evaluating",
        entity_version=cg.version,
        metadata_store=sqlite_store,
        artifact_store=localfs_store,  # type: ignore[arg-type]
        compute=FakeCompute(),  # type: ignore[arg-type]
        llm=None,
        config=EngineConfig(),
        checkpointer=None,
    )
    outcome = await handler(ctx)
    assert outcome.error is None  # dispatch handler does not raise — writes error file

    from smai_orchestrator.specs.cg_execution import EVALUATION_ERROR_KEY_TEMPLATE  # noqa: PLC0415

    error_key = EVALUATION_ERROR_KEY_TEMPLATE.format(cg_id=cg_id)
    assert await localfs_store.exists(error_key)
    payload = await localfs_store.get(error_key)
    assert b"ValidationConfig missing" in payload
    # Mechanical was attempted (recorded in trace); contextual was not.
    assert trace == ["mechanical_evaluator"]
