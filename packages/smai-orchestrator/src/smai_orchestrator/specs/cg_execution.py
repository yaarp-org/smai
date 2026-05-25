"""CG-execution :class:`PipelineSpec` instance — ``03-state-machine.md`` §3.

Composes the engine substrate (Task 2.C1), the worker loop (Task 2.C2),
the :class:`PipelineSpec` shape (Task 2.C3), the harness builder + technique
implementer dispatch handlers (Task 2.B3), the code-reviewer + contextual-
evaluator single-call agents (Task 2.B4), and the runtime workspace +
manifest helpers (Task 2.D1) into the concrete state machine that drives
:class:`ComparisonGroupRecord` entities through ``draft → implementing →
implemented → running → evaluating → complete`` plus the three failure
terminals per DEC-034 #1.

Two specs are registered:

* :func:`build_cg_execution_spec` — entity_kind ``"cg"`` per the
  :data:`smai_core.plugins.EntityKind` literal. Owns the CG-level lifecycle.
* :func:`build_cg_entries_spec` (in :mod:`.cg_entries`) — entity_kind
  ``"entry"``. Owns per-entry implementer dispatch.

Cross-spec coordination is via shared :class:`MetadataStore` and
:class:`ArtifactStore` per `05` §5.3 and DEC-004's separation principle.

Spec ambiguities resolved (one-line flips if a future reconciliation
disagrees):

* **Per-entity-kind structure:** two specs (``smai_cg_execution``,
  ``smai_cg_entries``) over a single discriminator-extended spec.
  :class:`PipelineSpec.entity_kind` is single-valued per `05` §5.1, so
  collapsing into one spec would force an engine extension.
* **Additive-baseline skip:** double-enforced. The
  :meth:`MetadataStore.get_ready_to_implement_entry` query filters
  baselines out at the SQL layer (DEC-013 + `07` §5.6.3); the
  technique-implementer dispatch handler also short-circuits on
  :func:`smai_runtime.is_additive_baseline` per Task 2.B3. Spec authors
  who reuse this spec inherit both layers.
* **Code-review caching across the three implemented-edge gates:**
  the gate-trio shares one :class:`CodeReviewResult` per cycle by writing
  the result to ArtifactStore at
  ``comparison-groups/{cg_id}/code-review-attempt-{N}.json`` (keyed on
  the CG's ``code_review_attempt`` counter so retries re-fire). The
  engine's :class:`Checkpointer` is not used because :class:`GateContext`
  doesn't carry it (`05` §1.2's gate signature predates
  :class:`Checkpointer`); the artifact-store cache plays the same role.
* **Manifest hash fan-out (DEC-033 #3):** lands as ``implemented``
  state's on-entry dispatch handler, NOT inside the success-gate body
  (the gate-rule contract per §1.3 is read-only). The handler uses
  :meth:`MetadataStore.transaction` for atomic per-entry CAS, satisfying
  the "atomically" requirement of DEC-033 #3 within the bounds of the
  current Protocol surface (no batch API).
* **Run lifecycle (production peer-spec coordination — Task 3.E3):**
  the ``running`` on-entry dispatch handler creates one
  :class:`RunRecord` per ``(entry × seed)`` tuple in ``pending`` state
  and returns. The :class:`RunRecord` sub-state-machine pipeline-spec
  (in :mod:`.run_record` per `03` §3.9 / DEC-034 #3) drives each run
  through ``pending → submitted → {succeeded | failed | inconclusive}``
  via the engine's standard write-first dispatch + phase-1 polling.
  Cross-spec coordination is via :meth:`MetadataStore.list_runs_for_entry`
  — the ``running → evaluating`` gate (`05` §1.3 sibling-read pattern)
  reads child run records and decides "all runs terminal" without
  needing a per-job-event hook (the engine's existing phase-2
  ``get_ready_to_evaluate`` discovery picks up CGs whose runs have
  all terminated on the next cycle, per `05` §3.2).
* **``evaluating → complete`` transition:** uses ``dispatch_time``
  edges off ``evaluating`` rather than the spec table's
  ``fires_on=job_succeeded`` (which would require a job handle for
  phase-1 polling — impossible for inline dispatches per the engine's
  :func:`phase1_step` contract). The two ``dispatch_time`` edges decide
  ``complete`` vs ``evaluation_failed`` based on which artifact the
  on-entry handler wrote (the result vs. an error file). Documented as
  a v1 design choice; the spec's intent is preserved (terminal state
  reached after evaluation runs).
* **LlmProvider injection:** factory parameters
  (``llm_for_code_reviewer`` / ``llm_for_contextual_evaluator``)
  captured in closures over the gate-rule and dispatch-handler
  callables. The brief flags ``EngineConfig.gate_extensions`` as the
  canonical place per `05` §6's "general extension surface"; v1 ships
  the closure-captured shape because adding ``gate_extensions`` to
  :class:`EngineConfig` would surface the seam without buying anything
  the closures don't already provide. A future reconciliation can lift
  closures into ``gate_extensions`` for runtime overridability if the
  CLI surface needs it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from smai_core.artifacts import HarnessContract, TechniqueContract, ValidationConfig
from smai_core.evaluation import EvaluationResult, RawMetrics
from smai_core.evaluation._evaluate import RawMetricsShapeError, evaluate
from smai_core.evaluation.raw_metrics import EntryMetrics, SeedRunOutcome
from smai_core.plugins import (
    ArtifactNotFound,
    LlmProvider,
)
from smai_core.plugins.metadata_store import ConflictError

from smai_orchestrator.engine.types import (
    ConcurrencyPool,
    DispatchAction,
    DispatchContext,
    DispatchOutcome,
    EdgeDef,
    GateContext,
    GateOutcome,
    RetryPolicy,
    SchedulingQueryRef,
    StateDef,
)
from smai_orchestrator.entities.tracking import (
    ComparisonGroupRecord,
    EntryRecord,
    RunRecord,
)
from smai_orchestrator.runtime.spec import PipelineSpec

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    from smai_agents.schemas.code_review import CodeReviewResult
    from smai_agents.schemas.contextual_verdict import ContextualVerdict


# === Constants ==============================================================

CG_EXECUTION_SPEC_NAME = "smai_cg_execution"
CG_ENTRIES_SPEC_NAME = "smai_cg_entries"

# Pool names — three pools for the CG-execution + entry specs (Phase 2
# scope; the proposal_pipeline / paper_ingestion pools land in Phase 3
# alongside their respective specs per DEC-034 #4).
POOL_AGENTS = "agents"
POOL_RUNS = "runs"
POOL_INLINE = "inline"

# v1 path conventions — mirror ``CLAUDE.md`` /
# ``designs/yaarp/...`` Artifact Layout for ``comparison-groups/{cg_id}/``.
HARNESS_CONTRACT_KEY_TEMPLATE = "comparison-groups/{cg_id}/harness/contract.json"
HARNESS_MANIFEST_KEY_TEMPLATE = "comparison-groups/{cg_id}/harness/manifest.json"
TECHNIQUE_CONTRACT_KEY_TEMPLATE = (
    "comparison-groups/{cg_id}/entries/{entry_id}/technique_contract.json"
)
TECHNIQUE_CODE_KEY_TEMPLATE = (
    "comparison-groups/{cg_id}/entries/{entry_id}/code/techniques/{technique_name}.py"
)
TECHNIQUE_VALIDATION_KEY_TEMPLATE = (
    "comparison-groups/{cg_id}/entries/{entry_id}/code/validation_results.json"
)
HARNESS_CODE_PREFIX_TEMPLATE = "comparison-groups/{cg_id}/harness/"
HARNESS_VALIDATION_KEY_TEMPLATE = "comparison-groups/{cg_id}/harness/validation_results.json"
VALIDATION_CONFIG_KEY_TEMPLATE = "comparison-groups/{cg_id}/validation_config.json"

CODE_REVIEW_KEY_TEMPLATE = "comparison-groups/{cg_id}/code-review-attempt-{attempt}.json"
EVALUATION_RESULT_KEY_TEMPLATE = "comparison-groups/{cg_id}/evaluation_result.json"
CONTEXTUAL_VERDICT_KEY_TEMPLATE = "comparison-groups/{cg_id}/contextual_verdict.json"
EVALUATION_ERROR_KEY_TEMPLATE = "comparison-groups/{cg_id}/evaluation_error.json"
RUN_METRICS_KEY_TEMPLATE = "comparison-groups/{cg_id}/runs/{entry_id}/{seed}/metrics.json"

# Comparison-rule values per `02-dsl-and-contracts.md` §7.6 — used by the
# contextual evaluator's variant selection.
ComparisonRule = Literal["compare_to_baseline", "compare_to_target"]

# Run-lifecycle terminal vs in-progress states per `03` §3.9 / `01` §5.5.
RUN_TERMINAL_STATES: frozenset[str] = frozenset({"succeeded", "failed", "inconclusive"})
RUN_IN_PROGRESS_STATES: frozenset[str] = frozenset({"pending", "submitted", "running"})


# === Helpers shared by gates and dispatch handlers ===========================


def read_validation_results_pass(payload: bytes) -> bool:
    """Best-effort parse of the runtime's ``validation_results.json``.

    The runtime emits at least ``{"passed": <bool>, ...}`` per
    ``10-runtime-and-templates.md`` §5.4. We tolerate extra fields and
    return ``False`` on any parse failure (the gate body's caller treats
    "no parseable pass" the same as "validation failed"). Avoiding a hard
    schema dependency here keeps the spec module decoupled from the
    runtime's exact emission shape.
    """
    import json  # noqa: PLC0415 — lazy on the cold path.

    try:
        body: object = json.loads(payload)
    except (ValueError, TypeError):
        return False
    if not isinstance(body, dict):
        return False
    body_dict = cast(dict[str, object], body)
    return bool(body_dict.get("passed"))


async def _all_runs_for_cg(
    metadata_store: Any,
    cg_id: str,
) -> list[RunRecord]:
    """Walk every entry of the CG and return its child :class:`RunRecord`s.

    The :class:`MetadataStore` Protocol exposes
    :meth:`list_runs_for_entry` (per-entry) but not ``list_runs_for_cg``;
    we compose by fanning out across :meth:`list_entries_for_cg`. Both
    pages are drained (cursor-paginated; we follow ``next_cursor`` until
    exhausted) so the predicate always sees the full child set.
    """
    runs: list[RunRecord] = []
    entries = await _list_all_entries_for_cg(metadata_store, cg_id)
    for entry in entries:
        cursor: str | None = None
        while True:
            page = await metadata_store.list_runs_for_entry(entry.id, limit=100, cursor=cursor)
            runs.extend(page.items)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
    return runs


async def _list_all_entries_for_cg(
    metadata_store: Any,
    cg_id: str,
) -> list[EntryRecord]:
    """Drain all pages of :meth:`MetadataStore.list_entries_for_cg`."""
    out: list[EntryRecord] = []
    cursor: str | None = None
    while True:
        page = await metadata_store.list_entries_for_cg(cg_id, limit=100, cursor=cursor)
        out.extend(page.items)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    return out


def _entries_terminal_with_at_least_one_implemented(entries: Sequence[EntryRecord]) -> bool:
    """Predicate: every entry is in a terminal state and ≥1 is ``implemented``.

    Per ``03-state-machine.md`` §3.3
    (``cg.all_entries_implemented_and_manifest_present``).
    """
    if not entries:
        return False
    if not all(e.state in {"implemented", "implementation_failed"} for e in entries):
        return False
    return any(e.state == "implemented" for e in entries)


def _entries_terminal_with_zero_implemented(entries: Sequence[EntryRecord]) -> bool:
    """Predicate: every entry is in a terminal state and zero are ``implemented``."""
    if not entries:
        return False
    if not all(e.state in {"implemented", "implementation_failed"} for e in entries):
        return False
    return not any(e.state == "implemented" for e in entries)


# === Code-review gate-trio cache (per-cycle dedup) ===========================


def _code_review_artifact_key(cg_id: str, attempt: int) -> str:
    return CODE_REVIEW_KEY_TEMPLATE.format(cg_id=cg_id, attempt=attempt)


async def _load_or_compute_code_review(
    *,
    ctx: GateContext,
    cg: ComparisonGroupRecord,
    code_reviewer_factor_type: Literal["additive", "substitutive"],
    code_reviewer_factor_dimension: str,
    llm_for_code_reviewer: LlmProvider,
) -> CodeReviewResult:
    """Shared cache for the implemented-edge gate trio.

    Reads ``comparison-groups/{cg_id}/code-review-attempt-{N}.json`` from
    ArtifactStore. Cache hit: deserialize and return.
    Cache miss: assemble :class:`CodeReviewerInput`, call
    :func:`run_code_review`, write the serialized result, return.

    Cache-key invalidation across retries is handled by the
    ``code_review_attempt`` field on :class:`ComparisonGroupRecord` —
    bumping it (in the retry-route gate body) produces a fresh key so
    the next cycle's review re-fires.
    """
    # Lazy imports per the smai-orchestrator → smai-agents dep direction
    # (implementation_plan.md §2.1).
    from smai_agents.agents.code_reviewer import (  # noqa: PLC0415
        CodeReviewerInput,
        EntryUnderReview,
        run_code_review,
    )
    from smai_agents.schemas.code_review import (  # noqa: PLC0415
        CodeReviewResult,
    )

    key = _code_review_artifact_key(cg.id, cg.code_review_attempt)
    try:
        cached = await ctx.artifact_store.get(key)
        return CodeReviewResult.model_validate_json(cached)
    except ArtifactNotFound:
        pass

    harness_contract_key = HARNESS_CONTRACT_KEY_TEMPLATE.format(cg_id=cg.id)
    harness_contract_bytes = await ctx.artifact_store.get(harness_contract_key)
    harness_contract = HarnessContract.model_validate_json(harness_contract_bytes)

    # Read harness code (every .py file under the harness/ prefix).
    harness_code: dict[str, str] = {}
    prefix = HARNESS_CODE_PREFIX_TEMPLATE.format(cg_id=cg.id)
    iterator = await ctx.artifact_store.list(prefix)
    async for full_key in iterator:
        rel = full_key[len(prefix) :]
        if not rel.endswith(".py"):
            continue
        body = await ctx.artifact_store.get(full_key)
        harness_code[rel] = body.decode("utf-8")

    entries = await _list_all_entries_for_cg(ctx.metadata_store, cg.id)
    review_entries: list[EntryUnderReview] = []
    for entry in entries:
        contract_key = TECHNIQUE_CONTRACT_KEY_TEMPLATE.format(cg_id=cg.id, entry_id=entry.id)
        try:
            contract_bytes = await ctx.artifact_store.get(contract_key)
        except ArtifactNotFound:
            continue
        technique_contract = TechniqueContract.model_validate_json(contract_bytes)
        technique_id = entry.technique_id
        technique_name = technique_id or "baseline"
        code: str | None = None
        if technique_id is not None:
            code_key = TECHNIQUE_CODE_KEY_TEMPLATE.format(
                cg_id=cg.id, entry_id=entry.id, technique_name=technique_name
            )
            try:
                code = (await ctx.artifact_store.get(code_key)).decode("utf-8")
            except ArtifactNotFound:
                code = None
        review_entries.append(
            EntryUnderReview(
                entry_id=entry.id,
                technique_id=technique_id,
                technique_name=technique_name,
                technique_description="",
                is_baseline=entry.is_baseline,
                technique_contract=technique_contract,
                code=code,
            )
        )

    review_input = CodeReviewerInput(
        cg_id=cg.id,
        factor_type=code_reviewer_factor_type,
        factor_dimension=code_reviewer_factor_dimension,
        harness_contract=harness_contract,
        harness_code=harness_code,
        entries=review_entries,
    )
    result = await run_code_review(llm=llm_for_code_reviewer, input=review_input)
    await ctx.artifact_store.put(key, result.model_dump_json().encode("utf-8"))
    return result


# === Factor-type read helper ================================================


async def _read_factor_meta(
    *,
    ctx: GateContext,
    cg: ComparisonGroupRecord,
) -> tuple[Literal["additive", "substitutive"], str]:
    """Read the factor type + dimension off the loaded :class:`HarnessContract`.

    The contract is the canonical source of factor framing per DEC-017 /
    `02-dsl-and-contracts.md` §7.4.
    """
    key = HARNESS_CONTRACT_KEY_TEMPLATE.format(cg_id=cg.id)
    bytes_ = await ctx.artifact_store.get(key)
    contract = HarnessContract.model_validate_json(bytes_)
    return (contract.body.factor.type, contract.body.factor.name)


# === Gate-rule callable factories ===========================================
# All gates resolve via closures over the spec-factory's args so the spec
# stays substrate-agnostic (no module-level singletons).


def _make_gate_draft_ready() -> Callable[[GateContext], Awaitable[GateOutcome]]:
    async def _gate(ctx: GateContext) -> GateOutcome:
        # Phase-3 always-fire entry gate: state==draft + no in-flight
        # harness handle. The scheduling query already filters for
        # ``harness_job_handle is null``, so we simply advance.
        return GateOutcome(advance=True, reason="draft → implementing always-fire")

    return _gate


def _make_gate_harness_succeeded_advance() -> Callable[[GateContext], Awaitable[GateOutcome]]:
    async def _gate(ctx: GateContext) -> GateOutcome:
        # Round 14: ``dispatch_time`` gate (the harness builder runs
        # in-process, so ``ctx.job_outcome`` is always None — there is
        # no external job). Success = the harness builder emitted a
        # manifest AND every entry is terminal AND ≥1 is implemented.
        # The ``emit_harness_manifest`` tool only writes the manifest
        # after a passing in-workspace validation run (manifest_tool §9
        # step 5), so manifest-presence already encodes "harness
        # validation passed" — no separate validation-results read.
        manifest_key = HARNESS_MANIFEST_KEY_TEMPLATE.format(cg_id=ctx.entity_id)
        if not await ctx.artifact_store.exists(manifest_key):
            return GateOutcome(advance=False, reason="manifest not yet written")
        entries = await _list_all_entries_for_cg(ctx.metadata_store, ctx.entity_id)
        if not _entries_terminal_with_at_least_one_implemented(entries):
            return GateOutcome(advance=False, reason="entries not all terminal yet")

        # DEC-033 #3 manifest-hash fanout: write ``harness_api_manifest_hash``
        # to every entry of the CG atomically before the engine CAS-
        # transitions the CG to ``implemented``. The brief flags this
        # work as the gate-body's responsibility ("the implementing →
        # implemented gate writes harness_api_manifest_hash to all
        # entries"); the engine's phase-1 path doesn't fire the target
        # state's on_entry_dispatch (only phase-3 does), so the
        # canonical place for this work is here. §1.3's "gate rules
        # MUST NOT write entity state" is interpreted strictly to
        # apply to the entity being evaluated (the CG); per-entry
        # writes are sibling-state writes that the brief explicitly
        # directs.
        await _fanout_manifest_hash(
            metadata_store=ctx.metadata_store,
            artifact_store=ctx.artifact_store,
            cg_id=ctx.entity_id,
            entries=entries,
        )
        return GateOutcome(advance=True, reason="all-entries-implemented + manifest fanned out")

    return _gate


def _make_gate_harness_succeeded_no_survivors() -> Callable[[GateContext], Awaitable[GateOutcome]]:
    async def _gate(ctx: GateContext) -> GateOutcome:
        # Round 14: ``dispatch_time`` gate (``ctx.job_outcome`` is always
        # None — in-process dispatch). Fires only when every entry has
        # reached a terminal implementation state and none is
        # ``implemented`` — which can only happen once the harness built
        # (entries cannot run without a manifest), so a harness-build
        # failure never reaches here (it routes through the dispatch
        # handler's error path + the RetryPolicy instead).
        entries = await _list_all_entries_for_cg(ctx.metadata_store, ctx.entity_id)
        if _entries_terminal_with_zero_implemented(entries):
            return GateOutcome(advance=True, reason="zero entries implemented")
        return GateOutcome(advance=False)

    return _gate


def _make_gate_review_pass(
    *,
    llm_for_code_reviewer: LlmProvider,
    require_human_approval: bool,
) -> Callable[[GateContext], Awaitable[GateOutcome]]:
    async def _gate(ctx: GateContext) -> GateOutcome:
        cg = await ctx.metadata_store.get_cg(ctx.entity_id)
        if cg is None:
            return GateOutcome(advance=False, reason="CG not found")
        factor_type, factor_dim = await _read_factor_meta(ctx=ctx, cg=cg)
        result = await _load_or_compute_code_review(
            ctx=ctx,
            cg=cg,
            code_reviewer_factor_type=factor_type,
            code_reviewer_factor_dimension=factor_dim,
            llm_for_code_reviewer=llm_for_code_reviewer,
        )
        if not result.overall_pass:
            return GateOutcome(advance=False, reason="code review failed")
        if require_human_approval:
            # Per DEC-003 / `03-state-machine.md` §3.3: deployment-
            # configurable human-approval branch. v1 ships the
            # auto-approve default; the flag's MetadataStore placement
            # is `01-data-model.md` Session-C territory.
            return GateOutcome(advance=False, reason="awaiting human approval")
        return GateOutcome(advance=True, reason="code review passed")

    return _gate


def _make_gate_review_fail_with_retry(
    *,
    llm_for_code_reviewer: LlmProvider,
    max_review_attempts: int,
) -> Callable[[GateContext], Awaitable[GateOutcome]]:
    async def _gate(ctx: GateContext) -> GateOutcome:
        cg = await ctx.metadata_store.get_cg(ctx.entity_id)
        if cg is None:
            return GateOutcome(advance=False)
        factor_type, factor_dim = await _read_factor_meta(ctx=ctx, cg=cg)
        result = await _load_or_compute_code_review(
            ctx=ctx,
            cg=cg,
            code_reviewer_factor_type=factor_type,
            code_reviewer_factor_dimension=factor_dim,
            llm_for_code_reviewer=llm_for_code_reviewer,
        )
        if result.overall_pass:
            return GateOutcome(advance=False, reason="review passed; retry inapplicable")

        critical_targets = {f.target_id for f in result.findings if f.severity == "critical"}
        if not critical_targets:
            return GateOutcome(advance=False, reason="no critical findings")
        entries = await _list_all_entries_for_cg(ctx.metadata_store, ctx.entity_id)
        retryable_entries: list[EntryRecord] = []
        for entry in entries:
            entry_targeted = entry.id in critical_targets or (
                entry.technique_id is not None and entry.technique_id in critical_targets
            )
            if not entry_targeted:
                continue
            if entry.implementation_attempt < max_review_attempts:
                retryable_entries.append(entry)
        if not retryable_entries:
            return GateOutcome(advance=False, reason="no entries with retry budget")

        # DEC-016 / brief: the gate body performs the per-entry resets
        # (increment ``implementation_attempt``, reset state to ``pending``)
        # plus bumps the CG's ``code_review_attempt`` so the next cycle's
        # review re-fires against the (newly-reset) entries with a fresh
        # cache key.
        for entry in retryable_entries:
            try:
                await ctx.metadata_store.transition_entry_state(
                    entry.id,
                    entry.version,
                    "pending",
                    implementation_attempt=entry.implementation_attempt + 1,
                    implementation_job_handle=None,
                )
            except ConflictError:
                continue
        try:
            await ctx.metadata_store.transition_cg_state(
                cg.id,
                cg.version,
                cg.state,  # pyright: ignore[reportArgumentType] — preserve current state for field-only update
                code_review_attempt=cg.code_review_attempt + 1,
            )
        except ConflictError:
            pass
        return GateOutcome(
            advance=True,
            reason=f"retry route fired for {len(retryable_entries)} entries",
        )

    return _gate


def _make_gate_review_fail_no_survivors(
    *,
    llm_for_code_reviewer: LlmProvider,
) -> Callable[[GateContext], Awaitable[GateOutcome]]:
    async def _gate(ctx: GateContext) -> GateOutcome:
        cg = await ctx.metadata_store.get_cg(ctx.entity_id)
        if cg is None:
            return GateOutcome(advance=False)
        factor_type, factor_dim = await _read_factor_meta(ctx=ctx, cg=cg)
        result = await _load_or_compute_code_review(
            ctx=ctx,
            cg=cg,
            code_reviewer_factor_type=factor_type,
            code_reviewer_factor_dimension=factor_dim,
            llm_for_code_reviewer=llm_for_code_reviewer,
        )
        if result.overall_pass:
            return GateOutcome(advance=False)
        critical_targets = {f.target_id for f in result.findings if f.severity == "critical"}
        entries = await _list_all_entries_for_cg(ctx.metadata_store, ctx.entity_id)
        survivor_count = sum(
            1
            for e in entries
            if e.state == "implemented"
            and e.id not in critical_targets
            and (e.technique_id is None or e.technique_id not in critical_targets)
        )
        if survivor_count > 0:
            return GateOutcome(advance=False, reason="≥1 survivor remains")
        return GateOutcome(advance=True, reason="no survivors after review failure")

    return _gate


def _make_gate_runs_all_terminal_with_success() -> Callable[[GateContext], Awaitable[GateOutcome]]:
    async def _gate(ctx: GateContext) -> GateOutcome:
        runs = await _all_runs_for_cg(ctx.metadata_store, ctx.entity_id)
        if not runs:
            return GateOutcome(advance=False, reason="no runs created yet")
        if not all(r.state in RUN_TERMINAL_STATES for r in runs):
            return GateOutcome(advance=False, reason="not all runs terminal")
        if not any(r.state == "succeeded" for r in runs):
            return GateOutcome(advance=False, reason="zero successful runs")
        return GateOutcome(advance=True, reason="all runs terminal; ≥1 succeeded")

    return _gate


def _make_gate_runs_all_terminal_zero_success() -> Callable[[GateContext], Awaitable[GateOutcome]]:
    async def _gate(ctx: GateContext) -> GateOutcome:
        runs = await _all_runs_for_cg(ctx.metadata_store, ctx.entity_id)
        if not runs:
            return GateOutcome(advance=False)
        if not all(r.state in RUN_TERMINAL_STATES for r in runs):
            return GateOutcome(advance=False)
        if any(r.state == "succeeded" for r in runs):
            return GateOutcome(advance=False)
        return GateOutcome(advance=True, reason="all runs terminal; zero succeeded")

    return _gate


def _make_gate_evaluation_complete() -> Callable[[GateContext], Awaitable[GateOutcome]]:
    async def _gate(ctx: GateContext) -> GateOutcome:
        result_key = EVALUATION_RESULT_KEY_TEMPLATE.format(cg_id=ctx.entity_id)
        verdict_key = CONTEXTUAL_VERDICT_KEY_TEMPLATE.format(cg_id=ctx.entity_id)
        if not await ctx.artifact_store.exists(result_key):
            return GateOutcome(advance=False, reason="EvaluationResult missing")
        if not await ctx.artifact_store.exists(verdict_key):
            return GateOutcome(advance=False, reason="ContextualVerdict missing")
        return GateOutcome(advance=True, reason="evaluation artifacts present")

    return _gate


def _make_gate_evaluation_failed() -> Callable[[GateContext], Awaitable[GateOutcome]]:
    async def _gate(ctx: GateContext) -> GateOutcome:
        error_key = EVALUATION_ERROR_KEY_TEMPLATE.format(cg_id=ctx.entity_id)
        if await ctx.artifact_store.exists(error_key):
            return GateOutcome(advance=True, reason="evaluation error file present")
        return GateOutcome(advance=False)

    return _gate


# === Dispatch-handler factories =============================================


async def _fanout_manifest_hash(
    *,
    metadata_store: Any,
    artifact_store: Any,
    cg_id: str,
    entries: Sequence[EntryRecord],
) -> None:
    """Write ``harness_api_manifest_hash`` to every entry atomically.

    Per DEC-033 #3: the "atomic per-entry write" semantic. Reads the
    manifest from ArtifactStore, extracts ``content_hash``, then
    issues per-entry transitions inside one
    :meth:`MetadataStore.transaction()` block.

    Used by both:

    * The ``implementing → implemented`` success-gate body
      (canonical phase-1 entry path), and
    * :func:`_make_dispatch_manifest_fanout`'s on-entry dispatch
      (kept for symmetry / phase-3 paths that may eventually exist;
      see DEC-033 #3 / `03` §3.3 design rationale).

    Raises if the manifest is missing or has an empty ``content_hash``;
    on per-entry CAS conflict, the transaction rolls back and the
    exception propagates (the gate body's caller treats this as
    advance=False so the next cycle re-evaluates).
    """
    from smai_runtime import HarnessAPIManifest  # noqa: PLC0415

    manifest_key = HARNESS_MANIFEST_KEY_TEMPLATE.format(cg_id=cg_id)
    manifest_bytes = await artifact_store.get(manifest_key)
    manifest = HarnessAPIManifest.model_validate_json(manifest_bytes)
    if not manifest.content_hash:
        raise ValueError("manifest.content_hash is empty")

    async with await metadata_store.transaction() as txn:
        for entry in entries:
            await txn.transition_entry_state(
                entry.id,
                entry.version,
                entry.state,  # pyright: ignore[reportArgumentType] — preserve current state
                harness_api_manifest_hash=manifest.content_hash,
            )


def _make_dispatch_manifest_fanout() -> Callable[[DispatchContext], Awaitable[DispatchOutcome]]:
    """``implemented`` on-entry dispatch — idempotent re-fanout.

    The canonical fanout happens inside the ``implementing →
    implemented`` success-gate body (see
    :func:`_make_gate_harness_succeeded_advance`) — the engine's
    phase-1 path doesn't fire ``on_entry_dispatch`` on the target
    state, so the gate body owns the work.

    This dispatch handler is the phase-3 idempotent counterpart: if a
    future amendment introduces a phase-3 path into ``implemented``,
    the dispatch fires here, reads the entries (which already have
    the hash from the gate body's earlier fanout), and skips entries
    whose hash already matches. The handler is a no-op in the v1
    happy path; documented as a structural seam.
    """

    async def _dispatch(ctx: DispatchContext) -> DispatchOutcome:
        cg_id = ctx.entity_id
        manifest_key = HARNESS_MANIFEST_KEY_TEMPLATE.format(cg_id=cg_id)
        try:
            entries = await _list_all_entries_for_cg(ctx.metadata_store, cg_id)
            await _fanout_manifest_hash(
                metadata_store=ctx.metadata_store,
                artifact_store=ctx.artifact_store,
                cg_id=cg_id,
                entries=entries,
            )
        except ArtifactNotFound:
            return DispatchOutcome(error=f"manifest not found at {manifest_key!r}")
        except ConflictError as exc:
            # All entries already at the right hash + version → CAS fails
            # because version mismatch. The gate body already wrote it
            # on phase-1 entry. We treat as success (idempotent).
            del exc
            return DispatchOutcome()
        except ValueError as exc:
            return DispatchOutcome(error=str(exc))
        return DispatchOutcome()

    return _dispatch


def _make_dispatch_runs(
    *,
    seeds: Sequence[int] = (0,),
    image: str = "smai-runtime:dev",
) -> Callable[[DispatchContext], Awaitable[DispatchOutcome]]:
    """``running`` on-entry dispatch — peer-spec coordination per `03` §3.9.

    Lifted from the Phase-2 inline approximation per Task 3.E3 / DEC-034
    #3. For each ``(entry × seed)`` pair where the entry is
    ``implemented``, this handler creates one :class:`RunRecord` in
    ``pending`` state and returns. The :class:`RunRecord` sub-spec
    (:mod:`.run_record`) drives each run through
    ``pending → submitted → {succeeded | failed | inconclusive}`` via
    the engine's standard write-first dispatch + phase-1 polling — no
    inline :class:`Compute` submit and no synchronous polling here.

    Cross-spec coordination is via the existing sibling-read pattern
    in the ``running → evaluating`` gate (per `05` §1.3): once every
    child run reaches a terminal state, the next CG-spec poll cycle's
    phase-2 ``get_ready_to_evaluate`` discovery picks up the CG and
    the gate fires.

    Args:
        seeds: Seeds to dispatch per entry. v1 default ``(0,)`` keeps
            the smoke-test fast; production deployments override.
        image: Reserved for forward-compatibility — the per-run
            container image now belongs to the run sub-spec's
            ``Compute.submit`` dispatch, not this handler. Kept on the
            signature so :func:`build_cg_execution_spec`'s
            ``runtime_image`` plumbing continues to flow through; can
            be removed once a future refactor consolidates the
            CG-execution and run sub-spec construction.
    """
    del image  # see docstring — image now flows through the run sub-spec

    async def _dispatch(ctx: DispatchContext) -> DispatchOutcome:
        from datetime import UTC, datetime  # noqa: PLC0415

        cg_id = ctx.entity_id
        entries = await _list_all_entries_for_cg(ctx.metadata_store, cg_id)
        runnable = [e for e in entries if e.state == "implemented"]
        if not runnable:
            return DispatchOutcome(error="no implemented entries to run")

        for entry in runnable:
            for seed in seeds:
                run_id = f"run_{cg_id}_{entry.id}_{seed}"
                now = datetime.now(UTC)
                run = RunRecord(
                    id=run_id,
                    cg_id=cg_id,
                    entry_id=entry.id,
                    seed=seed,
                    state="pending",
                    created_at=now,
                    updated_at=now,
                )
                await ctx.metadata_store.create_run(run)
        return DispatchOutcome()

    return _dispatch


def _make_dispatch_evaluation(
    *,
    llm_for_contextual_evaluator: LlmProvider,
    dispatch_trace: list[str] | None = None,
) -> Callable[[DispatchContext], Awaitable[DispatchOutcome]]:
    """``evaluating`` on-entry dispatch — DEC-034 #2.

    Sequential dispatch:

    1. **Mechanical evaluator first.** Reads :class:`ValidationConfig`
       from ArtifactStore + assembles :class:`RawMetrics` from per-run
       metrics artifacts; calls :func:`smai_core.evaluate`. Writes
       :class:`EvaluationResult` to ArtifactStore on success; writes an
       error file on failure.
    2. **Contextual evaluator second.** Calls
       :func:`run_contextual_evaluation` with the
       :class:`EvaluationResult` + CG metadata. Writes
       :class:`ContextualVerdict` to ArtifactStore on success; writes
       an error file on failure.

    On failure of either step the handler does not raise — it writes
    an error artifact and returns. The next cycle's
    ``evaluating → evaluation_failed`` ``dispatch_time`` edge picks up
    the error file and routes the CG to the failure terminal.

    The "ordering" assertion the acceptance criteria require is
    satisfied by the source-level sequence; tests inject a
    ``dispatch_trace`` list and assert the order of appends.

    Args:
        llm_for_contextual_evaluator: :class:`LlmProvider` for the
            contextual evaluator role.
        dispatch_trace: Optional spy hook — when provided, the handler
            appends ``"mechanical_evaluator"`` then
            ``"contextual_evaluator"`` to record the per-step ordering
            for test assertions. Production deployments leave this
            ``None``.
    """

    async def _dispatch(ctx: DispatchContext) -> DispatchOutcome:
        cg_id = ctx.entity_id
        result_key = EVALUATION_RESULT_KEY_TEMPLATE.format(cg_id=cg_id)
        verdict_key = CONTEXTUAL_VERDICT_KEY_TEMPLATE.format(cg_id=cg_id)
        error_key = EVALUATION_ERROR_KEY_TEMPLATE.format(cg_id=cg_id)

        # === Step 1: Mechanical evaluator (per DEC-034 #2 ordering) =========
        if dispatch_trace is not None:
            dispatch_trace.append("mechanical_evaluator")

        validation_config_key = VALIDATION_CONFIG_KEY_TEMPLATE.format(cg_id=cg_id)
        try:
            vc_bytes = await ctx.artifact_store.get(validation_config_key)
        except ArtifactNotFound as exc:
            await ctx.artifact_store.put(error_key, f"ValidationConfig missing: {exc}".encode())
            return DispatchOutcome()
        try:
            validation_config = ValidationConfig.model_validate_json(vc_bytes)
        except ValueError as exc:
            await ctx.artifact_store.put(
                error_key, f"ValidationConfig parse failed: {exc}".encode()
            )
            return DispatchOutcome()

        try:
            raw_metrics = await _build_raw_metrics_for_cg(ctx=ctx, cg_id=cg_id)
        except Exception as exc:  # noqa: BLE001 — assembly is best-effort
            await ctx.artifact_store.put(error_key, f"RawMetrics assembly failed: {exc}".encode())
            return DispatchOutcome()

        try:
            evaluation_result: EvaluationResult = evaluate(validation_config, raw_metrics)
        except RawMetricsShapeError as exc:
            await ctx.artifact_store.put(error_key, f"RawMetricsShapeError: {exc}".encode())
            return DispatchOutcome()
        except Exception as exc:  # noqa: BLE001 — programming-error catch-all per `06` §1
            await ctx.artifact_store.put(
                error_key,
                f"mechanical evaluator raised: {type(exc).__name__}: {exc}".encode(),
            )
            return DispatchOutcome()

        await ctx.artifact_store.put(
            result_key, evaluation_result.model_dump_json().encode("utf-8")
        )

        # === Step 2: Contextual evaluator =================================
        if dispatch_trace is not None:
            dispatch_trace.append("contextual_evaluator")

        from smai_agents.agents.contextual_evaluator import (  # noqa: PLC0415
            CGMetadata,
            ContextualEvaluatorEntry,
            ContextualEvaluatorInput,
            run_contextual_evaluation,
        )

        cg = await ctx.metadata_store.get_cg(cg_id)
        if cg is None:
            await ctx.artifact_store.put(
                error_key, f"CG {cg_id!r} not found for contextual eval".encode()
            )
            return DispatchOutcome()
        try:
            harness_contract_bytes = await ctx.artifact_store.get(
                HARNESS_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id)
            )
        except ArtifactNotFound as exc:
            await ctx.artifact_store.put(
                error_key,
                f"contextual eval failed: HarnessContract missing: {exc}".encode(),
            )
            return DispatchOutcome()
        harness_contract = HarnessContract.model_validate_json(harness_contract_bytes)
        entries = await _list_all_entries_for_cg(ctx.metadata_store, cg_id)

        ctx_entries: list[ContextualEvaluatorEntry] = []
        for entry in entries:
            technique_id = entry.technique_id
            technique_name = technique_id or "baseline"
            description = ""
            try:
                _ = await ctx.artifact_store.get(
                    TECHNIQUE_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id, entry_id=entry.id)
                )
                # The TechniqueContract has no description field per `02`
                # §7.5 (post-Decision-1 narrowing); the agent-side
                # `EntryUnderReview.technique_description` is informational
                # only. Fall back to an empty string; the LLM grounds its
                # narrative on the technique_id + name + body.
                description = ""
            except ArtifactNotFound:
                description = ""
            ctx_entries.append(
                ContextualEvaluatorEntry(
                    entry_id=entry.id,
                    technique_id=technique_id,
                    technique_name=technique_name,
                    technique_description=description,
                    is_baseline=entry.is_baseline,
                )
            )

        comparison_rule: ComparisonRule = validation_config.body.comparison.rule
        cg_metadata = CGMetadata(
            cg_id=cg_id,
            cg_name=cg_id,
            cg_description="",
            factor_dimension=harness_contract.body.factor.name,
            factor_type=harness_contract.body.factor.type,
            controlled_conditions=[],
            entries=ctx_entries,
        )
        contextual_input = ContextualEvaluatorInput(
            cg_metadata=cg_metadata,
            evaluation_result=evaluation_result,
            comparison_rule=comparison_rule,
            comparison_rationale=None,
        )

        try:
            verdict: ContextualVerdict = await run_contextual_evaluation(
                llm=llm_for_contextual_evaluator,
                input=contextual_input,
            )
        except Exception as exc:  # noqa: BLE001
            await ctx.artifact_store.put(
                error_key,
                f"contextual eval raised: {type(exc).__name__}: {exc}".encode(),
            )
            return DispatchOutcome()

        await ctx.artifact_store.put(verdict_key, verdict.model_dump_json().encode("utf-8"))
        return DispatchOutcome()

    return _dispatch


async def _build_raw_metrics_for_cg(
    *,
    ctx: DispatchContext,
    cg_id: str,
) -> RawMetrics:
    """Aggregate per-run metrics into :class:`RawMetrics` for evaluator input.

    Walks every :class:`EntryRecord` of the CG, then every
    :class:`RunRecord` of the entry. For ``succeeded`` runs the
    metrics-artifact key is derived from ``(cg_id, entry_id, seed)``
    via :data:`RUN_METRICS_KEY_TEMPLATE` (the run sub-spec doesn't
    persist :attr:`RunRecord.raw_metrics_artifact_key` per Task 3.E3 —
    phase-1's CAS path can't pass arbitrary fields from the gate body
    and §1.3 forbids gates from writing entity state, so the canonical
    derivation is the source of truth). Reads the metrics JSON and
    runs it through :func:`smai_runtime.build_seed_run_outcome` to
    project to the :class:`SeedRunOutcome` shape the evaluator expects.

    Failed / inconclusive runs are encoded as ``completed=False`` with
    the ``failure_reason`` carried through.
    """
    import json  # noqa: PLC0415

    from smai_runtime import build_seed_run_outcome  # noqa: PLC0415

    harness_bytes = await ctx.artifact_store.get(HARNESS_CONTRACT_KEY_TEMPLATE.format(cg_id=cg_id))
    harness_contract = HarnessContract.model_validate_json(harness_bytes)

    by_entry: dict[str, EntryMetrics] = {}
    entries = await _list_all_entries_for_cg(ctx.metadata_store, cg_id)
    for entry in entries:
        seed_outcomes: dict[int, SeedRunOutcome] = {}
        cursor: str | None = None
        while True:
            page = await ctx.metadata_store.list_runs_for_entry(entry.id, limit=100, cursor=cursor)
            for run in page.items:
                if run.state == "succeeded":
                    metrics_key = RUN_METRICS_KEY_TEMPLATE.format(
                        cg_id=cg_id, entry_id=run.entry_id, seed=run.seed
                    )
                    try:
                        body = await ctx.artifact_store.get(metrics_key)
                        metrics = json.loads(body)
                    except (ArtifactNotFound, ValueError):
                        seed_outcomes[run.seed] = SeedRunOutcome(
                            completed=False,
                            failure_reason="metrics artifact missing or unparseable",
                        )
                        continue
                    seed_outcomes[run.seed] = build_seed_run_outcome(
                        metrics, harness_contract, completed=True
                    )
                else:
                    seed_outcomes[run.seed] = SeedRunOutcome(
                        completed=False,
                        failure_reason=run.failure_reason or run.state,
                    )
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        if seed_outcomes:
            by_entry[entry.id] = EntryMetrics(entry_id=entry.id, seed_outcomes=seed_outcomes)
    return RawMetrics(by_entry=by_entry)


# === Public factories ========================================================


def build_cg_execution_spec(
    *,
    workspace_root: Path,
    llm_for_code_reviewer: LlmProvider,
    llm_for_contextual_evaluator: LlmProvider,
    seeds: Sequence[int] = (0,),
    runtime_image: str = "smai-runtime:dev",
    runtime_cpu_image: str = "smai-runtime-cpu:dev",
    max_review_attempts: int = 1,
    max_implementation_phase_attempts: int = 2,
    require_human_approval: bool = False,
    evaluation_dispatch_trace: list[str] | None = None,
    harness_builder_inline_runner: Any = None,
) -> PipelineSpec:
    """Build the SMAI CG-execution :class:`PipelineSpec`.

    Args:
        workspace_root: Filesystem root under which per-CG agent
            workspaces land. Threaded through to the harness builder
            dispatch handler factory (Task 2.B3).
        llm_for_code_reviewer: :class:`LlmProvider` the
            ``implemented → ...`` gate-trio uses for the code-review
            single-call agent (Task 2.B4). Production deployments
            instantiate one provider per role (typically
            ``BedrockProvider(model_id=get_model_for_task("code_reviewer"))``);
            tests inject a stub provider with canned responses.
        llm_for_contextual_evaluator: :class:`LlmProvider` for the
            contextual evaluator dispatched in the ``evaluating``
            state's on-entry handler.
        seeds: Per-entry training seeds the ``running`` dispatch
            iterates. Default ``(0,)`` for fast smoke tests.
        runtime_image: GPU container image for the runtime experiment
            jobs the ``running`` dispatch submits AND the harness
            builder's ``run_experiment`` validation tool when the
            CG's :class:`HarnessContract` requests GPU.
        runtime_cpu_image: CPU container image, used in place of
            ``runtime_image`` when the CG's harness contract requests
            CPU. Threaded into the harness-builder session so the
            round-16 gpu-flag derivation pairs cleanly with the image
            (silently keeping the GPU image on a CPU-only experiment
            attempts to pull a tag the operator never built).
        max_review_attempts: DEC-016 retry budget — the number of
            ``implemented → implementing`` retries an entry is allowed
            before being marked ``implementation_failed``.
        max_implementation_phase_attempts: Round-10 retry budget for the
            CG-level harness-builder dispatch. Each (re-)entry into
            ``implementing`` bumps :attr:`ComparisonGroupRecord.implementation_phase_attempt`
            via the engine-synthesized step-1 CAS; once the count
            reaches this cap a dispatch failure routes the CG to the
            ``implementation_failed`` terminal via the engine-
            synthesized retry-exhausted edge. Default ``2`` (one retry
            beyond initial). Closes the round-10 latent infinite-retry
            on CG implementing-phase dispatch failures.
        require_human_approval: DEC-003 deployment-configurable flag.
            When ``True`` the ``implemented → running`` review-pass
            edge holds the CG in ``implemented`` until human approval
            (the API surface that flips the approval flag is
            `09-cli.md` Session-C territory). v1 default ``False``
            (auto-approve).
        evaluation_dispatch_trace: Optional spy hook for the test
            asserting mechanical-then-contextual ordering per the
            acceptance criteria. Production deployments leave ``None``.
        harness_builder_inline_runner: Test-only runner override threaded
            into :func:`make_dispatch_harness_build` as ``inline_runner``
            — replaces :func:`run_loop` as the harness-builder session
            runner. Production leaves it ``None``. Typed ``Any`` to keep
            this module loadable without smai-agents installed.

    Per-entity-kind structure: this returns the CG-level spec
    (entity_kind=``"cg"``); the entry-level spec is built by
    :func:`build_cg_entries_spec` in :mod:`.cg_entries`. Both register
    independently and coordinate via shared :class:`MetadataStore` /
    :class:`ArtifactStore` per `05` §5.3.
    """
    # Lazy import — the harness builder dispatch factory is in smai-agents,
    # which depends transitively on smai-orchestrator (for tracking record
    # types). Lazy-loading here keeps the spec module loadable in
    # contexts where smai-agents isn't installed (e.g., the methodology-
    # layer Tier B integration path, where users construct PipelineSpec
    # objects programmatically without the agent fleet).
    from smai_agents.agents.harness_builder import (  # noqa: PLC0415
        make_dispatch_harness_build,
    )

    harness_builder_dispatch = make_dispatch_harness_build(
        workspace_root=workspace_root,
        runtime_image=runtime_image,
        runtime_cpu_image=runtime_cpu_image,
        inline_runner=harness_builder_inline_runner,
    )

    states: list[StateDef] = [
        StateDef(name="draft"),
        StateDef(
            name="implementing",
            on_entry_dispatch=DispatchAction(
                name="cg.dispatch_implementation_phase",
                handler=cast(
                    Callable[[DispatchContext], Awaitable[DispatchOutcome]],
                    harness_builder_dispatch,
                ),
                pool=POOL_AGENTS,
                handle_field="harness_job_handle",
                # Round 10: a dispatch failure here (e.g. Modal
                # ``Sandbox.create`` failing pre-submit, or the harness
                # builder agent's session raising before submitting a
                # validation job) used to infinitely roll back to
                # ``draft`` and re-dispatch; the synthesized retry-
                # exhausted terminal closes that loop.
                retry_policy=RetryPolicy(
                    attempt_counter_field="implementation_phase_attempt",
                    max_attempts=max_implementation_phase_attempts,
                    on_exhaustion_target_state="implementation_failed",
                    on_exhaustion_reason=("implementation-phase retry budget exhausted; terminal"),
                ),
            ),
        ),
        StateDef(
            name="implemented",
            on_entry_dispatch=DispatchAction(
                name="cg.dispatch_manifest_fanout",
                handler=_make_dispatch_manifest_fanout(),
                pool=POOL_INLINE,
            ),
        ),
        StateDef(
            name="running",
            on_entry_dispatch=DispatchAction(
                name="cg.dispatch_runs",
                handler=_make_dispatch_runs(seeds=seeds, image=runtime_image),
                pool=POOL_RUNS,
            ),
        ),
        StateDef(
            name="evaluating",
            on_entry_dispatch=DispatchAction(
                name="cg.dispatch_evaluation",
                handler=_make_dispatch_evaluation(
                    llm_for_contextual_evaluator=llm_for_contextual_evaluator,
                    dispatch_trace=evaluation_dispatch_trace,
                ),
                pool=POOL_INLINE,
            ),
        ),
        StateDef(name="complete", is_terminal=True),
        StateDef(name="implementation_failed", is_terminal=True),
        StateDef(name="running_failed", is_terminal=True),
        StateDef(name="evaluation_failed", is_terminal=True),
    ]

    edges: list[EdgeDef] = [
        EdgeDef(
            name="cg.draft → implementing",
            from_state="draft",
            target_state="implementing",
            gate_rule=_make_gate_draft_ready(),
            fires_on="dispatch_time",
        ),
        # Round 14: the harness builder runs in-process (no external
        # Compute job to poll), so these edges fire on ``dispatch_time``
        # — the next worker cycle re-evaluates them against the
        # artifacts the in-process session published, exactly like the
        # proposal spec's ``designing → designed`` edge. A *harness
        # build* failure (the agent did not emit a manifest) is surfaced
        # by the dispatch handler as a ``DispatchOutcome`` error and
        # handled by the engine's ``_handle_dispatch_failure`` + the
        # ``implementing`` state's RetryPolicy — there is no separate
        # ``harness build failed`` edge. Declared success-first so a
        # passing harness short-circuits before the no-survivors edge.
        EdgeDef(
            name="cg.implementing → implemented",
            from_state="implementing",
            target_state="implemented",
            gate_rule=_make_gate_harness_succeeded_advance(),
            fires_on="dispatch_time",
        ),
        EdgeDef(
            name="cg.implementing → implementation_failed (no survivors)",
            from_state="implementing",
            target_state="implementation_failed",
            gate_rule=_make_gate_harness_succeeded_no_survivors(),
            fires_on="dispatch_time",
        ),
        # implemented edge trio — success first, retry second, terminal third.
        EdgeDef(
            name="cg.implemented → running (review pass)",
            from_state="implemented",
            target_state="running",
            gate_rule=_make_gate_review_pass(
                llm_for_code_reviewer=llm_for_code_reviewer,
                require_human_approval=require_human_approval,
            ),
            fires_on="dispatch_time",
        ),
        EdgeDef(
            name="cg.implemented → implementing (review fail, retry budget)",
            from_state="implemented",
            target_state="implementing",
            gate_rule=_make_gate_review_fail_with_retry(
                llm_for_code_reviewer=llm_for_code_reviewer,
                max_review_attempts=max_review_attempts,
            ),
            fires_on="dispatch_time",
        ),
        EdgeDef(
            name="cg.implemented → implementation_failed (review fail, no survivors)",
            from_state="implemented",
            target_state="implementation_failed",
            gate_rule=_make_gate_review_fail_no_survivors(
                llm_for_code_reviewer=llm_for_code_reviewer,
            ),
            fires_on="dispatch_time",
        ),
        # running edges — all-runs-terminal predicates fire from dispatch_time
        # (inline approximation per `03` §3.9; the runs are progressed
        # synchronously by the runs dispatch handler).
        EdgeDef(
            name="cg.running → evaluating (≥1 succeeded)",
            from_state="running",
            target_state="evaluating",
            gate_rule=_make_gate_runs_all_terminal_with_success(),
            fires_on="dispatch_time",
        ),
        EdgeDef(
            name="cg.running → running_failed (zero succeeded)",
            from_state="running",
            target_state="running_failed",
            gate_rule=_make_gate_runs_all_terminal_zero_success(),
            fires_on="dispatch_time",
        ),
        # evaluating edges — handler writes artifacts, gate-pair decides
        # complete vs evaluation_failed on the next cycle.
        EdgeDef(
            name="cg.evaluating → complete",
            from_state="evaluating",
            target_state="complete",
            gate_rule=_make_gate_evaluation_complete(),
            fires_on="dispatch_time",
        ),
        EdgeDef(
            name="cg.evaluating → evaluation_failed",
            from_state="evaluating",
            target_state="evaluation_failed",
            gate_rule=_make_gate_evaluation_failed(),
            fires_on="dispatch_time",
        ),
    ]

    pools: list[ConcurrencyPool] = [
        ConcurrencyPool(name=POOL_RUNS, limit=4, priority=100),
        ConcurrencyPool(name=POOL_AGENTS, limit=6, priority=50),
        ConcurrencyPool(name=POOL_INLINE, limit=16, priority=70),
    ]

    scheduling_queries: dict[str, SchedulingQueryRef] = {
        "draft": SchedulingQueryRef(
            name="get_ready_for_harness_build",
            fn=drain_to_list("get_ready_for_harness_build"),
        ),
        "implementing": SchedulingQueryRef(
            name="get_in_flight_harness_build",
            fn=drain_to_list("get_in_flight_harness_build"),
        ),
        "implemented": SchedulingQueryRef(
            name="get_ready_for_review_and_run",
            fn=drain_to_list("get_ready_for_review_and_run"),
        ),
        "running": SchedulingQueryRef(
            name="get_ready_to_evaluate",
            fn=drain_to_list("get_ready_to_evaluate"),
        ),
        "evaluating": SchedulingQueryRef(
            name="get_in_flight_evaluation",
            fn=drain_to_list("get_in_flight_evaluation"),
        ),
    }

    return PipelineSpec(
        name=CG_EXECUTION_SPEC_NAME,
        entity_kind="cg",
        initial_state="draft",
        states=states,
        edges=edges,
        pools=pools,
        scheduling_queries=scheduling_queries,
    )


def drain_to_list(method_name: str) -> Callable[[Any], Awaitable[list[Any]]]:
    """Compose a :data:`SchedulingQueryRef.fn` callable that drains all pages.

    The :class:`SchedulingQueryRef` consumes a callable
    :class:`MetadataStore` → ``list[StateDrivenRecord]``; the Protocol's
    methods return a :class:`CursorPage` per DEC-035 #1. This wrapper
    walks all pages and concatenates the items so the worker loop sees
    the full candidate set per cycle.
    """

    async def _fn(store: Any) -> list[Any]:
        method = getattr(store, method_name)
        out: list[Any] = []
        cursor: str | None = None
        while True:
            page = await method(limit=100, cursor=cursor)
            out.extend(page.items)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        return out

    _fn.__qualname__ = f"drain_to_list[{method_name}].fn"
    return _fn


__all__ = [
    "CG_ENTRIES_SPEC_NAME",
    "CG_EXECUTION_SPEC_NAME",
    "CODE_REVIEW_KEY_TEMPLATE",
    "CONTEXTUAL_VERDICT_KEY_TEMPLATE",
    "EVALUATION_ERROR_KEY_TEMPLATE",
    "EVALUATION_RESULT_KEY_TEMPLATE",
    "HARNESS_CONTRACT_KEY_TEMPLATE",
    "HARNESS_MANIFEST_KEY_TEMPLATE",
    "HARNESS_VALIDATION_KEY_TEMPLATE",
    "POOL_AGENTS",
    "POOL_INLINE",
    "POOL_RUNS",
    "RUN_IN_PROGRESS_STATES",
    "RUN_METRICS_KEY_TEMPLATE",
    "RUN_TERMINAL_STATES",
    "TECHNIQUE_CONTRACT_KEY_TEMPLATE",
    "TECHNIQUE_VALIDATION_KEY_TEMPLATE",
    "VALIDATION_CONFIG_KEY_TEMPLATE",
    "build_cg_execution_spec",
]
