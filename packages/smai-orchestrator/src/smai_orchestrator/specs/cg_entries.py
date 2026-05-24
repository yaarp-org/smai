"""CG-entries :class:`PipelineSpec` instance — per-entry implementer dispatch.

Per ``03-state-machine.md`` §3.1 / §3.4 — the entry-level half of the
CG-execution lifecycle. :class:`PipelineSpec.entity_kind` is single-
valued per `05` §5.1, so the per-entity-kind structure ships as two
peer specs registered alongside each other:

* :func:`build_cg_execution_spec` (in :mod:`.cg_execution`) drives
  :class:`ComparisonGroupRecord` entities.
* :func:`build_cg_entries_spec` (this module) drives :class:`EntryRecord`
  entities through their lifecycle: ``pending → implementing →
  implemented | implementation_failed``.

Cross-spec coordination per `05` §5.3:

* The CG spec writes :class:`EntryRecord` rows via the proposal-pipeline
  registration transaction (Phase-3 territory; tests pre-create the
  rows directly).
* The CG spec's ``implementing → implemented`` success-gate predicate
  reads ``EntryRecord.state`` to decide all-children-terminal.
* The CG spec's ``implemented → implementing`` retry route writes
  ``EntryRecord.state="pending"`` + bumps ``implementation_attempt`` —
  this spec's phase-2 discovery (``get_ready_to_implement_entry``)
  picks them up on the next cycle.

The entry spec itself is narrow: per-entry technique-implementer
dispatch, phase-1 polling for completion, and a validation-results
gate that decides ``implemented`` vs ``implementation_failed``.

Spec ambiguities resolved:

* **Harness-manifest-committed gate:** an entry must not be dispatched
  to the technique implementer until the harness builder has committed
  the harness manifest — otherwise the implementer fails on a missing
  ``harness/manifest.json``. The scheduling query
  :meth:`MetadataStore.get_ready_to_implement_entry` only gates on the
  parent CG being ``implementing`` (a SQL predicate cannot see an
  :class:`ArtifactStore`), so this precondition is enforced in the
  ``pending → implementing`` gate body via an
  :class:`ArtifactStore.exists` check — see
  :func:`_make_entry_dispatch_ready_gate`. Round 15: the SQL query
  staying loose is a known perf-only point (an entry whose harness is
  not ready is discovered and held back each cycle rather than filtered
  at the query) — left for the backlog.
* **Additive-baseline pre-filter:** double-enforced (per `03` §3.4).
  The :meth:`MetadataStore.get_ready_to_implement_entry` query
  excludes baselines (``technique_id IS NULL``) at the SQL layer per
  `07` §5.6.3 / DEC-013, so they never enter this spec. The
  technique-implementer dispatch handler also short-circuits via
  :func:`smai_runtime.is_additive_baseline` per Task 2.B3, providing a
  belt-and-braces second layer for spec authors who reuse the dispatch
  factory in different contexts.
* **Validation-results gate:** the
  ``implementing → implemented`` success-gate reads
  ``code/validation_results.json`` from ArtifactStore per
  ``10-runtime-and-templates.md`` §5.4. The runtime emits
  ``{"passed": <bool>, ...}``; the gate body advances on
  ``passed=True``. The per-entry validation results are independent of
  the harness-level ``validation_results.json`` checked by the CG
  spec's harness-success gate.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

from smai_core.plugins import ArtifactNotFound

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
from smai_orchestrator.runtime.spec import PipelineSpec
from smai_orchestrator.specs.cg_execution import (
    CG_ENTRIES_SPEC_NAME,
    HARNESS_MANIFEST_KEY_TEMPLATE,
    POOL_AGENTS,
    POOL_INLINE,
    POOL_RUNS,
    TECHNIQUE_VALIDATION_KEY_TEMPLATE,
    drain_to_list,
    read_validation_results_pass,
)


def _make_entry_dispatch_ready_gate() -> Callable[[GateContext], Awaitable[GateOutcome]]:
    """``pending → implementing`` dispatch_time gate — entry side.

    The scheduling query :meth:`MetadataStore.get_ready_to_implement_entry`
    (per `07` §5.6.3) filters to entries whose state is ``pending``, whose
    parent CG is in ``implementing``, and which are not additive
    baselines. It does NOT — and cannot — check that the harness manifest
    exists: a SQL predicate has no view of an :class:`ArtifactStore`. So
    the "harness manifest committed" precondition is enforced HERE, in the
    gate body, via an :class:`ArtifactStore` ``exists`` check.

    Round 15: without this check an entry is dispatched while the harness
    builder is still running, the technique-implementer fails on a missing
    ``harness/manifest.json``, and the entry burns its round-11
    ``entry_dispatch_attempt`` budget on a transient precondition —
    terminally ``implementation_failed`` even though nothing about the
    entry's implementation actually failed. Returning ``advance=False``
    holds the entry at ``pending``: it is re-discovered and re-evaluated
    next worker cycle, and no ``entry_dispatch_attempt`` is consumed —
    that counter is bumped only by the round-10/11 RetryPolicy machinery
    on the dispatch CAS into ``implementing``, which an ``advance=False``
    gate never reaches.
    """

    async def _gate(ctx: GateContext) -> GateOutcome:
        entry = await ctx.metadata_store.get_entry(ctx.entity_id)
        if entry is None:
            return GateOutcome(advance=False, reason="entry not found")
        manifest_key = HARNESS_MANIFEST_KEY_TEMPLATE.format(cg_id=entry.cg_id)
        if not await ctx.artifact_store.exists(manifest_key):
            return GateOutcome(
                advance=False,
                reason=f"harness manifest not yet committed at {manifest_key!r}",
            )
        return GateOutcome(
            advance=True, reason="harness manifest committed; entry pending → implementing"
        )

    return _gate


async def _read_entry_validation(ctx: GateContext) -> tuple[bool, bool]:
    """Read the per-entry ``validation_results.json`` from ArtifactStore.

    Returns ``(present, passed)``. ``present`` is ``False`` when the
    artifact is missing — the technique-implementer dispatch publishes
    it after the in-process session, so absence means the agent never
    completed (a case the dispatch handler's completeness check already
    routes to ``_handle_dispatch_failure`` + the RetryPolicy).
    """
    entry = await ctx.metadata_store.get_entry(ctx.entity_id)
    if entry is None:
        return (False, False)
    validation_key = TECHNIQUE_VALIDATION_KEY_TEMPLATE.format(cg_id=entry.cg_id, entry_id=entry.id)
    try:
        payload = await ctx.artifact_store.get(validation_key)
    except ArtifactNotFound:
        return (False, False)
    return (True, read_validation_results_pass(payload))


def _make_entry_validation_pass_gate() -> Callable[[GateContext], Awaitable[GateOutcome]]:
    """``implementing → implemented`` dispatch_time gate — entry side.

    Round 14: the technique implementer runs in-process (no external
    Compute job — ``ctx.job_outcome`` is always None), so this is a
    ``dispatch_time`` gate re-evaluated each worker cycle. Reads the
    per-entry ``validation_results.json`` the dispatch handler
    published. Advances on ``{"passed": true}``; otherwise the
    failure-terminal edge fires.
    """

    async def _gate(ctx: GateContext) -> GateOutcome:
        present, passed = await _read_entry_validation(ctx)
        if not present:
            return GateOutcome(advance=False, reason="validation_results.json not present")
        if not passed:
            return GateOutcome(advance=False, reason="validation did not pass")
        return GateOutcome(advance=True, reason="validation passed")

    return _gate


def _make_entry_validation_fail_gate() -> Callable[[GateContext], Awaitable[GateOutcome]]:
    """``implementing → implementation_failed`` dispatch_time gate — entry side.

    Fires on a *positive* failure signal: ``validation_results.json`` is
    present AND reports ``passed != true``. It deliberately does NOT
    advance on a *missing* artifact — a missing result means the
    in-process agent did not complete (a hard failure), which the
    dispatch handler surfaces as a ``DispatchOutcome`` error so the
    engine's ``_handle_dispatch_failure`` + the ``implementing`` state's
    RetryPolicy retry/terminate it. Advancing here on absence would
    pre-empt that retry budget (round-11 ``entry_dispatch_attempt``).
    Declared after the success edge so a passing run short-circuits.
    """

    async def _gate(ctx: GateContext) -> GateOutcome:
        present, passed = await _read_entry_validation(ctx)
        if present and not passed:
            return GateOutcome(advance=True, reason="validation did not pass; terminal")
        return GateOutcome(advance=False, reason="no positive validation-failure signal")

    return _gate


def build_cg_entries_spec(
    *,
    workspace_root: Path,
    runtime_image: str = "smai-runtime:dev",
    max_entry_dispatch_attempts: int = 2,
    technique_implementer_inline_runner: Any = None,
) -> PipelineSpec:
    """Build the SMAI CG-entries :class:`PipelineSpec`.

    Args:
        workspace_root: Filesystem root for per-entry workspaces.
            Threaded into the technique-implementer dispatch handler
            so it materializes ``<workspace_root>/<cg_id>/<entry_id>/``
            for the in-process agent.
        runtime_image: Container image for ``run_experiment`` validation
            jobs the implementer agent submits during its multi-turn
            loop. v1 default ``smai-runtime:dev`` per Task 2.D1.
        technique_implementer_inline_runner: Test-only runner override
            threaded into :func:`make_dispatch_technique_implementation`
            as ``inline_runner``. Production leaves it ``None``. Typed
            ``Any`` to keep this module loadable without smai-agents.
        max_entry_dispatch_attempts: Round-11 retry budget for the
            per-entry technique-implementer dispatch. Each (re-)entry
            into ``implementing`` bumps
            :attr:`EntryRecord.entry_dispatch_attempt` via the
            engine-synthesized step-1 CAS; once the count reaches this
            cap a dispatch failure routes the entry to the
            ``implementation_failed`` terminal via the engine-
            synthesized retry-exhausted edge. Default ``2`` (one retry
            beyond the initial attempt), parity with
            ``cg_execution``'s ``max_implementation_phase_attempts``.
            Closes the round-10 latent infinite-retry on entry
            implementation-dispatch failures.

    Returns:
        A :class:`PipelineSpec` with ``entity_kind="entry"``,
        registered separately from the CG-execution spec per `05` §5.1's
        one-spec-per-entity-kind constraint.
    """
    from smai_agents.agents.technique_implementer import (  # noqa: PLC0415
        make_dispatch_technique_implementation,
    )

    technique_implementer_dispatch = make_dispatch_technique_implementation(
        workspace_root=workspace_root,
        run_experiment_image=runtime_image,
        inline_runner=technique_implementer_inline_runner,
    )

    states: list[StateDef] = [
        StateDef(name="pending"),
        StateDef(
            name="implementing",
            on_entry_dispatch=DispatchAction(
                name="entry.dispatch_technique_implementation",
                handler=cast(
                    Callable[[DispatchContext], Awaitable[DispatchOutcome]],
                    technique_implementer_dispatch,
                ),
                pool=POOL_AGENTS,
                handle_field="implementation_job_handle",
                # Round 11: a dispatch-handler failure here (the
                # technique-implementer agent's session raising before
                # it submits an implementation job) used to roll back
                # to ``pending`` and re-dispatch forever — the entry
                # spec declared no dispatch-failure terminal edge.
                # Round 10 left this ``retry_policy=None`` because the
                # entry's ``implementation_attempt`` counter is owned by
                # the CG-level review-retry gate (double-count hazard);
                # ``entry_dispatch_attempt`` is the separate counter that
                # closes the loop without colliding with that gate.
                retry_policy=RetryPolicy(
                    attempt_counter_field="entry_dispatch_attempt",
                    max_attempts=max_entry_dispatch_attempts,
                    on_exhaustion_target_state="implementation_failed",
                    on_exhaustion_reason=(
                        "entry implementation-dispatch retry budget exhausted; terminal"
                    ),
                ),
            ),
        ),
        StateDef(name="implemented", is_terminal=True),
        StateDef(name="implementation_failed", is_terminal=True),
    ]

    edges: list[EdgeDef] = [
        EdgeDef(
            name="entry.pending → implementing",
            from_state="pending",
            target_state="implementing",
            gate_rule=_make_entry_dispatch_ready_gate(),
            fires_on="dispatch_time",
        ),
        # Round 14: the technique implementer runs in-process (no
        # external Compute job to poll), so these edges fire on
        # ``dispatch_time`` — re-evaluated each worker cycle against the
        # ``validation_results.json`` the dispatch handler published. A
        # *hard* failure (the agent never produced validation_results)
        # is surfaced by the dispatch handler as a ``DispatchOutcome``
        # error and handled by ``_handle_dispatch_failure`` + the
        # ``implementing`` state's RetryPolicy — there is no separate
        # ``job failed`` edge. Declared validation-pass first so a
        # passing run short-circuits the failure edge.
        EdgeDef(
            name="entry.implementing → implemented (validation pass)",
            from_state="implementing",
            target_state="implemented",
            gate_rule=_make_entry_validation_pass_gate(),
            fires_on="dispatch_time",
        ),
        EdgeDef(
            name="entry.implementing → implementation_failed (validation fail)",
            from_state="implementing",
            target_state="implementation_failed",
            gate_rule=_make_entry_validation_fail_gate(),
            fires_on="dispatch_time",
        ),
    ]

    # Re-declare the same pool set as the CG-execution spec — the worker
    # loop's slot accounting aggregates across specs (per `05` §3.4), so
    # both specs must declare every pool they reference. ``runs`` and
    # ``inline`` aren't actually consumed by entry dispatches, but
    # declaring them keeps the two specs' pool definitions consistent
    # for the eventual worker-coordinator (Task 3.G3).
    pools: list[ConcurrencyPool] = [
        ConcurrencyPool(name=POOL_RUNS, limit=4, priority=100),
        ConcurrencyPool(name=POOL_AGENTS, limit=6, priority=50),
        ConcurrencyPool(name=POOL_INLINE, limit=16, priority=70),
    ]

    scheduling_queries: dict[str, SchedulingQueryRef] = {
        "pending": SchedulingQueryRef(
            name="get_ready_to_implement_entry",
            fn=drain_to_list("get_ready_to_implement_entry"),
        ),
        "implementing": SchedulingQueryRef(
            name="get_in_flight_entry_implementation",
            fn=drain_to_list("get_in_flight_entry_implementation"),
        ),
    }

    return PipelineSpec(
        name=CG_ENTRIES_SPEC_NAME,
        entity_kind="entry",
        initial_state="pending",
        states=states,
        edges=edges,
        pools=pools,
        scheduling_queries=scheduling_queries,
    )


__all__ = ["build_cg_entries_spec"]
