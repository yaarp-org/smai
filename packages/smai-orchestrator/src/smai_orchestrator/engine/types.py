"""Engine primitives — typed callables, contexts, and spec records.

Per ``designs/smai/05-orchestrator.md`` §1 (engine architecture) and §5
(pipeline-spec contract). The shapes here are the substrate Task 2.C3
composes into the full :class:`PipelineSpec` and Task 2.C2 wraps in the
worker loop. C1 ships only the primitive types plus a minimal
:class:`EngineSpec` umbrella sufficient for the synthetic-pipeline tests
in §3.3's acceptance criteria.

The engine is entity-kind-agnostic per `05` §1.1; the primitives here
carry ``entity_kind`` as a discriminator that the metadata-ops adapter
(:mod:`smai_orchestrator.engine._metadata_ops`) uses to dispatch per-kind
:class:`MetadataStore` methods. No SMAI-domain types appear in this
module; CG / proposal / paper concerns live in the pipeline-specs that
consume these types (Tasks 2.C4 / 3.E1 / 3.E2).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from smai_core.plugins import (
    ArtifactStore,
    Compute,
    EntityKind,
    JobHandle,
    JobStatus,
    LlmProvider,
    MetadataStore,
)

# Direct import of :class:`Checkpointer` so Pydantic can resolve the
# annotation at model-build time. The dependency stays acyclic because
# ``smai_orchestrator.checkpointer.types`` imports only stdlib + pydantic
# (not the engine), and the indirect path through ``checkpointer.__init__``
# resolves cleanly: ``engine/__init__.py`` loads ``engine.clock`` and
# ``engine.config`` before ``engine.types``, so by the time the
# checkpointer's artifact-flavor module asks for ``engine.clock`` it is
# already in :data:`sys.modules`.
from smai_orchestrator.checkpointer.types import Checkpointer
from smai_orchestrator.engine.config import EngineConfig

# Phase trigger discriminator for an :class:`EdgeDef`. Per `05` §1.2
# phase-keyed edge evaluation: phase-3 (dispatch_time) edges fire when
# the engine inspects an entity in a non-in-progress state; phase-1
# (job_succeeded / job_failed) edges fire when an external compute job
# attached to an in-progress state terminates.
PhaseTrigger = Literal["dispatch_time", "job_succeeded", "job_failed"]


# === Gate-rule shape (§1.2 / §1.3) ==========================================


class GateContext(BaseModel):
    """Read-only view passed to every gate rule (`05` §1.2).

    Gate rules are typed Python callables; they MUST NOT write entity
    state — state writes happen on transition (CAS in :mod:`dispatch`)
    and inside dispatch handlers. Reads are unrestricted: a gate rule
    may read entity state, sibling entities, and artifact bodies in
    deciding pass / fail.

    ``job_outcome`` is populated *only* for phase-1 evaluations
    (``fires_on=job_succeeded`` / ``fires_on=job_failed``) per `05` §9
    carry-forward #9 — the engine has already read the
    :class:`JobStatus` from :class:`Compute` and surfaces it here so
    phase-1 gate rules can branch on ``exit_code`` /
    ``failure_reason`` without re-reading.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    entity_kind: EntityKind
    entity_id: str
    entity_state: str
    entity_version: int
    metadata_store: MetadataStore
    artifact_store: ArtifactStore
    config: EngineConfig
    job_outcome: JobStatus | None = None


class GateOutcome(BaseModel):
    """Return value from a gate rule (`05` §1.2)."""

    model_config = ConfigDict(extra="forbid")

    advance: bool
    reason: str | None = None


GateRule = Callable[[GateContext], Awaitable[GateOutcome]]


class EdgeDef(BaseModel):
    """A typed transition between two states (`05` §1.2).

    Multiple edges may leave the same state; evaluation order is
    declaration order — the first edge whose gate returns
    ``advance=True`` wins. Edges are uniform across all gate kinds
    (artifact-contract checks, retry-budget rules, all-children-terminal
    checks, etc.) — the same callable shape, the same evaluation path.

    ``fires_on`` selects which phase evaluates the edge: phase-3 sees
    ``dispatch_time`` edges, phase-1 sees ``job_succeeded`` /
    ``job_failed`` edges.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    from_state: str
    target_state: str
    gate_rule: GateRule
    fires_on: PhaseTrigger = "dispatch_time"


# === Dispatch shape (§1.4) ==================================================


class DispatchContext(BaseModel):
    """Mutable-but-engine-coordinated context for a dispatch handler (`05` §1.4).

    The engine constructs one of these per dispatch invocation. Handlers
    perform side-effecting work — submit external compute jobs via
    :attr:`compute`, write artifacts via :attr:`artifact_store`, call
    LLMs via :attr:`llm`. The engine handles the write-first state
    transition and post-dispatch handle recording around the handler.

    ``llm`` is ``None`` when the deployment did not configure an
    :class:`LlmProvider`; handlers that require LLM access must guard.

    ``checkpointer`` is the engine-internal step-memoization seam (`05`
    §2 / DEC-025) — Task 2.C2 shipped :class:`Checkpointer` plus the
    two concrete flavors (:class:`MetadataStoreCheckpointer` /
    :class:`ArtifactStoreCheckpointer`). Handlers that want
    memoization wrap their work via
    :func:`smai_orchestrator.checkpointer.memoized`. ``None`` is the
    "no memoization configured" shape — handlers must guard.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    entity_kind: EntityKind
    entity_id: str
    entity_state: str
    entity_version: int
    metadata_store: MetadataStore
    artifact_store: ArtifactStore
    compute: Compute
    llm: LlmProvider | None
    config: EngineConfig
    checkpointer: Checkpointer | None = None


class DispatchOutcome(BaseModel):
    """Result returned by a :data:`DispatchHandler` (`05` §1.4).

    For external dispatches the handler submits one (or more) compute
    jobs and returns the resulting :class:`JobHandle` objects. The
    engine writes the first handle to the entity's configured
    :attr:`DispatchAction.handle_field` as step 3 of write-first
    ordering. For inline dispatches (single LLM call, no external
    compute) :attr:`submitted_handles` is empty and :attr:`handle_field`
    is ``None``.

    On failure the handler populates :attr:`error` with a brief
    diagnostic; the engine then rolls back the entity's state to the
    edge's ``from_state`` per `05` §1.4.

    The doc literally types this as ``list[str]`` in the §1.4 example;
    the records (``EntryRecord.implementation_job_handle``,
    ``RunRecord.compute_job_handle``, etc. per `01` §5) carry the typed
    :class:`JobHandle` Pydantic object, so we use ``list[JobHandle]``
    here for round-trip cleanliness. Flagged as a one-line spec
    reconciliation — the str/object choice is a single line in `05`
    §1.4 and a single line here.
    """

    model_config = ConfigDict(extra="forbid")

    submitted_handles: list[JobHandle] = []  # noqa: RUF012 — Pydantic deep-copies field defaults; safe.
    error: str | None = None


DispatchHandler = Callable[[DispatchContext], Awaitable[DispatchOutcome]]


class PostTerminalContext(BaseModel):
    """Read-only context the post-terminal hook receives (`05` §3.1 +
    agent_refactor Step 4 sub-PR B).

    Built by :func:`smai_orchestrator.engine.phase1.phase1_step` after a
    terminal-success CAS and handed to the dispatch action's
    :attr:`DispatchAction.post_terminal_handler` if one is configured.
    The hook performs side-effecting work that must happen exactly once
    per successful terminal observation (workspace harvest, in
    practice). It MUST NOT write entity state — state writes are the
    engine's territory.

    The ``dispatch_context`` field is a synthetic
    :class:`DispatchContext` reconstructed from the post-terminal
    entity. The hook's resolvers (e.g.
    :attr:`WorkspaceOutputs.destination`) consume it the same way the
    dispatch handler does so callers can share resolver implementations
    between dispatch and post-terminal.

    Best-effort by contract: the engine wraps the hook call in a
    try/except so a hook failure does not block the state-machine
    transition. The hook MAY raise; the engine logs and continues.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    entity_kind: EntityKind
    entity_id: str
    compute: Compute
    metadata_store: MetadataStore
    artifact_store: ArtifactStore
    config: EngineConfig
    job_handle: JobHandle
    dispatch_context: DispatchContext


PostTerminalHandler = Callable[[PostTerminalContext], Awaitable[None]]


class RetryPolicy(BaseModel):
    """Declarative retry bookkeeping for a :class:`DispatchAction` (round 10).

    Decentralized retry bookkeeping was the root cause of four rounds of
    "the system has no terminal failure gate for state X's retry budget"
    friction (rounds 5 / 6 / 8 / 10): each dispatched state had to manually
    bump its retry counter from inside the handler AND register a
    ``*_failed (retry exhausted)`` edge against itself. Forget either and
    the dispatch infinite-loops on errors. ``RetryPolicy`` centralizes
    both — the engine reads this declaration to:

    1. **Bump the counter on step-1 CAS.** When :func:`run_dispatch` fires
       the write-first transition into the dispatch state, the counter
       field listed here is included in the same ``fields=`` payload so
       the bump rides the state-transition CAS — no separate write, no
       version-bump race against the handler's own writes.
    2. **Synthesize a retry-exhausted terminal edge** in
       :func:`_handle_dispatch_failure`. When the handler fails, the
       engine prepends a synthesized edge to the dispatch state's
       outgoing-edge evaluation: ``<dispatch_state> → <on_exhaustion_target_state>``
       whose gate is "``record.<attempt_counter_field>`` >=
       :attr:`max_attempts`". The synthesized edge fires before any
       spec-declared manual edges, matching the round-8 declaration-
       order ordering ("retry-exhausted terminal pre-empts retry-one-
       more-time" — the gate that fires when the budget is exhausted
       MUST win over the gate that would otherwise re-fire the dispatch).

    Spec authors who want unbounded retries leave ``retry_policy=None``;
    that is the explicit opt-out shape. Every dispatched state with an
    ``*_attempt`` counter on its record SHOULD declare one — see
    :file:`packages/smai-orchestrator/src/smai_orchestrator/specs/RETRY_POLICY.md`.

    The companion CONTRIBUTING note: spec authors MUST NOT manually bump
    these counters in dispatch handler bodies, and MUST NOT manually
    declare ``*_failed (retry exhausted)`` edges — the engine does both
    when the policy is set, and double-bumping or duplicate-edge
    declarations are bugs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt_counter_field: str
    """Name of the integer field on the entity record to bump on each
    dispatch entry (e.g. ``"design_attempt"``). The field must exist on
    every record kind this :class:`DispatchAction` can target."""

    max_attempts: int
    """The retry-budget cap. The synthesized terminal edge fires when
    ``record.<attempt_counter_field>`` reaches this value — i.e. the
    *N*-th attempt has completed. Mirrors the manual gate predicate the
    round-5..8 friction installed:
    ``record.<attempt_counter_field> >= max_attempts``."""

    on_exhaustion_target_state: str
    """State name the synthesized terminal edge transitions into when
    the budget is exhausted (e.g. ``"failed"``,
    ``"implementation_failed"``). The state must be declared on the spec;
    the engine asserts this at spec construction."""

    on_exhaustion_reason: str
    """:class:`GateOutcome.reason` text recorded on the synthesized
    edge's ``transition_log`` row. Use the same wording the manual
    gates used pre-round-10 so audit queries (e.g.
    ``WHERE gate_outcome_reason LIKE '%retry budget exhausted%'``)
    keep matching."""


class DispatchAction(BaseModel):
    """A typed handler attached to a state's on-entry slot (`05` §1.4).

    The engine fires this when an entity transitions *into* the host
    state. ``pool`` resolution and slot accounting are C2's scope (Task
    2.C2) — the field exists for the worker loop to consume; C1 simply
    persists the value through the spec.

    ``handle_field`` is the name of the column on the underlying
    :class:`MetadataStore` record that should hold the submitted
    :class:`JobHandle` after step 2 of write-first ordering. Per `01`
    §5 the field name varies per entity-kind and per dispatch site
    (``harness_job_handle`` on a CG, ``implementation_job_handle`` on
    an entry, ``compute_job_handle`` on a run). The engine stays
    entity-kind-agnostic by reading this name from the spec rather
    than hard-coding the mapping. ``None`` for inline dispatches.

    ``retry_policy`` (round 10) makes per-state retry bookkeeping
    declarative — the engine bumps the counter on step-1 CAS and
    synthesizes a retry-exhausted terminal edge when the policy is set.
    ``None`` (the default) means "no retry budget" — the dispatch can
    fail indefinitely. See :class:`RetryPolicy`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    handler: DispatchHandler
    pool: str
    handle_field: str | None = None
    retry_policy: RetryPolicy | None = None
    post_terminal_handler: PostTerminalHandler | None = None
    """Optional hook invoked once on phase-1's terminal-success CAS.

    Wires workspace-harvest (and other "host attests-and-persists"
    side effects per architectural_decisions §7) onto a dispatch
    action without leaking the concern into the spec author's gate
    rules. ``None`` (the default) means "no post-terminal hook" — the
    seed-run shape. :func:`make_compute_dispatcher` populates this
    field via its returned :class:`DispatcherBundle` when
    :attr:`WorkspaceOutputs.destination` is non-``None``.

    Best-effort: phase-1 wraps the call in a try/except so a hook
    failure does not block the state-machine transition. Operator
    debugging surfaces the failure via the engine's logger and the
    transition_log audit trail (unaffected by the hook).
    """


class StateDef(BaseModel):
    """One state in the entity's lifecycle (`05` §1.1 / §5.1).

    Terminal states drop out of phase-2 discovery; their on-entry
    dispatch (if any) fires on the transition that lands the entity
    here, not on subsequent observation cycles.

    ``in_progress`` states are the targets of phase-3 dispatch
    transitions and the inputs to phase-1 polling; they should declare
    a non-``None`` :attr:`on_entry_dispatch`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    is_terminal: bool = False
    on_entry_dispatch: DispatchAction | None = None


# === EngineSpec — minimal pipeline-spec substrate (§5.1) ====================
#
# C1 ships a stripped-down spec carrying only the engine-driver primitives
# (states, edges, initial state, entity kind). Task 2.C3 composes the full
# `PipelineSpec` (concurrency pools, scheduling-query refs, name) per `05`
# §5.1 with this as the substrate. The split keeps C1's surface narrow and
# the C3 task's job clearly additive.


class ConcurrencyPool(BaseModel):
    """One named concurrency pool — pipeline-spec primitive (`05` §3.4).

    The model lives in :mod:`smai_orchestrator.engine.types` (alongside
    other spec primitives like :class:`StateDef` and :class:`EdgeDef`)
    rather than in :mod:`smai_orchestrator.worker.concurrency` because
    :class:`EngineSpec` references it on the :attr:`EngineSpec.pools`
    field — having the model in :mod:`engine.types` keeps the engine ↔
    worker dependency strictly one-way (worker imports engine, never
    the reverse).

    The slot-computation helpers (:func:`compute_pool_slots`,
    :func:`in_flight_states_for_pool`, :class:`PoolSlot`) live in
    :mod:`smai_orchestrator.worker.concurrency` and re-export this
    class for callers that prefer the worker-namespaced import.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    """Pool identifier referenced by :attr:`DispatchAction.pool`."""

    limit: int
    """Maximum number of in-flight jobs the pool may hold. The worker
    derives ``available = limit − count_with_in_flight_jobs(...)`` at
    each phase-3 cycle."""

    priority: int = 0
    """Sort key for inter-pool dispatch ordering — higher drains first
    (`05` §3.4). Default ``0`` is fine when only one pool exists."""

    fair_scheduling: bool = False
    """Per-pool opt-in for the engine's :attr:`EngineConfig.fair_scheduling`
    policy (`05` §3.4). ``True`` passes the engine policy through to
    the :class:`MetadataStore` plugin's discovery query as a hint;
    ``False`` overrides the engine policy to FIFO regardless of the
    global setting."""


class SchedulingQueryRef(BaseModel):
    """A pipeline-spec's reference to one :class:`MetadataStore` discovery
    query (`05` §3.2 / §5.1).

    The engine treats scheduling queries as opaque, named "give me
    candidates for state X" calls — :attr:`name` is the identifier the
    plugin's fair-scheduling discipline keys on, and :attr:`fn` is the
    callable the engine invokes per cycle. Per `05` §3.2 the query
    namespace is owned by the spec, not the engine; spec authors bind
    each :class:`MetadataStore` Protocol method to a :class:`SchedulingQueryRef`
    here.

    :attr:`fn` returns ``list[StateDrivenRecord]`` — the engine drains
    the page (or first page; cursor-based pagination is the spec
    author's concern under the C2 worker-cycle model). C3's full
    :class:`PipelineSpec` may revisit the cursor-handling shape.

    :attr:`fair_scheduling` is per-query opt-in for the engine's global
    :attr:`EngineConfig.fair_scheduling` policy (`05` §3.4) — passed
    through to the plugin as a hint; the plugin decides how to
    interpret. ``False`` (default) means FIFO regardless of the global
    policy.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    fn: Callable[[MetadataStore], Awaitable[list[Any]]]
    """Bound to ``Callable[[MetadataStore], Awaitable[list[StateDrivenRecord]]]``
    semantically — typed as ``list[Any]`` here to keep the field's
    runtime type out of the engine ↔ ``_metadata_ops`` import order.
    Worker code that consumes the return value treats each element as
    a :data:`StateDrivenRecord` (the union of pipeline-tracking record
    types from :mod:`smai_orchestrator.entities.tracking`)."""

    fair_scheduling: bool = False


class EngineSpec(BaseModel):
    """Pipeline-spec substrate driving the engine (`05` §5.1).

    C2 grew the C1 substrate with three fields:

    * :attr:`pools` — :class:`ConcurrencyPool` declarations consumed by
      the worker loop's slot accounting (`05` §3.4).
    * :attr:`phase1_queries` — state name → :class:`SchedulingQueryRef`
      for "find in-flight entities in this state" — used by phase 1 of
      the worker cycle (`05` §3.1).
    * :attr:`phase2_queries` — state name → :class:`SchedulingQueryRef`
      for "find ready candidates entering this state" — used by phase
      2 of the worker cycle (`05` §3.2).

    Composed into the full :class:`PipelineSpec` by Task 2.C3 with a
    top-level :attr:`name` field added; the C3 task may also collapse
    the per-phase scheduling-query maps if a unifying shape emerges
    from the SMAI CG-execution / proposal / paper-ingestion specs
    (Task 2.C4 / 3.E1 / 3.E2).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    entity_kind: EntityKind
    initial_state: str
    states: list[StateDef]
    edges: list[EdgeDef]
    pools: list[ConcurrencyPool] = Field(default_factory=list[ConcurrencyPool])
    phase1_queries: dict[str, SchedulingQueryRef] = Field(default_factory=dict)
    phase2_queries: dict[str, SchedulingQueryRef] = Field(default_factory=dict)

    def state_def(self, state: str) -> StateDef:
        """Look up a :class:`StateDef` by name; raises :class:`KeyError`
        if unregistered.
        """
        for s in self.states:
            if s.name == state:
                return s
        raise KeyError(f"unregistered state {state!r}")

    def edges_from(self, state: str, fires_on: PhaseTrigger) -> list[EdgeDef]:
        """Return the edges leaving ``state`` whose ``fires_on`` matches
        ``fires_on``, in declaration order (`05` §1.2 phase-keyed
        evaluation).
        """
        return [e for e in self.edges if e.from_state == state and e.fires_on == fires_on]

    def state_to_pool(self) -> dict[str, str]:
        """Derive the in-progress-state → pool-name map (`05` §3.4).

        Walks :attr:`states` and projects ``(state.name, state.on_entry_dispatch.pool)``
        for every state whose :attr:`StateDef.on_entry_dispatch` is
        non-None. The worker uses this to call
        :meth:`MetadataStore.count_with_in_flight_jobs` with the right
        in-flight-state list per pool (per DEC-035 #3).
        """
        return {
            s.name: s.on_entry_dispatch.pool for s in self.states if s.on_entry_dispatch is not None
        }


__all__ = [
    "ConcurrencyPool",
    "DispatchAction",
    "DispatchContext",
    "DispatchHandler",
    "DispatchOutcome",
    "EdgeDef",
    "EngineSpec",
    "GateContext",
    "GateOutcome",
    "GateRule",
    "PhaseTrigger",
    "RetryPolicy",
    "SchedulingQueryRef",
    "StateDef",
]
