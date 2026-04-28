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

from pydantic import BaseModel, ConfigDict
from smai_core.plugins import (
    ArtifactStore,
    Compute,
    EntityKind,
    JobHandle,
    JobStatus,
    LlmProvider,
    MetadataStore,
)

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

    ``checkpointer`` is C2's scope (Task 2.C2 ships :class:`Checkpointer`
    + the two flavors). C1 carries a placeholder typed ``Any | None``
    so handler signatures can already declare the slot. The C2 task
    replaces the placeholder with the concrete Protocol per `05` §2.
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
    checkpointer: Any | None = None


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
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    handler: DispatchHandler
    pool: str
    handle_field: str | None = None


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


class EngineSpec(BaseModel):
    """Minimal pipeline-spec substrate driving the engine (`05` §5.1).

    Composed into the full :class:`PipelineSpec` by Task 2.C3 with
    pools, scheduling-query refs, and a top-level name added.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    entity_kind: EntityKind
    initial_state: str
    states: list[StateDef]
    edges: list[EdgeDef]

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


__all__ = [
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
    "StateDef",
]
