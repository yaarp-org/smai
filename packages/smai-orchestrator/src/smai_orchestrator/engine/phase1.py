"""Phase-1 wait-for-job + orphan detection.

Per ``designs/smai/05-orchestrator.md`` §1.5 / §3.1. Phase-1 is the only
phase that observes external job completion. For each entity in an
in-progress state with a non-null ``job_handle``:

1. **Orphan detection.** If ``job_handle`` is null but the entity has
   been in the in-progress state past
   :attr:`EngineConfig.orphan_grace_seconds`, reset to the prior state
   via CAS (catches workers that crashed between dispatch steps 1 and
   2 of write-first ordering).
2. **Job status.** Call :meth:`Compute.status(handle)`. If ``RUNNING``,
   leave it. Otherwise read the :class:`JobStatus` and evaluate the
   matching ``fires_on`` edges leaving the in-progress state — phase-1
   uses ``"job_succeeded"`` for ``state == "succeeded"`` and
   ``"job_failed"`` for the failure-set ``{"failed", "cancelled",
   "timeout"}``. The first edge whose gate returns ``advance=True``
   wins; on pass, CAS to the edge's target.
3. **Cost / telemetry hooks.** If a per-state cost-record handler is
   attached to the spec, fire it on completion. C1 ships the substrate
   (a typed callable invocation) but does not register handlers; Task
   2.C2 / 2.C4 wires the SMAI-specific cost handlers into the spec.

Discovery and dispatch (phase-2 / phase-3 from `05` §3.2 / §3.3) live
in C2's worker loop; this module only handles the per-entity phase-1
step.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict
from smai_core.plugins import (
    ArtifactStore,
    Compute,
    JobHandle,
    JobStatus,
    MetadataStore,
)
from smai_core.plugins.metadata_store import ConflictError

from smai_orchestrator.engine._metadata_ops import (
    StateDrivenRecord,
    entity_id_for,
    transition_state,
)
from smai_orchestrator.engine.config import EngineConfig
from smai_orchestrator.engine.dispatch import reset_orphan
from smai_orchestrator.engine.types import (
    EdgeDef,
    EngineSpec,
    GateContext,
)

# Per-state cost-record handler shape. Spec authors may register one
# of these on a state's terminal phase-1 transition; the engine fires
# it after the transition succeeds. Fully optional — C1 ships the
# substrate, not any handlers.
CostRecordHandler = Callable[
    ["CostRecordContext"],
    Awaitable[None],
]


class CostRecordContext(BaseModel):
    """Read-only payload for a phase-1 cost-record handler (`05` §3.1.3)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    entity_id: str
    entity_state: str
    job_handle: JobHandle
    job_status: JobStatus


@dataclass
class Phase1Outcome:
    """Result of one phase-1 step against a single entity (`05` §3.1).

    Variants:

    * ``orphan_reset`` — the entity's ``job_handle`` was null and
      orphan-grace had elapsed; the engine reset it to the prior
      state via :func:`reset_orphan`.
    * ``orphan_no_grace`` — the entity's ``job_handle`` was null but
      orphan-grace has not yet elapsed; left in place.
    * ``running`` — :class:`Compute` reports the job is still running;
      left in place.
    * ``advanced`` — the job has terminated; an edge fired and the
      entity transitioned to the edge's target state.
    * ``terminated_no_match`` — the job has terminated but no
      ``fires_on`` edge passed; the entity stays in the in-progress
      state pending operator intervention or spec elaboration.
    * ``conflict`` — CAS lost during the transition; another worker
      handled it.
    """

    status: Literal[
        "orphan_reset",
        "orphan_no_grace",
        "running",
        "advanced",
        "terminated_no_match",
        "conflict",
    ]
    fired_edge: EdgeDef | None = None
    job_status: JobStatus | None = None
    error: str | None = None


def _failure_outcomes_to_trigger() -> frozenset[str]:
    """Job states that count as ``fires_on=job_failed`` triggers (`05` §3.1).

    Per :data:`smai_core.plugins.JobState`: ``failed``, ``cancelled``,
    and ``timeout`` are all "external job did not succeed" — the
    failure-edge's gate rule decides whether to retry, abandon, or
    escalate. ``submitted`` and ``running`` are not terminal.
    """
    return frozenset({"failed", "cancelled", "timeout"})


def _read_handle(record: StateDrivenRecord, handle_field: str) -> JobHandle | None:
    """Read the configured ``handle_field`` off the record.

    Returns ``None`` if the record's field is null (orphan-detection
    candidate). Raises :class:`AttributeError` if the field name is
    unknown to the record class — this is a spec-author bug
    (DispatchAction.handle_field doesn't match any column on the
    record); surface it loudly rather than silently.
    """
    handle = getattr(record, handle_field)
    if handle is None:
        return None
    if not isinstance(handle, JobHandle):
        raise TypeError(
            f"record.{handle_field} is not a JobHandle: got {type(handle).__name__}"
        )
    return handle


async def phase1_step(  # noqa: PLR0913
    *,
    spec: EngineSpec,
    metadata_store: MetadataStore,
    artifact_store: ArtifactStore,
    compute: Compute,
    config: EngineConfig,
    record: StateDrivenRecord,
    cost_handler: CostRecordHandler | None = None,
) -> Phase1Outcome:
    """Run one phase-1 wait-for-job step against ``record`` (`05` §3.1).

    Caller (the worker loop in C2) is responsible for filtering the
    candidate set to entities currently in an in-progress state; this
    function does not re-validate that.
    """
    entity_id = entity_id_for(record)
    entity_state = record.state
    entity_version = record.version

    state_def = spec.state_def(entity_state)
    action = state_def.on_entry_dispatch
    if action is None or action.handle_field is None:
        # Either the state has no on-entry dispatch (operator put the
        # entity here manually) or the dispatch is inline (no handle).
        # Both cases: phase-1 has nothing to observe.
        return Phase1Outcome(status="running")

    handle = _read_handle(record, action.handle_field)

    # ---- Orphan detection (handle is null) ------------------------------
    if handle is None:
        elapsed = (config.wall_clock() - record.updated_at).total_seconds()
        if elapsed < config.orphan_grace_seconds:
            return Phase1Outcome(status="orphan_no_grace")
        prior_state = _resolve_prior_state(spec, entity_state)
        rolled = await reset_orphan(
            spec=spec,
            metadata_store=metadata_store,
            edge_from_state=prior_state,
            entity_id=entity_id,
            current_version=entity_version,
        )
        return Phase1Outcome(
            status="orphan_reset" if rolled is not None else "conflict",
        )

    # ---- Live job: poll Compute -----------------------------------------
    job_status = await compute.status(handle)
    if job_status.state in ("submitted", "running"):
        return Phase1Outcome(status="running", job_status=job_status)

    fires_on = "job_succeeded" if job_status.state == "succeeded" else "job_failed"
    if job_status.state not in _failure_outcomes_to_trigger() | {"succeeded"}:
        # Only the listed terminal states map to phase-1 edge triggers.
        # Anything else is a Compute plugin bug; treat as no-match.
        return Phase1Outcome(status="terminated_no_match", job_status=job_status)

    gate_context = GateContext(
        entity_kind=spec.entity_kind,
        entity_id=entity_id,
        entity_state=entity_state,
        entity_version=entity_version,
        metadata_store=metadata_store,
        artifact_store=artifact_store,
        config=config,
        job_outcome=job_status,
    )

    fired: EdgeDef | None = None
    for edge in spec.edges_from(entity_state, fires_on=fires_on):
        outcome = await edge.gate_rule(gate_context)
        if outcome.advance:
            fired = edge
            break

    if fired is None:
        return Phase1Outcome(status="terminated_no_match", job_status=job_status)

    # CAS to the edge's target. Phase-1 transitions also clear the handle
    # field as part of the same UPDATE — once the job has terminated and
    # been observed, the handle is no longer load-bearing and leaving it
    # populated would confuse subsequent phase-1 polling.
    try:
        await transition_state(
            metadata_store,
            spec.entity_kind,
            entity_id,
            entity_version,
            fired.target_state,
            fields={action.handle_field: None},
        )
    except ConflictError:
        return Phase1Outcome(status="conflict", fired_edge=fired, job_status=job_status)

    if cost_handler is not None:
        await cost_handler(
            CostRecordContext(
                entity_id=entity_id,
                entity_state=entity_state,
                job_handle=handle,
                job_status=job_status,
            )
        )

    return Phase1Outcome(status="advanced", fired_edge=fired, job_status=job_status)


def _resolve_prior_state(spec: EngineSpec, in_progress_state: str) -> str:
    """Return the prior state for an orphaned entity (`05` §1.4 / §9 #8).

    The design (`05` §1.4) specifies "the prior state is taken from
    :attr:`EdgeDef.from_state` of the edge that fired" — but at orphan
    detection time the engine doesn't know which edge fired (the
    transition log is the audit surface, but `transition_log` CRUD is
    Session-C-pending per DEC-033 affected-area note). C1 resolves this
    by looking at the spec: it finds all edges whose ``target_state``
    is ``in_progress_state`` and takes the unique ``from_state``. If
    there are multiple distinct ``from_state`` values, the engine
    raises (the spec author must introduce a unique pre-in-progress
    sentinel, or `transition_log` CRUD must ship). C1's synthetic
    pipeline-spec satisfies the unique-from-state property; the SMAI
    CG-execution spec (Task 2.C4) does too per `03` §3.

    Carry-forward: this lookup needs revisiting if a future spec
    elaboration introduces multiple distinct edges into the same
    in-progress state. Surfaced in `05` §9 #8.
    """
    incoming = [e for e in spec.edges if e.target_state == in_progress_state]
    if not incoming:
        raise LookupError(
            f"engine cannot resolve prior state for orphaned in-progress "
            f"state {in_progress_state!r}: no edges target this state"
        )
    distinct_sources = {e.from_state for e in incoming}
    if len(distinct_sources) > 1:
        raise LookupError(
            f"engine cannot resolve prior state for orphaned in-progress "
            f"state {in_progress_state!r}: {len(distinct_sources)} distinct "
            f"from_states ({sorted(distinct_sources)}); spec authors must "
            f"either ship `transition_log` CRUD or introduce a unique "
            f"pre-in-progress sentinel"
        )
    return next(iter(distinct_sources))


__all__ = [
    "CostRecordContext",
    "CostRecordHandler",
    "Phase1Outcome",
    "phase1_step",
]
