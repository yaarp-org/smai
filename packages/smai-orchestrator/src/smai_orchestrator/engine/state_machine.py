"""State-machine driver — edge evaluation and phase-3 dispatch.

Per ``designs/smai/05-orchestrator.md`` §1.1 / §1.2 / §3.3. Given an
entity in a state, evaluate the outgoing edges in declaration order;
the first edge whose gate rule returns ``advance=True`` wins; if the
target state's :class:`StateDef.on_entry_dispatch` is non-``None`` the
engine fires it under write-first ordering (see :mod:`dispatch`).

This module ships the substrate primitives (`drive_entity_phase3`,
`evaluate_outgoing_edges`); Task 2.C2's worker loop wraps them in the
three-phase poll cycle (complete / discover / dispatch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from smai_core.plugins import (
    ArtifactStore,
    Compute,
    LlmProvider,
    MetadataStore,
)

from smai_orchestrator.engine._metadata_ops import StateDrivenRecord, entity_id_for
from smai_orchestrator.engine.config import EngineConfig
from smai_orchestrator.engine.dispatch import (
    DispatchOutcomeWire,
    run_dispatch_with_lease,
)
from smai_orchestrator.engine.types import (
    EdgeDef,
    EngineSpec,
    GateContext,
    GateOutcome,
    PhaseTrigger,
)


@dataclass
class DriveOutcome:
    """Result of one phase-3 drive of a single entity (`05` §3.3).

    The variants match the four ways the cycle can end for an entity:

    * ``no_match`` — every outgoing edge's gate returned
      ``advance=False``; the entity stays in its current state for the
      next cycle.
    * ``advanced`` — an edge fired and the entity transitioned to the
      edge's target state. If the new state had an on-entry dispatch,
      :attr:`dispatch_outcome` carries its result; otherwise ``None``.
    * ``conflict`` — another worker won the CAS race during the
      transition or rollback; this worker bails for this cycle.
    * ``dispatch_failed_rolled_back`` — the dispatch handler returned
      :class:`DispatchOutcome` with ``error`` non-None (or raised);
      the engine rolled the entity back to the edge's ``from_state``.
    * ``dispatch_failed_routed`` — the dispatch handler failed, but a
      spec-declared ``dispatch_time`` error/retry-exhausted edge fired
      (e.g. a ``*_failed`` terminal), so the entity advanced there with
      ``last_error`` recorded rather than rolling back. A *failure*
      outcome from the worker-stats point of view (tallied under
      ``phase3_dispatch_failed``, not ``phase3_advanced``) — the
      per-dispatch ``outcome=failed`` log line already covers the
      transition's visibility.
    * ``lease_held`` — phase-3 dispatch lease (per `05` §3.5 / DEC-035
      #2) is currently held by another worker. The dispatch never
      fired; the entity stays in its current state for the next cycle.
      Distinct from ``conflict`` (which is post-acquire CAS race lost)
      to give the worker-loop stats a clean "another worker has the
      lease" counter without overloading the CAS-conflict bucket.

    The :attr:`status` field discriminates and the optional fields
    carry the per-variant payload.
    """

    status: Literal[
        "no_match",
        "advanced",
        "conflict",
        "dispatch_failed_rolled_back",
        "dispatch_failed_routed",
        "lease_held",
    ]
    fired_edge: EdgeDef | None = None
    dispatch_outcome: DispatchOutcomeWire | None = None
    error: str | None = None


async def evaluate_outgoing_edges_with_outcome(
    *,
    spec: EngineSpec,
    entity_state: str,
    phase: PhaseTrigger,
    gate_context: GateContext,
) -> tuple[EdgeDef, GateOutcome] | None:
    """Like :func:`evaluate_outgoing_edges`, but also returns the
    advancing edge's :class:`GateOutcome` so the caller can thread
    :attr:`GateOutcome.reason` into the ``transition_log`` audit row.

    Gates are evaluated in declaration order and the first to return
    ``advance=True`` wins (it short-circuits, so gates with side
    effects past the winner do not run — same contract as
    :func:`evaluate_outgoing_edges`).
    """
    edges = spec.edges_from(entity_state, fires_on=phase)
    for edge in edges:
        outcome = await edge.gate_rule(gate_context)
        if outcome.advance:
            return edge, outcome
    return None


async def evaluate_outgoing_edges(
    *,
    spec: EngineSpec,
    entity_state: str,
    phase: PhaseTrigger,
    gate_context: GateContext,
) -> EdgeDef | None:
    """Evaluate the outgoing edges from ``entity_state`` for ``phase``,
    in declaration order; return the first one whose gate rule returns
    ``advance=True`` or ``None`` if none pass (`05` §1.2).

    The edges are filtered to those whose ``fires_on`` matches
    ``phase`` (phase-keyed evaluation per §1.2). Phase-3 callers pass
    ``"dispatch_time"``; phase-1 callers pass ``"job_succeeded"`` or
    ``"job_failed"`` (which one depends on the observed
    :class:`JobStatus`).
    """
    result = await evaluate_outgoing_edges_with_outcome(
        spec=spec,
        entity_state=entity_state,
        phase=phase,
        gate_context=gate_context,
    )
    return result[0] if result is not None else None


async def drive_entity_phase3(  # noqa: PLR0913
    *,
    spec: EngineSpec,
    metadata_store: MetadataStore,
    artifact_store: ArtifactStore,
    compute: Compute,
    llm: LlmProvider | None,
    config: EngineConfig,
    record: StateDrivenRecord,
    worker_id: str | None = None,
) -> DriveOutcome:
    """Run one phase-3 evaluate-and-dispatch step against ``record``
    (`05` §3.3).

    Steps:

    1. Evaluate outgoing ``dispatch_time`` edges in declaration order;
       if none pass, return :class:`DriveOutcome` with
       ``status="no_match"``.
    2. The winning edge's target state is the new state. If the target
       state's ``on_entry_dispatch`` is ``None``, the engine performs a
       single-step CAS transition (no external compute work). Otherwise
       it delegates to :func:`run_dispatch_with_lease` for the full
       lease-acquire / write-first / lease-release sequence (`05` §3.5
       + §1.4).
    3. On lease contention: return ``status="lease_held"`` (another
       worker holds the per-entity lease; this worker skips the entity
       this cycle without firing the handler).
    4. On CAS conflict: return ``status="conflict"`` (lease was
       acquired but the in-dispatch CAS race was lost — orphan reclaim
       or peer-reset).
    5. On dispatch handler failure: the engine rolls back to
       ``edge.from_state``; return
       ``status="dispatch_failed_rolled_back"``.

    ``worker_id`` is the stable per-worker identity used as
    :attr:`LeaseToken.lease_holder_id` for audit / debugging surfaces.
    ``None`` (the test default) auto-generates a fresh per-call UUID;
    production callers — :func:`smai_orchestrator.worker.run_worker_cycle`,
    :func:`smai_cli.runtime.Runtime.start_in_band` — pass the
    deployment-stable worker id so multi-worker contention shows up
    coherently in audit queries.

    Phase-1 wait-for-job semantics live in :mod:`phase1`; this driver
    only handles phase-3.
    """
    # ``id`` / ``arxiv_id`` is normalized via :func:`entity_id_for`;
    # ``version`` is on :class:`BasePipelineRecord`; ``state`` is per-
    # record but every member of :data:`StateDrivenRecord` declares it.
    entity_id = entity_id_for(record)
    entity_state = record.state
    entity_version = record.version

    gate_context = GateContext(
        entity_kind=spec.entity_kind,
        entity_id=entity_id,
        entity_state=entity_state,
        entity_version=entity_version,
        metadata_store=metadata_store,
        artifact_store=artifact_store,
        config=config,
        job_outcome=None,
    )

    evaluated = await evaluate_outgoing_edges_with_outcome(
        spec=spec,
        entity_state=entity_state,
        phase="dispatch_time",
        gate_context=gate_context,
    )
    if evaluated is None:
        return DriveOutcome(status="no_match")
    edge, gate_outcome = evaluated

    target_def = spec.state_def(edge.target_state)
    resolved_worker_id = worker_id if worker_id is not None else f"worker-{uuid4().hex[:8]}"
    return await run_dispatch_with_lease(
        spec=spec,
        metadata_store=metadata_store,
        artifact_store=artifact_store,
        compute=compute,
        llm=llm,
        config=config,
        edge=edge,
        target_state=target_def,
        entity_id=entity_id,
        expected_version=entity_version,
        worker_id=resolved_worker_id,
        gate_outcome_reason=gate_outcome.reason,
    )


__all__ = [
    "DriveOutcome",
    "drive_entity_phase3",
    "evaluate_outgoing_edges",
    "evaluate_outgoing_edges_with_outcome",
]
