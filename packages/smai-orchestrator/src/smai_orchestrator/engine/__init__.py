"""Orchestrator engine — state-machine driver, write-first dispatch,
phase-1 wait-for-job semantics.

Per ``designs/smai/05-orchestrator.md`` §1. The C1 surface ships:

* Typed primitives — :class:`GateContext`, :class:`GateOutcome`,
  :class:`GateRule`, :class:`EdgeDef`, :class:`StateDef`,
  :class:`DispatchContext`, :class:`DispatchOutcome`,
  :class:`DispatchHandler`, :class:`DispatchAction`, :class:`EngineSpec`.
* Engine config — :class:`EngineConfig`, :class:`RetryPolicy`, the
  :data:`TimeProvider` / :data:`WallClockProvider` seams.
* Phase-3 driver — :func:`evaluate_outgoing_edges`,
  :func:`drive_entity_phase3`, :class:`DriveOutcome`.
* Phase-1 driver — :func:`phase1_step`, :class:`Phase1Outcome`.
* Dispatch primitives — :func:`run_dispatch`, :func:`reset_orphan`,
  :class:`DispatchOutcomeWire`.

Out of scope, deferred:

* Worker loop (3-phase poll cycle) — Task 2.C2.
* :class:`Checkpointer` step memoization — Task 2.C2.
* Concurrency-pool slot accounting — Task 2.C2.
* Full :class:`PipelineSpec` composition (pools / scheduling-query refs
  / top-level name) — Task 2.C3.
* CG-execution / proposal / paper-ingestion pipeline-specs — Tasks
  2.C4 / 3.E1 / 3.E2.
* Multi-worker leasing integration — Task 3.G1.
"""

from smai_orchestrator.engine.clock import (
    TimeProvider,
    WallClockProvider,
    default_time_provider,
    default_wall_clock,
)
from smai_orchestrator.engine.config import EngineConfig, RetryPolicy
from smai_orchestrator.engine.dispatch import (
    DispatchOutcomeWire,
    reset_orphan,
    run_dispatch,
)
from smai_orchestrator.engine.phase1 import (
    CostRecordContext,
    CostRecordHandler,
    Phase1Outcome,
    phase1_step,
)
from smai_orchestrator.engine.state_machine import (
    DriveOutcome,
    drive_entity_phase3,
    evaluate_outgoing_edges,
)
from smai_orchestrator.engine.types import (
    ConcurrencyPool,
    DispatchAction,
    DispatchContext,
    DispatchHandler,
    DispatchOutcome,
    EdgeDef,
    EngineSpec,
    GateContext,
    GateOutcome,
    GateRule,
    PhaseTrigger,
    SchedulingQueryRef,
    StateDef,
)

__all__ = [
    "ConcurrencyPool",
    "CostRecordContext",
    "CostRecordHandler",
    "DispatchAction",
    "DispatchContext",
    "DispatchHandler",
    "DispatchOutcome",
    "DispatchOutcomeWire",
    "DriveOutcome",
    "EdgeDef",
    "EngineConfig",
    "EngineSpec",
    "GateContext",
    "GateOutcome",
    "GateRule",
    "Phase1Outcome",
    "PhaseTrigger",
    "RetryPolicy",
    "SchedulingQueryRef",
    "StateDef",
    "TimeProvider",
    "WallClockProvider",
    "default_time_provider",
    "default_wall_clock",
    "drive_entity_phase3",
    "evaluate_outgoing_edges",
    "phase1_step",
    "reset_orphan",
    "run_dispatch",
]
