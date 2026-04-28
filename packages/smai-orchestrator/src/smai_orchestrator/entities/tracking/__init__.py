"""Pipeline-tracking record types for the orchestrator engine.

Per ``designs/smai/01-data-model.md`` §5 — the runtime-instance
counterparts of the methodology entities in ``smai_core.entities``.

Six records, six state-Literal aliases, organized one-record-per-module:

* :class:`ComparisonGroupRecord` (:data:`CGState`) — §5.3
* :class:`EntryRecord` (:data:`EntryState`) — §5.4
* :class:`RunRecord` (:data:`RunState`) — §5.5
* :class:`ProposalRecord` (:data:`ProposalState`) — §5.6
* :class:`PaperRecord` (:data:`PaperState`) — §5.7
* :class:`FactorModelRecord` (no state Literal — degenerate lifecycle
  per DEC-031 #5) — §5.8

The state Literal sets here are the canonical enumeration; per §5.2.7
they are owned by ``03-state-machine.md`` (and
``08-novel-technique-pipeline.md`` for proposal / paper) — until ``03``
ships its own State module the Literals live alongside the record types
so consumers have one import path. Once ``03`` settles the orchestrator-
side state-machine driver, the Literal definitions move there and
record modules import them.
"""

from smai_orchestrator.entities.tracking.comparison_group import (
    CGState,
    ComparisonGroupRecord,
)
from smai_orchestrator.entities.tracking.entry import EntryRecord, EntryState
from smai_orchestrator.entities.tracking.factor_model import FactorModelRecord
from smai_orchestrator.entities.tracking.paper import PaperRecord, PaperState
from smai_orchestrator.entities.tracking.proposal import ProposalRecord, ProposalState
from smai_orchestrator.entities.tracking.run import RunRecord, RunState

__all__ = [
    "CGState",
    "ComparisonGroupRecord",
    "EntryRecord",
    "EntryState",
    "FactorModelRecord",
    "PaperRecord",
    "PaperState",
    "ProposalRecord",
    "ProposalState",
    "RunRecord",
    "RunState",
]
