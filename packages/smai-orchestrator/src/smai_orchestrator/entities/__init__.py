"""Pipeline-tracking entity types for the orchestrator engine.

Per ``designs/smai/01-data-model.md`` §5: the runtime-instance counterparts
of the methodology entities in ``smai_core.entities``. These records
carry identity, state, version, lock/lease fields, audit trail, and
content-hash references — everything the orchestrator needs to drive
methodology entities through a state machine via the ``MetadataStore``
plugin.

The dependency direction is one-way (DEC-029): pipeline-tracking entities
import methodology entities (to type their ``*_id`` foreign-key fields);
methodology entities never import pipeline-tracking entities.
"""

from smai_orchestrator.entities.tracking import (
    CGState,
    ComparisonGroupRecord,
    EntryRecord,
    EntryState,
    FactorModelRecord,
    PaperRecord,
    PaperState,
    ProposalRecord,
    ProposalState,
    RunRecord,
    RunState,
)

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
