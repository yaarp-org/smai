"""Engine, pipeline-spec format, worker loop, checkpointer; SMAI PipelineSpec instances.

Public surface (Task 1.10): the pipeline-tracking record types under
``smai_orchestrator.entities.tracking`` — runtime instances of the
methodology entities in ``smai_core.entities``, persisted via the
``MetadataStore`` plugin and driven through the orchestrator's state
machines per ``designs/smai/01-data-model.md`` §5.
"""

from smai_orchestrator.entities import (
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
