"""Record-to-spec projections shared across routers.

Each helper turns a ``smai_orchestrator`` record (the
:class:`MetadataStore`-shaped persistence type) into the matching
``smai_api_spec`` response model. Centralizing keeps the routers
terse and the field-mapping in one place.
"""

from __future__ import annotations

from smai_api_spec import EntrySummary, EntryWithRuns, RunSummary
from smai_orchestrator import EntryRecord, RunRecord


def run_record_to_summary(record: RunRecord) -> RunSummary:
    """Project a :class:`RunRecord` onto :class:`RunSummary`."""
    return RunSummary(
        id=record.id,
        cg_id=record.cg_id,
        entry_id=record.entry_id,
        seed=record.seed,
        state=record.state,
        duration_seconds=record.duration_seconds,
        raw_metrics_artifact_key=record.raw_metrics_artifact_key,
        started_at=record.started_at,
        completed_at=record.completed_at,
        failure_reason=record.failure_reason,
        run_attempt=record.run_attempt,
        updated_at=record.updated_at,
    )


def entry_record_to_summary(record: EntryRecord) -> EntrySummary:
    """Project an :class:`EntryRecord` onto :class:`EntrySummary`."""
    return EntrySummary(
        id=record.id,
        cg_id=record.cg_id,
        technique_id=record.technique_id,
        is_baseline=record.is_baseline,
        state=record.state,
        technique_contract_hash=record.technique_contract_hash,
        implementation_attempt=record.implementation_attempt,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def entry_record_to_with_runs(record: EntryRecord, runs: list[RunRecord]) -> EntryWithRuns:
    """Project an :class:`EntryRecord` + parented runs onto :class:`EntryWithRuns`."""
    return EntryWithRuns(
        id=record.id,
        cg_id=record.cg_id,
        technique_id=record.technique_id,
        is_baseline=record.is_baseline,
        state=record.state,
        technique_contract_hash=record.technique_contract_hash,
        harness_api_manifest_hash=record.harness_api_manifest_hash,
        implementation_attempt=record.implementation_attempt,
        runs=[run_record_to_summary(r) for r in runs],
        created_at=record.created_at,
        updated_at=record.updated_at,
        last_error=record.last_error,
        version=record.version,
    )


__all__ = [
    "entry_record_to_summary",
    "entry_record_to_with_runs",
    "run_record_to_summary",
]
