"""Entry response shapes per ``designs/smai/11-api.md`` §4.4.

Entries are NOT a top-level resource in v1 — they are always read in
the context of their parent CG. This module supplies the per-entry
projections embedded in :class:`smai_api_spec.comparison_groups.ComparisonGroupDetailResponse`,
plus the dedicated entry-detail response served at
``GET /api/v1/comparison-groups/{cg_id}/entries/{entry_id}``.
"""

from __future__ import annotations

from datetime import datetime

from smai_api_spec._common import APIBaseModel, BaseAuditedResponse, EntryState
from smai_api_spec.runs import RunSummary


class EntrySummary(APIBaseModel):
    """One item in ``GET /api/v1/comparison-groups/{cg_id}/entries``.

    Audit fields kept for SPA rendering. ``runs`` are not embedded on the
    list — fetch the entry detail or filter ``/runs?entry_id=`` for them.
    Per ``11`` §5.1.2 the lease triple is NOT exposed;
    ``implementation_job_handle`` is also withheld as an orchestrator-
    internal opaque token.
    """

    id: str
    cg_id: str
    technique_id: str | None
    is_baseline: bool
    state: EntryState
    technique_contract_hash: str | None
    implementation_attempt: int
    created_at: datetime
    updated_at: datetime


class EntryWithRuns(BaseAuditedResponse):
    """Per-entry projection embedded inside the CG-detail response.

    The full ``GET /api/v1/comparison-groups/{cg_id}`` response embeds a
    list of these (one per entry) with each entry's runs inlined. Per
    ``11`` §4.4 the embedding is always-on in v1; if real-world payloads
    grow beyond ~50 runs an opt-in ``?include=`` flag may be added (per
    OQ1, still open).
    """

    id: str
    cg_id: str
    technique_id: str | None
    is_baseline: bool
    state: EntryState
    technique_contract_hash: str | None
    harness_api_manifest_hash: str | None
    implementation_attempt: int
    runs: list[RunSummary]


class EntryDetailResponse(BaseAuditedResponse):
    """Full entry detail. Returned by
    ``GET /api/v1/comparison-groups/{cg_id}/entries/{entry_id}``.

    Mirrors :class:`EntryWithRuns` — same shape, exposed at a dedicated
    URL for clients that already know the entry id.
    """

    id: str
    cg_id: str
    technique_id: str | None
    is_baseline: bool
    state: EntryState
    technique_contract_hash: str | None
    harness_api_manifest_hash: str | None
    implementation_attempt: int
    runs: list[RunSummary]


__all__ = [
    "EntryDetailResponse",
    "EntrySummary",
    "EntryWithRuns",
]
