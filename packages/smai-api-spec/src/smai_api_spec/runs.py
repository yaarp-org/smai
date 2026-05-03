"""Run request / response shapes per ``designs/smai/11-api.md`` §4.5.

Runs are the leaf entity of the CG-execution pipeline (one per
entry × seed). They are also embedded inside :class:`ComparisonGroupDetailResponse`
and per-entry detail responses; the flat ``/api/v1/runs`` listing exists
for cross-CG views ("show me everything currently ``running``").

Endpoints covered:

* ``GET /api/v1/runs``           — :class:`ListRunsParams` → ``CursorPage[RunSummary]``
* ``GET /api/v1/runs/{run_id}``  — (no request body)      → :class:`RunDetailResponse`
"""

from __future__ import annotations

from datetime import datetime

from smai_api_spec._common import APIBaseModel, BaseAuditedResponse, RunState
from smai_api_spec.pagination import PaginationParams

# === GET /api/v1/runs =======================================================


class ListRunsParams(PaginationParams):
    """Query-param model for ``GET /api/v1/runs``."""

    state: RunState | None = None
    cg_id: str | None = None
    entry_id: str | None = None


class RunSummary(APIBaseModel):
    """Per-run summary — embedded in CG-detail and entry-detail responses,
    and returned as the page-item type for ``GET /api/v1/runs``.

    Per ``11`` §5.1.2 the lease triple is NOT exposed; ``compute_job_handle``
    is also withheld as an orchestrator-internal opaque token.
    """

    id: str
    cg_id: str
    entry_id: str
    seed: int
    state: RunState
    duration_seconds: float | None = None
    raw_metrics_artifact_key: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failure_reason: str | None = None
    run_attempt: int
    updated_at: datetime


# === GET /api/v1/runs/{run_id} ==============================================


class RunDetailResponse(BaseAuditedResponse):
    """Run detail — superset of :class:`RunSummary` adding the audit
    fields from :class:`BaseAuditedResponse`.
    """

    id: str
    cg_id: str
    entry_id: str
    seed: int
    state: RunState
    duration_seconds: float | None
    raw_metrics_artifact_key: str | None
    started_at: datetime | None
    completed_at: datetime | None
    failure_reason: str | None
    run_attempt: int


__all__ = [
    "ListRunsParams",
    "RunDetailResponse",
    "RunSummary",
]
