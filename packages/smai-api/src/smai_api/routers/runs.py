"""``/api/v1/runs`` router per ``designs/smai/11-api.md`` §4.5.

Two read endpoints; runs are leaf entities created by the worker.
"""

from __future__ import annotations

from fastapi import APIRouter
from smai_api_spec import (
    CursorPage,
    RunDetailResponse,
    RunState,
    RunSummary,
)
from smai_api_spec.paths import RUN_DETAIL, RUNS

from smai_api._deps import RuntimeDep
from smai_api._pagination import paginate
from smai_api._record_projections import run_record_to_summary

router = APIRouter()


# === GET /api/v1/runs ======================================================


@router.get(RUNS, response_model=CursorPage[RunSummary])
async def list_runs(
    runtime: RuntimeDep,
    cursor: str | None = None,
    limit: int | None = None,
    state: RunState | None = None,
    cg_id: str | None = None,
    entry_id: str | None = None,
) -> CursorPage[RunSummary]:
    """List active runs — filterable by state, parent CG, or parent entry.

    Filtered AND-combined per ``11`` §4.8. ``state="succeeded"`` /
    ``"failed"`` / ``"inconclusive"`` are terminal and not part of the
    in-flight aggregator's coverage; the call still returns 200 with an
    empty list (the contract is "filter ⊆ active set" — terminal runs
    are reachable only via parent-CG navigation per ``11`` §4.5).
    """
    runs = await runtime.status.list_active_runs()
    if state is not None:
        runs = [r for r in runs if r.state == state]
    if cg_id is not None:
        runs = [r for r in runs if r.cg_id == cg_id]
    if entry_id is not None:
        runs = [r for r in runs if r.entry_id == entry_id]
    summaries = [run_record_to_summary(r) for r in runs]
    return paginate(summaries, cursor=cursor, limit=limit)


# === GET /api/v1/runs/{run_id} =============================================


@router.get(RUN_DETAIL, response_model=RunDetailResponse)
async def get_run(
    run_id: str,
    runtime: RuntimeDep,
) -> RunDetailResponse:
    """Run detail — superset of :class:`RunSummary` plus audit fields."""
    record = await runtime.status.get_run_record(run_id)
    return RunDetailResponse(
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
        created_at=record.created_at,
        updated_at=record.updated_at,
        last_error=record.last_error,
        version=record.version,
    )


__all__ = ["router"]
