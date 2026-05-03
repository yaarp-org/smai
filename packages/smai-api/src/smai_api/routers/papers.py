"""``/api/v1/papers`` router per ``designs/smai/11-api.md`` §4.2."""

from __future__ import annotations

from fastapi import APIRouter
from smai_api_spec import (
    CursorPage,
    PaperDetailResponse,
    PaperState,
    PaperSubmissionResponse,
    PaperSummary,
    PromotePartialResponse,
    SubmitPaperRequest,
    TechniqueRefSummary,
)
from smai_api_spec.paths import (
    PAPER_DETAIL,
    PAPER_PROMOTE_PARTIAL,
    PAPERS,
)

from smai_api._deps import RuntimeDep
from smai_api._pagination import paginate

router = APIRouter()


# === POST /api/v1/papers ====================================================


@router.post(PAPERS, status_code=202, response_model=PaperSubmissionResponse)
async def submit_paper(
    body: SubmitPaperRequest,
    runtime: RuntimeDep,
) -> PaperSubmissionResponse:
    """Submit by ``arxiv_id``. Idempotent per ``11`` §4.2 / §4.8 —
    creates a new record on first submit, promotes a ``partial`` to
    ``submitted`` on resubmit, no-ops on a paper already in flight or
    in a terminal state."""
    await runtime.papers.submit(arxiv_id=body.arxiv_id)
    record = await runtime.papers.get(body.arxiv_id)
    return PaperSubmissionResponse(
        arxiv_id=record.arxiv_id,
        state=record.state,
        created_at=record.created_at,
    )


# === GET /api/v1/papers =====================================================


@router.get(PAPERS, response_model=CursorPage[PaperSummary])
async def list_papers(
    runtime: RuntimeDep,
    cursor: str | None = None,
    limit: int | None = None,
    state: PaperState | None = None,
) -> CursorPage[PaperSummary]:
    """List active papers — paginated, optionally filtered by state."""
    papers = await runtime.papers.list_active()
    if state is not None:
        papers = [p for p in papers if p.state == state]
    summaries = [
        PaperSummary(
            arxiv_id=p.arxiv_id,
            state=p.state,
            title=p.title,
            created_at=p.created_at,
            updated_at=p.updated_at,
            last_error=p.last_error,
            version=p.version,
        )
        for p in papers
    ]
    return paginate(summaries, cursor=cursor, limit=limit)


# === GET /api/v1/papers/{arxiv_id} =========================================


@router.get(PAPER_DETAIL, response_model=PaperDetailResponse)
async def get_paper(
    arxiv_id: str,
    runtime: RuntimeDep,
) -> PaperDetailResponse:
    """Detail endpoint. ``technique_refs`` is populated only once the
    paper reaches the terminal ``registered`` state — that is the
    contract surface for "paper is fully ingested; here are its
    techniques to reference in a reproduce-paper proposal" (per
    ``11`` §4.2 / DEC-032)."""
    record = await runtime.papers.get(arxiv_id)
    technique_refs: list[TechniqueRefSummary] = []
    if record.state == "registered":
        techniques = await runtime.papers.list_techniques(arxiv_id)
        technique_refs = [
            TechniqueRefSummary(
                technique_id=t.id,
                name=t.name,
                description=t.description,
            )
            for t in techniques
        ]
    return PaperDetailResponse(
        arxiv_id=record.arxiv_id,
        state=record.state,
        title=record.title,
        authors=list(record.authors),
        abstract=record.abstract,
        published_date=record.published_date,
        categories=list(record.categories),
        latex_bundle_artifact_key=record.latex_bundle_artifact_key,
        expanded_tex_artifact_key=record.expanded_tex_artifact_key,
        extracted_text_artifact_key=record.extracted_text_artifact_key,
        figures_artifact_key=record.figures_artifact_key,
        screen_result_decision=record.screen_result_decision,
        screen_result_reason=record.screen_result_reason,
        technique_buffer_artifact_key=record.technique_buffer_artifact_key,
        error_context_artifact_key=record.error_context_artifact_key,
        planning_attempt=record.planning_attempt,
        screening_attempt=record.screening_attempt,
        registration_attempt=record.registration_attempt,
        technique_refs=technique_refs,
        created_at=record.created_at,
        updated_at=record.updated_at,
        last_error=record.last_error,
        version=record.version,
    )


# === POST /api/v1/papers/{arxiv_id}/promote-partial ========================


@router.post(PAPER_PROMOTE_PARTIAL, response_model=PromotePartialResponse)
async def promote_partial_paper(
    arxiv_id: str,
    runtime: RuntimeDep,
) -> PromotePartialResponse:
    """Fire the ``partial → submitted`` edge per ``08`` §5.7."""
    record = await runtime.papers.promote_partial(arxiv_id)
    return PromotePartialResponse(
        arxiv_id=record.arxiv_id,
        state=record.state,
    )


__all__ = ["router"]
