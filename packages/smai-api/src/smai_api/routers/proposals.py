"""``/api/v1/proposals`` router per ``designs/smai/11-api.md`` §4.1.

Five endpoints — submit (POST), list (GET), detail (GET), approve
(POST RPC verb), reject (POST RPC verb). Submissions carry one of three
description forms (validated in the spec model);
``submission_kind="reproduce_paper"`` cross-validation against the
referenced paper's state happens here per OQ11 RESOLVED 2026-05-03 — a
missing or non-terminal paper surfaces as 409 + ``PAPER_NOT_READY``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter
from smai_api_spec import (
    ApproveProposalResponse,
    CursorPage,
    ProposalDetailResponse,
    ProposalState,
    ProposalSubmissionResponse,
    ProposalSummary,
    RejectProposalRequest,
    RejectProposalResponse,
    SubmitProposalRequest,
)
from smai_api_spec.paths import (
    PROPOSAL_APPROVE,
    PROPOSAL_DETAIL,
    PROPOSAL_REJECT,
    PROPOSALS,
)
from smai_cli.runtime import PaperNotFoundError

from smai_api._deps import RuntimeDep
from smai_api._pagination import paginate
from smai_api.errors import PaperNotReadyError

router = APIRouter()


_TERMINAL_PAPER_REGISTERED: frozenset[str] = frozenset({"registered"})


# === POST /api/v1/proposals =================================================


@router.post(PROPOSALS, status_code=202, response_model=ProposalSubmissionResponse)
async def submit_proposal(
    body: SubmitProposalRequest,
    runtime: RuntimeDep,
) -> ProposalSubmissionResponse:
    """Submit a proposal — primary input verb per DEC-032.

    Spec validation (exactly-one-description-form, cross-validate
    against ``submission_kind``) is enforced by Pydantic at the
    :class:`SubmitProposalRequest` boundary. We additionally check the
    reproduce-paper precondition per ``11`` §13 OQ11 RESOLVED
    2026-05-03: a missing or non-terminal paper surfaces as 409 +
    ``PAPER_NOT_READY``, distinct from ``INVALID_STATE`` so clients can
    render a paper-specific error message.
    """
    if body.submission_kind == "reproduce_paper":
        # The spec model already requires reproduce_paper_arxiv_id when
        # submission_kind is reproduce_paper; the assert pacifies pyright.
        assert body.reproduce_paper_arxiv_id is not None
        try:
            paper = await runtime.papers.get(body.reproduce_paper_arxiv_id)
        except PaperNotFoundError as exc:
            raise PaperNotReadyError(body.reproduce_paper_arxiv_id) from exc
        if paper.state not in _TERMINAL_PAPER_REGISTERED:
            raise PaperNotReadyError(body.reproduce_paper_arxiv_id, current_state=paper.state)

    proposal_id = body.proposal_id or _new_proposal_id()
    technique_description: dict[str, object] | str | None
    if body.technique_description is not None:
        technique_description = dict(body.technique_description)
    elif body.technique_description_text is not None:
        technique_description = body.technique_description_text
    else:
        # reproduce_paper carries no technique description payload —
        # the artifact key remains None on the resulting record.
        technique_description = None

    submission = await runtime.proposals.submit(
        proposal_id=proposal_id,
        submission_kind=body.submission_kind,
        technique_description=technique_description,
        reproduce_paper_arxiv_id=body.reproduce_paper_arxiv_id,
        submitted_by=body.submitted_by,
    )
    record = await runtime.proposals.get(submission.proposal_id)
    return ProposalSubmissionResponse(
        id=record.id,
        state=record.state,
        submission_kind=record.submission_kind,
        technique_description_artifact_key=record.technique_description_artifact_key,
        reproduce_paper_arxiv_id=record.reproduce_paper_arxiv_id,
        submitted_by=record.submitted_by,
        created_at=record.created_at,
    )


# === GET /api/v1/proposals ==================================================


@router.get(PROPOSALS, response_model=CursorPage[ProposalSummary])
async def list_proposals(
    runtime: RuntimeDep,
    cursor: str | None = None,
    limit: int | None = None,
    state: ProposalState | None = None,
) -> CursorPage[ProposalSummary]:
    """List active proposals — paginated, optionally filtered by state."""
    proposals = await runtime.proposals.list_active()
    if state is not None:
        proposals = [p for p in proposals if p.state == state]
    summaries = [
        ProposalSummary(
            id=p.id,
            state=p.state,
            submission_kind=p.submission_kind,
            submitted_by=p.submitted_by,
            reproduce_paper_arxiv_id=p.reproduce_paper_arxiv_id,
            created_at=p.created_at,
            updated_at=p.updated_at,
            last_error=p.last_error,
            version=p.version,
        )
        for p in proposals
    ]
    return paginate(summaries, cursor=cursor, limit=limit)


# === GET /api/v1/proposals/{proposal_id} ===================================


@router.get(PROPOSAL_DETAIL, response_model=ProposalDetailResponse)
async def get_proposal(
    proposal_id: str,
    runtime: RuntimeDep,
) -> ProposalDetailResponse:
    """Detail endpoint. ``registered_cg_ids`` is populated only once the
    proposal reaches ``registered`` (per ``08`` §3.3 — the user-decision
    write + 1+ CG inserts happen in one transaction). Pre-registration
    the field is present and empty per ``11`` §13 OQ2 RESOLVED 2026-05-03.
    """
    record = await runtime.proposals.get(proposal_id)
    cg_ids: list[str] = []
    if record.state == "registered":
        cgs = await runtime.proposals.list_cgs(proposal_id)
        cg_ids = [cg.id for cg in cgs]
    return ProposalDetailResponse(
        id=record.id,
        submitted_by=record.submitted_by,
        submission_kind=record.submission_kind,
        state=record.state,
        technique_description_artifact_key=record.technique_description_artifact_key,
        reproduce_paper_arxiv_id=record.reproduce_paper_arxiv_id,
        design_plan_artifact_key=record.design_plan_artifact_key,
        error_context_artifact_key=record.error_context_artifact_key,
        design_attempt=record.design_attempt,
        registration_attempt=record.registration_attempt,
        user_decision=record.user_decision,
        user_decided_at=record.user_decided_at,
        registered_cg_ids=cg_ids,
        created_at=record.created_at,
        updated_at=record.updated_at,
        last_error=record.last_error,
        version=record.version,
    )


# === POST /api/v1/proposals/{proposal_id}/approve ==========================


@router.post(PROPOSAL_APPROVE, response_model=ApproveProposalResponse)
async def approve_proposal(
    proposal_id: str,
    runtime: RuntimeDep,
) -> ApproveProposalResponse:
    """Fire the ``designed → registered`` edge. Synchronous transaction
    per ``08`` §3.3 — user-decision write + 1+ CG inserts atomic."""
    record = await runtime.proposals.approve(proposal_id)
    cgs = await runtime.proposals.list_cgs(proposal_id)
    # ``user_decided_at`` is set in the same transaction as the state
    # change; the spec model requires it on the response, so default to
    # a freshly-resolved ``now()`` if the underlying record's value
    # somehow ended up None (defensive — ProposalsService.approve
    # always sets it).
    decided_at = record.user_decided_at or datetime.now(UTC)
    return ApproveProposalResponse(
        id=record.id,
        state=record.state,
        cg_ids=[cg.id for cg in cgs],
        user_decided_at=decided_at,
    )


# === POST /api/v1/proposals/{proposal_id}/reject ===========================


@router.post(PROPOSAL_REJECT, response_model=RejectProposalResponse)
async def reject_proposal(
    proposal_id: str,
    runtime: RuntimeDep,
    body: RejectProposalRequest | None = None,
) -> RejectProposalResponse:
    """Fire the ``designed → rejected`` edge. The optional ``reason``
    body field is audit-trail informational per ``11`` §5.2.1."""
    reason = body.reason if body is not None else None
    record = await runtime.proposals.reject(proposal_id, reason=reason)
    decided_at = record.user_decided_at or datetime.now(UTC)
    return RejectProposalResponse(
        id=record.id,
        state=record.state,
        user_decided_at=decided_at,
    )


# === Helpers ===============================================================


def _new_proposal_id() -> str:
    """Mint a new proposal id when the caller didn't supply one.

    The ``ProposalRecord.id`` validator accepts ULID-shaped or human-
    readable strings (per ``01-data-model.md`` §5.2.2 — the format is
    permissive in v1). We use a UUID4 hex prefix; production callers
    typically supply their own ULIDs via ``proposal_id`` for
    idempotency.
    """
    return f"prop-{uuid4().hex[:16]}"


__all__ = ["router"]
