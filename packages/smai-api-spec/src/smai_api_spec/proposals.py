"""Proposal request / response shapes per ``designs/smai/11-api.md`` §4.1 / §5.2.1.

Per DEC-032, proposals are the **primary input path** in v2 — every
HTTP user-driven CG creation flows through this surface.

Endpoints covered (URL constants exported from :mod:`smai_api_spec.paths`):

* ``POST .../proposals`` — :class:`SubmitProposalRequest` →
  :class:`ProposalSubmissionResponse`
* ``GET  .../proposals`` — :class:`ListProposalsParams` →
  ``CursorPage[ProposalSummary]``
* ``GET  .../proposals/{id}`` → :class:`ProposalDetailResponse`
* ``POST .../proposals/{id}/approve`` → :class:`ApproveProposalResponse`
* ``POST .../proposals/{id}/reject`` —
  :class:`RejectProposalRequest` (optional) → :class:`RejectProposalResponse`
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import model_validator

from smai_api_spec._common import (
    APIBaseModel,
    BaseAuditedResponse,
    ProposalState,
    SubmissionKind,
    UserDecision,
)
from smai_api_spec.pagination import PaginationParams

# === POST /api/v1/proposals =================================================


class SubmitProposalRequest(APIBaseModel):
    """Submit a novel-technique proposal or a reproduce-paper proposal.

    Per ``11`` §5.2.1: exactly one of the three description fields must be
    populated; the populated field must be consistent with
    ``submission_kind``:

    * ``submission_kind="novel_technique"`` → exactly one of
      ``technique_description`` / ``technique_description_text``.
    * ``submission_kind="reproduce_paper"`` → ``reproduce_paper_arxiv_id``,
      and the referenced paper must exist + be in a terminal
      ``registered`` state. Implementations enforce the latter at submit
      time and respond with 409 + ``code: "PAPER_NOT_READY"`` if the
      paper is missing or non-terminal (per ``11`` §13 OQ11 RESOLVED
      2026-05-03).

    ``proposal_id`` lets callers pin their own ULID for idempotency: a
    repeat submit with the same id returns the existing proposal record
    rather than 409. Without a pinned id every POST creates a new record.

    ``submitted_by`` is accepted from the client and audit-logged.
    Multi-tenant implementations (Yaarp v2) overwrite this field via
    middleware with the auth-derived subject (per ``11`` §13 OQ6
    RESOLVED 2026-05-03 — same Pydantic shape on both sides).
    """

    submission_kind: SubmissionKind = "novel_technique"
    technique_description: dict[str, object] | None = None
    technique_description_text: str | None = None
    reproduce_paper_arxiv_id: str | None = None
    proposal_id: str | None = None
    submitted_by: str | None = None

    @model_validator(mode="after")
    def _exactly_one_description_form(self) -> Self:
        # Count populated description fields. ``technique_description``
        # is a dict; treat ``{}`` as populated (some callers may submit
        # an explicitly-empty design baseline) — the check is presence,
        # not truthiness.
        forms = {
            "technique_description": self.technique_description is not None,
            "technique_description_text": self.technique_description_text is not None,
            "reproduce_paper_arxiv_id": self.reproduce_paper_arxiv_id is not None,
        }
        populated = [name for name, present in forms.items() if present]
        if len(populated) == 0:
            raise ValueError(
                "exactly one of technique_description / technique_description_text / "
                "reproduce_paper_arxiv_id must be populated; got none"
            )
        if len(populated) > 1:
            raise ValueError(
                "exactly one of technique_description / technique_description_text / "
                f"reproduce_paper_arxiv_id must be populated; got {populated}"
            )
        # Cross-validate with submission_kind.
        if self.submission_kind == "reproduce_paper":
            if self.reproduce_paper_arxiv_id is None:
                raise ValueError(
                    "submission_kind='reproduce_paper' requires reproduce_paper_arxiv_id"
                )
        else:
            # submission_kind == "novel_technique"
            if self.reproduce_paper_arxiv_id is not None:
                raise ValueError(
                    "reproduce_paper_arxiv_id is only valid with submission_kind='reproduce_paper'"
                )
        return self


class ProposalSubmissionResponse(APIBaseModel):
    """``202 Accepted`` body for ``POST /api/v1/proposals``.

    The proposal lands in ``proposal_submitted``; the worker drives
    further transitions on its own cadence. Clients poll
    ``GET /proposals/{id}`` (or subscribe via SSE) for progress.
    """

    id: str
    state: ProposalState
    submission_kind: SubmissionKind
    technique_description_artifact_key: str | None
    reproduce_paper_arxiv_id: str | None
    submitted_by: str | None
    created_at: datetime


# === GET /api/v1/proposals ==================================================


class ListProposalsParams(PaginationParams):
    """Query-param model for ``GET /api/v1/proposals``."""

    state: ProposalState | None = None


class ProposalSummary(BaseAuditedResponse):
    """One item in the ``GET /api/v1/proposals`` list response."""

    id: str
    state: ProposalState
    submission_kind: SubmissionKind
    submitted_by: str | None = None
    reproduce_paper_arxiv_id: str | None = None


# === GET /api/v1/proposals/{id} =============================================


class ProposalDetailResponse(BaseAuditedResponse):
    """Full proposal detail. Per ``11`` §5.1.2 the lease triple is NOT
    exposed; ``planner_job_handle`` is also withheld as an
    orchestrator-internal opaque token.

    ``registered_cg_ids`` is populated once the proposal reaches
    ``registered`` — per ``08`` §3.3 the user-decision write + the 1+ CG
    inserts happen in one transaction.
    """

    id: str
    submitted_by: str | None
    submission_kind: SubmissionKind
    state: ProposalState
    technique_description_artifact_key: str | None
    reproduce_paper_arxiv_id: str | None
    design_plan_artifact_key: str | None
    error_context_artifact_key: str | None
    design_attempt: int
    registration_attempt: int
    user_decision: UserDecision | None
    user_decided_at: datetime | None
    registered_cg_ids: list[str] = []


# === POST /api/v1/proposals/{id}/approve ====================================


class ApproveProposalResponse(APIBaseModel):
    """``200 OK`` body for ``POST /api/v1/proposals/{id}/approve``.

    The ``designed → registered`` transition is synchronous (per ``08``
    §3.3) — the user-decision write + 1+ CG inserts happen atomically.
    """

    id: str
    state: ProposalState
    cg_ids: list[str]
    user_decided_at: datetime


# === POST /api/v1/proposals/{id}/reject =====================================


class RejectProposalRequest(APIBaseModel):
    """Optional body for ``POST /api/v1/proposals/{id}/reject``.

    The ``reason`` field is reserved for an audit trail in multi-tenant
    deployments; SMAI's OSS implementation accepts and stores it on the
    proposal record but does not require it.
    """

    reason: str | None = None


class RejectProposalResponse(APIBaseModel):
    """``200 OK`` body for ``POST /api/v1/proposals/{id}/reject``."""

    id: str
    state: ProposalState
    user_decided_at: datetime


__all__ = [
    "ApproveProposalResponse",
    "ListProposalsParams",
    "ProposalDetailResponse",
    "ProposalSubmissionResponse",
    "ProposalSummary",
    "RejectProposalRequest",
    "RejectProposalResponse",
    "SubmitProposalRequest",
]
