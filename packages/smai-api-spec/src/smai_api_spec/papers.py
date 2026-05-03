"""Paper request / response shapes per ``designs/smai/11-api.md`` §4.2.

Per DEC-032, paper ingestion is the **supporting utility** path. Papers
carry NO CG references — paper ingestion produces ``TechniqueRef``s only;
CGs are exclusively proposal-born.

Endpoints covered:

* ``POST .../papers`` — :class:`SubmitPaperRequest` → :class:`PaperSubmissionResponse`
* ``GET  .../papers`` — :class:`ListPapersParams`   → ``CursorPage[PaperSummary]``
* ``GET  .../papers/{arxiv_id}`` → :class:`PaperDetailResponse`
* ``POST .../papers/{arxiv_id}/promote-partial`` → :class:`PromotePartialResponse`
  (no request body in v1 — see ``11`` §13 OQ3)

URL constants for these are exported from :mod:`smai_api_spec.paths`.
"""

from __future__ import annotations

from datetime import datetime

from smai_api_spec._common import (
    APIBaseModel,
    BaseAuditedResponse,
    PaperState,
    ScreenDecision,
)
from smai_api_spec.pagination import PaginationParams

# === POST /api/v1/papers ====================================================


class SubmitPaperRequest(APIBaseModel):
    """Submit a paper for ingestion by ``arxiv_id``.

    Idempotent per ``11`` §4.2 / §4.8: creates a new record on first
    submit, promotes a ``partial`` record to ``submitted`` on resubmit,
    or no-ops when the paper is already in flight or in a terminal
    state.

    ``arxiv_id`` format is intentionally not locked down here — both
    ``YYMM.NNNNN[v#]`` and the legacy ``archive/YYMM###`` shapes exist
    and the plugin layer enforces per-substrate.
    """

    arxiv_id: str


class PaperSubmissionResponse(APIBaseModel):
    """``202 Accepted`` body for ``POST /api/v1/papers``."""

    arxiv_id: str
    state: PaperState
    created_at: datetime


# === GET /api/v1/papers =====================================================


class ListPapersParams(PaginationParams):
    """Query-param model for ``GET /api/v1/papers``."""

    state: PaperState | None = None


class PaperSummary(BaseAuditedResponse):
    """One item in the ``GET /api/v1/papers`` list response."""

    arxiv_id: str
    state: PaperState
    title: str | None = None


# === GET /api/v1/papers/{arxiv_id} ==========================================


class TechniqueRefSummary(APIBaseModel):
    """One paper-derived ``TechniqueRef`` exposed on the paper detail.

    Per DEC-032 / ``08`` §5: paper ingestion produces a buffer of
    ``TechniqueRef``s; the API exposes them on the paper-detail
    response so the SPA can render "this paper produced N techniques;
    submit a reproduce-paper proposal to run them."

    The internal record shape is wider than this projection — the
    contract surface here exposes only what the SPA needs to render +
    reference for a follow-up proposal.
    """

    technique_id: str
    name: str | None = None
    description: str | None = None


class PaperDetailResponse(BaseAuditedResponse):
    """Full paper detail. Job handles are NOT exposed (orchestrator-internal)."""

    arxiv_id: str
    state: PaperState
    title: str | None
    authors: list[str]
    abstract: str | None
    published_date: datetime | None
    categories: list[str]
    latex_bundle_artifact_key: str | None
    expanded_tex_artifact_key: str | None
    extracted_text_artifact_key: str | None
    figures_artifact_key: str | None
    screen_result_decision: ScreenDecision | None
    screen_result_reason: str | None
    technique_buffer_artifact_key: str | None
    error_context_artifact_key: str | None
    planning_attempt: int
    screening_attempt: int
    registration_attempt: int
    # Populated once state == "registered" (terminal accept) — the
    # technique pool produced by paper ingestion. Submit a reproduce-
    # paper proposal against an entry to run it.
    technique_refs: list[TechniqueRefSummary] = []


# === POST /api/v1/papers/{arxiv_id}/promote-partial =========================


class PromotePartialResponse(APIBaseModel):
    """``200 OK`` body for ``POST /api/v1/papers/{arxiv_id}/promote-partial``.

    Per ``11`` §13 OQ3 (open): the request body is currently empty for
    v1; an ``audit_reason`` field may be added in a minor bump if Yaarp
    v2 wants the audit hook.
    """

    arxiv_id: str
    state: PaperState


__all__ = [
    "ListPapersParams",
    "PaperDetailResponse",
    "PaperSubmissionResponse",
    "PaperSummary",
    "PromotePartialResponse",
    "SubmitPaperRequest",
    "TechniqueRefSummary",
]
