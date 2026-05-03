"""Comparison-group request / response shapes per ``designs/smai/11-api.md`` §4.4.

CGs have no CRUD on the contract — they are created exclusively by
``POST /api/v1/proposals/{id}/approve`` (proposal-born) or
``POST /api/v1/experiments`` (smai-run-born). Direct mutation is not on
the surface.

Endpoints covered:

* ``GET .../comparison-groups`` — :class:`ListCGsParams` → ``CursorPage[CGSummary]``
* ``GET .../comparison-groups/{cg_id}`` → :class:`ComparisonGroupDetailResponse`
* ``GET .../comparison-groups/{cg_id}/status`` → :class:`CGStatusResponse`
* ``GET .../comparison-groups/{cg_id}/agent-status`` → :class:`AgentStatusResponse`
* ``GET .../comparison-groups/{cg_id}/entries`` → ``CursorPage[EntrySummary]``
* ``GET .../comparison-groups/{cg_id}/entries/{entry_id}`` → :class:`EntryDetailResponse`
* ``GET .../comparison-groups/{cg_id}/evaluation`` → :class:`EvaluationResultResponse`
* ``GET .../comparison-groups/{cg_id}/artifacts`` —
  :class:`ListArtifactsParams` → :class:`ArtifactKeysResponse`
* ``GET .../comparison-groups/{cg_id}/artifacts/{path:path}`` →
  bytes (streams or 302-redirects; no Pydantic response body)

URL constants for these are exported from :mod:`smai_api_spec.paths`.
"""

from __future__ import annotations

from datetime import datetime

from smai_api_spec._common import APIBaseModel, BaseAuditedResponse, CGState
from smai_api_spec.entries import EntryWithRuns
from smai_api_spec.pagination import PaginationParams

# === GET /api/v1/comparison-groups ==========================================


class ListCGsParams(PaginationParams):
    """Query-param model for ``GET /api/v1/comparison-groups``."""

    state: CGState | None = None
    proposal_id: str | None = None


class CGSummary(BaseAuditedResponse):
    """One item in the ``GET /api/v1/comparison-groups`` list response."""

    id: str
    proposal_id: str
    state: CGState
    is_terminal: bool


# === GET /api/v1/comparison-groups/{cg_id} ==================================


class ComparisonGroupDetailResponse(BaseAuditedResponse):
    """Full CG detail with embedded entries + runs.

    Per ``11`` §4.4 the embedding is always-on in v1. Per ``11`` §5.1.2
    the lease triple is NOT exposed; ``harness_job_handle`` is also
    withheld as an orchestrator-internal opaque token. Artifacts are NOT
    inlined — fetch them via the per-CG artifact endpoints.
    """

    id: str
    proposal_id: str
    factor_model_id: str | None
    state: CGState
    is_terminal: bool
    experiment_definition_id: str
    experiment_plan_hash: str | None
    harness_contract_hash: str | None
    validation_config_hash: str | None
    code_review_attempt: int
    code_review_result_hash: str | None
    entries: list[EntryWithRuns]


# === GET /api/v1/comparison-groups/{cg_id}/status ===========================


class CGStatusResponse(APIBaseModel):
    """Narrow status snapshot for ``GET /comparison-groups/{cg_id}/status``.

    The dashboard uses this for the per-row state pill — cheaper than
    fetching the full detail body.
    """

    id: str
    state: CGState
    is_terminal: bool
    updated_at: datetime


# === GET /api/v1/comparison-groups/{cg_id}/agent-status =====================


class HarnessAgentStatus(APIBaseModel):
    """Mirrors what the harness builder writes to ``harness/status.json``.

    The full schema is owned by ``04-agents.md``; the contract surface
    here is a narrow projection of the fields the SPA renders. Each
    field is optional so partial / in-progress writes parse cleanly.

    Per ``11`` §5.2.3 the joining + parsing happens server-side so the
    SPA gets one round-trip instead of fanning out to N artifact reads.
    """

    state: str | None = None
    turn: int | None = None
    cost_usd: float | None = None
    last_message_at: datetime | None = None


class EntryAgentStatus(APIBaseModel):
    """Per-entry agent-status projection.

    ``status`` is the parsed contents of
    ``entries/{id}/code/status.json`` (a free-form JSON object whose
    schema is owned by ``04-agents.md``); ``None`` if the agent has not
    yet written it.
    """

    technique_id: str | None
    status: dict[str, object] | None


class AgentStatusResponse(APIBaseModel):
    """``200 OK`` body for ``GET /api/v1/comparison-groups/{cg_id}/agent-status``."""

    harness: HarnessAgentStatus | None
    entries: dict[str, EntryAgentStatus]


# === GET /api/v1/comparison-groups/{cg_id}/evaluation =======================


class EvaluationResultResponse(APIBaseModel):
    """``200 OK`` body for ``GET /api/v1/comparison-groups/{cg_id}/evaluation``.

    A 404 with ``code: "ARTIFACT_NOT_FOUND"`` is returned if
    ``evaluation_result.json`` has not yet been produced — this is the
    common case for CGs that are still ``running`` or ``evaluating``.

    The shape mirrors :class:`smai_core.entities.EvaluationResult`. The
    full nested types (``RawMetrics``, per-entry verdict objects) are
    methodology-owned (``06-mechanical-evaluation.md``); the API
    contract treats the inner dicts as opaque ``dict[str, object]`` so
    additive shape changes in ``smai-core`` do not require a contract
    bump.
    """

    cg_id: str
    verdict: str
    raw_metrics: dict[str, object]
    per_entry: dict[str, object]
    contextual_evaluation: dict[str, object] | None = None
    artifact_key: str | None = None


# === GET /api/v1/comparison-groups/{cg_id}/artifacts ========================


class ListArtifactsParams(APIBaseModel):
    """Query-param model for ``GET /comparison-groups/{cg_id}/artifacts``.

    Optional ``prefix`` filter narrows the listing to keys under a
    sub-namespace (e.g. ``entries/`` to list per-entry artifacts only).
    """

    prefix: str | None = None


class ArtifactKeysResponse(APIBaseModel):
    """``200 OK`` body listing artifact keys under the CG namespace.

    Per ``11`` §5.2.4 the trailing-slash variant of the artifact URL is
    the *list* form; the with-path variant streams or 302-redirects.
    Listing is always JSON; it does not paginate in v1 (CG artifact
    counts are bounded by the methodology — entries × runs × per-run
    files).
    """

    keys: list[str]


# Note: the artifact-fetch endpoint
# ``GET /api/v1/comparison-groups/{cg_id}/artifacts/{path:path}`` does
# not have a Pydantic response model — it streams bytes (LocalFs
# capability) or returns a 302 with ``Location:`` (S3 capability) per
# ``11`` §5.2.4. The ``ErrorEnvelope`` is returned on 404 (artifact not
# found).


__all__ = [
    "AgentStatusResponse",
    "ArtifactKeysResponse",
    "CGStatusResponse",
    "CGSummary",
    "ComparisonGroupDetailResponse",
    "EntryAgentStatus",
    "EvaluationResultResponse",
    "HarnessAgentStatus",
    "ListArtifactsParams",
    "ListCGsParams",
]
