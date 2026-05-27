"""``/api/v1/comparison-groups`` router per ``designs/smai/11-api.md`` §4.4.

CGs have NO CRUD on the contract — they are created exclusively by
``POST /api/v1/proposals/{id}/approve`` (proposal-born) or
``POST /api/v1/experiments`` (smai-run-born). The router below covers
the read surface plus the artifact transport.

The artifact endpoint per §5.2.4 is the one place where the API
behaves differently across plugin shapes — for ``ArtifactStore``
plugins that return HTTP-shaped presigned URLs we 302-redirect; for
plugins whose ``url_for`` returns a non-HTTP URL (e.g.
``LocalFsStore``'s ``file://...``) we stream the bytes back. The SPA
never knows which mode is active. See the inline comment on the
``GET /artifacts/{path}`` handler for the spec deviation we landed on
to keep ``smai dev``'s LocalFs deployment working with a browser.
"""

from __future__ import annotations

import json
import mimetypes
from typing import cast

from fastapi import APIRouter
from fastapi.responses import RedirectResponse, Response
from smai_api_spec import (
    AgentStatusResponse,
    ArtifactKeysResponse,
    CGState,
    CGStatusResponse,
    CGSummary,
    ComparisonGroupDetailResponse,
    CursorPage,
    EntryDetailResponse,
    EntrySummary,
    EntryWithRuns,
    EvaluationResultResponse,
    HarnessAgentStatus,
)
from smai_api_spec.comparison_groups import EntryAgentStatus
from smai_api_spec.paths import (
    COMPARISON_GROUP_AGENT_STATUS,
    COMPARISON_GROUP_ARTIFACT,
    COMPARISON_GROUP_ARTIFACTS,
    COMPARISON_GROUP_DETAIL,
    COMPARISON_GROUP_ENTRIES,
    COMPARISON_GROUP_ENTRY_DETAIL,
    COMPARISON_GROUP_EVALUATION,
    COMPARISON_GROUP_STATUS,
    COMPARISON_GROUPS,
)
from smai_cli.runtime import TERMINAL_CG_STATES
from smai_core.plugins.artifact_store import ArtifactNotFound

from smai_api._deps import RuntimeDep
from smai_api._pagination import paginate
from smai_api._record_projections import (
    entry_record_to_summary,
    entry_record_to_with_runs,
    run_record_to_summary,
)
from smai_api.errors import EntryNotFoundError

router = APIRouter()


# === GET /api/v1/comparison-groups =========================================


@router.get(COMPARISON_GROUPS, response_model=CursorPage[CGSummary])
async def list_cgs(
    runtime: RuntimeDep,
    cursor: str | None = None,
    limit: int | None = None,
    state: CGState | None = None,
    proposal_id: str | None = None,
) -> CursorPage[CGSummary]:
    """List active CGs — paginated, filterable by state and parent proposal."""
    cgs = await runtime.status.list_active_cgs()
    if state is not None:
        cgs = [c for c in cgs if c.state == state]
    if proposal_id is not None:
        cgs = [c for c in cgs if c.proposal_id == proposal_id]
    summaries = [
        CGSummary(
            id=c.id,
            proposal_id=c.proposal_id,
            state=c.state,
            is_terminal=c.state in TERMINAL_CG_STATES,
            created_at=c.created_at,
            updated_at=c.updated_at,
            last_error=c.last_error,
            version=c.version,
        )
        for c in cgs
    ]
    return paginate(summaries, cursor=cursor, limit=limit)


# === GET /api/v1/comparison-groups/{cg_id} =================================


@router.get(COMPARISON_GROUP_DETAIL, response_model=ComparisonGroupDetailResponse)
async def get_cg_detail(
    cg_id: str,
    runtime: RuntimeDep,
) -> ComparisonGroupDetailResponse:
    """Full CG detail with embedded entries + per-entry runs (per ``11`` §4.4)."""
    record = await runtime.status.get_cg_record(cg_id)
    entries = await runtime.status.list_entries_for_cg(cg_id)
    embedded: list[EntryWithRuns] = []
    for entry in entries:
        runs = await runtime.status.list_runs_for_entry(entry.id)
        embedded.append(entry_record_to_with_runs(entry, runs))
    return ComparisonGroupDetailResponse(
        id=record.id,
        proposal_id=record.proposal_id,
        factor_model_id=record.factor_model_id,
        state=record.state,
        is_terminal=record.state in TERMINAL_CG_STATES,
        experiment_definition_id=record.experiment_definition_id,
        experiment_plan_hash=record.experiment_plan_hash,
        harness_contract_hash=record.harness_contract_hash,
        validation_config_hash=record.validation_config_hash,
        code_review_attempt=record.code_review_attempt,
        code_review_result_hash=record.code_review_result_hash,
        entries=embedded,
        created_at=record.created_at,
        updated_at=record.updated_at,
        last_error=record.last_error,
        version=record.version,
    )


# === GET /api/v1/comparison-groups/{cg_id}/status ==========================


@router.get(COMPARISON_GROUP_STATUS, response_model=CGStatusResponse)
async def get_cg_status(
    cg_id: str,
    runtime: RuntimeDep,
) -> CGStatusResponse:
    """Narrow status snapshot for the per-row state pill."""
    snapshot = await runtime.status.get(cg_id)
    return CGStatusResponse(
        id=snapshot.cg_id,
        state=cast("CGState", snapshot.state),
        is_terminal=snapshot.is_terminal,
        updated_at=snapshot.updated_at,
    )


# === GET /api/v1/comparison-groups/{cg_id}/agent-status ===================


@router.get(COMPARISON_GROUP_AGENT_STATUS, response_model=AgentStatusResponse)
async def get_cg_agent_status(
    cg_id: str,
    runtime: RuntimeDep,
) -> AgentStatusResponse:
    """Composite agent-status read per ``11`` §5.2.3.

    Pulls one ``comparison-groups/{cg_id}/harness/status.json`` and one
    per-entry ``comparison-groups/{cg_id}/entries/{entry_id}/status.json``
    (the technique implementer's per-turn payload — see
    :data:`smai_orchestrator.sandboxed_dispatch.technique_implementer.DEFAULT_STATUS_KEY_TEMPLATE`,
    which has no ``code/`` segment), parses each, joins them. The SPA
    gets one round-trip instead of fanning out to N artifact reads.

    A 404 from the underlying CG is allowed to bubble (the central
    handler renders ``CG_NOT_FOUND``); missing per-entry status.json
    files surface as ``status=None`` in the response (NOT a 404 — the
    parent CG exists, just without that artifact yet).
    """
    # Validate the CG exists first so we 404 cleanly when it doesn't.
    await runtime.status.get_cg_record(cg_id)
    entries = await runtime.status.list_entries_for_cg(cg_id)
    artifact_store = runtime.plugins.artifact_store

    # Harness status.
    harness_key = f"comparison-groups/{cg_id}/harness/status.json"
    harness: HarnessAgentStatus | None = None
    try:
        raw = await artifact_store.get(harness_key)
        parsed = _safe_json(raw)
        if parsed is not None:
            harness = HarnessAgentStatus.model_validate(parsed)
    except ArtifactNotFound:
        harness = None

    # Per-entry status.
    entries_status: dict[str, EntryAgentStatus] = {}
    for entry in entries:
        entry_key = f"comparison-groups/{cg_id}/entries/{entry.id}/status.json"
        try:
            raw = await artifact_store.get(entry_key)
            parsed = _safe_json(raw)
            entries_status[entry.id] = EntryAgentStatus(
                technique_id=entry.technique_id,
                status=parsed,
            )
        except ArtifactNotFound:
            entries_status[entry.id] = EntryAgentStatus(
                technique_id=entry.technique_id,
                status=None,
            )

    return AgentStatusResponse(harness=harness, entries=entries_status)


# === GET /api/v1/comparison-groups/{cg_id}/entries =========================


@router.get(COMPARISON_GROUP_ENTRIES, response_model=CursorPage[EntrySummary])
async def list_entries(
    cg_id: str,
    runtime: RuntimeDep,
    cursor: str | None = None,
    limit: int | None = None,
) -> CursorPage[EntrySummary]:
    """List entries on a CG."""
    # Validate CG exists; otherwise 404.
    await runtime.status.get_cg_record(cg_id)
    entries = await runtime.status.list_entries_for_cg(cg_id)
    summaries = [entry_record_to_summary(e) for e in entries]
    return paginate(summaries, cursor=cursor, limit=limit)


# === GET /api/v1/comparison-groups/{cg_id}/entries/{entry_id} ==============


@router.get(COMPARISON_GROUP_ENTRY_DETAIL, response_model=EntryDetailResponse)
async def get_entry_detail(
    cg_id: str,
    entry_id: str,
    runtime: RuntimeDep,
) -> EntryDetailResponse:
    """Entry detail."""
    # Validate parent CG.
    await runtime.status.get_cg_record(cg_id)
    entries = await runtime.status.list_entries_for_cg(cg_id)
    record = next((e for e in entries if e.id == entry_id), None)
    if record is None:
        raise EntryNotFoundError(entry_id)
    runs = await runtime.status.list_runs_for_entry(entry_id)
    return EntryDetailResponse(
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


# === GET /api/v1/comparison-groups/{cg_id}/evaluation ======================


@router.get(COMPARISON_GROUP_EVALUATION, response_model=EvaluationResultResponse)
async def get_cg_evaluation(
    cg_id: str,
    runtime: RuntimeDep,
) -> EvaluationResultResponse:
    """Read ``evaluation_result.json`` for the CG. 404 +
    ``ARTIFACT_NOT_FOUND`` if it has not yet been produced (the common
    case for CGs still ``running`` / ``evaluating``)."""
    # Validate CG exists; the central handler 404s with CG_NOT_FOUND
    # if not.
    await runtime.status.get_cg_record(cg_id)
    artifact_store = runtime.plugins.artifact_store
    evaluation_key = f"comparison-groups/{cg_id}/evaluation_result.json"
    raw = await artifact_store.get(evaluation_key)
    parsed = _safe_json(raw)
    if parsed is None:
        # Treat unparseable evaluation_result.json as "not produced" —
        # the artifact is corrupt; the 404 + ARTIFACT_NOT_FOUND path is
        # the right surface for the SPA per ``11`` §4.4.
        raise ArtifactNotFound(evaluation_key)
    verdict = str(parsed.get("verdict", "unknown"))
    raw_metrics = parsed.get("raw_metrics", {})
    per_entry = parsed.get("per_entry", {})
    contextual = parsed.get("contextual_evaluation")
    if not isinstance(raw_metrics, dict):
        raw_metrics = {}
    if not isinstance(per_entry, dict):
        per_entry = {}
    if contextual is not None and not isinstance(contextual, dict):
        contextual = None
    return EvaluationResultResponse(
        cg_id=cg_id,
        verdict=verdict,
        raw_metrics=cast("dict[str, object]", raw_metrics),
        per_entry=cast("dict[str, object]", per_entry),
        contextual_evaluation=cast("dict[str, object] | None", contextual),
        artifact_key=evaluation_key,
    )


# === GET /api/v1/comparison-groups/{cg_id}/artifacts =======================


@router.get(COMPARISON_GROUP_ARTIFACTS, response_model=ArtifactKeysResponse)
async def list_artifacts(
    cg_id: str,
    runtime: RuntimeDep,
    prefix: str | None = None,
) -> ArtifactKeysResponse:
    """List artifact keys under the CG namespace; optional ``?prefix=``
    narrows the listing per ``11`` §5.2.4 (trailing-slash variant).

    Per ``11`` §4.4, the `?prefix=` filter is an additional path
    component anchored under the CG's namespace — e.g.
    ``?prefix=harness/`` returns ``harness/status.json``,
    ``harness/contract.json`` etc. We pass the CG-rooted full prefix to
    :meth:`ArtifactStore.list` and strip the namespace from the
    returned keys so the response is relative to the CG.
    """
    await runtime.status.get_cg_record(cg_id)
    namespace = f"comparison-groups/{cg_id}/"
    full_prefix = namespace + (prefix or "")
    artifact_store = runtime.plugins.artifact_store
    keys: list[str] = []
    iterator = await artifact_store.list(full_prefix)
    async for key in iterator:
        # Strip the CG namespace prefix; consumers use the relative key
        # directly with the artifact-fetch endpoint.
        if key.startswith(namespace):
            keys.append(key[len(namespace) :])
        else:
            keys.append(key)
    return ArtifactKeysResponse(keys=keys)


# === GET /api/v1/comparison-groups/{cg_id}/artifacts/{path:path} ===========


# The ``{path}`` placeholder in the spec's URL constant is a single-
# segment match by default in FastAPI; the artifact endpoint takes a
# multi-segment relative path (``harness/status.json``,
# ``entries/{id}/code/foo.py``), so the route registration uses the
# ``{path:path}`` converter form. The constant from smai_api_spec is
# preserved by callers; only the route registration substitutes.
_ARTIFACT_ROUTE = COMPARISON_GROUP_ARTIFACT.replace("{path}", "{path:path}")


@router.get(_ARTIFACT_ROUTE)
async def get_artifact(
    cg_id: str,
    path: str,
    runtime: RuntimeDep,
) -> Response:
    """Stream artifact bytes or 302-redirect to a presigned URL.

    Per ``11`` §5.2.4 the contract is "transparent to consumer";
    plugin-capability detection picks the mode. Our heuristic:

    * If the configured ``ArtifactStore`` exposes
      ``capabilities.supports_presigned_urls=True`` AND ``url_for``
      returns an HTTP/HTTPS URL, return ``302 Found`` with that URL in
      ``Location``.
    * Otherwise, stream the bytes via :meth:`ArtifactStore.get` with a
      Content-Type sniffed from the path's extension
      (``application/octet-stream`` fallback).

    Why the HTTP-scheme check on top of the spec's capability bit:
    ``LocalFsStore`` reports ``supports_presigned_urls=True`` because
    its ``url_for`` returns ``file://<absolute_path>`` (a usable URL,
    just not presigned with expiry semantics). A ``302`` to ``file://``
    is not useful for a browser caller, so we fall through to the
    streaming path in that case. Real S3-shaped presigned URLs (the
    deployment shape that benefits most from 302) are correctly
    detected as HTTPS and redirect.

    The ``ArtifactNotFound`` exception flows up to the central handler
    which renders ``404 ARTIFACT_NOT_FOUND``.
    """
    await runtime.status.get_cg_record(cg_id)
    namespace = f"comparison-groups/{cg_id}/"
    key = namespace + path
    artifact_store = runtime.plugins.artifact_store
    if artifact_store.capabilities.supports_presigned_urls:
        url = await artifact_store.url_for(key)
        if url.startswith("http://") or url.startswith("https://"):
            return RedirectResponse(url=url, status_code=302)
    data = await artifact_store.get(key)
    media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return Response(content=data, media_type=media_type)


# === Helpers ===============================================================


def _safe_json(raw: bytes) -> dict[str, object] | None:
    """Decode + JSON-parse ``raw``; ``None`` on failure or non-object payload.

    The agent-status / evaluation paths read JSON written by other
    pipelines; on a malformed write the safest surface is "treat as
    not present" rather than 500 — those artifacts are recovery-on-
    rerun anyway.
    """
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    try:
        parsed = json.loads(decoded)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return cast("dict[str, object]", parsed)


__all__ = ["router"]
