"""Spec-conformant canned responses for the package's own self-test.

The self-test at ``tests/test_self_conformance.py`` subclasses
:class:`smai_api_conformance.APIConformanceBase` with a fixture that
returns an ``httpx.AsyncClient(transport=httpx.MockTransport(handle))``
where ``handle`` is :func:`mock_handler` defined here.

The handler dispatches on ``request.method`` + ``request.url.path``
and returns a response that:

* uses the correct HTTP status code per ``11-api.md`` §4.8;
* has a body that parses cleanly into the matching
  :mod:`smai_api_spec` Pydantic model (or :class:`ErrorEnvelope` for
  non-2xx);
* is internally consistent (e.g., ``CursorPage.count == len(items)``,
  ``SystemMigrateStatusResponse.at_head == (current == head_revision)``).

For state-changing tests the handler maintains a small in-memory
state map (proposal IDs created → terminal-state behavior) so the
"approve fails on non-designed" test fires the right edge.

For the SSE test the handler returns a streaming response that
emits one canned ``state_change`` event followed by a heartbeat,
then closes the connection. This proves the test machinery handles
SSE correctly without needing a real worker.

This module is **for self-test only** — real implementations do not
import it. Living under ``src/`` rather than ``tests/`` is
intentional: tests in this package import from it, and the
``--import-mode=importlib`` setting at the workspace root rules out
naming it ``conftest.py``.

Per the per-task fixture filename hygiene convention, the
``_4_j2_<purpose>.py`` prefix avoids sys-path collisions with
sibling-conformance fixture files.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs

from httpx import AsyncByteStream, Request, Response
from smai_api_spec import paths

# === In-memory state map ====================================================
#
# Tracks proposals/papers created by the handler so subsequent GETs
# return them. A single module-level state is fine for the self-test —
# tests run sequentially against the same MockTransport instance.

_NOW = datetime(2026, 5, 3, 12, 0, 0, tzinfo=UTC)


class _State:
    """In-memory state for the mock handler."""

    def __init__(self) -> None:
        self.proposals: dict[str, dict[str, Any]] = {}
        self.papers: dict[str, dict[str, Any]] = {}
        self._next_id = 1

    def next_proposal_id(self) -> str:
        n = self._next_id
        self._next_id += 1
        return f"prop_self_{n:04d}"


_STATE = _State()

# Default fixtures the self-test subclass exposes via the
# ``existing_cg_id`` / ``existing_entry_id`` / ``existing_run_id``
# fixture overrides — the mock recognizes these IDs and returns the
# matching detail responses.
SELF_TEST_CG_ID = "cg_self_test_default"
SELF_TEST_ENTRY_ID = "entry_self_test_default"
SELF_TEST_RUN_ID = "run_self_test_default"


# === Helpers ================================================================


def _json_response(status: int, payload: dict[str, Any]) -> Response:
    return Response(
        status_code=status,
        content=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


def _error(status: int, code: str, message: str) -> Response:
    return _json_response(status, {"error": {"code": code, "message": message}})


def _isoformat(dt: datetime) -> str:
    return dt.isoformat()


def _audit_fields() -> dict[str, Any]:
    return {
        "created_at": _isoformat(_NOW),
        "updated_at": _isoformat(_NOW),
        "last_error": None,
        "version": 1,
    }


def _proposal_detail_payload(proposal_id: str, state: str = "proposal_submitted") -> dict[str, Any]:
    return _audit_fields() | {
        "id": proposal_id,
        "submitted_by": None,
        "submission_kind": "novel_technique",
        "state": state,
        "technique_description_artifact_key": (
            f"proposals/{proposal_id}/technique_description.json"
        ),
        "reproduce_paper_arxiv_id": None,
        "design_plan_artifact_key": None,
        "error_context_artifact_key": None,
        "design_attempt": 0,
        "registration_attempt": 0,
        "user_decision": None,
        "user_decided_at": None,
        "registered_cg_ids": [],
    }


def _paper_detail_payload(arxiv_id: str, state: str = "submitted") -> dict[str, Any]:
    return _audit_fields() | {
        "arxiv_id": arxiv_id,
        "state": state,
        "title": None,
        "authors": [],
        "abstract": None,
        "published_date": None,
        "categories": [],
        "latex_bundle_artifact_key": None,
        "expanded_tex_artifact_key": None,
        "extracted_text_artifact_key": None,
        "figures_artifact_key": None,
        "screen_result_decision": None,
        "screen_result_reason": None,
        "technique_buffer_artifact_key": None,
        "error_context_artifact_key": None,
        "planning_attempt": 0,
        "screening_attempt": 0,
        "registration_attempt": 0,
        "technique_refs": [],
    }


def _run_summary(run_id: str, cg_id: str, entry_id: str) -> dict[str, Any]:
    return {
        "id": run_id,
        "cg_id": cg_id,
        "entry_id": entry_id,
        "seed": 0,
        "state": "pending",
        "duration_seconds": None,
        "raw_metrics_artifact_key": None,
        "started_at": None,
        "completed_at": None,
        "failure_reason": None,
        "run_attempt": 0,
        "updated_at": _isoformat(_NOW),
    }


def _entry_with_runs(entry_id: str, cg_id: str) -> dict[str, Any]:
    return _audit_fields() | {
        "id": entry_id,
        "cg_id": cg_id,
        "technique_id": None,
        "is_baseline": True,
        "state": "pending",
        "technique_contract_hash": None,
        "harness_api_manifest_hash": None,
        "implementation_attempt": 0,
        "runs": [_run_summary(SELF_TEST_RUN_ID, cg_id, entry_id)],
    }


def _cg_detail_payload(cg_id: str) -> dict[str, Any]:
    return _audit_fields() | {
        "id": cg_id,
        "proposal_id": "prop_self_seed",
        "factor_model_id": None,
        "state": "draft",
        "is_terminal": False,
        "experiment_definition_id": cg_id,
        "experiment_plan_hash": None,
        "harness_contract_hash": None,
        "validation_config_hash": None,
        "code_review_attempt": 0,
        "code_review_result_hash": None,
        "entries": [_entry_with_runs(SELF_TEST_ENTRY_ID, cg_id)],
    }


def _query_params(request: Request) -> dict[str, str]:
    """Extract query params as a single-value dict (last wins)."""
    raw = request.url.query.decode("utf-8") if request.url.query else ""
    parsed = parse_qs(raw)
    return {k: v[-1] for k, v in parsed.items() if v}


def _read_json(request: Request) -> dict[str, Any]:
    """Read the request body as JSON, returning ``{}`` for an empty body."""
    body = request.content
    if not body:
        return {}
    try:
        parsed = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed  # type: ignore[return-value]


# === Endpoint handlers ======================================================


def _handle_post_proposals(request: Request) -> Response:
    body = _read_json(request)
    submission_kind = body.get("submission_kind", "novel_technique")

    # Validate "exactly one description form populated".
    desc_fields = (
        body.get("technique_description"),
        body.get("reproduce_paper_arxiv_id"),
    )
    populated = sum(1 for f in desc_fields if f is not None)
    if populated != 1:
        return _error(
            400,
            "VALIDATION_ERROR",
            "exactly one description field must be populated",
        )

    if submission_kind == "reproduce_paper":
        arxiv_id = body.get("reproduce_paper_arxiv_id")
        if arxiv_id and arxiv_id not in _STATE.papers:
            # Per OQ11 RESOLVED: 409 with PAPER_NOT_READY.
            return _error(
                409,
                "PAPER_NOT_READY",
                f"paper {arxiv_id!r} does not exist or is not in terminal state",
            )

    proposal_id = body.get("proposal_id") or _STATE.next_proposal_id()
    _STATE.proposals[proposal_id] = _proposal_detail_payload(proposal_id)
    return _json_response(
        202,
        {
            "id": proposal_id,
            "state": "proposal_submitted",
            "submission_kind": submission_kind,
            "technique_description_artifact_key": (
                f"proposals/{proposal_id}/technique_description.json"
            ),
            "reproduce_paper_arxiv_id": body.get("reproduce_paper_arxiv_id"),
            "submitted_by": body.get("submitted_by"),
            "created_at": _isoformat(_NOW),
        },
    )


def _handle_get_proposals(request: Request) -> Response:
    params = _query_params(request)
    state_filter = params.get("state")
    items: list[dict[str, Any]] = []
    for proposal in _STATE.proposals.values():
        if state_filter and proposal["state"] != state_filter:
            continue
        items.append(
            _audit_fields()
            | {
                "id": proposal["id"],
                "state": proposal["state"],
                "submission_kind": proposal["submission_kind"],
                "submitted_by": None,
                "reproduce_paper_arxiv_id": proposal["reproduce_paper_arxiv_id"],
            }
        )
    return _json_response(200, {"items": items, "next_cursor": None, "count": len(items)})


def _handle_get_proposal_detail(proposal_id: str) -> Response:
    proposal = _STATE.proposals.get(proposal_id)
    if proposal is None:
        return _error(404, "PROPOSAL_NOT_FOUND", f"proposal {proposal_id!r} not found")
    return _json_response(200, proposal)


def _handle_proposal_rpc(proposal_id: str, action: str) -> Response:
    """POST /proposals/{id}/{approve|reject}.

    For the self-test we model the typical contract: a freshly-created
    proposal is in ``proposal_submitted`` (NOT ``designed``), so both
    approve and reject return 409 INVALID_STATE. The 404 branch fires
    when the proposal doesn't exist.
    """
    proposal = _STATE.proposals.get(proposal_id)
    if proposal is None:
        return _error(404, "PROPOSAL_NOT_FOUND", f"proposal {proposal_id!r} not found")
    if proposal["state"] != "designed":
        return _error(
            409,
            "INVALID_STATE",
            f"proposal {proposal_id!r} is in state {proposal['state']!r}; "
            f"cannot {action} (requires designed)",
        )
    # Synthesized happy-path responses (not exercised by current tests
    # but kept here for completeness).
    if action == "approve":
        return _json_response(
            200,
            {
                "id": proposal_id,
                "state": "registered",
                "cg_ids": [SELF_TEST_CG_ID],
                "user_decided_at": _isoformat(_NOW),
            },
        )
    return _json_response(
        200,
        {
            "id": proposal_id,
            "state": "rejected",
            "user_decided_at": _isoformat(_NOW),
        },
    )


def _handle_post_papers(request: Request) -> Response:
    body = _read_json(request)
    arxiv_id = body.get("arxiv_id")
    if not arxiv_id or not isinstance(arxiv_id, str):
        return _error(400, "VALIDATION_ERROR", "arxiv_id is required")
    # Idempotent: store on first submit, no-op on resubmit.
    if arxiv_id not in _STATE.papers:
        _STATE.papers[arxiv_id] = _paper_detail_payload(arxiv_id)
    return _json_response(
        202,
        {
            "arxiv_id": arxiv_id,
            "state": "submitted",
            "created_at": _isoformat(_NOW),
        },
    )


def _handle_get_papers(request: Request) -> Response:
    params = _query_params(request)
    state_filter = params.get("state")
    limit_str = params.get("limit")
    cursor = params.get("cursor")
    limit = int(limit_str) if limit_str else None

    all_items = [
        _audit_fields()
        | {
            "arxiv_id": paper["arxiv_id"],
            "state": paper["state"],
            "title": None,
        }
        for paper in _STATE.papers.values()
        if not state_filter or paper["state"] == state_filter
    ]
    # Sort for deterministic pagination.
    all_items.sort(key=lambda p: str(p["arxiv_id"]))

    # Cursor = index into all_items as a string.
    start = int(cursor) if cursor and cursor.isdigit() else 0
    end = start + limit if limit else len(all_items)
    items = all_items[start:end]
    next_cursor = str(end) if end < len(all_items) else None
    return _json_response(
        200,
        {"items": items, "next_cursor": next_cursor, "count": len(items)},
    )


def _handle_get_paper_detail(arxiv_id: str) -> Response:
    paper = _STATE.papers.get(arxiv_id)
    if paper is None:
        return _error(404, "PAPER_NOT_FOUND", f"paper {arxiv_id!r} not found")
    return _json_response(200, paper)


def _handle_paper_promote(arxiv_id: str) -> Response:
    paper = _STATE.papers.get(arxiv_id)
    if paper is None:
        return _error(404, "PAPER_NOT_FOUND", f"paper {arxiv_id!r} not found")
    if paper["state"] != "partial":
        return _error(
            409,
            "INVALID_STATE",
            f"paper {arxiv_id!r} is in state {paper['state']!r}; cannot promote",
        )
    # Synthesized happy-path response (not exercised by current tests).
    return _json_response(200, {"arxiv_id": arxiv_id, "state": "submitted"})


def _handle_post_experiments_compile(request: Request) -> Response:
    body = _read_json(request)
    if not body.get("definition_text"):
        return _error(400, "VALIDATION_ERROR", "definition_text is required")
    return _json_response(
        200,
        {
            "compilations": [
                {
                    "cg_id": "cg_compile_self_test",
                    "experiment_plan": {"placeholder": True},
                    "harness_contract": {"placeholder": True},
                    "technique_contracts": [],
                    "validation_config": {"placeholder": True},
                }
            ]
        },
    )


def _handle_post_experiments(request: Request) -> Response:
    body = _read_json(request)
    if not body.get("definition_text"):
        return _error(400, "VALIDATION_ERROR", "definition_text is required")
    return _json_response(
        202,
        {
            "cgs": [
                {"cg_id": "cg_submit_self_test", "state": "draft"},
            ]
        },
    )


def _handle_get_cgs(request: Request) -> Response:
    params = _query_params(request)
    state_filter = params.get("state")
    items: list[dict[str, Any]] = []
    cg = {
        **_audit_fields(),
        "id": SELF_TEST_CG_ID,
        "proposal_id": "prop_self_seed",
        "state": "draft",
        "is_terminal": False,
    }
    if not state_filter or cg["state"] == state_filter:
        items.append(cg)
    return _json_response(200, {"items": items, "next_cursor": None, "count": len(items)})


def _handle_get_cg_detail(cg_id: str) -> Response:
    if cg_id != SELF_TEST_CG_ID:
        return _error(404, "CG_NOT_FOUND", f"comparison group {cg_id!r} not found")
    return _json_response(200, _cg_detail_payload(cg_id))


def _handle_get_cg_status(cg_id: str) -> Response:
    if cg_id != SELF_TEST_CG_ID:
        return _error(404, "CG_NOT_FOUND", f"comparison group {cg_id!r} not found")
    return _json_response(
        200,
        {
            "id": cg_id,
            "state": "draft",
            "is_terminal": False,
            "updated_at": _isoformat(_NOW),
        },
    )


def _handle_get_cg_agent_status(cg_id: str) -> Response:
    if cg_id != SELF_TEST_CG_ID:
        return _error(404, "CG_NOT_FOUND", f"comparison group {cg_id!r} not found")
    return _json_response(200, {"harness": None, "entries": {}})


def _handle_get_cg_entries(cg_id: str) -> Response:
    if cg_id != SELF_TEST_CG_ID:
        return _error(404, "CG_NOT_FOUND", f"comparison group {cg_id!r} not found")
    items = [
        {
            "id": SELF_TEST_ENTRY_ID,
            "cg_id": cg_id,
            "technique_id": None,
            "is_baseline": True,
            "state": "pending",
            "technique_contract_hash": None,
            "implementation_attempt": 0,
            "created_at": _isoformat(_NOW),
            "updated_at": _isoformat(_NOW),
        }
    ]
    return _json_response(200, {"items": items, "next_cursor": None, "count": len(items)})


def _handle_get_cg_entry_detail(cg_id: str, entry_id: str) -> Response:
    if cg_id != SELF_TEST_CG_ID:
        return _error(404, "CG_NOT_FOUND", f"comparison group {cg_id!r} not found")
    if entry_id != SELF_TEST_ENTRY_ID:
        return _error(404, "ENTRY_NOT_FOUND", f"entry {entry_id!r} not found")
    return _json_response(200, _entry_with_runs(entry_id, cg_id))


def _handle_get_cg_evaluation(cg_id: str) -> Response:
    # Evaluation result not yet produced for the self-test CG.
    if cg_id != SELF_TEST_CG_ID:
        return _error(404, "CG_NOT_FOUND", f"comparison group {cg_id!r} not found")
    return _error(
        404,
        "ARTIFACT_NOT_FOUND",
        f"evaluation_result.json not yet produced for {cg_id!r}",
    )


def _handle_get_cg_artifacts_list(cg_id: str, request: Request) -> Response:
    if cg_id != SELF_TEST_CG_ID:
        return _error(404, "CG_NOT_FOUND", f"comparison group {cg_id!r} not found")
    params = _query_params(request)
    prefix = params.get("prefix", "")
    keys = [
        "harness/contract.json",
        "harness/manifest.json",
        "harness/status.json",
        "entries/entry_self_test_default/code/__init__.py",
    ]
    if prefix:
        keys = [k for k in keys if k.startswith(prefix)]
    return _json_response(200, {"keys": keys})


def _handle_get_cg_artifact(cg_id: str, path: str) -> Response:
    if cg_id != SELF_TEST_CG_ID:
        return _error(404, "CG_NOT_FOUND", f"comparison group {cg_id!r} not found")
    if path == "harness/status.json":
        return Response(
            status_code=200,
            content=b'{"state": "starting", "turn": 0}',
            headers={"content-type": "application/json"},
        )
    return _error(404, "ARTIFACT_NOT_FOUND", f"artifact {path!r} not found")


def _handle_get_runs(request: Request) -> Response:
    params = _query_params(request)
    state_filter = params.get("state")
    cg_id_filter = params.get("cg_id")
    entry_id_filter = params.get("entry_id")

    candidate = _run_summary(SELF_TEST_RUN_ID, SELF_TEST_CG_ID, SELF_TEST_ENTRY_ID)
    matches = True
    if state_filter and candidate["state"] != state_filter:
        matches = False
    if cg_id_filter and candidate["cg_id"] != cg_id_filter:
        matches = False
    if entry_id_filter and candidate["entry_id"] != entry_id_filter:
        matches = False
    items = [candidate] if matches else []
    return _json_response(200, {"items": items, "next_cursor": None, "count": len(items)})


def _handle_get_run_detail(run_id: str) -> Response:
    if run_id != SELF_TEST_RUN_ID:
        return _error(404, "RUN_NOT_FOUND", f"run {run_id!r} not found")
    payload = _audit_fields() | {
        "id": run_id,
        "cg_id": SELF_TEST_CG_ID,
        "entry_id": SELF_TEST_ENTRY_ID,
        "seed": 0,
        "state": "pending",
        "duration_seconds": None,
        "raw_metrics_artifact_key": None,
        "started_at": None,
        "completed_at": None,
        "failure_reason": None,
        "run_attempt": 0,
    }
    return _json_response(200, payload)


def _handle_system_version() -> Response:
    return _json_response(
        200,
        {
            "smai_cli": "0.0.0+self-test",
            "smai_core": "0.0.0+self-test",
            "smai_api_spec": "0.1.0",
            "plugins": {},
        },
    )


def _handle_system_config() -> Response:
    return _json_response(200, {"config": {"placeholder": "self-test"}})


def _handle_system_plugins() -> Response:
    return _json_response(
        200,
        {
            "llm_providers": [],
            "metadata_stores": [],
            "artifact_stores": [],
            "computes": [],
        },
    )


def _handle_system_verify() -> Response:
    ok_result = {"ok": True, "reason": "ok", "latency_ms": 1.0}
    return _json_response(
        200,
        {
            "llm_provider": ok_result,
            "metadata_store": ok_result,
            "artifact_store": ok_result,
            "compute": ok_result,
            "overall_ok": True,
        },
    )


def _handle_system_dashboard() -> Response:
    return _json_response(
        200,
        {
            "counts": {
                "proposals_in_flight": 0,
                "cgs_in_flight": 0,
                "runs_in_flight": 0,
                "papers_in_flight": 0,
            },
            "recent_activity": [],
        },
    )


def _handle_system_migrate_status() -> Response:
    return _json_response(
        200,
        {"at_head": True, "head_revision": "rev_self_test", "current": "rev_self_test"},
    )


def _handle_system_health() -> Response:
    return _json_response(200, {"status": "ok"})


def _handle_events_stream() -> Response:
    """Stream a single canned StateChangeEvent then a heartbeat then close.

    Per ``11`` §8.1: SSE events are ``id:`` / ``event:`` / ``data:``
    line groups separated by blank lines. The stream closes after the
    heartbeat to keep the test deterministic — real implementations
    keep the stream open indefinitely.
    """
    state_change_payload = {
        "kind": "proposal",
        "id": "prop_self_test",
        "from": "proposal_submitted",
        "to": "designing",
        "ts": _isoformat(_NOW),
    }
    heartbeat_payload = {
        "cycle_id": 1,
        "cycles_processed": 1,
        "ts": _isoformat(_NOW),
    }
    body = (
        f"id: 1\nevent: state_change\ndata: {json.dumps(state_change_payload)}\n\n"
        f"id: 2\nevent: worker_heartbeat\ndata: {json.dumps(heartbeat_payload)}\n\n"
    ).encode()

    return Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        stream=_AsyncByteStreamFromBytes(body),
    )


class _AsyncByteStreamFromBytes(AsyncByteStream):
    """Async byte stream that yields one chunk then closes.

    httpx's AsyncClient requires that ``Response.stream`` is an
    :class:`AsyncByteStream` even when the response was constructed
    via :class:`MockTransport`. The default ``content=`` constructor
    builds a sync stream — for SSE-style responses we need an async
    one so ``aiter_lines()`` works.
    """

    def __init__(self, body: bytes) -> None:
        self._body = body

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._body

    async def aclose(self) -> None:
        pass


# === Top-level dispatch =====================================================


def mock_handler(request: Request) -> Response:
    """Dispatch ``request`` to the per-endpoint handler.

    Routing is path-prefix-based with parameter extraction inline —
    the URL space is small enough that a hand-rolled dispatch is
    cleaner than a framework here. Unknown URLs return 404 with a
    ``ErrorEnvelope`` so the cross-cutting envelope test passes.
    """
    method = request.method
    path = request.url.path

    # Static system endpoints first.
    if method == "GET" and path == paths.SYSTEM_VERSION:
        return _handle_system_version()
    if method == "GET" and path == paths.SYSTEM_CONFIG:
        return _handle_system_config()
    if method == "GET" and path == paths.SYSTEM_PLUGINS:
        return _handle_system_plugins()
    if method == "POST" and path == paths.SYSTEM_VERIFY:
        return _handle_system_verify()
    if method == "GET" and path == paths.SYSTEM_DASHBOARD:
        return _handle_system_dashboard()
    if method == "GET" and path == paths.SYSTEM_MIGRATE_STATUS:
        return _handle_system_migrate_status()
    if method == "GET" and path == paths.SYSTEM_HEALTH:
        return _handle_system_health()

    if method == "GET" and path == paths.EVENTS:
        return _handle_events_stream()

    # Proposals.
    if method == "POST" and path == paths.PROPOSALS:
        return _handle_post_proposals(request)
    if method == "GET" and path == paths.PROPOSALS:
        return _handle_get_proposals(request)
    if path.startswith(paths.PROPOSALS + "/"):
        # /proposals/{id}, /proposals/{id}/approve, /proposals/{id}/reject
        suffix = path[len(paths.PROPOSALS) + 1 :]
        parts = suffix.split("/")
        if len(parts) == 1:
            proposal_id = parts[0]
            if method == "GET":
                return _handle_get_proposal_detail(proposal_id)
        elif len(parts) == 2:
            proposal_id, action = parts
            if method == "POST" and action in {"approve", "reject"}:
                return _handle_proposal_rpc(proposal_id, action)

    # Papers.
    if method == "POST" and path == paths.PAPERS:
        return _handle_post_papers(request)
    if method == "GET" and path == paths.PAPERS:
        return _handle_get_papers(request)
    if path.startswith(paths.PAPERS + "/"):
        suffix = path[len(paths.PAPERS) + 1 :]
        parts = suffix.split("/")
        if len(parts) == 1:
            arxiv_id = parts[0]
            if method == "GET":
                return _handle_get_paper_detail(arxiv_id)
        elif len(parts) == 2:
            arxiv_id, action = parts
            if method == "POST" and action == "promote-partial":
                return _handle_paper_promote(arxiv_id)

    # Experiments.
    if method == "POST" and path == paths.EXPERIMENTS_COMPILE:
        return _handle_post_experiments_compile(request)
    if method == "POST" and path == paths.EXPERIMENTS:
        return _handle_post_experiments(request)

    # Comparison groups.
    if method == "GET" and path == paths.COMPARISON_GROUPS:
        return _handle_get_cgs(request)
    if path.startswith(paths.COMPARISON_GROUPS + "/"):
        suffix = path[len(paths.COMPARISON_GROUPS) + 1 :]
        parts = suffix.split("/")
        # /comparison-groups/{cg_id}
        if len(parts) == 1 and method == "GET":
            return _handle_get_cg_detail(parts[0])
        # /comparison-groups/{cg_id}/{sub}
        if len(parts) == 2 and method == "GET":
            cg_id, sub = parts
            if sub == "status":
                return _handle_get_cg_status(cg_id)
            if sub == "agent-status":
                return _handle_get_cg_agent_status(cg_id)
            if sub == "entries":
                return _handle_get_cg_entries(cg_id)
            if sub == "evaluation":
                return _handle_get_cg_evaluation(cg_id)
            if sub == "artifacts":
                return _handle_get_cg_artifacts_list(cg_id, request)
        # /comparison-groups/{cg_id}/entries/{entry_id}
        if len(parts) == 3 and method == "GET" and parts[1] == "entries":
            return _handle_get_cg_entry_detail(parts[0], parts[2])
        # /comparison-groups/{cg_id}/artifacts/{path:path}
        if len(parts) >= 3 and method == "GET" and parts[1] == "artifacts":
            artifact_path = "/".join(parts[2:])
            return _handle_get_cg_artifact(parts[0], artifact_path)

    # Runs.
    if method == "GET" and path == paths.RUNS:
        return _handle_get_runs(request)
    if path.startswith(paths.RUNS + "/") and method == "GET":
        run_id = path[len(paths.RUNS) + 1 :]
        return _handle_get_run_detail(run_id)

    # Unknown URL — envelope-shaped 404.
    return _error(404, "INTERNAL_ERROR", f"unknown URL: {method} {path}")


__all__ = [
    "SELF_TEST_CG_ID",
    "SELF_TEST_ENTRY_ID",
    "SELF_TEST_RUN_ID",
    "mock_handler",
]
