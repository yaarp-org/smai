"""HTTP route handlers for the ``smai serve`` dashboard.

One handler per page. Each handler:

1. Pulls the :class:`Runtime` from ``app.state.runtime`` (set by
   :func:`smai_cli.dashboard.app.build_app`).
2. Composes its data via the service surfaces on the runtime
   (:class:`StatusService` / :class:`ProposalsService` /
   :class:`PapersService`) — never reaches into :class:`MetadataStore`
   directly per the Task 3.H1 brief.
3. Renders the matching Jinja template.

Pages span the four pipeline-tracking entity kinds plus per-entity
detail views per `09-cli.md` §5.4.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from smai_cli.runtime import (
    CGNotFoundError,
    PaperNotFoundError,
    ProposalNotFoundError,
    RunNotFoundError,
    Runtime,
)


def register_routes(app: FastAPI) -> None:
    """Bind every dashboard page to ``app``.

    Routes are:

    * ``/`` — index (counts + nav)
    * ``/proposals/`` — proposals list (filterable by state)
    * ``/proposals/{proposal_id}`` — proposal detail
    * ``/cgs/`` — comparison-groups list
    * ``/cgs/{cg_id}`` — CG detail (entries, runs, artifact links)
    * ``/runs/`` — runs list
    * ``/runs/{run_id}`` — run detail
    * ``/papers/`` — papers list (partial papers visually distinct)
    * ``/papers/{arxiv_id}`` — paper detail (registered TechniqueRefs)
    """
    app.add_api_route("/", _index, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route("/proposals/", _proposals_list, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route(
        "/proposals/{proposal_id}",
        _proposal_detail,
        methods=["GET"],
        response_class=HTMLResponse,
    )
    app.add_api_route("/cgs/", _cgs_list, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route("/cgs/{cg_id}", _cg_detail, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route("/runs/", _runs_list, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route("/runs/{run_id}", _run_detail, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route("/papers/", _papers_list, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route(
        "/papers/{arxiv_id}",
        _paper_detail,
        methods=["GET"],
        response_class=HTMLResponse,
    )


# === Page handlers ===========================================================


async def _index(request: Request) -> HTMLResponse:
    runtime = _runtime_from(request)
    counts = await runtime.status.summary_counts()
    return _render(request, "index.html", {"counts": counts, "page_title": "smai dashboard"})


async def _proposals_list(request: Request, state: str | None = None) -> HTMLResponse:
    runtime = _runtime_from(request)
    proposals = await runtime.proposals.list_active()
    proposals = _filter_by_state(proposals, state)
    return _render(
        request,
        "proposals_list.html",
        {
            "proposals": proposals,
            "filter_state": state,
            "page_title": "proposals",
            "available_states": _PROPOSAL_LIST_STATES,
        },
    )


async def _proposal_detail(request: Request, proposal_id: str) -> HTMLResponse:
    runtime = _runtime_from(request)
    try:
        proposal = await runtime.proposals.get(proposal_id)
    except ProposalNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    cgs = await runtime.proposals.list_cgs(proposal_id)
    return _render(
        request,
        "proposal_detail.html",
        {
            "proposal": proposal,
            "cgs": cgs,
            "page_title": f"proposal {proposal.id}",
            "is_resting_at_human_gate": (
                proposal.state == "designed" and proposal.user_decision is None
            ),
            "is_decided_pending_registration": (
                proposal.state == "designed" and proposal.user_decision is not None
            ),
        },
    )


async def _cgs_list(request: Request, state: str | None = None) -> HTMLResponse:
    runtime = _runtime_from(request)
    cgs = await runtime.status.list_active_cgs()
    cgs = _filter_by_state(cgs, state)
    return _render(
        request,
        "cgs_list.html",
        {
            "cgs": cgs,
            "filter_state": state,
            "page_title": "comparison groups",
            "available_states": _CG_LIST_STATES,
        },
    )


async def _cg_detail(request: Request, cg_id: str) -> HTMLResponse:
    runtime = _runtime_from(request)
    try:
        cg = await runtime.status.get_cg_record(cg_id)
    except CGNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    entries = await runtime.status.list_entries_for_cg(cg_id)
    # Per-entry runs — the CG-detail page surfaces every run the
    # methodology produced. v1 entry counts × seed counts are bounded
    # so an inline materialization is fine.
    runs_per_entry = {
        entry.id: await runtime.status.list_runs_for_entry(entry.id) for entry in entries
    }
    return _render(
        request,
        "cg_detail.html",
        {
            "cg": cg,
            "entries": entries,
            "runs_per_entry": runs_per_entry,
            "page_title": f"CG {cg.id}",
        },
    )


async def _runs_list(request: Request, state: str | None = None) -> HTMLResponse:
    runtime = _runtime_from(request)
    runs = await runtime.status.list_active_runs()
    runs = _filter_by_state(runs, state)
    cg_id_filter = request.query_params.get("cg_id")
    if cg_id_filter:
        runs = [r for r in runs if r.cg_id == cg_id_filter]
    return _render(
        request,
        "runs_list.html",
        {
            "runs": runs,
            "filter_state": state,
            "filter_cg_id": cg_id_filter,
            "page_title": "runs",
            "available_states": _RUN_LIST_STATES,
        },
    )


async def _run_detail(request: Request, run_id: str) -> HTMLResponse:
    runtime = _runtime_from(request)
    try:
        run = await runtime.status.get_run_record(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _render(
        request,
        "run_detail.html",
        {
            "run": run,
            "page_title": f"run {run.id}",
        },
    )


async def _papers_list(request: Request, state: str | None = None) -> HTMLResponse:
    runtime = _runtime_from(request)
    papers = await runtime.papers.list_active()
    papers = _filter_by_state(papers, state)
    return _render(
        request,
        "papers_list.html",
        {
            "papers": papers,
            "filter_state": state,
            "page_title": "papers",
            "available_states": _PAPER_LIST_STATES,
        },
    )


async def _paper_detail(request: Request, arxiv_id: str) -> HTMLResponse:
    runtime = _runtime_from(request)
    try:
        paper = await runtime.papers.get(arxiv_id)
    except PaperNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    techniques = await runtime.papers.list_techniques(arxiv_id)
    return _render(
        request,
        "paper_detail.html",
        {
            "paper": paper,
            "techniques": techniques,
            "is_promotable": paper.state == "partial",
            "page_title": f"paper {paper.arxiv_id}",
        },
    )


# === Helpers =================================================================


def _runtime_from(request: Request) -> Runtime:
    """Pull the :class:`Runtime` off ``app.state``.

    :func:`smai_cli.dashboard.app.build_app` writes it on construction;
    handlers read it on each request. Surfaces a clear error when the
    runtime isn't bound (mis-configured caller).

    The return type is :class:`Runtime`; tests that supply a stub with
    the same service-shape (e.g., a duck-typed object exposing
    ``status``/``proposals``/``papers``) bypass the strict isinstance
    check by binding directly to ``app.state.runtime``.
    """
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError(
            "dashboard app has no runtime bound on app.state — was it "
            "constructed via smai_cli.dashboard.build_app(runtime)?"
        )
    return cast("Runtime", runtime)


def _render(request: Request, template_name: str, context: dict[str, Any]) -> HTMLResponse:
    """Render ``template_name`` with the per-route context.

    The base template carries the navigation chrome; per-page contexts
    carry only the page-specific data.
    """
    templates = request.app.state.templates
    if not isinstance(templates, Jinja2Templates):
        raise RuntimeError("dashboard app has no Jinja2Templates bound on app.state")
    # Modern Starlette/FastAPI: pass ``request`` positionally first per
    # the deprecation of the legacy ``(name, context)`` signature.
    return templates.TemplateResponse(request, template_name, context)


def _filter_by_state(items: Sequence[Any], state: str | None) -> list[Any]:
    """Apply the optional ``?state=`` query-param filter.

    Pages that accept a state filter call this with the parsed query
    param; ``None`` is the no-filter case.
    """
    if state is None or state == "":
        return list(items)
    return [item for item in items if getattr(item, "state", None) == state]


# Filter dropdowns. We hard-code the per-entity state catalogs from
# the ``*State`` Literals in :mod:`smai_orchestrator.entities.tracking`
# so the templates can iterate them. Terminal states are included for
# the CGs/runs/papers list dropdowns even though the active list won't
# return them — selecting a terminal state filters down to zero, which
# is the correct surface ("none of the active items are in this state").
_PROPOSAL_LIST_STATES: tuple[str, ...] = ("proposal_submitted", "designing", "designed")
_CG_LIST_STATES: tuple[str, ...] = (
    "draft",
    "implementing",
    "implemented",
    "running",
    "evaluating",
)
_RUN_LIST_STATES: tuple[str, ...] = ("pending", "submitted", "running")
_PAPER_LIST_STATES: tuple[str, ...] = (
    "submitted",
    "fetching",
    "screening",
    "planning",
    "partial",
)


__all__ = ["register_routes"]
