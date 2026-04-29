"""Shared fixtures for Task 3.E2 spec tests (paper-ingestion pipeline).

Per the workspace's pytest ``--import-mode=importlib`` filename-hygiene
rule (cross-package fixture filenames must not collide with siblings):
``_e2_*`` keeps these distinct from ``_e1_*`` (proposal — Task 3.E1),
``_e3_*`` (run-record — Task 3.E3), and ``_specs_fakes`` (Task 2.C4).

Builders shipped here:

* :class:`InProcessFakeFetcher` — :class:`PaperFetcher` that returns
  canned :class:`FetchedPaper` content. Used by spec tests + the
  integration round-trip test to keep the fetch stage offline.
* :func:`make_paper_record` — :class:`PaperRecord` builder.
* :func:`build_finalized_paper_buffer_payload` — JSON-shaped buffer
  matching :class:`smai_agents.agents.planner.PlannerBuffer` on
  ``finalize_paper_techniques`` success. Used to pre-stage a
  ``planning → registered`` advance without driving the planner loop.
* :func:`make_paper_planner_responses` — canned LLM responses driving
  the paper-ingestion-variant planner through a happy-path finalize:
  ``draft_create_technique`` then ``finalize_paper_techniques``.
* :func:`make_screener_response` — canned LLM response emitting a
  :class:`ScreenResult` tool_use.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from smai_core.plugins import (
    ModelResponse,
    NormalizedMessage,
    StopReason,
    TextContent,
    TokenUsage,
    ToolUseContent,
)
from smai_orchestrator.entities.tracking import PaperRecord
from smai_orchestrator.specs.paper_ingestion import FetchedPaper, PaperFetcher


def make_paper_record(
    *,
    arxiv_id: str,
    state: str = "submitted",
    title: str | None = None,
) -> PaperRecord:
    now = datetime.now(UTC)
    return PaperRecord(
        arxiv_id=arxiv_id,
        title=title,
        state=state,  # type: ignore[arg-type]
        version=0,
        created_at=now,
        updated_at=now,
    )


class InProcessFakeFetcher(PaperFetcher):
    """:class:`PaperFetcher` that returns canned content per arxiv id.

    The default constructor ships a single canned paper; tests can pass
    a mapping for multi-paper fixtures (e.g., source-paper enrichment
    tests).
    """

    def __init__(
        self,
        *,
        default_text: str = (
            "Default fake paper text — empirical comparison of method A vs. method B."
        ),
        default_title: str | None = "Fake Paper Title",
        default_abstract: str | None = "Fake abstract — empirical comparative claim.",
        per_paper: dict[str, FetchedPaper] | None = None,
    ) -> None:
        self._default = FetchedPaper(
            title=default_title,
            abstract=default_abstract,
            authors=["Fake Author"],
            paper_text=default_text,
            figures=[],
            expanded_tex=None,
        )
        self._per_paper = dict(per_paper or {})
        self.fetch_log: list[str] = []

    async def fetch(self, arxiv_id: str) -> FetchedPaper:
        self.fetch_log.append(arxiv_id)
        return self._per_paper.get(arxiv_id, self._default)


def _model_response(
    *,
    tool_uses: list[tuple[str, str, dict[str, Any]]] | None = None,
    text: str | None = None,
    stop_reason: StopReason = "tool_use",
) -> ModelResponse:
    from smai_core.plugins import NormalizedContent  # noqa: PLC0415

    content: list[NormalizedContent] = []
    if text is not None:
        content.append(TextContent(text=text))
    if tool_uses is not None:
        for tu_id, tu_name, tu_input in tool_uses:
            content.append(ToolUseContent(id=tu_id, name=tu_name, input=dict(tu_input)))
    return ModelResponse(
        message=NormalizedMessage(role="assistant", content=content),
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )


def make_screener_response(
    *,
    decision: str = "accept",
    rejection_reason: str | None = None,
    summary: str = "Empirical comparison paper; in-scope.",
    tool_name: str = "submit_screening",
) -> ModelResponse:
    payload: dict[str, Any] = {"decision": decision, "summary": summary}
    if rejection_reason is not None:
        payload["rejection_reason"] = rejection_reason
    return _model_response(tool_uses=[("call_1", tool_name, payload)])


def make_enricher_response(
    *,
    implementability: str = "high",
    method_extraction: str = "## Algorithm\n1. Apply standard component.\n",
    refined_description: str | None = None,
    blocked_reason: str | None = None,
    tool_name: str = "submit_enrichment",
) -> ModelResponse:
    payload: dict[str, Any] = {
        "implementability": implementability,
        "method_extraction": method_extraction,
    }
    if refined_description is not None:
        payload["refined_description"] = refined_description
    if blocked_reason is not None:
        payload["blocked_reason"] = blocked_reason
    return _model_response(tool_uses=[("call_1", tool_name, payload)])


def make_paper_planner_responses(
    *,
    arxiv_id: str,
    contribution_techniques: Sequence[dict[str, Any]] | None = None,
) -> list[ModelResponse]:
    """Canned LLM responses driving the paper-ingestion-variant planner
    through a happy-path finalize.

    For each contribution technique: one ``draft_create_technique`` call
    seeded with a :class:`PaperFidelityAnchor` matching ``arxiv_id``.
    Then one ``finalize_paper_techniques`` call (no symbol arg —
    finalizes the whole buffer per `08` §5.4) and one ``finish``.
    """
    techniques = (
        list(contribution_techniques)
        if contribution_techniques is not None
        else [
            {
                "symbolic_name": f"{arxiv_id}-tech-contrib",
                "name": "Contribution Technique",
                "description": "Primary contribution technique extracted from the paper.",
                "category": "augmentation",
                "compatible_factor_types": ["additive"],
                "standard": False,
                "fidelity_anchor": {
                    "kind": "paper",
                    "arxiv_id": arxiv_id,
                    "doi": f"arxiv:{arxiv_id}",
                },
            }
        ]
    )

    responses: list[ModelResponse] = []
    for idx, tech in enumerate(techniques):
        responses.append(
            _model_response(
                tool_uses=[
                    (
                        f"tu-create-{idx}",
                        "draft_create_technique",
                        dict(tech),
                    ),
                ],
            )
        )

    # ``finalize_paper_techniques`` takes no input fields per
    # FinalizePaperTechniquesInput. Send an empty dict.
    responses.append(
        _model_response(
            tool_uses=[("tu-finalize", "finalize_paper_techniques", {})],
        )
    )
    responses.append(
        _model_response(
            tool_uses=[
                (
                    "tu-finish",
                    "finish",
                    {"success": True, "summary": "paper ingestion complete"},
                )
            ],
        )
    )
    return responses


def build_finalized_paper_buffer_payload(
    *,
    arxiv_id: str,
    techniques: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a JSON-shaped finalized :class:`PlannerBuffer` payload for
    the paper-ingestion variant.

    Used by spec tests that pre-stage the buffer artifact at
    ``papers/{arxiv_id}/draft_techniques.json`` to drive the
    ``planning → registered`` gate without running the planner loop.
    """
    techniques_dict: dict[str, dict[str, Any]] = {}
    techs = (
        list(techniques)
        if techniques is not None
        else [
            {
                "symbolic_name": f"{arxiv_id}-tech-contrib",
                "name": "Contribution Technique",
                "description": "Primary contribution technique extracted from the paper.",
                "category": "augmentation",
                "compatible_factor_types": ["additive"],
                "standard": False,
                "fidelity_anchor": {
                    "kind": "paper",
                    "arxiv_id": arxiv_id,
                    "doi": f"arxiv:{arxiv_id}",
                },
                "affects_extension_points": [],
                "implies_controlled": [],
            }
        ]
    )
    for tech in techs:
        techniques_dict[tech["symbolic_name"]] = dict(tech)
    return {
        "proposal_id": None,
        "paper_arxiv_id": arxiv_id,
        "techniques": techniques_dict,
        "comparison_groups": [],
        "classification": None,
        "follow_ups": [],
        "finalized": True,
    }


__all__ = [
    "InProcessFakeFetcher",
    "build_finalized_paper_buffer_payload",
    "make_enricher_response",
    "make_paper_planner_responses",
    "make_paper_record",
    "make_screener_response",
]
