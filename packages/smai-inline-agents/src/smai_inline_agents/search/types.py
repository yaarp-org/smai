"""SMAI-owned Pydantic result models for the planner search-tool surface.

Per ``planner_refactor/design_notes/search_tool_surface.md`` §2.

These result types are deliberately substrate-independent: provider
implementations (Tavily / OpenAlex / Semantic Scholar / arXiv / Jina
Reader / direct httpx) map their native response shapes into these.
The PydanticAI tool wrapping uses these shapes; the agent's prompt and
conversation trace never see provider-native types.

D11 amendment (2026-05-27): v1 does NOT register a default ``web_search``
provider. :class:`SearchResult` and the ``web_search`` surface still
exist so operators can opt in via ``planner.search.web_provider`` later
without re-shaping the contract.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

# =============================================================================
# Result types -- SMAI-owned, never re-exports of provider-native shapes (D3a).
# =============================================================================


class SearchResult(BaseModel):
    """One general-web search hit. Emitted by ``web_search``.

    v1 has no default web_search provider (D11); this model exists so
    the surface stays callable when an operator opts a provider in.
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    url: HttpUrl
    snippet: str = Field(
        description=(
            "Provider-supplied summary, capped at 500 chars by the type-mapper "
            "if the provider returns more."
        ),
    )
    full_text: str | None = Field(
        default=None,
        description=(
            "Cleaned full-page text when the provider returns it; None when "
            "not requested or not available."
        ),
    )
    source: str = Field(
        description=(
            "Provider id for audit: 'tavily' | 'brave' | 'exa' | 'serper' | "
            "... Lower-case, matches the dispatch.py config Literal."
        ),
    )
    relevance_score: float | None = Field(
        default=None,
        description=(
            "Provider-supplied 0.0-1.0 score if any; pass-through, never "
            "synthesized by the type-mapper."
        ),
    )


class AcademicResult(BaseModel):
    """One academic paper hit. Emitted by ``academic_search``."""

    model_config = ConfigDict(extra="forbid")

    title: str
    authors: list[str] = Field(
        default_factory=list,
        description=(
            "Author display strings in paper order. Empty list (not None) "
            "when the provider returns no author info."
        ),
    )
    year: int | None = None
    arxiv_id: str | None = Field(
        default=None,
        description=(
            "arXiv ID without version suffix (e.g. '2504.00255'). Populated "
            "whenever any provider returns one; the fan-out dedupe keys on "
            "this when present."
        ),
    )
    doi: str | None = Field(
        default=None,
        description=(
            "DOI without any URL prefix (e.g. '10.1145/3589334.3645374'). "
            "Fan-out dedupe keys on this when arxiv_id is absent."
        ),
    )
    abstract: str = Field(
        default="",
        description=(
            "Empty string (not None) when the provider returns no abstract; "
            "absent-abstract is a frequent OpenAlex case."
        ),
    )
    citation_count: int | None = None
    pdf_url: HttpUrl | None = Field(
        default=None,
        description=(
            "Direct PDF URL when the provider exposes one. arXiv and "
            "OpenAlex commonly do; Semantic Scholar sometimes does."
        ),
    )
    source: str = Field(
        description=(
            "Provider id: 'openalex' | 'semantic_scholar' | 'arxiv'. Merged "
            "results carry the highest-confidence source (arxiv > openalex "
            "> semantic_scholar)."
        ),
    )


class FetchedDocument(BaseModel):
    """One URL fetch result. Emitted by ``fetch_url``."""

    model_config = ConfigDict(extra="forbid")

    url: HttpUrl
    title: str | None = Field(
        default=None,
        description=(
            "Extracted from <title> / first H1 / PDF metadata when available; "
            "None when extraction failed."
        ),
    )
    content_type: Literal["html", "pdf", "markdown", "latex", "json", "other"] = Field(
        description=(
            "Determined from Content-Type header for direct fetches, or from "
            "Jina Reader's response metadata for the fallback path."
        ),
    )
    text: str = Field(description="Extracted/cleaned text.")
    raw: str | None = Field(
        default=None,
        description=(
            "Original content when it fits the retention cap (default 100KB); None when truncated."
        ),
    )
    source: Literal["direct", "jina_reader"] = Field(
        description=(
            "Fetch path id: 'direct' (httpx + markdownify) or 'jina_reader' (reader fallback)."
        ),
    )


# =============================================================================
# Egress log event schema.
# =============================================================================


class EgressEvent(BaseModel):
    """Schema for individual JSONL records in the egress audit log.

    Per ``search_tool_surface.md`` §8. Every search query and URL fetch
    appends one event to
    ``proposals/{proposal_id}/planner/search_log.jsonl`` (Step 1 spec
    path; the design note's prose uses ``egress_log.jsonl``, the
    implementation_plan §1 task list specifies ``search_log.jsonl`` --
    Step 1 follows the plan).
    """

    model_config = ConfigDict(extra="forbid")

    ts: datetime
    type: Literal["web_search", "academic_search", "fetch_url"]
    session_id: str
    provider: str
    query: str | None = None
    url: str | None = None
    max_results: int | None = None
    result_count: int | None = None
    status: int | None = None
    content_type: str | None = None
    duration_ms: int
    error: str | None = None
    providers_succeeded: list[str] | None = None
    providers_failed: list[str] | None = None
    fallback_reason: str | None = None


__all__ = [
    "AcademicResult",
    "EgressEvent",
    "FetchedDocument",
    "SearchResult",
]
