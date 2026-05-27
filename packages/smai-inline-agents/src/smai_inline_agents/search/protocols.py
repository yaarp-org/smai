"""Provider Protocols for the planner search-tool surface.

Per ``planner_refactor/design_notes/search_tool_surface.md`` §4.

Three Protocols, one per generic tool capability:

* :class:`WebSearchProvider` -- implements ``web_search``.
* :class:`AcademicSearchProvider` -- one source for ``academic_search``
  (multiple instances run in parallel under one generic tool, fan-out
  + dedupe per §4 of the design note).
* :class:`UrlFetcher` -- implements ``fetch_url`` (Jina Reader fallback
  is wrapped inside the direct-fetch implementation, not a separate
  Protocol instance).

These are runtime-checkable Protocols (``@runtime_checkable``) so the
fan-out dispatcher can verify provider instances at construction time
in unit tests.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from smai_inline_agents.search.types import (
    AcademicResult,
    FetchedDocument,
    SearchResult,
)


@runtime_checkable
class WebSearchProvider(Protocol):
    """Implements the generic ``web_search`` capability.

    Per D11 (2026-05-27): v1 registers no default provider. The
    Protocol stays so operators can opt in to Tavily / Brave / Exa /
    Serper via ``planner.search.web_provider: <name>`` in ``smai.yaml``.
    """

    name: str

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Run one general-web search; map provider response to
        :class:`SearchResult`.
        """
        ...


@runtime_checkable
class AcademicSearchProvider(Protocol):
    """Implements one source for ``academic_search``.

    Multiple instances run in parallel under one generic tool; the
    fan-out dispatcher in ``dispatch.py`` dedupes by
    ``arxiv_id > doi > normalized_title``.
    """

    name: str

    async def search(self, query: str, max_results: int) -> list[AcademicResult]:
        """Run one academic-database search; map provider response to
        :class:`AcademicResult`.
        """
        ...


@runtime_checkable
class UrlFetcher(Protocol):
    """Implements the generic ``fetch_url`` capability.

    Jina Reader fallback is wrapped INSIDE the direct-fetch
    implementation (see
    :class:`~smai_inline_agents.search.providers.direct_fetch.DirectThenJinaFetcher`),
    not a separate Protocol instance.
    """

    name: str

    async def fetch(self, url: str) -> FetchedDocument:
        """Fetch the document at ``url`` and return cleaned text + metadata.

        The dispatcher enforces the D11 domain allowlist BEFORE this
        method is invoked; implementations may assume the host has
        already been validated.
        """
        ...


__all__ = [
    "AcademicSearchProvider",
    "UrlFetcher",
    "WebSearchProvider",
]
