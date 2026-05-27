"""Tavily general-web search provider -- REFERENCE STUB.

Per ``planner_refactor/notes/07_settled_decisions.md`` D11
(2026-05-27): Tavily is NOT registered in the v1 default dispatch.
The original D3 settlement made Tavily the v1 default; D11 supersedes
that portion of D3 -- academic providers (OpenAlex + Semantic Scholar
+ arXiv) cover the dominant ML-research use case, dropping general
web search shrinks the prompt-injection surface (``architectural_decisions.md`` §10),
and ``fetch_url`` against a short domain allowlist covers the residual.

Operators opt back in via ``planner.search.web_provider: tavily`` in
``smai.yaml`` (see :class:`~smai_inline_agents.search.config.PlannerSearchConfig`).
When opted in, the dispatcher in
:mod:`smai_inline_agents.search.dispatch` instantiates this class.

This module deliberately:

* Does NOT import ``tavily.AsyncTavilyClient`` at module scope -- the
  ``tavily-python`` package is NOT a v1 dependency per D11. The import
  is lazy inside :meth:`TavilyProvider.search`, so the stub is
  type-checkable and importable without the package installed; only
  operator opt-in pulls the dep.
* Does NOT use :class:`pydantic_ai.common_tools.tavily.tavily_search_tool`:
  the first-party helper returns provider-native shapes, violating
  D3a's SMAI-owned-result-types requirement (per the design note §5).

Implementation sketch -- production wraps ``tavily.AsyncTavilyClient``
and maps its native response shape into
:class:`~smai_inline_agents.search.types.SearchResult`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from smai_inline_agents.search.types import SearchResult

if TYPE_CHECKING:
    # tavily-python is intentionally NOT in v1 deps per D11; the import
    # is type-checking-only so the stub stays importable without the
    # package.
    pass


class TavilyProvider:
    """REFERENCE STUB. Not registered in the v1 default dispatch.

    Operator-opt-in path: set ``planner.search.web_provider: tavily`` in
    ``smai.yaml`` AND install ``tavily-python`` AND provision
    ``TAVILY_API_KEY``. The dispatcher then constructs this class.

    Production wraps ``tavily.AsyncTavilyClient.search(...)`` and maps
    the native ``{"results": [{"title", "url", "content", "score"}, ...]}``
    response to :class:`SearchResult`.
    """

    name: str = "tavily"

    def __init__(
        self,
        api_key: str,
        *,
        search_depth: Literal["basic", "advanced"] = "basic",
    ) -> None:
        self._api_key = api_key
        self._search_depth = search_depth

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Production wraps ``tavily.AsyncTavilyClient``.

        Sketch:

        .. code-block:: python

            from tavily import AsyncTavilyClient

            client = AsyncTavilyClient(api_key=self._api_key)
            payload = await client.search(
                query=query,
                max_results=max_results,
                search_depth=self._search_depth,
            )
            return [self._map(hit) for hit in payload["results"]]
        """
        raise NotImplementedError(
            "TavilyProvider is a reference stub (D11). Operator opt-in via "
            "planner.search.web_provider: tavily in smai.yaml plus "
            "`pip install tavily-python` is required before production use."
        )

    @staticmethod
    def _map(hit: dict[str, Any]) -> SearchResult:
        """Map a Tavily-native hit to :class:`SearchResult`.

        Kept on the class so the stub documents the shape the
        production implementation needs to produce.
        """
        snippet_raw = str(hit.get("content") or "")
        snippet = snippet_raw[:500]
        score_raw = hit.get("score")
        relevance_score = float(score_raw) if isinstance(score_raw, (int, float)) else None
        return SearchResult(
            title=str(hit.get("title") or ""),
            url=str(hit.get("url") or ""),  # type: ignore[arg-type] - HttpUrl coerces
            snippet=snippet,
            full_text=hit.get("raw_content"),
            source="tavily",
            relevance_score=relevance_score,
        )


__all__ = ["TavilyProvider"]
