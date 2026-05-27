"""Semantic Scholar academic-search provider.

Per ``planner_refactor/design_notes/search_tool_surface.md`` §6.

* Endpoint: ``https://api.semanticscholar.org/graph/v1/paper/search``.
* Auth: optional API key (env ``SEMANTIC_SCHOLAR_API_KEY``); without
  the key the public pool runs at ~100 req / 5 min, with a key 1
  req/sec. Rate-limit enforcement lives in
  :class:`~smai_inline_agents.search.budgets.BudgetTracker`; the
  provider here just sends the header when a key is configured.
* Fields requested:
  ``title,authors,year,abstract,externalIds,citationCount,openAccessPdf``.
* External ids: ``externalIds.ArXiv`` and ``externalIds.DOI`` populate
  the fan-out dedupe keys.
"""

from __future__ import annotations

from typing import Any, cast

import httpx

from smai_inline_agents.search.types import AcademicResult


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce ``value`` to a ``dict[str, Any]`` or return an empty dict.

    Pyright-friendly idiom for traversing untyped JSON without a
    flood of ``cast`` boilerplate.
    """
    if isinstance(value, dict):
        return cast("dict[str, Any]", value)
    return {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return cast("list[Any]", value)
    return []


_S2_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_FIELDS = "title,authors,year,abstract,externalIds,citationCount,openAccessPdf"


class SemanticScholarProvider:
    """v1 Semantic Scholar academic provider."""

    name: str = "semantic_scholar"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def search(self, query: str, max_results: int) -> list[AcademicResult]:
        params: dict[str, str | int] = {
            "query": query,
            "limit": max(1, min(max_results, 100)),
            "fields": _S2_FIELDS,
        }
        headers: dict[str, str] = {}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        response = await self._client.get(_S2_BASE, params=params, headers=headers)
        response.raise_for_status()
        payload = cast("dict[str, Any]", response.json())
        data = _as_list(payload.get("data"))
        out: list[AcademicResult] = []
        for p in data[:max_results]:
            if isinstance(p, dict):
                out.append(self._map(cast("dict[str, Any]", p)))
        return out

    def _map(self, paper: dict[str, Any]) -> AcademicResult:
        title = str(paper.get("title") or "")
        authors_raw = _as_list(paper.get("authors"))
        authors: list[str] = []
        for a in authors_raw:
            inner = _as_dict(a) if isinstance(a, dict) else {}
            name = inner.get("name")
            if isinstance(name, str) and name:
                authors.append(name)
        year_raw = paper.get("year")
        year = int(year_raw) if isinstance(year_raw, int) else None
        ext_ids = _as_dict(paper.get("externalIds"))
        arxiv_raw = ext_ids.get("ArXiv")
        arxiv_id = _trim_arxiv_version(arxiv_raw) if isinstance(arxiv_raw, str) else None
        doi_raw = ext_ids.get("DOI")
        doi = doi_raw.lower() if isinstance(doi_raw, str) and doi_raw else None
        abstract = str(paper.get("abstract") or "")
        cites_raw = paper.get("citationCount")
        citation_count = int(cites_raw) if isinstance(cites_raw, int) else None
        pdf_url = _extract_pdf_url(paper)
        return AcademicResult(
            title=title,
            authors=authors,
            year=year,
            arxiv_id=arxiv_id,
            doi=doi,
            abstract=abstract,
            citation_count=citation_count,
            pdf_url=pdf_url,  # type: ignore[arg-type] - HttpUrl coerces from str
            source=self.name,
        )


def _trim_arxiv_version(raw: str) -> str | None:
    candidate = raw.strip().strip("/")
    if not candidate:
        return None
    if "v" in candidate:
        head, tail = candidate.rsplit("v", 1)
        if tail.isdigit():
            return head
    return candidate


def _extract_pdf_url(paper: dict[str, Any]) -> str | None:
    oa = _as_dict(paper.get("openAccessPdf"))
    url = oa.get("url")
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        return url
    return None


__all__ = ["SemanticScholarProvider"]
