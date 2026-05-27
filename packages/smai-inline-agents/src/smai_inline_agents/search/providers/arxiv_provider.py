"""arXiv academic-search provider.

Per ``planner_refactor/design_notes/search_tool_surface.md`` §6.

Uses the ``arxiv`` Python package. The package is synchronous;
production wraps ``client.results(search)`` in ``asyncio.to_thread`` to
keep the §4 fan-out ``asyncio.gather`` parallel.

``delay_seconds=3`` is arXiv's documented politeness floor; the
``arxiv`` package enforces it internally.

Note: ``arxiv.Result.download_source()`` (LaTeX tarball fetch) is NOT
exposed through ``academic_search`` -- that capability lives in the
ingestion subagent (``architectural_decisions §9``, ingestion Step 3).
Search returns metadata + PDF URL only.

The ``arxiv`` Python package ships no type stubs (as of v2.x); we
import it dynamically and ascribe :class:`typing.Any` to its return
values so pyright doesn't reject the module.
"""

from __future__ import annotations

import asyncio
from typing import Any


def _load_arxiv() -> Any:
    """Lazy + type-erased import of the ``arxiv`` package.

    Kept in a helper so the module is importable on systems without
    the package (the v1 dep declaration installs it, but tests can
    monkey-patch this seam to avoid pulling in the real package).
    """
    import arxiv as _arxiv  # type: ignore[import-untyped] # noqa: PLC0415

    return _arxiv


from smai_inline_agents.search.types import AcademicResult  # noqa: E402


class ArxivProvider:
    """v1 arXiv preprint-discovery provider.

    The ``arxiv`` package's :class:`arxiv.Client` is constructed lazily
    on the first :meth:`search` call so the import cost (and the
    package's internal HTTP session) is paid only when an arxiv-using
    config is actually exercised.
    """

    name: str = "arxiv"

    def __init__(
        self,
        *,
        delay_seconds: float = 3.0,
    ) -> None:
        self._delay_seconds = delay_seconds
        self._client: Any | None = None  # arxiv.Client when populated.

    def _get_client(self) -> Any:
        if self._client is None:
            arxiv_mod: Any = _load_arxiv()
            self._client = arxiv_mod.Client(
                page_size=100,
                delay_seconds=self._delay_seconds,
                num_retries=2,
            )
        return self._client

    async def search(self, query: str, max_results: int) -> list[AcademicResult]:
        return await asyncio.to_thread(self._search_sync, query, max_results)

    def _search_sync(self, query: str, max_results: int) -> list[AcademicResult]:
        arxiv_mod: Any = _load_arxiv()
        client: Any = self._get_client()
        search: Any = arxiv_mod.Search(
            query=query,
            max_results=max(1, min(max_results, 100)),
            sort_by=arxiv_mod.SortCriterion.Relevance,
        )
        out: list[AcademicResult] = []
        results_iter: Any = client.results(search)
        for result in results_iter:
            out.append(self._map(result))
            if len(out) >= max_results:
                break
        return out

    def _map(self, result: Any) -> AcademicResult:
        # ``result`` is an arxiv.Result with these attributes (per the
        # arxiv package docs): title, authors (list[arxiv.Result.Author]),
        # published (datetime), entry_id (URL), summary, doi, pdf_url.
        title = str(getattr(result, "title", "") or "").strip()
        author_attrs: list[Any] = list(getattr(result, "authors", []) or [])
        authors = [str(getattr(a, "name", "") or "") for a in author_attrs]
        authors = [a for a in authors if a]
        published: Any = getattr(result, "published", None)
        year = int(published.year) if published is not None else None
        entry_id = str(getattr(result, "entry_id", "") or "")
        arxiv_id = _arxiv_id_from_entry_id(entry_id)
        doi_raw: Any = getattr(result, "doi", None)
        doi = doi_raw.lower() if isinstance(doi_raw, str) and doi_raw else None
        abstract = str(getattr(result, "summary", "") or "").strip()
        pdf_url: Any = getattr(result, "pdf_url", None)
        pdf_url_str = str(pdf_url) if isinstance(pdf_url, str) and pdf_url else None
        return AcademicResult(
            title=title,
            authors=authors,
            year=year,
            arxiv_id=arxiv_id,
            doi=doi,
            abstract=abstract,
            citation_count=None,  # arXiv API does not surface citation counts.
            pdf_url=pdf_url_str,  # type: ignore[arg-type] - HttpUrl coerces from str
            source=self.name,
        )


def _arxiv_id_from_entry_id(entry_id: str) -> str | None:
    """Pull the arXiv id out of an entry URL like
    ``http://arxiv.org/abs/2504.00255v1``.

    Strips the trailing ``vN`` version suffix.
    """
    if not entry_id:
        return None
    if "/abs/" not in entry_id:
        return None
    candidate = entry_id.rsplit("/abs/", 1)[-1].split("?", 1)[0].strip("/")
    if "v" in candidate:
        head, tail = candidate.rsplit("v", 1)
        if tail.isdigit():
            return head
    return candidate or None


__all__ = ["ArxivProvider"]
