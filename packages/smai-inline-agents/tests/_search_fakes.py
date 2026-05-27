"""Test fakes for the search-tool surface (Step 1 of planner_refactor).

Per CLAUDE.md "Testing patterns": per-task fixture filename hygiene
``_<task>_<purpose>.py``. Sibling-plugin fakes under ``_fakes.py``
collide under ``--import-mode=importlib``; this file's ``_search_``
prefix dodges that.
"""

from __future__ import annotations

from typing import Any

from smai_inline_agents.search.protocols import (
    AcademicSearchProvider,
    UrlFetcher,
    WebSearchProvider,
)
from smai_inline_agents.search.types import (
    AcademicResult,
    FetchedDocument,
    SearchResult,
)


class FakeArtifactStore:
    """Minimal in-memory ArtifactStore-shaped fake for egress-log roundtrip
    tests.

    Records the bytes / content-types passed to :meth:`put` so the test
    can assert the egress-log JSONL blob was written as expected.
    """

    def __init__(self) -> None:
        self.puts: list[tuple[str, bytes, str | None]] = []

    async def put(self, key: str, data: bytes, content_type: str | None = None) -> None:
        self.puts.append((key, data, content_type))


class FakeWebProvider:
    """Records calls + returns a fixed result list."""

    name: str = "fake_web"

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        self.calls.append((query, max_results))
        return list(self.results)


class FlakyWebProvider:
    """Raises a configured exception on :meth:`search`. Used to exercise the
    web-search degrade-to-empty path."""

    name: str = "flaky_web"

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        self.calls.append((query, max_results))
        raise self._exc


class FakeAcademicProvider:
    """Records calls + returns a fixed result list.

    ``name`` is configurable so the fan-out dedupe + source-priority
    tests can simulate multiple providers (e.g. arxiv + openalex +
    semantic_scholar) and assert the merge keeps the higher-priority
    record.
    """

    def __init__(self, name: str, results: list[AcademicResult] | None = None) -> None:
        self.name = name
        self.results = results or []
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, max_results: int) -> list[AcademicResult]:
        self.calls.append((query, max_results))
        return list(self.results)


class FlakyAcademicProvider:
    """Raises a configured exception on :meth:`search`. Used for fan-out
    failure-tolerance tests."""

    def __init__(self, name: str, exc: Exception) -> None:
        self.name = name
        self._exc = exc
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, max_results: int) -> list[AcademicResult]:
        self.calls.append((query, max_results))
        raise self._exc


class FakeUrlFetcher:
    """Records URL fetches + returns a fixed document. Used to assert that
    the dispatcher's allowlist check fires BEFORE the fetcher is invoked."""

    name: str = "fake_fetcher"

    def __init__(self, document: FetchedDocument | None = None) -> None:
        self.document = document
        self.calls: list[str] = []

    async def fetch(self, url: str) -> FetchedDocument:
        self.calls.append(url)
        if self.document is None:
            raise RuntimeError(f"FakeUrlFetcher has no document configured for {url!r}")
        return self.document


def make_search_result(
    *,
    title: str = "Example",
    url: str = "https://example.com/article",
    snippet: str = "Example snippet",
    source: str = "fake_web",
    relevance_score: float | None = None,
    full_text: str | None = None,
) -> SearchResult:
    return SearchResult(
        title=title,
        url=url,  # type: ignore[arg-type] - HttpUrl coerces
        snippet=snippet,
        full_text=full_text,
        source=source,
        relevance_score=relevance_score,
    )


def make_academic_result(
    *,
    title: str = "Example Paper",
    arxiv_id: str | None = None,
    doi: str | None = None,
    source: str = "fake_academic",
    abstract: str = "",
    authors: list[str] | None = None,
    year: int | None = None,
    pdf_url: str | None = None,
    citation_count: int | None = None,
) -> AcademicResult:
    return AcademicResult(
        title=title,
        authors=authors or [],
        year=year,
        arxiv_id=arxiv_id,
        doi=doi,
        abstract=abstract,
        citation_count=citation_count,
        pdf_url=pdf_url,  # type: ignore[arg-type]
        source=source,
    )


def make_fetched_document(
    *,
    url: str = "https://arxiv.org/abs/1607.00133",
    title: str | None = "Example",
    content_type: Any = "html",
    text: str = "example body",
    raw: str | None = None,
    source: Any = "direct",
) -> FetchedDocument:
    return FetchedDocument(
        url=url,  # type: ignore[arg-type]
        title=title,
        content_type=content_type,
        text=text,
        raw=raw,
        source=source,
    )


# Static type-check that the fakes satisfy the Protocols at construction.
# Pure typing exercise; if the Protocols drift, mypy/pyright catches it.
_assert_web: WebSearchProvider = FakeWebProvider()
_assert_academic: AcademicSearchProvider = FakeAcademicProvider("x")
_assert_fetch: UrlFetcher = FakeUrlFetcher()
