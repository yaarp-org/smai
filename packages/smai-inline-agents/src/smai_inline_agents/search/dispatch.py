"""Search-tool dispatcher -- the one-file swap target (D3a).

Per ``planner_refactor/design_notes/search_tool_surface.md`` §1, §4, §5.

This module:

* Maps a :class:`PlannerSearchConfig` onto concrete provider instances
  via :func:`build_search_deps`.
* Provides the three generic typed entry points
  (:func:`web_search`, :func:`academic_search`, :func:`fetch_url`) the
  literature-review agent's PydanticAI tool decorators wrap.
* Enforces the D11 fetch allowlist on :func:`fetch_url`.
* Enforces per-session budgets + per-provider rate limits via
  :class:`~smai_inline_agents.search.budgets.BudgetTracker`.
* Runs the academic_search fan-out: parallel ``asyncio.gather`` across
  registered providers, dedupe by ``arxiv_id > doi > normalized_title``,
  merge with source-priority ``arxiv > openalex > semantic_scholar``,
  tolerate 1-2 provider failures.
* Logs every search query + URL fetch through the
  :class:`~smai_inline_agents.search.egress_log.EgressLogger`.

Adding / replacing a provider is editing one ``match`` arm in
:func:`build_search_deps`; the generic surfaces below, the Pydantic
result models, and the Protocols all stay verbatim.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from urllib.parse import urlparse

from smai_inline_agents.search.budgets import BudgetTracker
from smai_inline_agents.search.config import PlannerSearchConfig
from smai_inline_agents.search.deps import SearchDeps
from smai_inline_agents.search.egress_log import ArtifactStoreLike, EgressLogger
from smai_inline_agents.search.errors import (
    AcademicSearchAllProvidersFailed,
    NotConfiguredError,
    UrlFetchFailed,
    UrlNotAllowlisted,
)
from smai_inline_agents.search.protocols import (
    AcademicSearchProvider,
    UrlFetcher,
    WebSearchProvider,
)
from smai_inline_agents.search.providers.arxiv_provider import ArxivProvider
from smai_inline_agents.search.providers.direct_fetch import DirectThenJinaFetcher
from smai_inline_agents.search.providers.jina import JinaOnlyFetcher
from smai_inline_agents.search.providers.openalex import OpenAlexProvider
from smai_inline_agents.search.providers.semantic_scholar import SemanticScholarProvider
from smai_inline_agents.search.providers.tavily import TavilyProvider
from smai_inline_agents.search.types import AcademicResult, FetchedDocument, SearchResult

# =============================================================================
# build_search_deps -- the one-file swap target.
# =============================================================================


def build_search_deps(
    config: PlannerSearchConfig,
    *,
    session_id: str,
    proposal_id: str | None = None,
    artifact_store: ArtifactStoreLike | None = None,
) -> SearchDeps:
    """Map a :class:`PlannerSearchConfig` onto concrete provider instances.

    THIS FUNCTION IS THE ONE-FILE SWAP TARGET (D3a). Swapping providers
    is editing match arms here; the result types, the Protocols, the
    tool wrappers, the prompts, and the agent code all stay verbatim.

    Per D11: when ``config.web_provider`` is ``None`` (the v1 default),
    no web-search provider is constructed and the
    :attr:`SearchDeps.web_provider` field stays ``None``;
    :func:`web_search` then raises
    :class:`~smai_inline_agents.search.errors.NotConfiguredError`.
    """
    # --- Web provider ---------------------------------------------------------
    web: WebSearchProvider | None
    match config.web_provider:
        case None:
            # D11: no v1 default. web_search() raises NotConfiguredError.
            web = None
        case "tavily":
            web = TavilyProvider(
                api_key=os.environ[config.providers.tavily.api_key_env],
                search_depth=config.providers.tavily.search_depth,
            )
        case "brave":
            raise NotImplementedError(
                "Brave provider not implemented in v1; see "
                "planner_refactor/notes/07_settled_decisions.md D11."
            )
        case "exa":
            raise NotImplementedError(
                "Exa provider not implemented in v1; see "
                "planner_refactor/notes/07_settled_decisions.md D11."
            )
        case "serper":
            raise NotImplementedError(
                "Serper provider not implemented in v1; see "
                "planner_refactor/notes/07_settled_decisions.md D11."
            )

    # --- Academic providers (fan-out list) ------------------------------------
    academic: list[AcademicSearchProvider] = []
    for name in config.academic_providers:
        match name:
            case "openalex":
                academic.append(
                    OpenAlexProvider(
                        polite_email=os.environ.get(
                            config.providers.openalex.polite_email_env,
                        ),
                    ),
                )
            case "semantic_scholar":
                academic.append(
                    SemanticScholarProvider(
                        api_key=os.environ.get(
                            config.providers.semantic_scholar.api_key_env,
                        ),
                    ),
                )
            case "arxiv":
                academic.append(
                    ArxivProvider(delay_seconds=config.providers.arxiv.delay_seconds),
                )

    # --- URL fetcher ----------------------------------------------------------
    fetcher: UrlFetcher
    jina_api_key = os.environ.get(config.providers.jina_reader.api_key_env)
    match config.url_fetch:
        case "direct":
            fetcher = DirectThenJinaFetcher(
                enable_jina_fallback=True,
                jina_api_key=jina_api_key,
            )
        case "direct_only":
            fetcher = DirectThenJinaFetcher(
                enable_jina_fallback=False,
                jina_api_key=jina_api_key,
            )
        case "jina_only":
            fetcher = JinaOnlyFetcher(api_key=jina_api_key)

    return SearchDeps(
        academic_providers=academic,
        url_fetcher=fetcher,
        egress_log=EgressLogger.open(
            session_id,
            proposal_id=proposal_id,
            artifact_store=artifact_store,
            enabled=config.log_queries,
        ),
        budgets=BudgetTracker.from_config(config.budgets),
        session_id=session_id,
        fetch_allowlist=tuple(config.fetch_allowlist),
        web_provider=web,
    )


# =============================================================================
# Generic typed tool entry points -- the PydanticAI @agent.tool wrappers
# eventually call these. Each applies budget + egress around the underlying
# provider call.
# =============================================================================


async def web_search(
    query: str,
    *,
    max_results: int = 5,
    deps: SearchDeps,
) -> list[SearchResult]:
    """Generic web-search tool.

    Per D11: when no web provider is registered (the v1 default),
    raises :class:`NotConfiguredError`. Operators opt in via
    ``planner.search.web_provider: <name>`` in ``smai.yaml``.

    Per-call failures degrade-to-empty (the lit-review prompt handles
    this gracefully); the underlying provider error is logged to the
    egress log but the agent observes ``[]``.
    """
    if deps.web_provider is None:
        raise NotConfiguredError(
            "web_search has no provider configured (D11 default); set "
            "planner.search.web_provider in smai.yaml to opt in",
        )
    await deps.budgets.check_and_increment("web_search", deps.web_provider.name)
    started = _now_utc()
    try:
        results = await deps.web_provider.search(query, max_results)
    except Exception as exc:  # noqa: BLE001 - degrade-to-empty per §9
        await deps.egress_log.log(
            type="web_search",
            provider=deps.web_provider.name,
            query=query,
            max_results=max_results,
            duration_ms=_ms_since(started),
            error=repr(exc),
        )
        return []
    await deps.egress_log.log(
        type="web_search",
        provider=deps.web_provider.name,
        query=query,
        max_results=max_results,
        result_count=len(results),
        duration_ms=_ms_since(started),
    )
    return results


async def academic_search(
    query: str,
    *,
    max_results: int = 5,
    deps: SearchDeps,
) -> list[AcademicResult]:
    """Generic academic-search tool. Fans out across
    :attr:`SearchDeps.academic_providers` in parallel; deduplicates by
    ``arxiv_id > doi > normalized_title``; merges with source-priority
    ``arxiv > openalex > semantic_scholar``.

    Tolerates 1-2 provider failures. Raises
    :class:`AcademicSearchAllProvidersFailed` only when ALL registered
    providers fail.
    """
    import asyncio  # noqa: PLC0415 - lazy: only async fan-out users pay this cost

    await deps.budgets.check_and_increment("academic_search", "fan_out")
    started = _now_utc()
    if not deps.academic_providers:
        await deps.egress_log.log(
            type="academic_search",
            provider="fan_out",
            query=query,
            max_results=max_results,
            result_count=0,
            duration_ms=_ms_since(started),
            providers_succeeded=[],
            providers_failed=[],
        )
        raise AcademicSearchAllProvidersFailed(
            failures=[],
        )
    tasks = [p.search(query, max_results) for p in deps.academic_providers]
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    by_key: dict[str, AcademicResult] = {}
    succeeded: list[str] = []
    failed: list[str] = []
    failures: list[tuple[str, BaseException]] = []
    for provider, result in zip(deps.academic_providers, raw, strict=True):
        if isinstance(result, BaseException):
            failed.append(provider.name)
            failures.append((provider.name, result))
            continue
        succeeded.append(provider.name)
        for hit in result:
            key = _dedupe_key(hit)
            existing = by_key.get(key)
            if existing is None or _source_priority(hit.source) > _source_priority(
                existing.source,
            ):
                by_key[key] = hit

    await deps.egress_log.log(
        type="academic_search",
        provider="fan_out",
        query=query,
        max_results=max_results,
        result_count=len(by_key),
        duration_ms=_ms_since(started),
        providers_succeeded=succeeded,
        providers_failed=failed,
    )

    if not succeeded:
        raise AcademicSearchAllProvidersFailed(failures)

    return list(by_key.values())[:max_results]


async def fetch_url(url: str, *, deps: SearchDeps) -> FetchedDocument:
    """Generic URL-fetch tool.

    Per D11: enforces the operator-configured domain allowlist BEFORE
    issuing the HTTP request. Non-allowlisted hosts raise
    :class:`UrlNotAllowlisted` and DO NOT count against the per-session
    fetch_url cap (the allowlist check happens before the budget
    increment; see the implementation below).

    On a successful direct-then-jina fetch path, returns the
    :class:`FetchedDocument` produced by the fetcher. On total failure
    raises :class:`UrlFetchFailed`.
    """
    _enforce_allowlist(url, deps.fetch_allowlist)
    await deps.budgets.check_and_increment("fetch_url", deps.url_fetcher.name)
    started = _now_utc()
    try:
        doc = await deps.url_fetcher.fetch(url)
    except UrlNotAllowlisted:
        # Allowlist check is enforced above; rethrow defensively if a
        # provider implementation ever surfaces it from inside fetch().
        raise
    except Exception as exc:  # noqa: BLE001
        await deps.egress_log.log(
            type="fetch_url",
            provider=deps.url_fetcher.name,
            url=url,
            duration_ms=_ms_since(started),
            error=repr(exc),
        )
        raise UrlFetchFailed(f"fetch_url({url!r}) failed: {exc!r}") from exc
    await deps.egress_log.log(
        type="fetch_url",
        provider=doc.source,
        url=url,
        content_type=doc.content_type,
        duration_ms=_ms_since(started),
        status=200,
    )
    return doc


# =============================================================================
# Allowlist enforcement.
# =============================================================================


def _enforce_allowlist(url: str, allowlist: tuple[str, ...]) -> None:
    """Raise :class:`UrlNotAllowlisted` if the host of ``url`` is not on
    the allowlist.

    Matching rules:

    * Exact match against any allowlist entry.
    * Subdomain match: ``arxiv.org`` on the allowlist matches the
      host ``www.arxiv.org``, ``export.arxiv.org`` (suffix match with
      a dot boundary). It does NOT match ``evilarxiv.org`` (no dot
      boundary).
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        raise UrlNotAllowlisted(host, allowlist)
    for entry in allowlist:
        entry_l = entry.lower()
        if host == entry_l or host.endswith("." + entry_l):
            return
    raise UrlNotAllowlisted(host, allowlist)


# =============================================================================
# Dedupe + source-priority helpers (§4 of the design note).
# =============================================================================


def _dedupe_key(result: AcademicResult) -> str:
    """Key used to merge fan-out results: arxiv_id > doi > normalized title."""
    if result.arxiv_id:
        return f"arxiv:{result.arxiv_id}"
    if result.doi:
        return f"doi:{result.doi.lower()}"
    return f"title:{_normalize_title(result.title)}"


def _normalize_title(title: str) -> str:
    return " ".join(title.lower().split())


def _source_priority(source: str) -> int:
    """Source-priority ordering for merge-with-richest-source policy.

    Empirical for v1: arXiv carries pdf_url + abstract reliably;
    OpenAlex carries citation graph + DOI; Semantic Scholar carries
    abstracts well but fewer fields.
    """
    return {"arxiv": 3, "openalex": 2, "semantic_scholar": 1}.get(source, 0)


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _ms_since(started: datetime) -> int:
    return int((_now_utc() - started).total_seconds() * 1000)


__all__ = [
    "academic_search",
    "build_search_deps",
    "fetch_url",
    "web_search",
]
