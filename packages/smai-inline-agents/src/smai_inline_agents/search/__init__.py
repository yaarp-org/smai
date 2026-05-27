"""Planner search-tool surface.

Per ``planner_refactor/architectural_decisions.md`` §4, §8, §10, §11
and ``planner_refactor/design_notes/search_tool_surface.md`` (Step 1
of the planner_refactor implementation_plan).

Three generic typed tool functions the PydanticAI ``@agent.tool``
decorators wrap in the lit-review phase agent:

* :func:`web_search` -- general-web search (no v1 default provider per
  D11; raises :class:`NotConfiguredError` until an operator opts in).
* :func:`academic_search` -- fan-out across OpenAlex + Semantic Scholar
  + arXiv with dedupe + source-priority merge.
* :func:`fetch_url` -- direct httpx + Jina Reader fallback URL fetch
  with an operator-configured domain allowlist (D11).

Plus:

* :class:`SearchDeps` -- the dependency payload constructed once per
  planner session by :func:`build_search_deps`.
* SMAI-owned result Pydantic models: :class:`SearchResult`,
  :class:`AcademicResult`, :class:`FetchedDocument`.
* The Provider Protocols: :class:`WebSearchProvider`,
  :class:`AcademicSearchProvider`, :class:`UrlFetcher`.
* :class:`PlannerSearchConfig` -- the ``smai.yaml -> planner.search``
  block.
* The exception taxonomy.

The Tavily provider stub stays in the package as a reference
implementation (per D11); it is NOT registered in the default dispatch
and import is only triggered when an operator sets
``planner.search.web_provider: tavily``.
"""

from __future__ import annotations

from smai_inline_agents.search.budgets import (
    DEFAULT_PROVIDER_RATE_LIMITS,
    BudgetTracker,
    RateLimit,
)
from smai_inline_agents.search.config import (
    DEFAULT_FETCH_ALLOWLIST,
    AcademicProviderName,
    ArxivProviderConfig,
    BudgetsConfig,
    JinaReaderProviderConfig,
    OpenAlexProviderConfig,
    PlannerSearchConfig,
    ProvidersBlock,
    SemanticScholarProviderConfig,
    TavilyProviderConfig,
    UrlFetchMode,
    WebProviderName,
)
from smai_inline_agents.search.deps import SearchDeps
from smai_inline_agents.search.dispatch import (
    academic_search,
    build_search_deps,
    fetch_url,
    web_search,
)
from smai_inline_agents.search.egress_log import EgressLogger
from smai_inline_agents.search.errors import (
    AcademicSearchAllProvidersFailed,
    BudgetExhausted,
    ContentExtractionFailed,
    NotConfiguredError,
    RateLimitExceeded,
    UrlFetchFailed,
    UrlNotAllowlisted,
    WebSearchProviderDown,
)
from smai_inline_agents.search.protocols import (
    AcademicSearchProvider,
    UrlFetcher,
    WebSearchProvider,
)
from smai_inline_agents.search.types import (
    AcademicResult,
    EgressEvent,
    FetchedDocument,
    SearchResult,
)

__all__ = [
    "DEFAULT_FETCH_ALLOWLIST",
    "DEFAULT_PROVIDER_RATE_LIMITS",
    "AcademicProviderName",
    "AcademicResult",
    "AcademicSearchAllProvidersFailed",
    "AcademicSearchProvider",
    "ArxivProviderConfig",
    "BudgetExhausted",
    "BudgetTracker",
    "BudgetsConfig",
    "ContentExtractionFailed",
    "EgressEvent",
    "EgressLogger",
    "FetchedDocument",
    "JinaReaderProviderConfig",
    "NotConfiguredError",
    "OpenAlexProviderConfig",
    "PlannerSearchConfig",
    "ProvidersBlock",
    "RateLimit",
    "RateLimitExceeded",
    "SearchDeps",
    "SearchResult",
    "SemanticScholarProviderConfig",
    "TavilyProviderConfig",
    "UrlFetchFailed",
    "UrlFetchMode",
    "UrlFetcher",
    "UrlNotAllowlisted",
    "WebProviderName",
    "WebSearchProvider",
    "WebSearchProviderDown",
    "academic_search",
    "build_search_deps",
    "fetch_url",
    "web_search",
]
