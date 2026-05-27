"""Config models for the planner search-tool surface.

Mirrors the ``smai.yaml -> planner.search`` block per
``planner_refactor/design_notes/search_tool_surface.md`` §10.

These models live in ``smai-inline-agents`` rather than
``smai-orchestrator``'s ``EngineConfig`` because they are agent-tool
configuration -- consumed by the literature-review phase of the
planner, not by the engine. The CLI's config-layering pipeline can
embed a :class:`PlannerSearchConfig` under ``RuntimeConfig.planner`` in
a follow-up step (Step 4 of the planner_refactor, when the lit-review
phase wires this in); Step 1 ships the model + factory in isolation.

D11 amendment (2026-05-27): ``web_provider`` defaults to ``None`` (no
v1 default); ``fetch_allowlist`` defaults to a short list of hosts the
planner is expected to reach for legitimate research grounding.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Default v1 fetch allowlist per D11. Operator-extensible via
# ``planner.search.fetch_allowlist`` in ``smai.yaml``.
DEFAULT_FETCH_ALLOWLIST: tuple[str, ...] = (
    "arxiv.org",
    "github.com",
    "raw.githubusercontent.com",
    "huggingface.co",
    "paperswithcode.com",
    "openreview.net",
    "docs.pytorch.org",
    "pytorch.org",
    "tensorflow.org",
    "scikit-learn.org",
)


# Provider name Literals -- the type-narrowed surface used by ``dispatch.py``'s
# ``match`` arms.
WebProviderName = Literal["tavily", "brave", "exa", "serper"]
AcademicProviderName = Literal["openalex", "semantic_scholar", "arxiv"]
UrlFetchMode = Literal["direct", "direct_only", "jina_only"]


class TavilyProviderConfig(BaseModel):
    """Reference config for the Tavily provider stub.

    Per D11, Tavily is NOT registered in the default dispatch; this
    config block exists so the stub stays importable and operators
    opting in via ``planner.search.web_provider: tavily`` can hand it
    real values.
    """

    model_config = ConfigDict(extra="forbid")
    api_key_env: str = "TAVILY_API_KEY"
    search_depth: Literal["basic", "advanced"] = "basic"


class OpenAlexProviderConfig(BaseModel):
    """OpenAlex polite-pool config. Email is OPTIONAL; absent leaves
    the provider in the standard (uncapped) pool."""

    model_config = ConfigDict(extra="forbid")
    polite_email_env: str = "OPENALEX_POLITE_EMAIL"


class SemanticScholarProviderConfig(BaseModel):
    """Semantic Scholar config. API key OPTIONAL: absent runs the
    public-pool rate cap (100 req / 5 min); present runs the keyed
    rate cap (1 req / sec)."""

    model_config = ConfigDict(extra="forbid")
    api_key_env: str = "SEMANTIC_SCHOLAR_API_KEY"


class ArxivProviderConfig(BaseModel):
    """arXiv provider config. ``delay_seconds`` is the documented
    politeness floor; the ``arxiv`` Python package enforces it
    internally too."""

    model_config = ConfigDict(extra="forbid")
    delay_seconds: float = 3.0


class JinaReaderProviderConfig(BaseModel):
    """Jina Reader (https://r.jina.ai/) config. API key OPTIONAL:
    absent runs the free tier."""

    model_config = ConfigDict(extra="forbid")
    api_key_env: str = "JINA_API_KEY"


class ProvidersBlock(BaseModel):
    """Per-provider config blocks. Each field defaults so partial
    config in ``smai.yaml`` works."""

    model_config = ConfigDict(extra="forbid")
    tavily: TavilyProviderConfig = Field(default_factory=TavilyProviderConfig)
    openalex: OpenAlexProviderConfig = Field(default_factory=OpenAlexProviderConfig)
    semantic_scholar: SemanticScholarProviderConfig = Field(
        default_factory=SemanticScholarProviderConfig,
    )
    arxiv: ArxivProviderConfig = Field(default_factory=ArxivProviderConfig)
    jina_reader: JinaReaderProviderConfig = Field(
        default_factory=JinaReaderProviderConfig,
    )


class BudgetsConfig(BaseModel):
    """Per-session call caps. The session cap is enforced by
    :class:`~smai_inline_agents.search.budgets.BudgetTracker` before
    each provider call; hitting the cap raises
    :class:`~smai_inline_agents.search.errors.BudgetExhausted`."""

    model_config = ConfigDict(extra="forbid")
    web_search_cap_per_session: int = 10
    academic_search_cap_per_session: int = 20
    fetch_url_cap_per_session: int = 20


class PlannerSearchConfig(BaseModel):
    """``smai.yaml -> planner.search`` block.

    Lands under SMAI's existing env-var convention:
    ``SMAI_PLANNER__SEARCH__WEB_PROVIDER=brave`` etc. Step 1 ships the
    model + factory; the CLI wires this into ``RuntimeConfig.planner``
    when Step 4 of the planner_refactor needs it.

    Per D11 (2026-05-27): ``web_provider`` defaults to ``None`` (no v1
    default registered); ``fetch_allowlist`` defaults to
    :data:`DEFAULT_FETCH_ALLOWLIST` (10 hosts covering legitimate
    research-grounding fetches). Operators opt into general web-search
    by setting ``web_provider`` to a registered provider name.
    """

    model_config = ConfigDict(extra="forbid")

    web_provider: WebProviderName | None = None
    """Per D11: no v1 default. Set to a provider name to enable open-web
    search; the corresponding provider's config block (e.g.
    ``providers.tavily``) must then carry credentials."""

    academic_providers: list[AcademicProviderName] = Field(
        default_factory=lambda: ["openalex", "semantic_scholar", "arxiv"],
    )
    """The fan-out list. All three are registered by default; operators
    drop entries to reduce the fan-out (e.g. omit ``semantic_scholar``
    when no API key is provisioned and the public pool is too slow)."""

    url_fetch: UrlFetchMode = "direct"
    """``direct`` -- httpx + markdownify with Jina Reader fallback (the
    default). ``direct_only`` -- httpx only, no fallback.
    ``jina_only`` -- force the Jina path uniformly."""

    fetch_allowlist: list[str] = Field(
        default_factory=lambda: list(DEFAULT_FETCH_ALLOWLIST),
    )
    """D11 domain allowlist. Hosts NOT on this list raise
    :class:`~smai_inline_agents.search.errors.UrlNotAllowlisted` at
    fetch time, BEFORE the underlying HTTP request issues."""

    log_queries: bool = True
    """Egress-log toggle. Default ``True``; production deployments should
    not disable this (audit-trail obligation per
    ``architectural_decisions.md`` §10)."""

    budgets: BudgetsConfig = Field(default_factory=BudgetsConfig)
    providers: ProvidersBlock = Field(default_factory=ProvidersBlock)


__all__ = [
    "AcademicProviderName",
    "ArxivProviderConfig",
    "BudgetsConfig",
    "DEFAULT_FETCH_ALLOWLIST",
    "JinaReaderProviderConfig",
    "OpenAlexProviderConfig",
    "PlannerSearchConfig",
    "ProvidersBlock",
    "SemanticScholarProviderConfig",
    "TavilyProviderConfig",
    "UrlFetchMode",
    "WebProviderName",
]
