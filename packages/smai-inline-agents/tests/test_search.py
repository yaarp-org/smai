"""Unit tests for the search-tool surface (Step 1 of planner_refactor).

Coverage:

* Result types: SMAI-owned Pydantic models round-trip + reject extra
  fields per ``ConfigDict(extra="forbid")``.
* Provider mocks: per-provider response fixture maps to typed
  ``AcademicResult`` / ``FetchedDocument`` / ``SearchResult``.
* ``build_search_deps``: v1 default config produces no
  ``web_provider`` (D11); academic providers default to the 3-way
  fan-out; allowlist defaults to the 10-host D11 list.
* ``web_search``: raises ``NotConfiguredError`` by default;
  degrades-to-empty on provider failure when configured.
* ``academic_search``: fan-out parallel, dedupe by arxiv_id > doi >
  normalized title, source-priority arxiv > openalex >
  semantic_scholar, tolerates 1-2 provider failures, raises
  ``AcademicSearchAllProvidersFailed`` only when all providers fail.
* ``fetch_url``: enforces D11 allowlist BEFORE httpx call (asserted
  by inspecting fake URL fetcher's call list); raises
  ``UrlFetchFailed`` on fetcher exception; logs to egress log on
  success.
* Allowlist enforcement: subdomain match, missing host, non-allowlist
  hosts all rejected.
* Budgets: per-session cap raises ``BudgetExhausted``; per-provider
  rate-limit window raises ``RateLimitExceeded``.
* Egress logging: JSONL artifact written to ArtifactStore with the
  expected event sequence.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError
from smai_inline_agents.search import (
    DEFAULT_FETCH_ALLOWLIST,
    AcademicSearchAllProvidersFailed,
    BudgetExhausted,
    BudgetTracker,
    EgressEvent,
    EgressLogger,
    NotConfiguredError,
    PlannerSearchConfig,
    RateLimit,
    RateLimitExceeded,
    SearchDeps,
    UrlFetchFailed,
    UrlNotAllowlisted,
    academic_search,
    build_search_deps,
    fetch_url,
    web_search,
)
from smai_inline_agents.search.config import (
    BudgetsConfig,
)
from smai_inline_agents.search.providers.arxiv_provider import (
    ArxivProvider,
    _arxiv_id_from_entry_id,
)
from smai_inline_agents.search.providers.direct_fetch import (
    DirectThenJinaFetcher,
    _classify_content_type,
    _extract_title,
    _html_to_text,
)
from smai_inline_agents.search.providers.jina import JinaOnlyFetcher, jina_fetch
from smai_inline_agents.search.providers.openalex import (
    OpenAlexProvider,
    _reconstruct_abstract,
    _strip_doi_prefix,
)
from smai_inline_agents.search.providers.semantic_scholar import SemanticScholarProvider
from smai_inline_agents.search.providers.tavily import TavilyProvider
from smai_inline_agents.search.types import (
    AcademicResult,
    FetchedDocument,
    SearchResult,
)

from ._search_fakes import (
    FakeAcademicProvider,
    FakeArtifactStore,
    FakeUrlFetcher,
    FakeWebProvider,
    FlakyAcademicProvider,
    FlakyWebProvider,
    make_academic_result,
    make_fetched_document,
    make_search_result,
)

# =============================================================================
# Result type round-trip.
# =============================================================================


def test_search_result_round_trip() -> None:
    r = make_search_result(
        title="DP-SGD",
        url="https://example.com/dp-sgd",
        snippet="Overview of DP-SGD.",
        source="tavily",
        relevance_score=0.87,
    )
    dumped = r.model_dump_json()
    parsed = SearchResult.model_validate_json(dumped)
    assert parsed == r


def test_academic_result_round_trip() -> None:
    a = make_academic_result(
        title="Deep Learning with DP",
        arxiv_id="1607.00133",
        doi="10.1145/2976749.2978318",
        source="arxiv",
        abstract="We extend the moments accountant...",
        authors=["Abadi", "Chu", "Goodfellow"],
        year=2016,
        pdf_url="https://arxiv.org/pdf/1607.00133",
        citation_count=4023,
    )
    parsed = AcademicResult.model_validate_json(a.model_dump_json())
    assert parsed == a


def test_fetched_document_round_trip() -> None:
    fd = make_fetched_document()
    parsed = FetchedDocument.model_validate_json(fd.model_dump_json())
    assert parsed == fd


def test_search_result_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SearchResult.model_validate(
            {
                "title": "x",
                "url": "https://example.com",
                "snippet": "y",
                "source": "tavily",
                "unknown_field": "boom",
            },
        )


# =============================================================================
# PlannerSearchConfig defaults (D11).
# =============================================================================


def test_planner_search_config_defaults_d11() -> None:
    cfg = PlannerSearchConfig()
    assert cfg.web_provider is None
    assert cfg.academic_providers == ["openalex", "semantic_scholar", "arxiv"]
    assert cfg.url_fetch == "direct"
    assert tuple(cfg.fetch_allowlist) == DEFAULT_FETCH_ALLOWLIST
    assert cfg.log_queries is True
    assert cfg.budgets.web_search_cap_per_session == 10
    assert cfg.budgets.academic_search_cap_per_session == 20
    assert cfg.budgets.fetch_url_cap_per_session == 20


def test_planner_search_config_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        PlannerSearchConfig.model_validate({"unknown": "value"})


def test_default_fetch_allowlist_covers_d11_hosts() -> None:
    # D11 names the v1 hosts explicitly; sanity-check they're all present.
    assert "arxiv.org" in DEFAULT_FETCH_ALLOWLIST
    assert "github.com" in DEFAULT_FETCH_ALLOWLIST
    assert "raw.githubusercontent.com" in DEFAULT_FETCH_ALLOWLIST
    assert "huggingface.co" in DEFAULT_FETCH_ALLOWLIST
    assert "paperswithcode.com" in DEFAULT_FETCH_ALLOWLIST
    assert "openreview.net" in DEFAULT_FETCH_ALLOWLIST
    assert "pytorch.org" in DEFAULT_FETCH_ALLOWLIST
    assert "tensorflow.org" in DEFAULT_FETCH_ALLOWLIST


# =============================================================================
# build_search_deps.
# =============================================================================


def test_build_search_deps_default_no_web_provider() -> None:
    cfg = PlannerSearchConfig()
    deps = build_search_deps(cfg, session_id="s1")
    assert deps.web_provider is None
    assert {p.name for p in deps.academic_providers} == {
        "openalex",
        "semantic_scholar",
        "arxiv",
    }
    assert deps.url_fetcher.name == "direct"
    assert tuple(deps.fetch_allowlist) == DEFAULT_FETCH_ALLOWLIST
    assert deps.session_id == "s1"


def test_build_search_deps_drop_academic_providers() -> None:
    cfg = PlannerSearchConfig(academic_providers=["arxiv"])
    deps = build_search_deps(cfg, session_id="s2")
    assert [p.name for p in deps.academic_providers] == ["arxiv"]


def test_build_search_deps_jina_only_fetcher() -> None:
    cfg = PlannerSearchConfig(url_fetch="jina_only")
    deps = build_search_deps(cfg, session_id="s3")
    assert deps.url_fetcher.name == "jina_reader"


def test_build_search_deps_brave_not_implemented() -> None:
    cfg = PlannerSearchConfig(web_provider="brave")
    with pytest.raises(NotImplementedError):
        build_search_deps(cfg, session_id="s4")


# =============================================================================
# web_search behavior.
# =============================================================================


def _make_deps_with(
    *,
    web_provider: object | None = None,
    academic_providers: list[FakeAcademicProvider | FlakyAcademicProvider] | None = None,
    url_fetcher: FakeUrlFetcher | None = None,
    fetch_allowlist: tuple[str, ...] = DEFAULT_FETCH_ALLOWLIST,
    artifact_store: FakeArtifactStore | None = None,
    proposal_id: str | None = "prop-1",
    log_queries: bool = True,
    budgets: BudgetsConfig | None = None,
) -> SearchDeps:
    """Construct a :class:`SearchDeps` from fakes; the production
    :func:`build_search_deps` path is exercised separately."""
    return SearchDeps(
        academic_providers=list(academic_providers or []),  # type: ignore[arg-type]
        url_fetcher=url_fetcher or FakeUrlFetcher(),
        egress_log=EgressLogger.open(
            "sess",
            proposal_id=proposal_id,
            artifact_store=artifact_store,
            enabled=log_queries,
        ),
        budgets=BudgetTracker.from_config(budgets or BudgetsConfig()),
        session_id="sess",
        fetch_allowlist=fetch_allowlist,
        web_provider=web_provider,  # type: ignore[arg-type]
    )


async def test_web_search_raises_not_configured_by_default() -> None:
    deps = _make_deps_with()
    with pytest.raises(NotConfiguredError):
        await web_search("query", deps=deps)


async def test_web_search_returns_provider_results() -> None:
    provider = FakeWebProvider(
        results=[make_search_result(title="hit1"), make_search_result(title="hit2")],
    )
    deps = _make_deps_with(web_provider=provider)
    results = await web_search("query", max_results=3, deps=deps)
    assert [r.title for r in results] == ["hit1", "hit2"]
    assert provider.calls == [("query", 3)]


async def test_web_search_degrades_to_empty_on_provider_failure() -> None:
    provider = FlakyWebProvider(RuntimeError("upstream down"))
    deps = _make_deps_with(web_provider=provider)
    results = await web_search("query", deps=deps)
    assert results == []
    # The egress log records the failure.
    events = deps.egress_log.events
    assert len(events) == 1
    assert events[0].type == "web_search"
    assert events[0].error is not None
    assert "upstream down" in events[0].error


# =============================================================================
# academic_search behavior (fan-out, dedupe, source-priority, tolerance).
# =============================================================================


async def test_academic_search_fan_out_dedupes_by_arxiv_id_priority() -> None:
    # arxiv > openalex > semantic_scholar source priority.
    p_s2 = FakeAcademicProvider(
        "semantic_scholar",
        results=[make_academic_result(arxiv_id="2504.00255", source="semantic_scholar")],
    )
    p_oa = FakeAcademicProvider(
        "openalex",
        results=[make_academic_result(arxiv_id="2504.00255", source="openalex")],
    )
    p_arxiv = FakeAcademicProvider(
        "arxiv",
        results=[make_academic_result(arxiv_id="2504.00255", source="arxiv")],
    )
    deps = _make_deps_with(academic_providers=[p_s2, p_oa, p_arxiv])
    results = await academic_search("dp-sgd", max_results=5, deps=deps)
    assert len(results) == 1
    assert results[0].source == "arxiv"


async def test_academic_search_dedupes_by_doi_when_no_arxiv_id() -> None:
    p1 = FakeAcademicProvider(
        "openalex",
        results=[make_academic_result(doi="10.1145/2976749.2978318", source="openalex")],
    )
    p2 = FakeAcademicProvider(
        "semantic_scholar",
        results=[make_academic_result(doi="10.1145/2976749.2978318", source="semantic_scholar")],
    )
    deps = _make_deps_with(academic_providers=[p1, p2])
    results = await academic_search("q", max_results=5, deps=deps)
    assert len(results) == 1
    assert results[0].source == "openalex"  # openalex > semantic_scholar


async def test_academic_search_dedupes_by_normalized_title_when_no_id() -> None:
    p1 = FakeAcademicProvider(
        "openalex",
        results=[make_academic_result(title="Deep Learning with DP", source="openalex")],
    )
    p2 = FakeAcademicProvider(
        "semantic_scholar",
        results=[
            make_academic_result(title="DEEP LEARNING   with DP", source="semantic_scholar"),
        ],
    )
    deps = _make_deps_with(academic_providers=[p1, p2])
    results = await academic_search("q", max_results=5, deps=deps)
    assert len(results) == 1


async def test_academic_search_tolerates_partial_provider_failure() -> None:
    p_ok = FakeAcademicProvider(
        "arxiv",
        results=[make_academic_result(arxiv_id="2401.12345", source="arxiv")],
    )
    p_fail1 = FlakyAcademicProvider("openalex", RuntimeError("503"))
    p_fail2 = FlakyAcademicProvider("semantic_scholar", RuntimeError("timeout"))
    deps = _make_deps_with(academic_providers=[p_ok, p_fail1, p_fail2])
    results = await academic_search("q", max_results=5, deps=deps)
    assert len(results) == 1
    assert results[0].arxiv_id == "2401.12345"
    # Egress log records the partial failure.
    events = deps.egress_log.events
    assert len(events) == 1
    ev = events[0]
    assert ev.type == "academic_search"
    assert ev.providers_succeeded == ["arxiv"]
    assert set(ev.providers_failed or []) == {"openalex", "semantic_scholar"}


async def test_academic_search_raises_when_all_providers_fail() -> None:
    p1 = FlakyAcademicProvider("openalex", RuntimeError("503"))
    p2 = FlakyAcademicProvider("semantic_scholar", RuntimeError("503"))
    p3 = FlakyAcademicProvider("arxiv", RuntimeError("503"))
    deps = _make_deps_with(academic_providers=[p1, p2, p3])
    with pytest.raises(AcademicSearchAllProvidersFailed) as exc_info:
        await academic_search("q", max_results=5, deps=deps)
    assert len(exc_info.value.failures) == 3


async def test_academic_search_no_providers_registered_raises() -> None:
    deps = _make_deps_with(academic_providers=[])
    with pytest.raises(AcademicSearchAllProvidersFailed):
        await academic_search("q", deps=deps)


async def test_academic_search_truncates_to_max_results() -> None:
    p = FakeAcademicProvider(
        "arxiv",
        results=[
            make_academic_result(arxiv_id="1111.0001", source="arxiv"),
            make_academic_result(arxiv_id="1111.0002", source="arxiv"),
            make_academic_result(arxiv_id="1111.0003", source="arxiv"),
        ],
    )
    deps = _make_deps_with(academic_providers=[p])
    results = await academic_search("q", max_results=2, deps=deps)
    assert len(results) == 2


# =============================================================================
# fetch_url + allowlist enforcement.
# =============================================================================


async def test_fetch_url_rejects_non_allowlisted_host() -> None:
    fetcher = FakeUrlFetcher(document=make_fetched_document())
    deps = _make_deps_with(
        url_fetcher=fetcher,
        fetch_allowlist=("arxiv.org",),
    )
    with pytest.raises(UrlNotAllowlisted) as exc_info:
        await fetch_url("https://malicious.example.com/page", deps=deps)
    # Critical: the underlying fetcher must NEVER be invoked.
    assert fetcher.calls == []
    assert exc_info.value.host == "malicious.example.com"


async def test_fetch_url_allows_exact_match() -> None:
    fetcher = FakeUrlFetcher(document=make_fetched_document(url="https://arxiv.org/abs/1"))
    deps = _make_deps_with(
        url_fetcher=fetcher,
        fetch_allowlist=("arxiv.org",),
    )
    doc = await fetch_url("https://arxiv.org/abs/1", deps=deps)
    assert doc.url == fetcher.document.url  # type: ignore[union-attr]
    assert fetcher.calls == ["https://arxiv.org/abs/1"]


async def test_fetch_url_allows_subdomain_match() -> None:
    fetcher = FakeUrlFetcher(
        document=make_fetched_document(url="https://export.arxiv.org/abs/2401.1"),
    )
    deps = _make_deps_with(
        url_fetcher=fetcher,
        fetch_allowlist=("arxiv.org",),
    )
    await fetch_url("https://export.arxiv.org/abs/2401.1", deps=deps)
    assert fetcher.calls == ["https://export.arxiv.org/abs/2401.1"]


async def test_fetch_url_rejects_substring_lookalike_host() -> None:
    # ``evilarxiv.org`` is NOT a subdomain of ``arxiv.org`` (no dot
    # boundary); must be rejected.
    fetcher = FakeUrlFetcher(document=make_fetched_document())
    deps = _make_deps_with(
        url_fetcher=fetcher,
        fetch_allowlist=("arxiv.org",),
    )
    with pytest.raises(UrlNotAllowlisted):
        await fetch_url("https://evilarxiv.org/abs/1", deps=deps)
    assert fetcher.calls == []


async def test_fetch_url_rejects_empty_host() -> None:
    fetcher = FakeUrlFetcher()
    deps = _make_deps_with(
        url_fetcher=fetcher,
        fetch_allowlist=("arxiv.org",),
    )
    with pytest.raises(UrlNotAllowlisted):
        # Not a valid URL; urlparse yields empty hostname.
        await fetch_url("not://a/host", deps=deps)
    assert fetcher.calls == []


async def test_fetch_url_raises_url_fetch_failed_on_fetcher_exception() -> None:
    class BoomFetcher:
        name = "boom"

        async def fetch(self, url: str) -> FetchedDocument:
            raise RuntimeError(f"fetcher down ({url})")

    deps = _make_deps_with(
        url_fetcher=BoomFetcher(),  # type: ignore[arg-type]
        fetch_allowlist=("arxiv.org",),
    )
    with pytest.raises(UrlFetchFailed):
        await fetch_url("https://arxiv.org/abs/1", deps=deps)


async def test_fetch_url_logs_to_egress_on_success() -> None:
    fetcher = FakeUrlFetcher(
        document=make_fetched_document(
            url="https://arxiv.org/abs/1",
            content_type="html",
            source="direct",
        ),
    )
    deps = _make_deps_with(
        url_fetcher=fetcher,
        fetch_allowlist=("arxiv.org",),
    )
    await fetch_url("https://arxiv.org/abs/1", deps=deps)
    events = deps.egress_log.events
    assert len(events) == 1
    assert events[0].type == "fetch_url"
    assert events[0].provider == "direct"
    assert events[0].status == 200
    assert events[0].content_type == "html"


# =============================================================================
# Budget tracker.
# =============================================================================


async def test_budget_session_cap_raises_budget_exhausted() -> None:
    tracker = BudgetTracker(web_search_cap_per_session=2)
    await tracker.check_and_increment("web_search", "fake")
    await tracker.check_and_increment("web_search", "fake")
    with pytest.raises(BudgetExhausted):
        await tracker.check_and_increment("web_search", "fake")


async def test_budget_per_provider_rate_limit_raises_rate_limit_exceeded() -> None:
    # Three calls in the same window for a 1-per-second provider hits
    # the second call.
    fake_clock = [0.0]
    tracker = BudgetTracker(
        academic_search_cap_per_session=10,
        per_provider_rate_limit={"semantic_scholar": RateLimit(max_calls=1, window_seconds=1.0)},
        clock=lambda: fake_clock[0],
    )
    await tracker.check_and_increment("academic_search", "semantic_scholar")
    with pytest.raises(RateLimitExceeded):
        # Same instant, second call within the 1s window.
        await tracker.check_and_increment("academic_search", "semantic_scholar")


async def test_budget_per_provider_rate_limit_recovers_after_window() -> None:
    fake_clock = [0.0]
    tracker = BudgetTracker(
        academic_search_cap_per_session=10,
        per_provider_rate_limit={"semantic_scholar": RateLimit(max_calls=1, window_seconds=1.0)},
        clock=lambda: fake_clock[0],
    )
    await tracker.check_and_increment("academic_search", "semantic_scholar")
    # Advance the clock past the window.
    fake_clock[0] = 2.0
    # Should succeed -- the previous timestamp aged out.
    await tracker.check_and_increment("academic_search", "semantic_scholar")


async def test_budget_unknown_capability_raises_value_error() -> None:
    tracker = BudgetTracker()
    with pytest.raises(ValueError):
        await tracker.check_and_increment("unknown_capability", "p")


def test_budget_remaining_and_get_count() -> None:
    tracker = BudgetTracker(web_search_cap_per_session=3)
    assert tracker.remaining("web_search") == 3
    assert tracker.get_count("web_search") == 0


# =============================================================================
# Egress logging roundtrip.
# =============================================================================


async def test_egress_log_writes_jsonl_to_artifact_store_on_close() -> None:
    store = FakeArtifactStore()
    logger = EgressLogger.open(
        "sess-1",
        proposal_id="prop-x",
        artifact_store=store,
        enabled=True,
    )
    await logger.log(
        type="web_search",
        provider="tavily",
        query="dp-sgd",
        max_results=5,
        result_count=3,
        duration_ms=42,
    )
    await logger.log(
        type="fetch_url",
        provider="direct",
        url="https://arxiv.org/abs/1",
        content_type="html",
        status=200,
        duration_ms=120,
    )
    await logger.close()
    assert len(store.puts) == 1
    key, data, content_type = store.puts[0]
    assert key == "proposals/prop-x/planner/search_log.jsonl"
    assert content_type == "application/x-ndjson"
    lines = [line for line in data.decode("utf-8").splitlines() if line]
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["type"] == "web_search"
    assert first["query"] == "dp-sgd"
    assert first["result_count"] == 3
    second = json.loads(lines[1])
    assert second["type"] == "fetch_url"
    assert second["url"] == "https://arxiv.org/abs/1"
    # Each JSON line is a valid EgressEvent.
    EgressEvent.model_validate(first)
    EgressEvent.model_validate(second)


async def test_egress_log_disabled_skips_put() -> None:
    store = FakeArtifactStore()
    logger = EgressLogger.open(
        "sess-1",
        proposal_id="prop-x",
        artifact_store=store,
        enabled=False,
    )
    await logger.log(
        type="web_search",
        provider="tavily",
        query="q",
        max_results=5,
        result_count=1,
        duration_ms=10,
    )
    await logger.close()
    assert store.puts == []
    assert logger.events == []


async def test_egress_log_close_is_idempotent() -> None:
    store = FakeArtifactStore()
    logger = EgressLogger.open(
        "sess-1",
        proposal_id="prop-x",
        artifact_store=store,
        enabled=True,
    )
    await logger.log(
        type="web_search",
        provider="tavily",
        query="q",
        max_results=5,
        result_count=1,
        duration_ms=10,
    )
    await logger.close()
    await logger.close()  # second close is a no-op.
    assert len(store.puts) == 1


async def test_egress_log_close_swallows_artifact_store_failure() -> None:
    """The audit log must not break the planner when the ArtifactStore
    put fails. Guards the silent-catch in :meth:`EgressLogger.close`
    against future narrowing -- if someone tightens that ``except`` to
    a specific exception class, this test catches it."""

    class FailingStore:
        async def put(
            self,
            key: str,
            data: bytes,
            content_type: str | None = None,
        ) -> None:
            raise RuntimeError("storage backend down")

    logger = EgressLogger.open(
        "sess-fail",
        proposal_id="prop-fail",
        artifact_store=FailingStore(),
        enabled=True,
    )
    await logger.log(
        type="web_search",
        provider="tavily",
        query="q",
        max_results=5,
        result_count=1,
        duration_ms=10,
    )
    # Must NOT propagate the put failure.
    await logger.close()


async def test_egress_log_session_fallback_key_when_no_proposal_id() -> None:
    store = FakeArtifactStore()
    logger = EgressLogger.open(
        "sess-2",
        proposal_id=None,
        artifact_store=store,
        enabled=True,
    )
    await logger.log(
        type="web_search",
        provider="tavily",
        query="q",
        max_results=5,
        result_count=1,
        duration_ms=10,
    )
    await logger.close()
    assert store.puts[0][0] == "sessions/sess-2/planner/search_log.jsonl"


# =============================================================================
# OpenAlex type-mapper helpers (pure functions).
# =============================================================================


def test_openalex_strip_doi_prefix_handles_url_form() -> None:
    assert _strip_doi_prefix("https://doi.org/10.1145/Foo") == "10.1145/foo"
    assert _strip_doi_prefix("doi:10.5555/BaR") == "10.5555/bar"
    assert _strip_doi_prefix("") is None


def test_openalex_reconstruct_abstract_from_inverted_index() -> None:
    inverted = {"Deep": [0], "Learning": [1, 4], "is": [2], "fun": [3]}
    out = _reconstruct_abstract(inverted)
    assert out == "Deep Learning is fun Learning"


def test_openalex_reconstruct_abstract_handles_non_dict() -> None:
    assert _reconstruct_abstract(None) == ""
    assert _reconstruct_abstract("not an inverted index") == ""


async def test_openalex_provider_maps_full_response() -> None:
    """End-to-end via httpx.MockTransport: a single OpenAlex-shaped
    response maps to the typed :class:`AcademicResult`."""
    payload = {
        "results": [
            {
                "title": "Deep Learning with DP",
                "publication_year": 2016,
                "cited_by_count": 4023,
                "ids": {
                    "openalex": "https://openalex.org/W1234",
                    "doi": "https://doi.org/10.1145/2976749.2978318",
                },
                "abstract_inverted_index": {"Deep": [0], "DP": [1]},
                "authorships": [
                    {"author": {"display_name": "Abadi"}},
                    {"author": {"display_name": "Goodfellow"}},
                ],
                "primary_location": {
                    "pdf_url": "https://example.com/pdf",
                    "landing_page_url": "https://arxiv.org/abs/1607.00133",
                },
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/works"
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    provider = OpenAlexProvider(client=client)
    results = await provider.search("dp", 5)
    assert len(results) == 1
    r = results[0]
    assert r.title == "Deep Learning with DP"
    assert r.year == 2016
    assert r.citation_count == 4023
    assert r.doi == "10.1145/2976749.2978318"
    assert r.arxiv_id == "1607.00133"
    assert r.abstract == "Deep DP"
    assert r.authors == ["Abadi", "Goodfellow"]
    assert r.source == "openalex"
    await client.aclose()


# =============================================================================
# Semantic Scholar type-mapper (httpx.MockTransport).
# =============================================================================


async def test_semantic_scholar_provider_maps_full_response() -> None:
    payload = {
        "data": [
            {
                "title": "DP-SGD",
                "year": 2021,
                "citationCount": 100,
                "authors": [{"name": "A. Author"}, {"name": "B. Author"}],
                "externalIds": {
                    "ArXiv": "2401.12345v2",
                    "DOI": "10.1145/XYZ",
                },
                "abstract": "An abstract.",
                "openAccessPdf": {"url": "https://example.com/pdf"},
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    provider = SemanticScholarProvider(client=client, api_key="x")
    results = await provider.search("dp", 5)
    assert len(results) == 1
    r = results[0]
    assert r.title == "DP-SGD"
    assert r.year == 2021
    assert r.citation_count == 100
    assert r.arxiv_id == "2401.12345"  # ``v2`` stripped
    assert r.doi == "10.1145/xyz"
    assert r.authors == ["A. Author", "B. Author"]
    assert r.source == "semantic_scholar"
    await client.aclose()


async def test_semantic_scholar_provider_sends_api_key_header() -> None:
    seen_headers: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(dict(request.headers))
        return httpx.Response(200, json={"data": []})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    provider = SemanticScholarProvider(client=client, api_key="secret-key")
    await provider.search("q", 5)
    assert seen_headers[0].get("x-api-key") == "secret-key"
    await client.aclose()


async def test_semantic_scholar_provider_omits_api_key_header_when_absent() -> None:
    seen_headers: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(dict(request.headers))
        return httpx.Response(200, json={"data": []})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    provider = SemanticScholarProvider(client=client, api_key=None)
    await provider.search("q", 5)
    assert "x-api-key" not in seen_headers[0]
    await client.aclose()


# =============================================================================
# arXiv id parsing.
# =============================================================================


def test_arxiv_id_from_entry_id() -> None:
    assert _arxiv_id_from_entry_id("http://arxiv.org/abs/2504.00255v1") == "2504.00255"
    assert _arxiv_id_from_entry_id("https://arxiv.org/abs/2401.12345") == "2401.12345"
    assert _arxiv_id_from_entry_id("not-a-url") is None
    assert _arxiv_id_from_entry_id("") is None


def test_arxiv_provider_lazy_client_construction() -> None:
    # Constructing the provider does NOT import the arxiv package.
    provider = ArxivProvider(delay_seconds=3.0)
    assert provider.name == "arxiv"


# =============================================================================
# direct_fetch helpers.
# =============================================================================


def test_classify_content_type() -> None:
    assert _classify_content_type("text/html; charset=utf-8", "https://x.com/page") == "html"
    assert _classify_content_type("application/pdf", "https://x.com/p.pdf") == "pdf"
    assert _classify_content_type("application/json", "https://x.com/api") == "json"
    assert _classify_content_type("", "https://x.com/file.md") == "markdown"
    assert _classify_content_type("", "https://x.com/file.tex") == "latex"
    assert _classify_content_type("application/octet-stream", "https://x.com/blob") == "other"


def test_extract_title() -> None:
    assert _extract_title("<html><title>Hello &amp; Goodbye</title></html>") == "Hello & Goodbye"
    assert _extract_title("<html><body>no title</body></html>") is None


def test_html_to_text_strips_scripts_and_tags() -> None:
    body = """
    <html>
        <head><title>X</title><script>evil()</script></head>
        <body>
            <p>Hello <b>world</b></p>
            <style>.foo {}</style>
        </body>
    </html>
    """
    out = _html_to_text(body)
    assert "evil()" not in out
    assert ".foo" not in out
    assert "Hello" in out and "world" in out


async def test_direct_fetcher_html_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><title>T</title><body><p>Hello</p></body></html>",
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    fetcher = DirectThenJinaFetcher(
        enable_jina_fallback=False,
        client=client,
    )
    doc = await fetcher.fetch("https://arxiv.org/abs/1")
    assert doc.content_type == "html"
    assert doc.title == "T"
    assert "Hello" in doc.text
    assert doc.source == "direct"
    await client.aclose()


async def test_direct_fetcher_falls_back_to_jina_on_http_error() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if str(request.url).startswith("https://r.jina.ai/"):
            return httpx.Response(200, text="# Rendered markdown")
        return httpx.Response(503, text="server error")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    fetcher = DirectThenJinaFetcher(
        enable_jina_fallback=True,
        client=client,
    )
    doc = await fetcher.fetch("https://arxiv.org/abs/1")
    assert doc.source == "jina_reader"
    assert doc.content_type == "markdown"
    assert "Rendered markdown" in doc.text
    # Direct path tried first, then Jina.
    assert any("r.jina.ai" in c for c in calls)
    await client.aclose()


async def test_direct_fetcher_disabled_fallback_propagates_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="server error")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    fetcher = DirectThenJinaFetcher(
        enable_jina_fallback=False,
        client=client,
    )
    with pytest.raises(httpx.HTTPError):
        await fetcher.fetch("https://arxiv.org/abs/1")
    await client.aclose()


async def test_jina_only_fetcher_returns_markdown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith("https://r.jina.ai/")
        return httpx.Response(200, text="# md content")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    fetcher = JinaOnlyFetcher(client=client)
    doc = await fetcher.fetch("https://arxiv.org/abs/1")
    assert doc.source == "jina_reader"
    assert doc.content_type == "markdown"
    assert "md content" in doc.text
    await client.aclose()


# =============================================================================
# Redirect re-validation (D11 follow-up): the dispatcher only sees the
# initial URL; without an event hook on the httpx client, a 302 to a
# non-allowlisted host would silently exfiltrate. Cover both the rejected
# redirect path and the permitted same-allowlist redirect path.
# =============================================================================


async def test_direct_fetcher_redirect_to_non_allowlisted_host_rejected() -> None:
    seen_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(request.url.host)
        if request.url.host == "arxiv.org":
            return httpx.Response(
                302,
                headers={"location": "https://evil.example.com/exfil"},
            )
        # Hook should fire before the redirect target reaches the
        # transport; this branch must never execute.
        return httpx.Response(200, text="should-never-be-returned")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, follow_redirects=True)
    fetcher = DirectThenJinaFetcher(
        enable_jina_fallback=False,
        client=client,
        fetch_allowlist=("arxiv.org",),
    )
    with pytest.raises(UrlNotAllowlisted) as exc_info:
        await fetcher.fetch("https://arxiv.org/abs/1")
    assert exc_info.value.host == "evil.example.com"
    # Initial request reached the transport; the redirect target did NOT.
    assert seen_hosts == ["arxiv.org"]
    await client.aclose()


async def test_direct_fetcher_redirect_to_allowlisted_host_followed() -> None:
    seen_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(request.url.host)
        if request.url.host == "github.com":
            return httpx.Response(
                302,
                headers={"location": "https://raw.githubusercontent.com/o/r/main/file"},
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><title>RAW</title><body>final</body></html>",
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, follow_redirects=True)
    fetcher = DirectThenJinaFetcher(
        enable_jina_fallback=False,
        client=client,
        fetch_allowlist=("github.com", "raw.githubusercontent.com"),
    )
    doc = await fetcher.fetch("https://github.com/o/r/blob/main/file")
    assert doc.source == "direct"
    assert "final" in doc.text
    assert seen_hosts == ["github.com", "raw.githubusercontent.com"]
    await client.aclose()


async def test_jina_url_escapes_special_chars() -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(str(request.url))
        return httpx.Response(200, text="# md")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    # URL with fragment + query + a pre-encoded sequence. Raw concat
    # would let the ``#`` get parsed as a fragment of the r.jina.ai URL
    # rather than as part of the target path argument.
    target_url = "https://arxiv.org/abs/1?foo=bar%20baz#frag"
    await jina_fetch(client, target_url, api_key=None)
    assert len(captured) == 1
    sent = captured[0]
    assert sent.startswith("https://r.jina.ai/")
    after_prefix = sent[len("https://r.jina.ai/") :]
    # Fragment and query separators in the user-supplied URL must be
    # percent-encoded so Jina sees a single path argument.
    assert "#" not in after_prefix
    assert "?" not in after_prefix
    assert "%23" in after_prefix  # encoded '#'
    assert "%3F" in after_prefix  # encoded '?'
    await client.aclose()


# =============================================================================
# Tavily stub.
# =============================================================================


async def test_tavily_provider_is_a_stub_not_registered() -> None:
    # Default config does NOT register Tavily (D11).
    cfg = PlannerSearchConfig()
    assert cfg.web_provider is None
    deps = build_search_deps(cfg, session_id="s")
    assert deps.web_provider is None

    # Direct construction returns NotImplementedError on search.
    provider = TavilyProvider("dummy-key")
    with pytest.raises(NotImplementedError):
        await provider.search("q", 5)


def test_tavily_provider_map_helper_documents_shape() -> None:
    # _map exists on the stub to document the production shape.
    hit = {
        "title": "T",
        "url": "https://example.com",
        "content": "snippet",
        "score": 0.5,
        "raw_content": None,
    }
    result = TavilyProvider._map(hit)
    assert result.title == "T"
    assert result.snippet == "snippet"
    assert result.source == "tavily"
    assert result.relevance_score == 0.5


# =============================================================================
# SearchDeps lifecycle.
# =============================================================================


async def test_search_deps_aclose_flushes_egress_log() -> None:
    store = FakeArtifactStore()
    deps = _make_deps_with(artifact_store=store)
    await deps.egress_log.log(
        type="web_search",
        provider="x",
        query="q",
        max_results=5,
        result_count=1,
        duration_ms=10,
    )
    await deps.aclose()
    assert len(store.puts) == 1


async def test_search_deps_aclose_is_idempotent() -> None:
    store = FakeArtifactStore()
    deps = _make_deps_with(artifact_store=store)
    await deps.aclose()
    await deps.aclose()  # no error, no double-put.
    assert store.puts == []  # nothing was logged, so nothing was put.
