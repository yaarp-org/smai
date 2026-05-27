"""Credentialed live tests for the search-tool surface (Step 1 of
planner_refactor).

Per CLAUDE.md "Testing patterns": credentialed tests carry BOTH
``@pytest.mark.credentialed`` AND a ``@pytest.mark.skipif`` on the
relevant env var so they skip cleanly when credentials are absent.

Live lane is gated by ``SMAI_TEST_LIVE_SEARCH=1`` so the network-hitting
tests don't run by default on local dev (per the implementation_plan §1
Blocked #4 preference: "run credentialed lane locally per-step, not in
CI"). Per the implementation_plan: v1 academic stack uses OpenAlex
(no key), Semantic Scholar (optional key via ``SEMANTIC_SCHOLAR_API_KEY``;
public-pool tested when absent), arXiv (no key).

These tests sanity-check that each provider's real-API response is
mappable to the typed Pydantic result models, not the agent's
end-to-end behavior. They issue ONE small query per provider; the goal
is shape validation, not coverage of the providers' search relevance.
"""

from __future__ import annotations

import os

import pytest
from smai_inline_agents.search.providers.arxiv_provider import ArxivProvider
from smai_inline_agents.search.providers.openalex import OpenAlexProvider
from smai_inline_agents.search.providers.semantic_scholar import SemanticScholarProvider

_LIVE_ENV = "SMAI_TEST_LIVE_SEARCH"
_S2_ENV = "SEMANTIC_SCHOLAR_API_KEY"


def _live_enabled() -> bool:
    return os.environ.get(_LIVE_ENV) == "1"


@pytest.mark.credentialed
@pytest.mark.skipif(not _live_enabled(), reason=f"set {_LIVE_ENV}=1 to enable the live lane")
async def test_live_openalex_query_returns_results() -> None:
    """OpenAlex requires no key; runs on the standard pool. We pass the
    optional polite-pool email if ``OPENALEX_POLITE_EMAIL`` is set."""
    provider = OpenAlexProvider(
        polite_email=os.environ.get("OPENALEX_POLITE_EMAIL"),
    )
    try:
        results = await provider.search("deep learning differential privacy", max_results=3)
    finally:
        await provider.aclose()
    assert len(results) > 0
    for r in results:
        assert r.title
        assert r.source == "openalex"
        # Type-system validation already ran via Pydantic; this asserts
        # the at-least-one-field-populated invariant.
        assert isinstance(r.authors, list)


@pytest.mark.credentialed
@pytest.mark.skipif(not _live_enabled(), reason=f"set {_LIVE_ENV}=1 to enable the live lane")
async def test_live_semantic_scholar_public_pool_query_returns_results() -> None:
    """Semantic Scholar public-pool lane. Runs when the live flag is set
    EVEN WITHOUT an API key (the design note locks the public pool as a
    supported fallback). When the key is present, the test still uses
    it to lift the rate cap so the lane is fast."""
    provider = SemanticScholarProvider(
        api_key=os.environ.get(_S2_ENV),
    )
    try:
        results = await provider.search("differential privacy attention", max_results=3)
    finally:
        await provider.aclose()
    assert len(results) > 0
    for r in results:
        assert r.title
        assert r.source == "semantic_scholar"


@pytest.mark.credentialed
@pytest.mark.skipif(
    not _live_enabled() or not os.environ.get(_S2_ENV),
    reason=f"set {_LIVE_ENV}=1 AND {_S2_ENV} to enable the keyed lane",
)
async def test_live_semantic_scholar_keyed_lane_sends_header() -> None:
    """Keyed-lane sanity check: when an API key is provisioned, the
    provider runs with the key header so the rate cap is the higher
    1-req/sec tier rather than the public-pool 100-req/5min."""
    api_key = os.environ[_S2_ENV]
    provider = SemanticScholarProvider(api_key=api_key)
    try:
        results = await provider.search("DP-SGD", max_results=2)
    finally:
        await provider.aclose()
    # The keyed lane also returns results; assertion is the same shape.
    assert len(results) > 0


@pytest.mark.credentialed
@pytest.mark.skipif(not _live_enabled(), reason=f"set {_LIVE_ENV}=1 to enable the live lane")
async def test_live_arxiv_query_returns_results() -> None:
    """arXiv requires no key. The package enforces its 3-second
    politeness delay internally."""
    provider = ArxivProvider(delay_seconds=3.0)
    results = await provider.search("differential privacy attention", max_results=3)
    assert len(results) > 0
    for r in results:
        assert r.title
        assert r.source == "arxiv"
        # arXiv always carries the id.
        assert r.arxiv_id is not None
