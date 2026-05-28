"""Opt-in live ingestion-subagent smoke test (Step 3, Sub-PR B).

Runs the real ``run_ingestion_subagent`` end-to-end against a small,
LaTeX-source-available ML paper (Cutout, arXiv:1708.04552) with a real
LLM behind both the PydanticAI Paper Agent loop and the in-tool
sub-extractions / screener.

Gated by BOTH:

* ``ANTHROPIC_API_KEY`` — the credential the ``ingestion`` role's
  provider needs (the dev-default ingestion role is Bedrock/Sonnet, but
  this test pins Anthropic so a single key runs the whole path; override
  the models via the env vars below).
* ``SMAI_TEST_LIVE_INGEST`` — the live-network opt-in (the run fetches
  the arXiv LaTeX source). Mirrors ``SMAI_TEST_LIVE_SEARCH`` from the
  Step-1 search live tests.

Marked :mod:`pytest.mark.credentialed` per the no-credentials-in-CI
convention — CI never sets either env var, so this skips cleanly.

Env overrides:

* ``SMAI_TEST_LIVE_INGEST_MODEL`` — PydanticAI model spec for the outer
  Paper Agent loop (default ``anthropic:claude-sonnet-4-5``).
* ``SMAI_TEST_LIVE_INGEST_SUBMODEL`` — SMAI Anthropic ``model_id`` for
  the in-tool sub-extractions + screener (default ``claude-sonnet-4-6``).
* ``SMAI_TEST_LIVE_INGEST_ARXIV`` — the paper to ingest (default
  ``1708.04552``).
"""

from __future__ import annotations

import os

import pytest
from smai_inline_agents.ingestion import (
    IngestionResult,
    PaperAgentDepsFactory,
    fetch_paper_corpus,
    run_ingestion_subagent,
)

_SKIP_REASON = "live ingestion test — set ANTHROPIC_API_KEY and SMAI_TEST_LIVE_INGEST=1 to run"
_LIVE_ENABLED = bool(os.environ.get("ANTHROPIC_API_KEY")) and bool(
    os.environ.get("SMAI_TEST_LIVE_INGEST")
)


@pytest.mark.credentialed
@pytest.mark.skipif(not _LIVE_ENABLED, reason=_SKIP_REASON)
async def test_live_ingestion_extracts_paper_extract_with_source_excerpts() -> None:
    """A real run yields >=1 paper_extract technique with verbatim excerpts."""
    # Imported here (not at module top) so the module imports cleanly even
    # when the optional smai-llm-anthropic plugin is absent; the env gate
    # already guarantees this body only runs when credentials are present.
    anthropic_mod = pytest.importorskip("smai_llm_anthropic")
    submodel = os.environ.get("SMAI_TEST_LIVE_INGEST_SUBMODEL", "claude-sonnet-4-6")
    outer_model = os.environ.get("SMAI_TEST_LIVE_INGEST_MODEL", "anthropic:claude-sonnet-4-5")
    arxiv_id = os.environ.get("SMAI_TEST_LIVE_INGEST_ARXIV", "1708.04552")

    sub_llm = anthropic_mod.AnthropicProvider(model_id=submodel)
    factory = PaperAgentDepsFactory(
        sub_extraction_llm=sub_llm,
        screener_llm=sub_llm,
        corpus_fetcher=fetch_paper_corpus,
        model=outer_model,
    )

    # Sanity: the fetch path must actually return a corpus for the gate to
    # be meaningful (surface a clear failure if the paper has no LaTeX).
    corpus = await fetch_paper_corpus(arxiv_id)
    assert corpus is not None, f"no LaTeX corpus fetched for {arxiv_id}"

    result = await run_ingestion_subagent(arxiv_id, factory)

    assert isinstance(result, IngestionResult)
    assert result.error_reason is None, result.error_reason
    assert result.screening is not None
    assert result.screening.decision == "accept"
    assert len(result.techniques) >= 1

    paper_extracts = [t for t in result.techniques if t.context_kind == "paper_extract"]
    assert paper_extracts, "expected at least one context_kind='paper_extract' technique"
    technique = paper_extracts[0]
    # paper_extract requires non-empty verbatim algorithm excerpts.
    assert technique.algorithm.source_excerpts
    assert all(ex.text.strip() for ex in technique.algorithm.source_excerpts)
    assert technique.source_arxiv_id is not None
