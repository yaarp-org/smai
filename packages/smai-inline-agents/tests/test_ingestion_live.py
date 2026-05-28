"""Opt-in live ingestion-subagent smoke test (Step 3, Sub-PR B).

Runs the real ``run_ingestion_subagent`` end-to-end against a small,
LaTeX-source-available ML paper (Cutout, arXiv:1708.04552) with a real
LLM behind both the PydanticAI Paper Agent loop and the in-tool
sub-extractions / screener.

Two variants exercise the same flow against two providers:

* ``test_live_ingestion_extracts_paper_extract_with_source_excerpts`` —
  the cheap Anthropic-direct path (single ``ANTHROPIC_API_KEY``), kept
  for local checks.
* ``test_live_ingestion_bedrock_extracts_paper_extract_with_source_excerpts``
  — the production path: the ``ingestion`` role defaults to Bedrock/Sonnet
  (``TASK_DEFAULTS["ingestion"] = ("bedrock", "us.anthropic.claude-sonnet-4-6")``),
  which is what ``smai ingest`` actually runs. This variant drives the
  real Bedrock ``LlmProvider`` so the production provider path has
  automated coverage.

Gating — the Anthropic-direct variant needs BOTH:

* ``ANTHROPIC_API_KEY`` — the credential the Anthropic-pinned variant's
  provider needs.
* ``SMAI_TEST_LIVE_INGEST`` — the live-network opt-in (the run fetches
  the arXiv LaTeX source). Mirrors ``SMAI_TEST_LIVE_SEARCH`` from the
  Step-1 search live tests.

The Bedrock variant needs BOTH:

* ``BEDROCK_LIVE_TESTS=1`` — the project's canonical Bedrock live-call
  gate (mirrors ``plugins/smai-llm-bedrock/tests/test_live.py``); AWS
  credentials must be discoverable by botocore.
* ``SMAI_TEST_LIVE_INGEST`` — the same live-network opt-in.

Both are marked :mod:`pytest.mark.credentialed` per the
no-credentials-in-CI convention — CI never sets these env vars, so they
skip cleanly.

Env overrides:

* ``SMAI_TEST_LIVE_INGEST_MODEL`` — PydanticAI model spec for the outer
  Anthropic-direct Paper Agent loop (default ``anthropic:claude-sonnet-4-5``).
* ``SMAI_TEST_LIVE_INGEST_SUBMODEL`` — SMAI Anthropic ``model_id`` for
  the in-tool sub-extractions + screener (default ``claude-sonnet-4-6``).
* ``SMAI_TEST_LIVE_INGEST_BEDROCK_MODEL`` — Bedrock ``model_id`` for both
  the outer Paper Agent (as the ``bedrock:<model_id>`` PydanticAI spec)
  and the in-tool sub-extraction provider (default
  ``us.anthropic.claude-sonnet-4-6``, matching the ``ingestion`` default).
* ``AWS_REGION`` — Bedrock region for the sub-extraction provider
  (default ``us-east-1``, matching the canonical Bedrock live test).
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

_BEDROCK_SKIP_REASON = (
    "live Bedrock ingestion test — set BEDROCK_LIVE_TESTS=1 and SMAI_TEST_LIVE_INGEST=1 to run"
)
_BEDROCK_LIVE_ENABLED = os.environ.get("BEDROCK_LIVE_TESTS") == "1" and bool(
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


@pytest.mark.credentialed
@pytest.mark.skipif(not _BEDROCK_LIVE_ENABLED, reason=_BEDROCK_SKIP_REASON)
async def test_live_ingestion_bedrock_extracts_paper_extract_with_source_excerpts() -> None:
    """The production Bedrock path yields >=1 paper_extract with excerpts.

    Mirrors the Anthropic-direct test but drives the real Bedrock
    ``LlmProvider`` — the provider ``smai ingest`` actually uses (the
    ``ingestion`` role defaults to Bedrock/Sonnet). Both the outer Paper
    Agent loop (``bedrock:<model_id>`` PydanticAI spec) and the in-tool
    sub-extractions / screener run against Bedrock.
    """
    # Imported here (not at module top) so the module imports cleanly even
    # when the optional smai-llm-bedrock plugin is absent; the env gate
    # already guarantees this body only runs when credentials are present.
    bedrock_mod = pytest.importorskip("smai_llm_bedrock")
    model_id = os.environ.get(
        "SMAI_TEST_LIVE_INGEST_BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-6"
    )
    region = os.environ.get("AWS_REGION", "us-east-1")
    outer_model = f"bedrock:{model_id}"
    arxiv_id = os.environ.get("SMAI_TEST_LIVE_INGEST_ARXIV", "1708.04552")

    sub_llm = bedrock_mod.BedrockProvider(region=region, model_id=model_id)
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
