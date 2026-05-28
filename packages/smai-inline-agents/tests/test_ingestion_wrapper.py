"""Fake-LLM tests for run_ingestion_subagent + deep recursion (Step 3, Sub-PR B).

Covers the orchestration wrapper's three branches (fetch-fail, reject,
happy path) and ``search_literature``'s deep recursive enrichment mode
(design note §4). Reuses the Sub-PR A fakes (``FunctionModel`` for the
tool loop + ``StubLlmProvider`` for the in-tool sub-extractions /
screener single-call).
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from _agent_fakes import StubLlmProvider  # type: ignore[import-not-found]
from _ingestion_fakes import (  # type: ignore[import-not-found]
    AlwaysSnippetsLlm,
    FakeCiteResolver,
    make_corpus,
    make_corpus_fetcher,
    make_paper_extraction_args,
    screener_response,
)
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from smai_inline_agents.ingestion import (
    IngestionResult,
    PaperAgentDepsFactory,
    PaperCorpus,
    ScreeningOutcome,
    run_ingestion_subagent,
)
from smai_inline_agents.ingestion.paper_agent import _do_search_literature
from smai_inline_agents.ingestion.schemas import CitedPaperRef, SectionRef
from smai_inline_agents.schemas.screener import ScreenResult

_CITED_ARXIV = "2501.00001"
_TOP_ARXIV = "1708.04552"


def _emit_extraction_model() -> FunctionModel:
    """A FunctionModel that emits the output tool with a paper_extract technique."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=make_paper_extraction_args(),
                )
            ]
        )

    return FunctionModel(model_fn)


def _raise_if_called_model() -> FunctionModel:
    """A FunctionModel that fails the test if the agent loop ever runs it."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError("the Paper Agent must not run on the fetch-fail / reject branch")

    return FunctionModel(model_fn)


def _accept_outcome() -> ScreeningOutcome:
    return ScreeningOutcome.from_screen_result(
        ScreenResult(decision="accept", rejection_reason=None, summary="in scope")
    )


# === fetch-fail branch ======================================================


async def test_wrapper_fetch_fail_returns_empty_without_calling_llm() -> None:
    """corpus None -> techniques=[], error_reason='fetch_failed', LLM untouched."""
    factory = PaperAgentDepsFactory(
        sub_extraction_llm=StubLlmProvider([]),  # raises if called
        screener_llm=StubLlmProvider([]),  # raises if called
        corpus_fetcher=make_corpus_fetcher(None),
        model=_raise_if_called_model(),
    )
    result = await run_ingestion_subagent(_TOP_ARXIV, factory)
    assert isinstance(result, IngestionResult)
    assert result.techniques == []
    assert result.error_reason == "fetch_failed"
    assert result.screening is None  # never screened


# === reject branch (internal screening) =====================================


async def test_wrapper_reject_returns_empty_with_rationale() -> None:
    """Internal screener reject -> techniques=[], error_reason=rationale, no agent."""
    corpus = make_corpus(arxiv_id=_TOP_ARXIV)
    factory = PaperAgentDepsFactory(
        sub_extraction_llm=StubLlmProvider([]),
        screener_llm=StubLlmProvider(
            [screener_response(decision="reject", rejection_reason="survey paper, no comparison")]
        ),
        corpus_fetcher=make_corpus_fetcher(corpus),
        model=_raise_if_called_model(),
    )
    result = await run_ingestion_subagent(_TOP_ARXIV, factory)
    assert result.techniques == []
    assert result.screening is not None
    assert result.screening.decision == "reject"
    assert result.error_reason == "survey paper, no comparison"


# === happy path: prescreened, no re-screen =================================


async def test_wrapper_happy_path_prescreened_does_not_rescreen() -> None:
    """screening passed in -> agent runs -> IngestionResult assembled.

    The screener_llm would raise on call (StubLlmProvider([])); the test
    passing proves the pipeline path does NOT re-screen.
    """
    corpus = make_corpus(arxiv_id=_TOP_ARXIV)
    factory = PaperAgentDepsFactory(
        sub_extraction_llm=StubLlmProvider([]),
        screener_llm=StubLlmProvider([]),  # must NOT be called (would raise)
        corpus_fetcher=make_corpus_fetcher(corpus),
        model=_emit_extraction_model(),
    )
    result = await run_ingestion_subagent(_TOP_ARXIV, factory, screening=_accept_outcome())
    assert result.error_reason is None
    assert result.screening is not None
    assert result.screening.decision == "accept"
    assert len(result.techniques) == 1
    technique = result.techniques[0]
    assert technique.context_kind == "paper_extract"
    assert technique.algorithm.source_excerpts  # paper_extract requires excerpts
    assert result.paper_title == corpus.title


# === deep recursive enrichment ==============================================


def _outer_corpus_with_cite() -> PaperCorpus:
    """A top-level corpus whose cite resolver resolves wang2025 -> cited arxiv."""
    return PaperCorpus(
        arxiv_id=_TOP_ARXIV,
        title="Improved Regularization with Cutout",
        abstract="We introduce Cutout.",
        full_latex="We adopt the metric of \\cite{wang2025}.",
        sections=[SectionRef(section_id="1", title="Method", depth=1)],
        sections_by_id={"1": "Method body."},
        cite_resolver=FakeCiteResolver(
            {"wang2025": CitedPaperRef(cite_key="wang2025", arxiv_id=_CITED_ARXIV)}
        ),
    )


async def test_deep_recursion_at_depth_0_splices_cited_techniques() -> None:
    """request_deep_extraction=True at depth 0 recurses on the cited paper and
    splices its techniques into deps.enriched_techniques (design note §4)."""
    cited = make_corpus(arxiv_id=_CITED_ARXIV)
    factory = PaperAgentDepsFactory(
        sub_extraction_llm=AlwaysSnippetsLlm(["a snippet"]),
        screener_llm=StubLlmProvider([screener_response(decision="accept")]),
        corpus_fetcher=make_corpus_fetcher(cited),
        model=_emit_extraction_model(),
        enrichment_depth=0,
        enrichment_depth_limit=1,
    )
    deps = factory.build(corpus=_outer_corpus_with_cite())
    assert deps.enrichment_depth == 0
    assert deps.subagent_factory is not None

    out = await _do_search_literature(deps, "wang2025", "the proposed metric", True)

    assert len(deps.enriched_techniques) == 1
    assert deps.enriched_techniques[0].name == "cutout"
    assert _CITED_ARXIV in out  # the summary pointer names the cited paper


async def test_deep_recursion_depth_limit_stops_second_level() -> None:
    """At depth == depth_limit, request_deep_extraction falls back to the
    lightweight extraction (no second-level recurse)."""
    cited = make_corpus(
        arxiv_id=_CITED_ARXIV,
        full_latex="The proposed metric is the macro F-score.",
    )
    factory = PaperAgentDepsFactory(
        sub_extraction_llm=AlwaysSnippetsLlm(["the proposed metric is the macro F-score"]),
        screener_llm=StubLlmProvider([screener_response(decision="accept")]),
        corpus_fetcher=make_corpus_fetcher(cited),
        model=_emit_extraction_model(),
        enrichment_depth=1,
        enrichment_depth_limit=1,
    )
    deps = replace(factory, enrichment_depth=1).build(corpus=_outer_corpus_with_cite())
    assert deps.enrichment_depth == 1

    out = await _do_search_literature(deps, "wang2025", "the proposed metric", True)

    # No recursion fired -> lightweight extraction tagged Relevant Literature.
    assert deps.enriched_techniques == []
    assert "Source: Relevant Literature" in out


def test_pool_snapshot_threads_into_factory() -> None:
    """A non-empty PoolSnapshot survives factory.build (dedup-vs-pool seam)."""
    from smai_inline_agents.ingestion import PoolSnapshot

    snap = PoolSnapshot(technique_names=["cutout"], summary_markdown="- t1 | cutout")
    factory = PaperAgentDepsFactory(
        sub_extraction_llm=StubLlmProvider([]),
        corpus_fetcher=make_corpus_fetcher(None),
        pool_snapshot=snap,
    )
    deps = factory.build(corpus=make_corpus(arxiv_id=_TOP_ARXIV))
    assert deps.pool_snapshot.technique_names == ["cutout"]


@pytest.mark.parametrize("decision", ["accept", "reject"])
def test_screener_response_helper_roundtrips(decision: str) -> None:
    """The screener_response fake builds a parseable submit_screening payload."""
    resp = screener_response(
        decision=decision,
        rejection_reason="x" if decision == "reject" else None,
    )
    assert resp.message.content[0].name == "submit_screening"  # type: ignore[union-attr]
