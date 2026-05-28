"""Fake-LLM tests for the ingestion Paper Agent (Step 3, Sub-PR A).

Drives ``build_paper_agent()`` with PydanticAI's :class:`FunctionModel`
(the test facility for tool-using ReAct agents -- the agent-runtime
single-call steps monkeypatch their runner, but a tool loop wants a
scripted model). Asserts:

* the agent emits a :class:`PaperExtraction` with the expected
  paper_extract :class:`TechniqueDescription`;
* the three retrieval tools are reachable through the loop and map
  fixture corpus content to verbatim snippets (search_paper /
  search_literature via the in-tool ``structured_call`` sub-extraction;
  search_section via a fuzzy lookup);
* the forced-synthesis mutation drops the SearchX verbs after
  :data:`INGESTION_FORCED_SYNTHESIS_AFTER` tool calls.

In-tool sub-extractions are faked with the queue-driven
``StubLlmProvider`` carried in ``PaperAgentDeps.sub_extraction_llm``.
"""

from __future__ import annotations

from _agent_fakes import StubLlmProvider  # type: ignore[import-not-found]
from _ingestion_fakes import (  # type: ignore[import-not-found]
    AlwaysSnippetsLlm,
    FakeCiteResolver,
    make_corpus,
    make_corpus_fetcher,
    make_deps,
    make_paper_extraction,
    make_paper_extraction_args,
    snippets_response,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from smai_inline_agents.ingestion import (
    INGESTION_FORCED_SYNTHESIS_AFTER,
    INGESTION_USAGE_LIMITS,
    IngestionResult,
    PaperExtraction,
    ScreeningOutcome,
    build_paper_agent,
    load_paper_agent_prompt,
)
from smai_inline_agents.ingestion.paper_agent import (
    _NO_INFO_SENTINEL,
    _do_search_literature,
    _do_search_paper,
    _do_search_section,
)
from smai_inline_agents.ingestion.schemas import CitedPaperRef
from smai_inline_agents.prompts import render_initial_user_message
from smai_inline_agents.schemas.screener import ScreenResult

_SEARCH_VERBS = {"search_paper", "search_section", "search_literature"}


def _tool_returns(messages: list[ModelMessage]) -> list[str]:
    """Collect the string content of every ToolReturnPart in the trace."""
    out: list[str] = []
    for message in messages:
        for part in getattr(message, "parts", []):
            if isinstance(part, ToolReturnPart):
                out.append(str(part.content))
    return out


# === Agent emits a PaperExtraction ==========================================


async def test_agent_emits_paper_extraction_with_paper_extract_technique() -> None:
    """The agent (no tool calls) emits a PaperExtraction whose single
    technique carries context_kind='paper_extract'."""
    agent = build_paper_agent(provider="bedrock", model_id="test-model")

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=make_paper_extraction_args(),
                )
            ]
        )

    result = await agent.run(
        "Extract techniques for arXiv 1708.04552",
        deps=make_deps(sub_extraction_llm=StubLlmProvider([])),
        model=FunctionModel(model_fn),
        usage_limits=INGESTION_USAGE_LIMITS,
    )

    assert isinstance(result.output, PaperExtraction)
    assert len(result.output.techniques) == 1
    technique = result.output.techniques[0]
    assert technique.name == "cutout"
    assert technique.context_kind == "paper_extract"
    assert technique.algorithm.source_excerpts  # paper_extract requires excerpts
    assert technique.source_arxiv_id == "1708.04552"


# === Tool reachability through the agent loop ===============================


async def test_search_section_reachable_through_agent_loop() -> None:
    """A scripted model calls search_section; the section's raw text comes
    back as a ToolReturnPart, proving the tool is wired into the loop and
    maps the fixture corpus to its content."""
    agent = build_paper_agent(provider="bedrock", model_id="test-model")
    state = {"searched": False}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not state["searched"]:
            state["searched"] = True
            return ModelResponse(
                parts=[ToolCallPart(tool_name="search_section", args={"title": "Method"})]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(tool_name=info.output_tools[0].name, args=make_paper_extraction_args())
            ]
        )

    deps = make_deps(sub_extraction_llm=StubLlmProvider([]))
    result = await agent.run(
        "Extract techniques",
        deps=deps,
        model=FunctionModel(model_fn),
        usage_limits=INGESTION_USAGE_LIMITS,
    )

    returns = _tool_returns(result.all_messages())
    assert any("Method body text describing the masking procedure." in r for r in returns)
    assert deps.tool_call_count == 1


# === search_paper sub-extraction ============================================


async def test_search_paper_returns_verbatim_snippets() -> None:
    """search_paper runs a structured-call sub-extraction over the full
    LaTeX and returns the verbatim snippets tagged Source: Target Paper."""
    llm = StubLlmProvider(
        [snippets_response(["we randomly mask out square regions of input during training"])]
    )
    deps = make_deps(sub_extraction_llm=llm)
    out = await _do_search_paper(deps, "the cutout masking procedure")
    assert "Source: Target Paper" in out
    assert "we randomly mask out square regions of input during training" in out


async def test_search_paper_no_match_returns_sentinel() -> None:
    """found=False maps to the No-relevant-information sentinel."""
    llm = StubLlmProvider([snippets_response([], found=False)])
    deps = make_deps(sub_extraction_llm=llm)
    out = await _do_search_paper(deps, "something not in the paper")
    assert out == _NO_INFO_SENTINEL


# === search_section helper (fuzzy / disambiguation / miss) ==================


def test_search_section_exact_id_and_fuzzy_title() -> None:
    deps = make_deps(sub_extraction_llm=StubLlmProvider([]))
    by_id = _do_search_section(deps, "3.1")
    assert "Cutout details" in by_id
    by_title = _do_search_section(deps, "Method")
    assert "Method body text" in by_title


def test_search_section_miss_lists_available_sections() -> None:
    deps = make_deps(sub_extraction_llm=StubLlmProvider([]))
    out = _do_search_section(deps, "Nonexistent Appendix Q")
    assert "No section matching" in out
    assert "Method" in out  # the available-sections listing


def test_search_section_multiple_matches_disambiguates() -> None:
    from smai_inline_agents.ingestion.schemas import SectionRef

    sections = [
        SectionRef(section_id="2", title="Experiments", depth=1),
        SectionRef(section_id="2.1", title="Experiments Setup", depth=2),
    ]
    sections_by_id = {"2": "exp body", "2.1": "setup body"}
    deps = make_deps(
        sub_extraction_llm=StubLlmProvider([]),
        sections=sections,
        sections_by_id=sections_by_id,
    )
    out = _do_search_section(deps, "Experiments")
    assert "Multiple sections match" in out
    assert "2.1" in out


# === search_literature lightweight mode =====================================


async def test_search_literature_lightweight_returns_literature_snippets() -> None:
    """Resolves the cite key, fetches the cited corpus, sub-extracts, and
    tags the result Source: Relevant Literature."""
    cited = make_corpus(full_latex="The proposed metric is the macro F-score over classes.")
    llm = StubLlmProvider([snippets_response(["the proposed metric is the macro F-score"])])
    deps = make_deps(
        sub_extraction_llm=llm,
        corpus_fetcher=make_corpus_fetcher(cited),
    )
    out = await _do_search_literature(deps, "wang2025", "the proposed metric")
    assert "Source: Relevant Literature" in out
    assert "the proposed metric is the macro F-score" in out


async def test_search_literature_unknown_cite_key() -> None:
    deps = make_deps(sub_extraction_llm=StubLlmProvider([]))
    out = await _do_search_literature(deps, "no_such_key", "anything")
    assert "No paper found" in out


async def test_search_literature_unresolvable_arxiv_id_returns_sentinel() -> None:
    """A cite key with no recoverable arXiv id is un-fetchable -> sentinel."""
    deps = make_deps(
        sub_extraction_llm=StubLlmProvider([]),
        cite_resolver=FakeCiteResolver({"noid": CitedPaperRef(cite_key="noid", arxiv_id=None)}),
    )
    out = await _do_search_literature(deps, "noid", "anything")
    assert out == _NO_INFO_SENTINEL


async def test_search_literature_deep_extraction_falls_back_to_lightweight() -> None:
    """request_deep_extraction=True is a Sub-PR B seam; in A it runs the
    lightweight path rather than recursing."""
    cited = make_corpus(full_latex="The proposed metric is the macro F-score.")
    llm = StubLlmProvider([snippets_response(["the proposed metric is the macro F-score"])])
    deps = make_deps(sub_extraction_llm=llm, corpus_fetcher=make_corpus_fetcher(cited))
    out = await _do_search_literature(
        deps, "wang2025", "the proposed metric", request_deep_extraction=True
    )
    assert "Source: Relevant Literature" in out
    assert deps.enriched_techniques == []  # no recursion happened in Sub-PR A


# === Forced-synthesis tool-budget exhaustion ================================


async def test_forced_synthesis_drops_search_tools_after_budget() -> None:
    """After INGESTION_FORCED_SYNTHESIS_AFTER retrieval calls, the three
    SearchX verbs disappear from the toolkit, leaving only the output exit."""
    agent = build_paper_agent(provider="bedrock", model_id="test-model")
    # search_paper drives the sub-extraction LLM on every call; the
    # inexhaustible fake keeps all 20 retrievals served.
    deps = make_deps(sub_extraction_llm=AlwaysSnippetsLlm(["a verbatim snippet"]))
    tool_lists: list[set[str]] = []
    call_id = {"n": 0}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        names = {t.name for t in info.function_tools}
        tool_lists.append(names)
        if "search_paper" in names:
            # Emit 5 search_paper calls per turn so 4 turns reach the
            # 20-call budget without tripping the request_limit.
            parts: list[ToolCallPart] = []
            for _ in range(5):
                call_id["n"] += 1
                parts.append(
                    ToolCallPart(
                        tool_name="search_paper",
                        args={"query": "the masking procedure"},
                        tool_call_id=f"c{call_id['n']}",
                    )
                )
            return ModelResponse(parts=parts)
        return ModelResponse(
            parts=[
                ToolCallPart(tool_name=info.output_tools[0].name, args=make_paper_extraction_args())
            ]
        )

    result = await agent.run(
        "Extract techniques",
        deps=deps,
        model=FunctionModel(model_fn),
        usage_limits=INGESTION_USAGE_LIMITS,
    )

    assert isinstance(result.output, PaperExtraction)
    assert deps.tool_call_count == INGESTION_FORCED_SYNTHESIS_AFTER
    # The final model request saw no SearchX verbs (only the output exit).
    assert not (_SEARCH_VERBS & tool_lists[-1])
    # Earlier requests did have them.
    assert _SEARCH_VERBS & tool_lists[0]


# === Budget constants =======================================================


def test_ingestion_usage_limits_values() -> None:
    assert INGESTION_USAGE_LIMITS.request_limit == 15
    assert INGESTION_USAGE_LIMITS.tool_calls_limit == 30
    assert INGESTION_USAGE_LIMITS.total_tokens_limit == 200_000
    assert INGESTION_FORCED_SYNTHESIS_AFTER == 20


# === IngestionResult assembly + ScreeningOutcome mapping ====================


def test_ingestion_result_from_extraction_folds_in_screening() -> None:
    """IngestionResult.from_extraction (the Sub-PR B assembly seam) folds
    the agent's PaperExtraction with the carried screening + identity."""
    screening = ScreeningOutcome.from_screen_result(
        ScreenResult(decision="accept", rejection_reason=None, summary="in scope")
    )
    extraction = make_paper_extraction()
    result = IngestionResult.from_extraction(
        extraction,
        arxiv_id="1708.04552",
        paper_title="Cutout",
        paper_abstract="We introduce Cutout.",
        screening=screening,
    )
    assert isinstance(result, IngestionResult)
    assert result.arxiv_id == "1708.04552"
    assert result.screening is not None
    assert result.screening.decision == "accept"
    assert result.paper_level_summary == extraction.paper_summary
    assert len(result.techniques) == 1


def test_screening_outcome_maps_reject() -> None:
    screening = ScreeningOutcome.from_screen_result(
        ScreenResult(decision="reject", rejection_reason="survey paper", summary="out of scope")
    )
    assert screening.decision == "reject"
    assert screening.rejection_reason == "survey paper"


# === Prompt rendering =======================================================


def test_paper_agent_prompt_renders_with_budget_counters() -> None:
    """The initial-user-message template renders against the runtime
    variables with no UndefinedError (field-name parity with the loop)."""
    cfg = load_paper_agent_prompt()
    rendered = render_initial_user_message(
        cfg,
        paper_arxiv_id="1708.04552",
        section_list="- 3: Method",
        current_section_latex="we mask square regions",
        pool_summary="(empty)",
        current_tool_call_count=0,
        tool_call_cap=30,
        search_tools_available="search_paper, search_section, search_literature",
        time_pressure_threshold=20,
    )
    assert "1708.04552" in rendered
    assert "Round 0/30" in rendered


def test_paper_agent_prompt_time_pressure_block() -> None:
    cfg = load_paper_agent_prompt()
    rendered = render_initial_user_message(
        cfg,
        paper_arxiv_id="x",
        section_list="- 3: Method",
        current_section_latex="body",
        pool_summary="(empty)",
        current_tool_call_count=25,
        tool_call_cap=30,
        search_tools_available="emit_ingestion_result",
        time_pressure_threshold=20,
    )
    assert "past the time-pressure threshold" in rendered
