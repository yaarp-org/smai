"""The ``run_ingestion_subagent`` orchestration wrapper (Step 3, Sub-PR B).

Fans the fetch -> (optionally pre-screened) screener gate -> build deps
-> Paper Agent run -> :class:`IngestionResult` assembly per
``planner_refactor/design_notes/ingestion_subagent.md`` §4 / §6. Sub-PR
A shipped the library (schemas, fetch preprocessing, the Paper Agent +
its three retrieval tools, the prompt, the budget); this wrapper is the
host-side seam that drives them and is the single code path shared by:

* ``smai ingest`` (the paper-ingestion pipeline's ``planning`` state
  passes the screening-state verdict so the paper is screened once).
* The ``search_literature`` deep-extraction mode (design note §4 mode
  2), which recurses on a cited paper with a depth-bumped factory and
  rolls subagent token spend under the outer cap (``usage=ctx.usage``).
* (future, step 8) the planner's literature-review delegation.

Screening double-call resolution (design note §4 vs §7): the wrapper
screens internally ONLY when ``screening`` is not supplied. The pipeline
path passes its screening-state result, so the paper is never screened
twice; the recursion / standalone paths pass ``screening=None`` and
screen internally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from smai_inline_agents.agents.screener import ScreenerInput, run_paper_screening
from smai_inline_agents.ingestion.paper_agent import (
    INGESTION_FORCED_SYNTHESIS_AFTER,
    INGESTION_USAGE_LIMITS,
    build_paper_agent,
    load_paper_agent_prompt,
)
from smai_inline_agents.ingestion.schemas import (
    IngestionResult,
    PaperAgentDepsFactory,
    ScreeningOutcome,
    SectionRef,
)
from smai_inline_agents.prompts import render_initial_user_message

if TYPE_CHECKING:
    from pydantic_ai.usage import RunUsage


async def run_ingestion_subagent(
    arxiv_id: str,
    deps_factory: PaperAgentDepsFactory,
    *,
    screening: ScreeningOutcome | None = None,
    usage: RunUsage | None = None,
) -> IngestionResult:
    """Run one paper-ingestion extraction end-to-end (design note §4 / §6).

    Flow:

    1. Fetch the paper corpus via ``deps_factory.corpus_fetcher``. On
       ``None`` (every fetch path failed), return an empty result with
       ``error_reason="fetch_failed"`` WITHOUT calling the LLM.
    2. Screening. When ``screening`` is supplied (the pipeline path that
       already screened in the ``screening`` state) it is used verbatim
       and the screener is NOT re-run. When ``screening is None`` (the
       recursion / standalone path) the screener runs here against
       ``deps_factory.screener_llm``. Either way a ``reject`` verdict
       short-circuits to an empty result carrying the verdict +
       ``error_reason``.
    3. Build :class:`PaperAgentDeps` from the corpus and run the Paper
       Agent under :data:`INGESTION_USAGE_LIMITS` (rolling ``usage`` up
       for recursion-budget accounting).
    4. Fold the agent's :class:`PaperExtraction` into an
       :class:`IngestionResult` and splice in any techniques discovered
       by ``search_literature``'s deep-extraction recursion.
    """
    corpus = await deps_factory.corpus_fetcher(arxiv_id)
    if corpus is None:
        return IngestionResult(arxiv_id=arxiv_id, techniques=[], error_reason="fetch_failed")

    effective_screening = screening
    if effective_screening is None:
        if deps_factory.screener_llm is None:
            raise ValueError(
                "run_ingestion_subagent called without a pre-computed screening and "
                "deps_factory.screener_llm is None; supply one or the other"
            )
        screen = await run_paper_screening(
            llm=deps_factory.screener_llm,
            input=ScreenerInput(
                arxiv_id=arxiv_id,
                title=corpus.title or None,
                abstract=corpus.abstract or None,
                paper_text=corpus.full_latex,
            ),
        )
        effective_screening = ScreeningOutcome.from_screen_result(screen)

    if effective_screening.decision == "reject":
        return IngestionResult(
            arxiv_id=arxiv_id,
            paper_title=corpus.title,
            paper_abstract=corpus.abstract,
            screening=effective_screening,
            techniques=[],
            error_reason=effective_screening.rejection_reason or "screened out as out-of-scope",
        )

    deps = deps_factory.build(corpus=corpus)
    agent = build_paper_agent()
    user_message = render_initial_user_message(
        load_paper_agent_prompt(),
        paper_arxiv_id=arxiv_id,
        paper_title=corpus.title or "(title unavailable)",
        paper_abstract=corpus.abstract or "(abstract unavailable)",
        section_list=_format_section_list(corpus.sections),
        current_section_latex=corpus.full_latex,
        pool_summary=deps_factory.pool_snapshot.summary_markdown,
        current_tool_call_count=0,
        tool_call_cap=INGESTION_USAGE_LIMITS.tool_calls_limit,
        search_tools_available="search_paper, search_section, search_literature",
        time_pressure_threshold=INGESTION_FORCED_SYNTHESIS_AFTER,
    )
    run_result = await agent.run(
        user_message,
        deps=deps,
        usage_limits=INGESTION_USAGE_LIMITS,
        usage=usage,
        model=deps_factory.model,
    )
    result = IngestionResult.from_extraction(
        run_result.output,
        arxiv_id=arxiv_id,
        paper_title=corpus.title,
        paper_abstract=corpus.abstract,
        screening=effective_screening,
    )
    # Splice in techniques discovered by search_literature deep recursion
    # (design note §4 mode 2). model_copy avoids re-validating the
    # spliced list against techniques' max_length cap.
    if deps.enriched_techniques:
        result = result.model_copy(
            update={"techniques": [*result.techniques, *deps.enriched_techniques]}
        )
    return result


def _format_section_list(sections: list[SectionRef]) -> str:
    """Render the parsed section list as the prompt's bullet block."""
    if not sections:
        return "(no sections parsed from the source)"
    indent = {1: "", 2: "  ", 3: "    ", 4: "      ", 5: "        ", 6: "          "}
    return "\n".join(f"{indent.get(s.depth, '')}- {s.section_id}: {s.title}" for s in sections)


def summarize_recursive_result(arxiv_id: str, result: IngestionResult) -> str:
    """Format a ``search_literature`` deep-extraction return for the agent.

    The cited paper's extracted techniques are spliced into the outer
    paper's pool (host-side); this string tells the agent which
    technique names became available so it can reference them.
    """
    if not result.techniques:
        reason = result.error_reason or "no comparison-relevant technique found"
        return (
            f"Deep extraction of arXiv {arxiv_id} produced no technique "
            f"({reason}). Source: Relevant Literature."
        )
    names = ", ".join(t.name for t in result.techniques)
    return (
        f"Deep extraction of arXiv {arxiv_id} produced "
        f"{len(result.techniques)} technique(s): {names}. These have been "
        "added to this paper's technique pool; reference them by name. "
        "Source: Relevant Literature."
    )


__all__ = ["run_ingestion_subagent", "summarize_recursive_result"]
