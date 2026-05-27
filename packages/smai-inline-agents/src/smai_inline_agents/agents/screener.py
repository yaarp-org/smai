"""Paper screener single-call agent — ``08`` §5.2 / ``04-agents.md`` §6.

Per ``04-agents.md`` §6: single-call structured-output agent that
classifies an arXiv paper as in-scope / out-of-scope for the SMAI
technique pool. Per ``08-novel-technique-pipeline.md`` §5.2 the
screener fires from the paper-ingestion pipeline-spec's ``screening``
state on-entry handler; the next dispatch-time gate body reads the
returned :class:`ScreenResult.decision` and routes to ``planning`` (on
``accept``) or ``rejected`` (on ``reject``).

Per DEC-032 paper ingestion is a **supporting utility** that produces
``TechniqueRef``s only — NOT ``ExperimentDefinition``s,
``ComparisonGroupRecord``s, or ``EntryRecord``s. The screener's
question is "should this paper feed the technique pool?", not "should
any CG be created?" — CGs are exclusively proposal-born. Reject
verdicts are therefore comparatively cheap (a paper that turns out to
be CG-worthy can be re-ingested as a manual operator action; the
screener exists to prune obviously non-comparative papers, not to
adjudicate their CG-creation worthiness).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from smai_core.plugins import LlmProvider

from smai_inline_agents.prompts import PromptConfig, load_prompt_config
from smai_inline_agents.schemas.screener import ScreenResult
from smai_inline_agents.structured_call import structured_call

_SCREENER_ROLE = "screener"


class ScreenerInput(BaseModel):
    """Assembled inputs to :func:`run_paper_screening`.

    Carries the paper identity + extracted text the screener needs to
    classify in/out of scope. The dispatch handler in Task 3.E2's
    ``screening`` state populates this from ``ArtifactStore`` reads
    against the ``papers/{arxiv_id}/extracted/...`` keys produced by
    the prior ``fetching`` stage.
    """

    model_config = ConfigDict(extra="forbid")

    arxiv_id: str
    title: str | None = None
    abstract: str | None = None
    paper_text: str
    """Extracted paper text — typically the body of
    ``papers/{arxiv_id}/extracted/paper_text.json``. The screener gets
    enough text to make an empirical-comparative-claims judgment. The
    dispatch handler is welcome to truncate to a budget (the screener
    does not need the full paper to decide on scope; the planner does)."""


async def run_paper_screening(
    *,
    llm: LlmProvider,
    input: ScreenerInput,
    prompt_config: PromptConfig | None = None,
) -> ScreenResult:
    """Run the screener agent for one paper.

    Single-call structured output via :func:`structured_call`; no
    multi-turn loop. Steps:

    1. Resolve ``prompt_config`` — caller-supplied if present,
       otherwise :func:`smai_inline_agents.prompts.load_prompt_config` is
       called with ``role="screener"`` and no variant (the screener
       has only the base prompt; per ``04-agents.md`` §10.2 a single-
       call agent ships only the base layer unless deployment iteration
       surfaces the need for a variant).
    2. Read the structured-output tool's ``name`` + ``description``
       from :attr:`PromptConfig.structured_output_tool` (§10.2).
    3. Render the user message from ``input``.
    4. Call :func:`structured_call` with ``output_schema=ScreenResult``.
    5. Return the parsed verdict.

    Inherits :func:`structured_call`'s discipline: retry-once on
    text-instead-of-tool-call or schema-invalid response per DEC-018,
    :class:`smai_inline_agents.StructuredCallFailed` on second-attempt failure
    (no silent fallback).
    """
    cfg = prompt_config if prompt_config is not None else load_prompt_config(role=_SCREENER_ROLE)
    sot = cfg.structured_output_tool
    if sot is None:
        raise ValueError(
            "screener prompt config is missing structured_output_tool; single-call "
            "agents require it per 04-agents.md §10.2"
        )
    user_message = _build_user_message(input)
    return await structured_call(
        llm=llm,
        system=cfg.system_prompt,
        user_message=user_message,
        output_schema=ScreenResult,
        tool_name=sot.name,
        tool_description=sot.description,
    )


def _build_user_message(inp: ScreenerInput) -> str:
    """Render ``inp`` into the LLM's user-message text.

    Mirrors the shape used by other single-call agents
    (:func:`smai_inline_agents.agents.contextual_evaluator._build_user_message`):
    a structured Markdown body the LLM can scan top-to-bottom without
    needing to fence-strip or pattern-match.
    """
    parts: list[str] = []
    parts.append(f"# Paper Screening: {inp.arxiv_id}")
    if inp.title:
        parts.append("## Title")
        parts.append(inp.title)
    parts.append("")
    if inp.abstract:
        parts.append("## Abstract")
        parts.append(inp.abstract)
        parts.append("")
    parts.append("## Paper Text (extracted)")
    parts.append(inp.paper_text)
    parts.append("")
    parts.append("## Decision Criteria")
    parts.append(
        "Classify the paper as 'accept' if it makes empirical comparative claims "
        "between machine-learning techniques that SMAI's pipeline can evaluate; "
        "'reject' otherwise. Reject reasons should be concrete (e.g., 'pure "
        "theory paper, no empirical comparison', 'survey paper', 'no usable "
        "LaTeX content', or 'subject area outside ML')."
    )
    return "\n".join(parts)


__all__ = ["ScreenerInput", "run_paper_screening"]
