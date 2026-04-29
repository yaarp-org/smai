"""Comparison-technique enricher single-call agent — ``08`` §5.6 / ``04-agents.md`` §6.

Per ``04-agents.md`` §6: single-call structured-output agent that reads
a cited source paper and produces a focused method-extraction for one
specific technique. Per ``08-novel-technique-pipeline.md`` §5.6 the
enricher fires from the paper-ingestion pipeline-spec's ``planning``
state's dispatch handler (Task 3.E2), once per ``draft_ensure_technique``
skeleton in the planner buffer that points at a non-standard cited
source paper.

**Single-call vs. multi-turn judgment.** v1's enricher (per
``designs/yaarp/paper_ingestion.md`` §6) was traditionally a multi-step
operation across a longer source-paper context. The brief explicitly
asks the implementer to surface this judgment. v2 ships single-call:
the input is bounded (one technique's worth of source-paper context),
the output is bounded (one ``method_extraction.json``), and the failure
mode is graceful (``implementability="blocked"`` is a first-class
outcome). Multi-turn discipline would add complexity without changing
what the planner stage downstream needs. If a deployment surfaces an
enricher prompt that is reliably running over its single-call window,
the natural follow-up is to introduce an ``enricher`` multi-turn
variant — the schema and dispatch site are stable.

Per DEC-032 paper ingestion produces ``TechniqueRef``s only — the
enricher's job is to make the cited technique implementable from a
spec the registration step can persist; it does NOT decide CG-creation
worthiness.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from smai_core.plugins import LlmProvider

from smai_agents.prompts import PromptConfig, load_prompt_config
from smai_agents.schemas.enricher import EnrichmentResult
from smai_agents.structured_call import structured_call

_ENRICHER_ROLE = "enricher"


class EnricherInput(BaseModel):
    """Assembled inputs to :func:`run_technique_enrichment`.

    The ``planning`` dispatch handler builds one of these per skeleton
    in the planner buffer that needs enrichment, populates ``source_paper_text``
    from the source paper's extracted-text artifact, and invokes the
    enricher once per technique.
    """

    model_config = ConfigDict(extra="forbid")

    technique_id: str
    technique_name: str
    technique_description: str
    """The planner's draft description for this technique. Often
    cursory — the enricher refines via :attr:`EnrichmentResult.refined_description`
    based on the actual source paper."""

    citing_paper_arxiv_id: str
    """The arXiv id of the paper that referenced this technique in
    its comparison set. Surfaced for context — the enricher knows
    which paper surfaced the technique, in case the description in
    the citing paper itself is informative."""

    source_paper_arxiv_id: str
    source_paper_text: str
    """Extracted text of the source paper that introduced the technique.
    Bounded by the dispatch handler — typically the body of
    ``papers/{source_arxiv_id}/extracted/paper_text.json``. The
    enricher does not need the full paper for one technique's
    extraction; the dispatch handler may truncate to a budget."""


async def run_technique_enrichment(
    *,
    llm: LlmProvider,
    input: EnricherInput,
    prompt_config: PromptConfig | None = None,
) -> EnrichmentResult:
    """Run the enricher agent for one technique.

    Single-call structured output via :func:`structured_call`; no
    multi-turn loop (see module docstring for the single-call vs.
    multi-turn judgment). Steps:

    1. Resolve ``prompt_config`` — caller-supplied if present,
       otherwise :func:`smai_agents.prompts.load_prompt_config` is
       called with ``role="enricher"`` and no variant.
    2. Read the structured-output tool's ``name`` + ``description``
       from :attr:`PromptConfig.structured_output_tool` (§10.2).
    3. Render the user message from ``input``.
    4. Call :func:`structured_call` with
       ``output_schema=EnrichmentResult``.
    5. Return the parsed enrichment.

    Inherits :func:`structured_call`'s discipline: retry-once on
    text-instead-of-tool-call or schema-invalid response per DEC-018,
    :class:`smai_agents.StructuredCallFailed` on second-attempt failure
    (no silent fallback).
    """
    cfg = prompt_config if prompt_config is not None else load_prompt_config(role=_ENRICHER_ROLE)
    sot = cfg.structured_output_tool
    if sot is None:
        raise ValueError(
            "enricher prompt config is missing structured_output_tool; single-call "
            "agents require it per 04-agents.md §10.2"
        )
    user_message = _build_user_message(input)
    return await structured_call(
        llm=llm,
        system=cfg.system_prompt,
        user_message=user_message,
        output_schema=EnrichmentResult,
        tool_name=sot.name,
        tool_description=sot.description,
    )


def _build_user_message(inp: EnricherInput) -> str:
    """Render ``inp`` into the LLM's user-message text.

    Mirrors the single-call agent shape used elsewhere in the fleet.
    """
    parts: list[str] = []
    parts.append(f"# Technique Enrichment: {inp.technique_id}")
    parts.append("## Technique")
    parts.append(f"- name: {inp.technique_name}")
    parts.append(f"- planner draft description: {inp.technique_description}")
    parts.append("")
    parts.append("## Citing Paper")
    parts.append(f"- arxiv id: {inp.citing_paper_arxiv_id}")
    parts.append("")
    parts.append("## Source Paper")
    parts.append(f"- arxiv id: {inp.source_paper_arxiv_id}")
    parts.append("")
    parts.append("## Source Paper Text (extracted)")
    parts.append(inp.source_paper_text)
    return "\n".join(parts)


__all__ = ["EnricherInput", "run_technique_enrichment"]
