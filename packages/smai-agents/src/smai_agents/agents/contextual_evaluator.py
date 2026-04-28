"""Contextual evaluator single-call agent — §2.5 / DEC-034 #2.

Per ``04-agents.md`` §2.5: single-call structured-output agent that
consumes the mechanical evaluator's :class:`EvaluationResult`
(``Verdict`` + ``VerdictContext``) and produces the narrative
:class:`ContextualVerdict`. Per DEC-034 #2 the agent is invoked from
the single ``evaluating`` state's dispatch handler, after the
mechanical evaluator runs — Task 2.C4 wires that sequencing; this
module ships the role-shaped wrapper.

The contextual evaluator does NOT regenerate verdict numbers — it
consumes the deterministic ``verdict_context`` (per-seed values,
statistical summary, anomalies, cost telemetry, trend observations)
and produces the English narrative. v1's ``evaluationMode`` framing
(reproduction vs. discovery) is replaced by the
methodology-layer ``ComparisonRule`` per DEC-031 #8: the
``compare_to_baseline`` and ``compare_to_target`` variants drive
**only** prompt-config selection (the schema is rule-agnostic).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict
from smai_core.evaluation import EvaluationResult
from smai_core.plugins import LlmProvider

from smai_agents.prompts import PromptConfig, load_prompt_config
from smai_agents.schemas.contextual_verdict import ContextualVerdict
from smai_agents.structured_call import structured_call


class ContextualEvaluatorEntry(BaseModel):
    """Per-entry identity bundle the user message renders inline.

    Mirrors v1's ``ContextualEvaluatorInput.entries`` shape: the
    information the LLM needs to write per-entry rationale beyond
    what's in :class:`EvaluationResult` (entry_id, technique name,
    description, baseline flag).
    """

    model_config = ConfigDict(extra="forbid")

    entry_id: str
    technique_id: str | None
    technique_name: str
    technique_description: str
    is_baseline: bool


class CGMetadata(BaseModel):
    """The CG metadata bundle assembled by the dispatch handler.

    Per §2.5 "Inputs": factor (dimension + type), entries, controlled
    conditions. The dispatch handler in Task 2.C4 populates this from
    ``MetadataStore``; we accept it as a typed bundle so the wrapper
    stays substrate-agnostic (no direct ``MetadataStore`` reads).
    """

    model_config = ConfigDict(extra="forbid")

    cg_id: str
    cg_name: str
    cg_description: str
    factor_dimension: str
    factor_type: Literal["additive", "substitutive"]
    controlled_conditions: list[str]
    """One human-readable line per controlled condition (e.g.,
    ``"dataset: CIFAR-10"`` — flat strings rather than a structured
    record because the LLM only needs the text)."""

    entries: list[ContextualEvaluatorEntry]


class ContextualEvaluatorInput(BaseModel):
    """Assembled inputs to :func:`run_contextual_evaluation`.

    Mirrors §2.5's "Inputs": the CG metadata (factor, entries,
    controlled conditions), the full :class:`EvaluationResult`
    (verdict + verdict_context), the comparison rule, and the optional
    comparison rationale (``compare_to_baseline`` only — captured at
    planning time).
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    cg_metadata: CGMetadata
    evaluation_result: EvaluationResult
    comparison_rule: Literal["compare_to_baseline", "compare_to_target"]
    comparison_rationale: str | None = None
    """Optional, ``compare_to_baseline``-only: why this baseline was chosen
    and what direction of effect was hypothesized. Captured by the
    planner / experiment-designer at planning time and surfaced here so
    the contextual evaluator can ground its narrative in the original
    intent."""


async def run_contextual_evaluation(
    *,
    llm: LlmProvider,
    input: ContextualEvaluatorInput,
    prompt_config: PromptConfig | None = None,
) -> ContextualVerdict:
    """Run the contextual-evaluator agent for one CG.

    Single-call structured output via :func:`structured_call`; no
    multi-turn loop. Steps:

    1. Resolve ``prompt_config`` — caller-supplied if present,
       otherwise :func:`smai_agents.prompts.load_prompt_config` is
       called with ``role="contextual_evaluator"`` and
       ``variant_name=input.comparison_rule`` (per ``04-agents.md``
       §2.5 / DEC-031 #8: variant selection is keyed by comparison
       rule; the schema is rule-agnostic).
    2. Read the structured-output tool's ``name`` + ``description``
       from :attr:`PromptConfig.structured_output_tool` (§10.2).
    3. Render the user message from ``input``.
    4. Call :func:`structured_call` with
       ``output_schema=ContextualVerdict``.
    5. Return the parsed verdict.

    The function inherits :func:`structured_call`'s discipline:
    retry-once-on-text-or-schema-invalid response per DEC-018, and
    :class:`smai_agents.StructuredCallFailed` on second-attempt
    failure (no silent fallback).
    """
    cfg = (
        prompt_config
        if prompt_config is not None
        else load_prompt_config(
            role="contextual_evaluator",
            variant_name=input.comparison_rule,
        )
    )
    sot = cfg.structured_output_tool
    if sot is None:
        raise ValueError(
            "contextual_evaluator prompt config is missing structured_output_tool; "
            "single-call agents require it per 04-agents.md §10.2"
        )
    user_message = _build_user_message(input)

    return await structured_call(
        llm=llm,
        system=cfg.system_prompt,
        user_message=user_message,
        output_schema=ContextualVerdict,
        tool_name=sot.name,
        tool_description=sot.description,
    )


def _build_user_message(inp: ContextualEvaluatorInput) -> str:
    """Render ``inp`` into the LLM's user-message text.

    Follows v1's :file:`packages/agents/src/contextual-evaluator.ts`
    structure (cg framing + factor + controlled conditions +
    mechanical-result excerpt + per-entry detail), with v2-shaped
    :class:`EvaluationResult` excerpts (``Verdict`` and
    ``VerdictContext`` together) replacing v1's ad-hoc shape.
    """
    md = inp.cg_metadata
    er = inp.evaluation_result

    parts: list[str] = []
    parts.append(f"# Contextual Evaluation for CG: {md.cg_id}")
    parts.append(f"## Experiment: {md.cg_name}")
    parts.append(md.cg_description)
    parts.append("")
    parts.append(f"## Comparison Rule: {inp.comparison_rule}")
    if inp.comparison_rationale:
        parts.append("### Comparison Rationale")
        parts.append(inp.comparison_rationale)
    parts.append("")
    parts.append("## Experimental Factor")
    parts.append(f"- dimension: {md.factor_dimension}")
    parts.append(f"- factor_type: {md.factor_type}")
    parts.append("")
    parts.append("## Controlled Conditions")
    if not md.controlled_conditions:
        parts.append("(none recorded)")
    else:
        for line in md.controlled_conditions:
            parts.append(f"- {line}")
    parts.append("")
    parts.append("## Mechanical Verdict (locked — do not regenerate)")
    parts.append("```json")
    parts.append(er.verdict.model_dump_json(indent=2))
    parts.append("```")
    parts.append("")
    parts.append("## Verdict Context (deterministic narrative-grounding)")
    parts.append("```json")
    parts.append(er.verdict_context.model_dump_json(indent=2))
    parts.append("```")
    parts.append("")
    parts.append("## Entries")
    for entry in md.entries:
        baseline_tag = " [BASELINE]" if entry.is_baseline else ""
        parts.append(
            f"- {entry.technique_name}{baseline_tag} "
            f"(entry_id={entry.entry_id}, technique_id={entry.technique_id})"
        )
        parts.append(f"  - {entry.technique_description}")
    return "\n".join(parts)


__all__ = [
    "CGMetadata",
    "ContextualEvaluatorEntry",
    "ContextualEvaluatorInput",
    "run_contextual_evaluation",
]
