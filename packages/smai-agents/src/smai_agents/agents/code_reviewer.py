"""Code reviewer single-call agent — §2.4 / §11.

Per ``04-agents.md`` §2.4: single-call structured-output agent reviewing
all technique implementations within a CG simultaneously. Per §11 it
runs as the body of an ``implemented → running`` edge gate (DEC-016) —
this module ships the role-shaped wrapper that the gate-rule body in
Task 2.C4 calls. The wrapper itself does NOT touch the engine, the
``MetadataStore``, or the ``ArtifactStore``; it consumes the inputs
(contracts, code, factor type) the gate-rule body assembled and
produces the structured :class:`CodeReviewResult`.

Factor-type-aware behavior per DEC-017:

* ``additive`` baselines (``technique_id is None`` and ``is_baseline``
  per DEC-013) have no code to review — the harness no-op IS the
  baseline. The user-message builder filters them out and explicitly
  flags the omission so the LLM doesn't hallucinate findings about
  absent code.
* ``substitutive`` baselines have real technique modules and are
  reviewed normally.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict
from smai_core.artifacts import HarnessContract, TechniqueContract
from smai_core.plugins import LlmProvider

from smai_agents.prompts import PromptConfig, load_prompt_config
from smai_agents.schemas.code_review import CodeReviewResult
from smai_agents.structured_call import structured_call


class EntryUnderReview(BaseModel):
    """One entry's review-bundle: identity + contract + code.

    The gate-rule body (Task 2.C4) assembles these from
    ``MetadataStore`` (entry / technique identity, ``is_baseline``
    flag) + ``ArtifactStore`` (technique-module source + per-entry
    :class:`TechniqueContract`).
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    entry_id: str
    technique_id: str | None
    technique_name: str
    technique_description: str
    is_baseline: bool
    technique_contract: TechniqueContract
    code: str | None
    """Source of ``techniques/<name>.py``; ``None`` for ``additive`` baselines.

    The user-message builder treats ``None`` and the empty string
    identically — both indicate "no code to review for this entry."
    """


class CodeReviewerInput(BaseModel):
    """Assembled inputs to :func:`run_code_review` (gate-body's view).

    The fields mirror §2.4's "Inputs" paragraph: factor type,
    HarnessContract, per-entry TechniqueContracts (carried inside
    :class:`EntryUnderReview`), entries' technique-module code, and
    harness code. The gate-rule body — Task 2.C4 — populates this from
    ``MetadataStore`` + ``ArtifactStore``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    cg_id: str
    factor_type: Literal["additive", "substitutive"]
    factor_dimension: str
    harness_contract: HarnessContract
    harness_code: dict[str, str]
    """Path → source for harness-side files (``harness/*.py`` typically)."""

    entries: list[EntryUnderReview]


async def run_code_review(
    *,
    llm: LlmProvider,
    input: CodeReviewerInput,
    prompt_config: PromptConfig | None = None,
) -> CodeReviewResult:
    """Run the code-review agent for one CG.

    Single-call structured output via :func:`structured_call`; no
    multi-turn loop. The function is purely a wrapper:

    1. Assemble the user message from ``input``.
    2. Resolve ``prompt_config`` — caller-supplied or, when ``None``,
       :func:`smai_agents.prompts.load_prompt_config` for the
       ``code_reviewer`` role (per ``04-agents.md`` §10).
    3. Read the structured-output tool's ``name`` + ``description``
       from :attr:`PromptConfig.structured_output_tool` (§10.2).
    4. Call :func:`structured_call` with
       ``output_schema=CodeReviewResult``.
    5. Return the parsed verdict.

    The function inherits :func:`structured_call`'s discipline:
    retry-once-on-text-or-schema-invalid response per DEC-018, and
    :class:`smai_agents.StructuredCallFailed` on second-attempt
    failure (no silent fallback). The
    :class:`smai_agents.schemas.code_review.CodeReviewResult` validator
    enforces the ``overall_pass`` ↔ critical-findings invariant.
    """
    cfg = (
        prompt_config
        if prompt_config is not None
        else load_prompt_config(
            role="code_reviewer",
        )
    )
    sot = cfg.structured_output_tool
    if sot is None:
        raise ValueError(
            "code_reviewer prompt config is missing structured_output_tool; "
            "single-call agents require it per 04-agents.md §10.2"
        )
    user_message = _build_user_message(input)

    return await structured_call(
        llm=llm,
        system=cfg.system_prompt,
        user_message=user_message,
        output_schema=CodeReviewResult,
        tool_name=sot.name,
        tool_description=sot.description,
    )


def _build_user_message(inp: CodeReviewerInput) -> str:
    """Render ``inp`` into the LLM's user-message text.

    The structure mirrors v1's :file:`packages/agents/src/code-review.ts`
    (cg id + factor framing + harness code + per-entry sections), with
    v2-shaped contract excerpts (``HarnessContract.body`` /
    ``TechniqueContract.body``) replacing v1's ad-hoc sketch.

    Factor-type-aware filtering (DEC-017): additive baselines with
    ``code is None`` (or empty) are explicitly flagged as "no code to
    review" rather than omitted silently — the LLM should see the
    accounting so it doesn't manufacture findings.
    """
    parts: list[str] = []
    parts.append(f"# Code Review for CG: {inp.cg_id}")
    parts.append("")
    parts.append("## Factor")
    parts.append(f"- dimension: {inp.factor_dimension}")
    parts.append(f"- factor_type: {inp.factor_type}")
    parts.append("")
    parts.append("## HarnessContract (locked)")
    parts.append("```json")
    parts.append(inp.harness_contract.body.model_dump_json(indent=2))
    parts.append("```")
    parts.append("")
    parts.append("## Harness Code")
    if not inp.harness_code:
        parts.append("(no harness sources provided)")
    else:
        for path in sorted(inp.harness_code):
            parts.append(f"### {path}")
            parts.append("```python")
            parts.append(inp.harness_code[path])
            parts.append("```")
    parts.append("")
    parts.append("## Entries")
    if not inp.entries:
        parts.append("(no entries — no review possible)")
    for entry in inp.entries:
        parts.append("")
        baseline_tag = " [BASELINE]" if entry.is_baseline else ""
        parts.append(
            f"### {entry.technique_name}{baseline_tag} (entry_id={entry.entry_id}, "
            f"technique_id={entry.technique_id})"
        )
        parts.append(f"- description: {entry.technique_description}")
        parts.append("")
        parts.append("TechniqueContract (locked):")
        parts.append("```json")
        parts.append(entry.technique_contract.body.model_dump_json(indent=2))
        parts.append("```")
        if _is_additive_baseline_with_no_code(inp.factor_type, entry):
            parts.append("")
            parts.append(
                "**No code to review for this entry.** This is an "
                "``additive`` baseline (DEC-013): the harness's no-op "
                "default IS the baseline. Do not return findings against "
                "this entry."
            )
        elif not entry.code:
            parts.append("")
            parts.append(
                "**Implementation source is missing.** This is unusual "
                "for a non-baseline / substitutive entry; flag as "
                '``critical`` (target_kind="entry") if the absence is '
                "unexpected."
            )
        else:
            parts.append("")
            parts.append("Implementation:")
            parts.append("```python")
            parts.append(entry.code)
            parts.append("```")
    return "\n".join(parts)


def _is_additive_baseline_with_no_code(
    factor_type: Literal["additive", "substitutive"],
    entry: EntryUnderReview,
) -> bool:
    """DEC-013: ``additive`` baseline ``technique_id is None`` ⇒ no code."""
    return (
        factor_type == "additive"
        and entry.is_baseline
        and entry.technique_id is None
        and not entry.code
    )


__all__ = [
    "CodeReviewerInput",
    "EntryUnderReview",
    "run_code_review",
]
