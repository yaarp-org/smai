"""The SciReplicate-shaped paper-ingestion Paper Agent (Step 3, Sub-PR A).

A PydanticAI ``Agent[PaperAgentDeps, PaperExtraction]`` with the three
SciReplicate retrieval verbs (``search_paper`` / ``search_section`` /
``search_literature``) plus the implicit ``output_type`` exit. The agent
reads a pre-fetched :class:`PaperCorpus` (via deps) and emits a
:class:`PaperExtraction`; the Sub-PR B wrapper folds that into an
:class:`IngestionResult`.

Budget control (design note §5 / architectural_decisions §11):

* :data:`INGESTION_USAGE_LIMITS` -- PydanticAI hard ceilings.
* :data:`INGESTION_FORCED_SYNTHESIS_AFTER` -- after this many retrieval
  tool calls, the per-tool ``prepare`` callback drops the three SearchX
  verbs, leaving only the output-tool exit (the SciReplicate force-emit
  pattern lifted into PydanticAI's tool-prepare API).

Source-grounding discipline (design note §8) lives in the system prompt
(``prompts/ingestion/paper_agent.yaml``); the retrieval sub-extractions
use the homegrown :func:`smai_inline_agents.structured_call.structured_call`
pattern with the LLM carried in :attr:`PaperAgentDeps.sub_extraction_llm`.
"""

from __future__ import annotations

import difflib
from importlib import resources
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, RunContext, ToolOutput
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import UsageLimits

from smai_inline_agents.ingestion.schemas import (
    PaperAgentDeps,
    PaperExtraction,
    SectionRef,
)
from smai_inline_agents.model_selection import get_model_for_task
from smai_inline_agents.prompts import (
    PromptConfig,
    PromptConfigValidationError,
    PromptLayer,
)
from smai_inline_agents.structured_call import structured_call

if TYPE_CHECKING:
    from pydantic_ai.usage import RunUsage
    from smai_core.plugins import LlmProvider

# ---------------------------------------------------------------------------
# Budget (design note §5).
# ---------------------------------------------------------------------------

INGESTION_USAGE_LIMITS = UsageLimits(
    request_limit=15,
    tool_calls_limit=30,
    total_tokens_limit=200_000,
)
"""Hard ceilings for one paper extraction (design note §5)."""

INGESTION_FORCED_SYNTHESIS_AFTER = 20
"""Tool-call count after which the SearchX verbs are dropped from the
toolkit, forcing the agent to synthesize from what it has."""

_NO_INFO_SENTINEL = "No relevant information found."

_PROMPT_PACKAGE = "smai_inline_agents.prompts.ingestion"
_PROMPT_RESOURCE = "paper_agent.yaml"


# ---------------------------------------------------------------------------
# Retrieval sub-extraction (the in-tool structured-call grounding).
# ---------------------------------------------------------------------------


class _RetrievedSnippets(BaseModel):
    """Output of a retrieval sub-extraction (SciReplicate's retrieve-concept).

    Internal to the retrieval tools: ``found=False`` maps to the
    "No relevant information found." sentinel; otherwise ``snippets``
    carries the verbatim quotations to surface to the agent.
    """

    model_config = ConfigDict(extra="forbid")

    found: bool
    snippets: list[str] = Field(default_factory=list[str], max_length=10)


# SciReplicate's ``condtruct_prompt_for_retrieve_Concept`` adapted: a
# focused verbatim-snippet retrieval over a body of LaTeX. The
# "maintain exact text and LaTeX formatting" discipline (design note §8)
# is load-bearing -- the downstream implementer reads these snippets as
# ground truth.
_RETRIEVAL_SUBPROMPT = (
    "You are a retrieval assistant for a paper-ingestion agent. Given a "
    "QUERY and a body of paper SOURCE TEXT (LaTeX), find the passages that "
    "answer the query. MAINTAIN EXACT TEXT AND LATEX FORMATTING -- do not "
    "paraphrase, summarize, or add commentary. Return each relevant passage "
    "as a verbatim snippet. If the source text does not contain information "
    "answering the query, set found=false and return an empty snippet list. "
    "Do not fabricate."
)


async def _run_retrieval_subextraction(
    *,
    llm: LlmProvider,
    source_text: str,
    query: str,
    source_label: str,
) -> str:
    """Run one structured-call sub-extraction and format the result.

    Returns the verbatim snippets tagged with ``source_label`` (the
    SciReplicate ``Source: Target Paper | Relevant Literature``
    discipline), or the "No relevant information found." sentinel.
    """
    result = await structured_call(
        llm=llm,
        system=_RETRIEVAL_SUBPROMPT,
        user_message=f"QUERY:\n{query}\n\nSOURCE TEXT:\n{source_text}",
        output_schema=_RetrievedSnippets,
        tool_name="emit_snippets",
        tool_description=(
            "Emit the verbatim snippets from the source text that answer the "
            "query, or found=false if none."
        ),
    )
    if not result.found or not result.snippets:
        return _NO_INFO_SENTINEL
    body = "\n\n".join(f"[{i + 1}] {snippet}" for i, snippet in enumerate(result.snippets))
    return f"Source: {source_label}\n{body}"


# ---------------------------------------------------------------------------
# Tool implementations (module-level helpers; thin agent wrappers below).
# ---------------------------------------------------------------------------


async def _do_search_paper(deps: PaperAgentDeps, query: str) -> str:
    """Structured-call sub-extraction over the full target-paper LaTeX."""
    return await _run_retrieval_subextraction(
        llm=deps.sub_extraction_llm,
        source_text=deps.full_latex,
        query=query,
        source_label="Target Paper",
    )


def _do_search_section(deps: PaperAgentDeps, title: str) -> str:
    """Fuzzy-title (or exact-id) lookup against the parsed section list."""
    matches = _match_sections(deps.section_list, title)
    if not matches:
        return f"No section matching {title!r}. Available sections: " + ", ".join(
            f"{s.section_id} {s.title!r}" for s in deps.section_list
        )
    if len(matches) > 1:
        listing = "\n".join(f"- {s.section_id}: {s.title}" for s in matches)
        return f"Multiple sections match {title!r}; specify one by id or exact title:\n{listing}"
    section = matches[0]
    body = deps.sections_by_id.get(section.section_id, "")
    return f"Section {section.section_id} ({section.title}):\n{body}"


async def _do_search_literature(
    deps: PaperAgentDeps,
    cite_key: str,
    query: str,
    request_deep_extraction: bool = False,
    *,
    usage: RunUsage | None = None,
) -> str:
    """Cited-paper extraction (design note §4 modes 1 + 2).

    Resolves ``cite_key`` to an arXiv id, then either:

    * **Deep (mode 2):** when ``request_deep_extraction`` is set, the
      recursion budget is not exhausted (``deps.enrichment_depth <
      deps.enrichment_depth_limit``), and a recursion factory is wired
      (``deps.subagent_factory``), recursively invoke
      ``run_ingestion_subagent`` on the cited paper with a depth-bumped
      factory and ``usage`` rolled up under the outer cap. The recursive
      result's techniques splice into ``deps.enriched_techniques`` and a
      summary pointer returns so the agent can reference them by name.
    * **Lightweight (mode 1, the default + the at-depth-limit fallback):**
      fetch the cited paper's corpus via the injected fetcher and run a
      focused sub-extraction for ``query``, tagged ``Source: Relevant
      Literature`` (the precedence rule prefers the target paper).
    """
    ref = deps.cite_resolver.resolve(cite_key)
    if ref is None:
        return f"No paper found for cite key {cite_key!r}."
    if ref.arxiv_id is None:
        return _NO_INFO_SENTINEL
    if (
        request_deep_extraction
        and deps.subagent_factory is not None
        and deps.enrichment_depth < deps.enrichment_depth_limit
    ):
        # Lazy import breaks the run.py <-> paper_agent.py import cycle
        # (run.py imports build_paper_agent from this module).
        from smai_inline_agents.ingestion.run import (  # noqa: PLC0415
            run_ingestion_subagent,
            summarize_recursive_result,
        )

        sub_factory = deps.subagent_factory.for_recursion()
        sub_result = await run_ingestion_subagent(ref.arxiv_id, sub_factory, usage=usage)
        deps.enriched_techniques.extend(sub_result.techniques)
        return summarize_recursive_result(ref.arxiv_id, sub_result)
    cited = await deps.corpus_fetcher(ref.arxiv_id)
    if cited is None:
        return _NO_INFO_SENTINEL
    return await _run_retrieval_subextraction(
        llm=deps.sub_extraction_llm,
        source_text=cited.full_latex,
        query=query,
        source_label="Relevant Literature",
    )


def _match_sections(sections: list[SectionRef], query: str) -> list[SectionRef]:
    """Match ``query`` against sections by id, substring, then fuzzy title."""
    stripped = query.strip()
    by_id = {s.section_id: s for s in sections}
    if stripped in by_id:
        return [by_id[stripped]]
    lowered = stripped.lower()
    substring = [s for s in sections if lowered in s.title.lower() or s.title.lower() in lowered]
    if substring:
        return substring
    titles = [s.title for s in sections]
    close = set(difflib.get_close_matches(stripped, titles, n=5, cutoff=0.6))
    return [s for s in sections if s.title in close]


# ---------------------------------------------------------------------------
# Forced-synthesis prepare callback (design note §5).
# ---------------------------------------------------------------------------


async def _forced_synthesis_prepare(
    ctx: RunContext[PaperAgentDeps],
    tool_def: ToolDefinition,
) -> ToolDefinition | None:
    """Drop the retrieval tool once the tool-call budget is spent.

    Returning ``None`` removes the tool from the toolkit for this model
    request, so after :data:`INGESTION_FORCED_SYNTHESIS_AFTER` retrieval
    calls only the output-tool exit remains.
    """
    if ctx.deps.tool_call_count >= INGESTION_FORCED_SYNTHESIS_AFTER:
        return None
    return tool_def


# ---------------------------------------------------------------------------
# Prompt loading.
# ---------------------------------------------------------------------------


def load_paper_agent_prompt() -> PromptConfig:
    """Load + compose the ``ingestion/paper_agent.yaml`` prompt config.

    The ingestion prompt lives outside the role-keyed
    ``prompts/<role>/base.yaml`` tree (it is a single standalone prompt,
    not a base+variant set), so it gets a dedicated loader that still
    validates against the shared :class:`PromptLayer` schema and yields a
    composed :class:`PromptConfig` the same way the role loader does.
    """
    raw = (resources.files(_PROMPT_PACKAGE) / _PROMPT_RESOURCE).read_text(encoding="utf-8")
    layer = PromptLayer.model_validate(yaml.safe_load(raw))
    if layer.system_prompt is None or layer.initial_user_message_template is None:
        raise PromptConfigValidationError(
            location=f"{_PROMPT_PACKAGE}/{_PROMPT_RESOURCE}",
            detail="ingestion paper_agent prompt must set system_prompt + "
            "initial_user_message_template",
        )
    return PromptConfig(
        role=layer.role,
        system_prompt=layer.system_prompt,
        initial_user_message_template=layer.initial_user_message_template,
        tools=list(layer.tools or []),
        structured_output_tool=layer.structured_output_tool,
        layer_chain=[layer.name],
    )


# ---------------------------------------------------------------------------
# Agent factory.
# ---------------------------------------------------------------------------


def build_paper_agent(
    *,
    provider: str | None = None,
    model_id: str | None = None,
    prompt_config: PromptConfig | None = None,
) -> Agent[PaperAgentDeps, PaperExtraction]:
    """Construct the Paper Agent bound to ``output_type=PaperExtraction``.

    ``provider`` + ``model_id`` form the PydanticAI model spec string
    (``"<provider>:<model_id>"``); when omitted they resolve via
    ``get_model_for_task("ingestion")`` (the ``role_models["ingestion"]``
    config knob, design note §1). ``defer_model_check=True`` so
    construction never instantiates a provider client -- tests run with
    ``model=FunctionModel(...)`` and Sub-PR B passes the resolved
    production model.

    The three SciReplicate retrieval verbs are registered with the
    forced-synthesis ``prepare`` callback; the output-tool exit is always
    available.
    """
    if provider is None or model_id is None:
        resolved_provider, resolved_model = get_model_for_task("ingestion")
        provider = provider or resolved_provider
        model_id = model_id or resolved_model

    cfg = prompt_config if prompt_config is not None else load_paper_agent_prompt()

    agent: Agent[PaperAgentDeps, PaperExtraction] = Agent(
        model=f"{provider}:{model_id}",
        deps_type=PaperAgentDeps,
        # Name the output tool emit_ingestion_result so the YAML prompt's
        # "call emit_ingestion_result" instruction matches the wire tool.
        output_type=ToolOutput(PaperExtraction, name="emit_ingestion_result"),
        system_prompt=cfg.system_prompt,
        defer_model_check=True,
    )

    @agent.tool(prepare=_forced_synthesis_prepare)
    async def search_paper(ctx: RunContext[PaperAgentDeps], query: str) -> str:
        """Search the full paper for a variable / concept / definition.

        Use when a variable, equation symbol, or concept appears in the
        section you are extracting from without a full definition. Runs a
        verbatim-snippet retrieval over the full paper and returns the
        matching snippets, or "No relevant information found." Prefer
        targeted queries (e.g. "the definition of the importance weight
        in equation 4") over vague ones.
        """
        ctx.deps.tool_call_count += 1
        return await _do_search_paper(ctx.deps, query)

    @agent.tool(prepare=_forced_synthesis_prepare)
    async def search_section(ctx: RunContext[PaperAgentDeps], title: str) -> str:
        """Fetch a specific section's raw LaTeX text by title or id.

        Use when the section under extraction references another section
        by name or number. Fuzzy title match; multiple matches return a
        disambiguation list so you can re-call with a precise id.
        """
        ctx.deps.tool_call_count += 1
        return _do_search_section(ctx.deps, title)

    @agent.tool(prepare=_forced_synthesis_prepare)
    async def search_literature(
        ctx: RunContext[PaperAgentDeps],
        cite_key: str,
        query: str,
        request_deep_extraction: bool = False,
    ) -> str:
        r"""Extract information from a cited paper (\\cite{key}).

        Use when the current section cites another paper and you need
        specific information from it to ground a field. Fetches the cited
        paper and returns verbatim snippets tagged "Source: Relevant
        Literature". When the cited paper conflicts with the target
        paper, prefer the target paper. Set request_deep_extraction=True
        only for a full recursive extraction of a comparison-relevant
        cited technique not yet in the pool: that runs the whole
        ingestion subagent on the cited paper (depth-limited) and adds
        its techniques to this paper's pool.
        """
        ctx.deps.tool_call_count += 1
        return await _do_search_literature(
            ctx.deps, cite_key, query, request_deep_extraction, usage=ctx.usage
        )

    # The decorator registers each tool as a side effect; bind to ``_`` so
    # the unused-name lint stays quiet (mirrors smai-agent-runtime).
    _ = (search_paper, search_section, search_literature)
    return agent


__all__ = [
    "INGESTION_FORCED_SYNTHESIS_AFTER",
    "INGESTION_USAGE_LIMITS",
    "build_paper_agent",
    "load_paper_agent_prompt",
]
