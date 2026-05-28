"""Schemas + runtime deps for the ingestion subagent (planner-refactor Step 3).

The SciReplicate-shaped Paper Agent reads a paper's pre-fetched LaTeX
corpus and emits a :class:`PaperExtraction` (one or more
:class:`TechniqueDescription` records plus paper-level prose). The host
wrapper (Sub-PR B) assembles that into an :class:`IngestionResult` by
adding the :class:`ScreeningOutcome` carried from the screener and the
paper-identity metadata it already holds.

Design references:

* ``planner_refactor/design_notes/ingestion_subagent.md`` §1 (agent shape
  / ``PaperAgentDeps``), §3 (``IngestionResult`` is a list-per-invocation),
  §4 (screener carried through as :class:`ScreeningOutcome`).
* ``planner_refactor/design_notes/per_phase_prompts.md`` §5 (the
  ``[Output: IngestionResult]`` block — ``techniques`` /
  ``paper_summary`` / ``extraction_audit``).
* ``planner_refactor/architectural_decisions.md`` §7 (hybrid field
  discipline), §9 (subagent boundary).

The reusable :class:`TechniqueDescription` is imported from
``..planner.schemas`` (it shipped in Step 2); it is NOT redefined here.

Deviation from the design's ``Agent[PaperAgentDeps, IngestionResult]``
notation (documented for Sub-PR B): the agent's ``output_type`` is the
narrower :class:`PaperExtraction` so the LLM never has to echo
host-known fields (``arxiv_id`` / ``screening`` / paper title +
abstract). :class:`IngestionResult` is the host-assembled record the
Sub-PR B wrapper returns; :meth:`IngestionResult.from_extraction` is the
assembly seam.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field
from smai_core.plugins import LlmProvider

from smai_inline_agents.planner.schemas import TechniqueDescription
from smai_inline_agents.schemas.screener import ScreenResult

if TYPE_CHECKING:
    from smai_inline_agents.ingestion.corpus import PaperCorpus


# ---------------------------------------------------------------------------
# Screening verdict carried through from the screener preprocessing step.
# ---------------------------------------------------------------------------


class ScreeningOutcome(BaseModel):
    """The screener's accept/reject verdict, carried into the result.

    Mapped from the existing screener output
    :class:`smai_inline_agents.schemas.screener.ScreenResult` (the
    screener runs as a host-side preprocessor before the Paper Agent, per
    design note §4). Defined fresh here (rather than re-using
    ``ScreenResult`` directly) so :class:`IngestionResult` carries the
    verdict as ingestion-owned data without coupling its wire shape to
    the screener's schema.
    """

    model_config = ConfigDict(extra="forbid")

    decision: Literal["accept", "reject"]
    rejection_reason: str | None = None
    summary: str

    @classmethod
    def from_screen_result(cls, screen: ScreenResult) -> ScreeningOutcome:
        """Map a screener :class:`ScreenResult` into the carried verdict."""
        return cls(
            decision=screen.decision,
            rejection_reason=screen.rejection_reason,
            summary=screen.summary,
        )


# ---------------------------------------------------------------------------
# Paper-structure primitives (section list + cite resolution).
# ---------------------------------------------------------------------------


class SectionRef(BaseModel):
    """One section of the paper, as parsed from the LaTeX source.

    ``section_id`` is the synthesized dotted id (``"1"``, ``"1.1"``,
    ``"2"`` ...) used by ``search_section`` to fetch the raw text from
    :attr:`PaperAgentDeps.sections_by_id`. ``depth`` is 1 for
    ``\\section``, 2 for ``\\subsection``, etc.
    """

    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=400)
    depth: int = Field(ge=1, le=6)


class CitedPaperRef(BaseModel):
    """A resolved ``\\cite{key}`` reference.

    ``arxiv_id`` is populated best-effort by the LaTeX cite resolver
    (parsing the bibliography for an arXiv id); ``None`` when the cite
    key is known but no arXiv id could be recovered — in which case
    ``search_literature`` cannot fetch the cited paper and returns the
    "No relevant information found." sentinel.
    """

    model_config = ConfigDict(extra="forbid")

    cite_key: str = Field(min_length=1, max_length=200)
    arxiv_id: str | None = Field(default=None, max_length=40)
    title: str | None = Field(default=None, max_length=400)
    raw_entry: str | None = Field(default=None, max_length=2000)


@runtime_checkable
class CiteResolver(Protocol):
    """Maps a ``\\cite{key}`` to a :class:`CitedPaperRef`.

    Populated host-side from the paper's ``\\cite{}`` keys (and, when
    available, the bibliography) by ``fetch._parse_latex_corpus``.
    ``search_literature`` consults it to resolve a cite key into a
    fetchable arXiv id. A :class:`Protocol` so tests can supply a fake
    without parsing real LaTeX.
    """

    def resolve(self, cite_key: str) -> CitedPaperRef | None: ...

    def known_keys(self) -> list[str]: ...


# ---------------------------------------------------------------------------
# Pool snapshot (minimal placeholder in Sub-PR A; dedup-vs-pool is B's).
# ---------------------------------------------------------------------------


class PoolSnapshot(BaseModel):
    """A snapshot of the existing technique pool, surfaced to the agent.

    Minimal in Sub-PR A: the dedup-vs-pool logic (so the agent does not
    re-emit a technique already in the pool) is exercised by Sub-PR B's
    pipeline wiring. For now it carries the canonical names already in
    the pool plus a pre-rendered Markdown summary the prompt template
    drops in verbatim.
    """

    model_config = ConfigDict(extra="forbid")

    technique_names: list[str] = Field(default_factory=list[str], max_length=2000)
    summary_markdown: str = Field(default="", max_length=20000)

    @classmethod
    def empty(cls) -> PoolSnapshot:
        """An empty pool (no techniques registered yet)."""
        return cls(technique_names=[], summary_markdown="(the technique pool is empty)")


# ---------------------------------------------------------------------------
# Agent run-time dependencies (constructed host-side; design note §1).
# ---------------------------------------------------------------------------


# The cited-paper fetcher signature. The lightweight ``search_literature``
# mode calls this to pull a cited paper's corpus for a focused
# sub-extraction. Defaults to ``fetch.fetch_paper_corpus`` host-side;
# injectable so tests need not hit the network.
CorpusFetcher = Callable[[str], "Awaitable[PaperCorpus | None]"]


@dataclass
class PaperAgentDeps:
    """Per-run dependencies for the Paper Agent (design note §1).

    Built host-side by the Sub-PR B preprocessing helper (fetch +
    screen) and passed to ``paper_agent.run(deps=...)``. The agent never
    reaches the network for paper content; the retrieval tools work over
    the in-deps corpus and the injected sub-extraction LLM.

    :attr:`sub_extraction_llm` powers the ``search_paper`` /
    ``search_literature`` structured-call sub-extractions (the homegrown
    :func:`smai_inline_agents.structured_call.structured_call` pattern,
    distinct from the PydanticAI model that drives the outer loop).

    :attr:`tool_call_count` is the forced-synthesis counter: each
    retrieval tool bumps it, and the per-tool ``prepare`` callback drops
    the three SearchX verbs once it reaches
    :data:`INGESTION_FORCED_SYNTHESIS_AFTER`.
    """

    arxiv_id: str
    paper_title: str
    paper_abstract: str
    section_list: list[SectionRef]
    full_latex: str
    sections_by_id: dict[str, str]
    cite_resolver: CiteResolver
    pool_snapshot: PoolSnapshot
    sub_extraction_llm: LlmProvider
    corpus_fetcher: CorpusFetcher
    enrichment_depth: int = 0
    enrichment_depth_limit: int = 1
    tool_call_count: int = 0
    enriched_techniques: list[TechniqueDescription] = field(
        default_factory=list[TechniqueDescription]
    )


# ---------------------------------------------------------------------------
# Agent output (PaperExtraction) + host-assembled record (IngestionResult).
# ---------------------------------------------------------------------------


class ExtractionAuditEntry(BaseModel):
    """One retrieval move the agent self-reports (per_phase_prompts §5).

    The agent records its ``search_paper`` / ``search_section`` /
    ``search_literature`` moves so a host-side reviewer can audit the
    grounding. Loosely auditable against the tool-call log.
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["search_paper", "search_section", "search_literature"]
    query: str = Field(max_length=2000)
    found_count: int = Field(ge=0)


class PaperExtraction(BaseModel):
    """What the Paper Agent emits via its ``output_type``.

    Narrow on purpose (see module docstring): the host knows the paper
    identity and the screening verdict, so the LLM only produces the
    extracted techniques plus the paper-level prose and its retrieval
    audit. The Sub-PR B wrapper folds this into an
    :class:`IngestionResult`.
    """

    model_config = ConfigDict(extra="forbid")

    techniques: list[TechniqueDescription] = Field(
        default_factory=list[TechniqueDescription], max_length=20
    )
    paper_summary: str = Field(
        min_length=1,
        max_length=4000,
        description="One-paragraph plain-prose paraphrase of the paper's contribution.",
    )
    extraction_caveats: list[str] = Field(default_factory=list[str], max_length=50)
    extraction_audit: list[ExtractionAuditEntry] = Field(
        default_factory=list[ExtractionAuditEntry], max_length=200
    )


class IngestionResult(BaseModel):
    """Host-assembled result of one paper-ingestion run (design note §3).

    Carries the agent's extracted techniques plus the
    :class:`ScreeningOutcome` carried through from the screener
    preprocessing step and the paper-identity metadata the host holds.
    ``techniques`` is empty (with ``error_reason`` set) for the reject /
    fetch-fail branches the Sub-PR B wrapper constructs.

    Sub-PR A defines + exports this and the :meth:`from_extraction`
    assembly seam; the wrapper that fans fetch / screen / run into it
    lands in Sub-PR B.
    """

    model_config = ConfigDict(extra="forbid")

    arxiv_id: str = Field(max_length=40)
    paper_title: str = Field(default="", max_length=400)
    paper_abstract: str = Field(default="", max_length=8000)
    paper_level_summary: str = Field(default="", max_length=4000)
    screening: ScreeningOutcome | None = None
    techniques: list[TechniqueDescription] = Field(
        default_factory=list[TechniqueDescription], max_length=20
    )
    error_reason: str | None = Field(default=None, max_length=2000)
    extraction_caveats: list[str] = Field(default_factory=list[str], max_length=50)
    extraction_audit: list[ExtractionAuditEntry] = Field(
        default_factory=list[ExtractionAuditEntry], max_length=200
    )

    @classmethod
    def from_extraction(
        cls,
        extraction: PaperExtraction,
        *,
        arxiv_id: str,
        paper_title: str,
        paper_abstract: str,
        screening: ScreeningOutcome | None,
    ) -> IngestionResult:
        """Fold the agent's :class:`PaperExtraction` into a full record.

        The host supplies the authoritative identity + screening fields;
        the agent supplied the techniques + prose + audit.
        """
        return cls(
            arxiv_id=arxiv_id,
            paper_title=paper_title,
            paper_abstract=paper_abstract,
            paper_level_summary=extraction.paper_summary,
            screening=screening,
            techniques=list(extraction.techniques),
            error_reason=None,
            extraction_caveats=list(extraction.extraction_caveats),
            extraction_audit=list(extraction.extraction_audit),
        )


__all__ = [
    "CitedPaperRef",
    "CiteResolver",
    "CorpusFetcher",
    "ExtractionAuditEntry",
    "IngestionResult",
    "PaperAgentDeps",
    "PaperExtraction",
    "PoolSnapshot",
    "ScreeningOutcome",
    "SectionRef",
]
