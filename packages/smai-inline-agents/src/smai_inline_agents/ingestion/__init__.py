"""Ingestion subagent: the SciReplicate-shaped Paper Agent (Step 3, Sub-PR A).

Takes a pre-fetched paper corpus and produces a typed
:class:`IngestionResult` carrying one or more
:class:`smai_inline_agents.planner.schemas.TechniqueDescription` records.
PydanticAI-based, fresh-context per
``planner_refactor/architectural_decisions.md §9``.

Sub-PR A shipped the library: schemas, LaTeX-source-first fetch
preprocessing (S2ORC + Marker stubbed/deferred), the Paper Agent with
its three retrieval tools + forced-synthesis budget, and the prompt.
Sub-PR B adds the ``run_ingestion_subagent`` orchestration wrapper
(fetch -> optional-prescreened screener gate -> build deps -> run ->
:class:`IngestionResult` assembly) plus ``search_literature``'s deep
recursive enrichment mode; both are exported here.
"""

from __future__ import annotations

from smai_inline_agents.ingestion.corpus import PaperCorpus
from smai_inline_agents.ingestion.fetch import (
    LatexCiteResolver,
    fetch_paper_corpus,
)
from smai_inline_agents.ingestion.paper_agent import (
    INGESTION_FORCED_SYNTHESIS_AFTER,
    INGESTION_USAGE_LIMITS,
    build_paper_agent,
    load_paper_agent_prompt,
)
from smai_inline_agents.ingestion.run import (
    run_ingestion_subagent,
    summarize_recursive_result,
)
from smai_inline_agents.ingestion.schemas import (
    CitedPaperRef,
    CiteResolver,
    CorpusFetcher,
    ExtractionAuditEntry,
    IngestionResult,
    PaperAgentDeps,
    PaperAgentDepsFactory,
    PaperExtraction,
    PoolSnapshot,
    ScreeningOutcome,
    SectionRef,
)

__all__ = [
    "INGESTION_FORCED_SYNTHESIS_AFTER",
    "INGESTION_USAGE_LIMITS",
    "CiteResolver",
    "CitedPaperRef",
    "CorpusFetcher",
    "ExtractionAuditEntry",
    "IngestionResult",
    "LatexCiteResolver",
    "PaperAgentDeps",
    "PaperAgentDepsFactory",
    "PaperCorpus",
    "PaperExtraction",
    "PoolSnapshot",
    "ScreeningOutcome",
    "SectionRef",
    "build_paper_agent",
    "fetch_paper_corpus",
    "load_paper_agent_prompt",
    "run_ingestion_subagent",
    "summarize_recursive_result",
]
