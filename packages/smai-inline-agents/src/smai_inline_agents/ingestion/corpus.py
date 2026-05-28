"""The pre-fetched, parsed paper corpus the Paper Agent reads.

Produced by ``fetch.fetch_paper_corpus`` (LaTeX-source-first
preprocessing) and consumed by the Sub-PR B deps-builder + the
``search_literature`` lightweight mode. A dataclass rather than a
Pydantic model because it carries a live :class:`CiteResolver` object
(not wire data) and is never persisted.

Design reference: ``planner_refactor/design_notes/ingestion_subagent.md``
§1 (the fields ``PaperAgentDeps`` is built from) + §6 (fetch contract).
"""

from __future__ import annotations

from dataclasses import dataclass

from smai_inline_agents.ingestion.schemas import CiteResolver, SectionRef


@dataclass
class PaperCorpus:
    """One paper's parsed content + structure (fetch preprocessing output)."""

    arxiv_id: str
    title: str
    abstract: str
    full_latex: str
    sections: list[SectionRef]
    sections_by_id: dict[str, str]
    cite_resolver: CiteResolver


__all__ = ["PaperCorpus"]
