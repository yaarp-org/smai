""":class:`EnrichmentResult` — comparison-technique enricher's locked output.

Per ``04-agents.md`` §6 (single-call structured output via ``tool_use``)
and ``08-novel-technique-pipeline.md`` §5.6 (the enrichment sub-step of
the paper-ingestion pipeline-spec's ``planning`` state per DEC-032).
Consumed by:

* :func:`smai_agents.agents.enricher.run_technique_enrichment` — the
  role-shaped wrapper that hands this to
  :func:`smai_agents.structured_call` as ``output_schema``.
* Task 3.E2's paper-ingestion ``planning`` dispatch handler, which calls
  the enricher once per ``draft_ensure_technique`` skeleton in the
  planner's buffer that points at a non-standard cited source paper
  (per ``08`` §5.6's decision tree).

The schema covers the three substantive properties §5.6 commits to:

* **Implementability assessment.** ``"high" | "medium" | "blocked"`` —
  blocked means the technique is registered but cannot be implemented
  from the source paper alone (proprietary data, specialized hardware,
  or critical method details missing).
* **Method extraction.** A focused method-extraction body the planner
  writes to ``papers/{source_arxiv_id}/techniques/{technique_id}/method_extraction.json``
  per DEC-032 OQ5 (eager persistence at enrichment time).
* **Standard-flag judgment** is the planner's, not the enricher's
  (per ``08`` §5.6 decision tree and DEC-015) — the enricher only
  runs on non-standard skeletons. The schema therefore does not carry
  a standard flag; the planner's judgment from the prior buffer is
  authoritative.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

ImplementabilityLevel = Literal["high", "medium", "blocked"]


class EnrichmentResult(BaseModel):
    """Locked enricher output (``08`` §5.6 / ``04-agents.md`` §6).

    The enricher reads the cited source paper's content and produces a
    focused method-extraction for one specific technique (the skeleton
    the planner referenced). Outputs:

    * :attr:`implementability` — ``"high"`` (clear algorithm, standard
      components), ``"medium"`` (some ambiguity but feasible),
      ``"blocked"`` (proprietary data / specialized hardware / critical
      details missing). Blocked techniques are still registered (so
      future ``draft_ensure_technique`` calls find them via dedup) but
      flagged for downstream filtering.
    * :attr:`method_extraction` — the focused extraction body. Free-
      form Markdown organized into sections the implementer agent can
      read; the dispatch handler writes this verbatim to ArtifactStore.
    * :attr:`refined_description` — optional; if present, the
      registration step uses this in place of the planner's draft
      description (since the enricher has actually read the source
      paper, its description is typically more accurate).
    * :attr:`blocked_reason` — required when ``implementability ==
      "blocked"``, must be ``None`` otherwise. Surfaced in the runbook
      / dashboard so an operator knows why the entry citing this
      technique would be non-runnable downstream.
    """

    model_config = ConfigDict(extra="forbid")

    implementability: ImplementabilityLevel
    method_extraction: str
    """Free-form Markdown body. The dispatch handler writes this
    verbatim to ``papers/{source_arxiv_id}/techniques/{technique_id}/method_extraction.json``
    per DEC-032 OQ5. v1's convention organized this into ``## Inputs``,
    ``## Algorithm``, ``## Hyperparameters``, ``## Notes`` sections; v2
    keeps the format flexible (the implementer agent reads the body,
    not specific sections)."""

    refined_description: str | None = None
    """Optional refined description for the :class:`TechniqueRef`. When
    non-null, the registration step prefers this over the planner's
    draft description (the enricher actually read the source paper, so
    its description is typically more accurate). When null, the
    planner's draft description stands."""

    blocked_reason: str | None = None
    """Required when ``implementability == "blocked"``; must be null
    otherwise. Brief actionable explanation (e.g., "uses proprietary
    dataset not publicly available", "method requires specialized TPU
    hardware", "critical hyperparameter values redacted")."""

    def model_post_init(self, __context: object) -> None:
        # Enforce the implementability/blocked_reason invariant —
        # mirroring :class:`ScreenResult`'s decision/rejection_reason
        # discipline. v1 carry-forward of the "no silent fallback"
        # rule (DEC-018).
        if self.implementability == "blocked" and not self.blocked_reason:
            raise ValueError(
                "EnrichmentResult.implementability == 'blocked' requires a non-empty blocked_reason"
            )
        if self.implementability != "blocked" and self.blocked_reason is not None:
            raise ValueError(
                "EnrichmentResult.blocked_reason must be null when "
                "implementability is 'high' or 'medium'"
            )


__all__ = ["EnrichmentResult", "ImplementabilityLevel"]
