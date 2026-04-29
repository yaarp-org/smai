""":class:`ScreenResult` — paper screener's locked output schema.

Per ``04-agents.md`` §6 (single-call structured output via ``tool_use``)
and ``08-novel-technique-pipeline.md`` §5.2 (the screening stage of
the paper-ingestion supporting utility per DEC-032). Consumed by:

* :func:`smai_agents.agents.screener.run_paper_screening` — the role-
  shaped wrapper that hands this to :func:`smai_agents.structured_call`
  as ``output_schema``.
* Task 3.E2's paper-ingestion pipeline-spec ``screening → planning`` /
  ``screening → rejected`` gate rules (per ``03-state-machine.md`` §5.2
  edges 5 / 6: ``paper.screener_pass`` reads ``decision="accept"``;
  ``paper.screener_reject`` reads ``decision="reject"``).

DEC-032 framing: paper ingestion produces ``TechniqueRef``s only — the
screener decides whether the paper is in scope for that purpose, NOT
whether any CG should be created from it. CGs are exclusively proposal-
born.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ScreenResult(BaseModel):
    """Locked screener output (``08`` §5.2 / ``04-agents.md`` §6).

    The screener returns a structured pass/reject verdict with a brief
    rationale. The pipeline-spec gate rules read :attr:`decision` and
    route accordingly. ``rejection_reason`` is required when ``decision
    == "reject"`` (so the rejection terminal carries actionable runbook
    context) and forbidden otherwise — enforced by a Pydantic validator
    so the gate caller does not have to defend against drift.

    Per ``08`` §5.2 the screening criterion is "is this an empirical
    comparative-claims ML paper amenable to SMAI's question shape?"
    Sub-criteria:

    * The paper makes empirical claims comparing techniques (not a
      pure theory or survey paper).
    * The claims are machine-learning-flavored (not adjacent fields
      whose technique pool is out-of-scope for SMAI).
    * The paper has the substrate the planner needs: at least one
      contribution technique with enough text to extract a
      :class:`TechniqueRef` from.
    """

    model_config = ConfigDict(extra="forbid")

    decision: Literal["accept", "reject"]
    """``"accept"`` advances the paper to ``planning`` per ``08`` §5.2
    edge 5; ``"reject"`` lands the paper in the terminal ``rejected``
    state per edge 6."""

    rejection_reason: str | None = None
    """Brief actionable explanation when ``decision == "reject"``. Cited
    by the runbook so an operator can decide whether to manually push
    the paper back to ``submitted`` (e.g., the screener was wrong about
    the empirical-comparative-claim criterion) or accept the rejection.
    Required when ``decision == "reject"``; must be ``None`` when
    ``decision == "accept"`` (a reason field for an accept verdict
    would be confusing — the planner stage is where positive rationale
    accumulates)."""

    summary: str
    """One-paragraph free-text summary of what the paper claims and why
    the screener decided as it did. Always populated; surfaced in the
    paper detail dashboard (Task 3.H1). Independent of the
    accept/reject decision — the summary is the in-scope abstract a
    human reader can scan to confirm the screener's verdict."""

    def model_post_init(self, __context: object) -> None:
        # Enforce the decision/rejection_reason invariant: reject must
        # carry a reason; accept must not. v1 carry-forward of the
        # "no silent fallback" discipline (DEC-018) — drift here would
        # show up as a confusing dashboard or a missing runbook hint.
        if self.decision == "reject" and not self.rejection_reason:
            raise ValueError(
                "ScreenResult.decision == 'reject' requires a non-empty rejection_reason"
            )
        if self.decision == "accept" and self.rejection_reason is not None:
            raise ValueError("ScreenResult.decision == 'accept' must not carry a rejection_reason")


__all__ = ["ScreenResult"]
