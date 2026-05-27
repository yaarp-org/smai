""":class:`ContextualVerdict` — contextual evaluator's locked output.

Per ``04-agents.md`` §2.5 (verbatim shape) and §6 (single-call structured
output via ``tool_use``). Consumed by Task 2.C4's ``evaluating``-state
dispatch handler, which runs the mechanical evaluator first then this
agent (DEC-034 #2 — single ``evaluating`` state with sequential
mechanical-then-contextual dispatches).

The schema is **rule-agnostic**: per §2.5 the same shape ships for
``compare_to_baseline`` and ``compare_to_target`` CGs; only the
prompt-config variant differs (per DEC-031 #8 — v1's ``evaluationMode``
field is collapsed into the methodology-layer ``ComparisonRule``). The
contextual evaluator does not generate fresh numbers — it consumes the
deterministic ``EvaluationResult.verdict_context`` and produces the
narrative interpretation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class EntryRanking(BaseModel):
    """One entry's ranking within the CG (§2.5 verbatim)."""

    model_config = ConfigDict(extra="forbid")

    entry_id: str
    rank: int
    rationale: str


class ContextualVerdict(BaseModel):
    """Locked contextual-verdict schema (§2.5 verbatim).

    Five output fields beyond ``overall_verdict``:

    * ``summary`` — the 2-4 sentence interpretation written for a
      researcher who wants to quickly understand what happened.
    * ``rankings`` — entries ordered best to worst with rationale.
    * ``insights`` — patterns, surprises, concerns; each item is a
      short structured English statement (the v1 ``{type, description}``
      tuple is folded into a single narrative string per the §2.5
      schema).
    * ``limitations`` — factors that limit the strength of conclusions
      (e.g., small seed counts, missing runs).
    * ``suggested_follow_ups`` — concrete next experiments or
      investigations that would strengthen / extend the findings.

    Per `00-vision.md` §2.1 / `06-mechanical-evaluation.md` §5.1, this
    is tier 3 (agent-validated) output — informational, NOT gating. The
    mechanical ``Verdict`` is the verdict-path commitment; this artifact
    grounds the writeup that the outer pipeline produces.
    """

    model_config = ConfigDict(extra="forbid")

    overall_verdict: Literal["promising", "inconclusive", "not_promising"]
    summary: str
    rankings: list[EntryRanking]
    insights: list[str]
    limitations: list[str]
    suggested_follow_ups: list[str]


__all__ = ["ContextualVerdict", "EntryRanking"]
