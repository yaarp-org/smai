"""Planner subtree of ``smai-inline-agents``.

Per ``architectural_decisions.md §4`` (planner-refactor design), the
``planner/`` subtree holds the new four-phase PydanticAI-based planner.

Step 2 of the planner-refactor implementation plan (this commit) lands
only :mod:`smai_inline_agents.planner.schemas` — the producer-side
:class:`TechniqueDescription` schema consumed by both the planner's
phase-3 output and the ingestion subagent's output. The planner's phase
modules (``phases/literature_review.py`` and the others) land at
Steps 4-6.
"""

from __future__ import annotations

from smai_inline_agents.planner.schemas import (
    AlgorithmSpec,
    ConfidenceFlag,
    DatasetReference,
    Hyperparameter,
    LossSpec,
    SourceExcerpt,
    SourceLocation,
    SourceLocationEquation,
    SourceLocationInline,
    SourceLocationPage,
    SourceLocationSection,
    TechniqueDescription,
    TrainingRecipe,
)

__all__ = [
    "AlgorithmSpec",
    "ConfidenceFlag",
    "DatasetReference",
    "Hyperparameter",
    "LossSpec",
    "SourceExcerpt",
    "SourceLocation",
    "SourceLocationEquation",
    "SourceLocationInline",
    "SourceLocationPage",
    "SourceLocationSection",
    "TechniqueDescription",
    "TrainingRecipe",
]
