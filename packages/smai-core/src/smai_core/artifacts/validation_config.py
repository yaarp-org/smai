""":class:`ValidationConfig` — what the mechanical evaluator consumes.

Per ``02-dsl-and-contracts.md`` §7.6. Locked at compile time; the evaluator's
only inputs are this artifact + raw multi-seed metrics.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from smai_core.artifacts._envelope import ArtifactEnvelope
from smai_core.entities.metric import MetricRef
from smai_core.entities.validation import (
    AggregationRule,
    ComparisonRule,
    TrendCheck,
)


class ValidationConfigBody(BaseModel):
    """Body fields of :class:`ValidationConfig` (§7.6).

    ``ComparisonRule.baseline_entry_id`` is filled in by the compiler at
    emission time per §2.5; the embedded :class:`ComparisonRule` validates
    in artifact mode (the default) so the absence-when-needed check fires.
    """

    model_config = ConfigDict(extra="forbid")

    parent_experiment_hash: str
    metric: MetricRef
    direction: Literal["higher_is_better", "lower_is_better"]
    aggregation: AggregationRule
    comparison: ComparisonRule
    seed_count_required: int
    rationale: str | None = None
    trend_check: TrendCheck | None = None


class ValidationConfig(BaseModel):
    """Verdict-path configuration artifact (§7.6)."""

    model_config = ConfigDict(extra="forbid")

    envelope: ArtifactEnvelope
    body: ValidationConfigBody
