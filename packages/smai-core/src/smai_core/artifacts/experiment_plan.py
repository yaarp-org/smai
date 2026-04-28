""":class:`ExperimentPlan` — the canonical experiment declaration artifact.

Per ``02-dsl-and-contracts.md`` §7.3. Substantively identical to the verified
:class:`ExperimentDefinition`, plus an envelope. Faithfully preserves all
design-time intent.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from smai_core.artifacts._envelope import ArtifactEnvelope
from smai_core.entities.conditions import ControlledConditions
from smai_core.entities.factor import Entry, Factor
from smai_core.entities.validation import ValidationCriteria


class ExperimentPlanBody(BaseModel):
    """Body fields of :class:`ExperimentPlan` (§7.3)."""

    model_config = ConfigDict(extra="forbid")

    hypothesis: str
    factor_model_id: str | None = None
    factors: list[Factor]
    controlled_conditions: ControlledConditions
    entries: list[Entry]
    validation: ValidationCriteria


class ExperimentPlan(BaseModel):
    """Canonical declaration of an experiment (§7.3)."""

    model_config = ConfigDict(extra="forbid")

    envelope: ArtifactEnvelope
    body: ExperimentPlanBody
