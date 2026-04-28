"""Top-level methodology entities.

Per ``designs/smai/01-data-model.md`` §3.7 (``ExperimentDefinition``,
``VerifiedExperimentDefinition``) and §3.9 (``FactorModel``).
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from smai_core.entities.conditions import ControlledConditions
from smai_core.entities.factor import Entry, Factor
from smai_core.entities.validation import ValidationCriteria


class ExperimentDefinition(BaseModel):
    """One CG's full spec.

    ``factors`` is length-1 in v1 (DEC-031 sub-decision #4); the constraint is
    enforced at Pass 1 by Pydantic ``Field`` constraints. Lifting the
    constraint in vN is an additive change.

    ``hypothesis`` is recorded surface — consumed by the contextual evaluator
    agent for narrative writeup, never by the verdict path.
    ``factor_model_id`` is an optional bidirectional reference to a
    ``FactorModel`` (consistency-checked at Pass-2 verification, Task 1.5).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    hypothesis: str
    factor_model_id: str | None = None
    factors: Annotated[list[Factor], Field(min_length=1, max_length=1)]
    controlled_conditions: ControlledConditions
    entries: list[Entry]
    validation: ValidationCriteria


class VerifiedExperimentDefinition(ExperimentDefinition):
    """Marker subclass returned by Pass-2 verification.

    Type-system precondition for ``emit_artifacts(...)`` (Task 1.6). Adds no
    fields; the wrapper exists purely to encode the precondition in the type
    system. The verifier (Task 1.5) returns ``VerifiedExperimentDefinition``
    on success.
    """


class FactorModel(BaseModel):
    """Optional grouping above CG.

    Per DEC-014 / DEC-031 sub-decision #5: purely organizational in v1. Does
    not emit its own contract artifact; does not aggregate verdicts across
    CGs. ``shared_conditions`` declares values comprising CGs are expected to
    share — the compiler verifies match-where-overlapping rather than strict
    equality. Bidirectional reference: this entity's ``comparison_group_ids``
    must agree with each comprising CG's ``factor_model_id`` (consistency
    check at Pass-2 verification).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    research_question: str
    shared_conditions: ControlledConditions | None = None
    comparison_group_ids: list[str]
