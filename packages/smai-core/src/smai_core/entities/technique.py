"""Technique references and the fidelity-anchor discriminated union.

Per ``designs/smai/01-data-model.md`` §3.2 and DEC-032 (``fidelity_anchor``
generalization replacing v1's ``source_paper_reference``).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

TechniqueParamValue: TypeAlias = str | int | float | bool | None
"""JSON-shaped primitives admitted in per-entry technique configuration."""

TechniqueParams: TypeAlias = dict[str, TechniqueParamValue]
"""Per-entry technique configuration; flat dict of primitives.

Validated against ``TechniqueRef.parameter_schema`` (JSON Schema Draft
2020-12) at compile time (Task 1.5 verification).
"""


class PaperFidelityAnchor(BaseModel):
    """Anchor: a published paper grounds the technique spec."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["paper"] = "paper"
    doi: str
    arxiv_id: str | None = None
    title: str | None = None


class ProposalFidelityAnchor(BaseModel):
    """Anchor: a novel-technique proposal grounds the technique spec."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["proposal"] = "proposal"
    proposal_id: str
    submitted_by: str | None = None


class ReviewerAttestedFidelityAnchor(BaseModel):
    """Anchor: a reviewer-vetted self-contained spec grounds the technique."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["reviewer_attested"] = "reviewer_attested"
    spec_text: str
    attested_by: str | None = None


FidelityAnchor = Annotated[
    PaperFidelityAnchor | ProposalFidelityAnchor | ReviewerAttestedFidelityAnchor,
    Field(discriminator="kind"),
]
"""Discriminated union of the three v1 fidelity-anchor kinds (DEC-032).

Required on ``TechniqueRef`` when ``standard=False``. The pipeline-layer
review gate dispatches on ``kind`` rather than assuming a paper.
"""

FidelityAnchorAdapter: TypeAdapter[FidelityAnchor] = TypeAdapter(FidelityAnchor)
"""``TypeAdapter`` for the anchor union; use to validate raw dicts."""


class TechniqueRef(BaseModel):
    """A registered technique. Global, source-independent.

    ``category`` is a closed v1 enum encoded as ``str`` (the closed set lives
    in §3.2 of the data model and is verified by Task 1.5 rather than locked
    via ``Literal`` — extending the set is an additive registry change rather
    than a schema change). ``compatible_factor_types`` is a small ``Literal``
    set verified at compile time against the CG's factor type. ``standard``
    techniques (DEC-015) need no fidelity anchor; non-standard techniques
    require one (verified by Task 1.5).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str
    category: str
    compatible_factor_types: list[Literal["additive", "substitutive"]]
    standard: bool = False
    fidelity_anchor: FidelityAnchor | None = None
    affects_extension_points: list[str]
    implies_controlled: list[str] = []
    parameter_schema: dict[str, Any] | None = None
