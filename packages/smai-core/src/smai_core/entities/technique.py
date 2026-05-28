"""Technique references and the fidelity-anchor discriminated union.

Per ``designs/smai/01-data-model.md`` §3.2 and DEC-032 (``fidelity_anchor``
generalization replacing v1's ``source_paper_reference``).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

ContextKind: TypeAlias = Literal["paper_extract", "proposal", "reviewer_attested", "standard"]
"""Per ``agent_refactor/upstream_requirements.md`` §1: explicit grounding-flavor
discriminator on :class:`TechniqueRef` and :class:`TechniqueContractBody`,
populated by the planner / ingestion subagent and read directly by the
methodology compiler. Replaces the inference-from-anchor logic that lived
on the implementer side.

The four-way set mirrors :class:`TechniqueDescription.context_kind` from
``smai_inline_agents.planner.schemas``. The fifth ``no_op_baseline`` shape
is workflow-derived (DEC-013 / DEC-017) and not a value of this Literal —
additive baselines have no :class:`TechniqueRef` to discriminate.
"""

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
    context_kind: ContextKind | None = None
    """Explicit grounding flavor (``upstream_requirements §1``). Derived
    from ``standard`` + ``fidelity_anchor.kind`` by
    :meth:`_derive_context_kind` when constructed without one; producers
    (planner / ingestion subagent) set it explicitly. The methodology
    compiler reads this field rather than re-inferring at compile time.
    """

    @model_validator(mode="after")
    def _derive_context_kind(self) -> TechniqueRef:
        """Backfill / consistency-check ``context_kind``.

        The mapping (``upstream_requirements §1``):

        * :class:`PaperFidelityAnchor` → ``"paper_extract"``
        * :class:`ProposalFidelityAnchor` → ``"proposal"``
        * :class:`ReviewerAttestedFidelityAnchor` → ``"reviewer_attested"``
        * ``fidelity_anchor=None`` + ``standard=True`` → ``"standard"``

        Non-standard refs without an anchor leave ``context_kind`` at
        ``None`` here; Task 1.5 ``technique.fidelity_anchor_present_or_standard``
        verification rejects them downstream so no production path
        survives with a ``None`` context_kind.

        Existing-data ``TechniqueRef``s persisted before Step 2 of the
        planner_refactor get backfilled on read via this validator
        (D10: no data migration; backfill at hydration is the v1 posture).
        New producers set ``context_kind`` explicitly; a mismatch with
        the anchor / standard mapping raises here.
        """
        derived = _derive_context_kind_from_anchor(self.fidelity_anchor, self.standard)
        if self.context_kind is None:
            if derived is not None:
                object.__setattr__(self, "context_kind", derived)
        elif derived is not None and self.context_kind != derived:
            raise ValueError(
                f"context_kind {self.context_kind!r} disagrees with "
                f"fidelity_anchor / standard mapping ({derived!r}); "
                "see upstream_requirements §1 for the canonical mapping"
            )
        return self


def _derive_context_kind_from_anchor(
    anchor: FidelityAnchor | None, standard: bool
) -> ContextKind | None:
    """Map an anchor + standard flag onto :data:`ContextKind`.

    Returns ``None`` when no mapping applies (non-standard + anchor-less,
    which is rejected by Task 1.5 verification downstream).
    """
    if anchor is not None:
        if anchor.kind == "paper":
            return "paper_extract"
        if anchor.kind == "proposal":
            return "proposal"
        if anchor.kind == "reviewer_attested":
            return "reviewer_attested"
    elif standard:
        return "standard"
    return None
