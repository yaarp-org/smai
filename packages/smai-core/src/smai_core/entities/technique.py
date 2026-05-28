"""Technique references and the fidelity-anchor discriminated union.

Per ``designs/smai/01-data-model.md`` §3.2 and DEC-032 (``fidelity_anchor``
generalization replacing v1's ``source_paper_reference``).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

ContextKind: TypeAlias = Literal[
    "paper_extract", "proposal", "reviewer_attested", "standard", "no_op_baseline"
]
"""Per ``agent_refactor/upstream_requirements.md`` §1: explicit grounding-flavor
discriminator on :class:`TechniqueRef` and :class:`TechniqueContractBody`,
populated by the planner / ingestion subagent and read directly by the
methodology compiler. Replaces the inference-from-anchor logic that lived
on the implementer side.

The five-way set:

* ``paper_extract`` — :class:`PaperFidelityAnchor`-anchored ref.
* ``proposal`` — :class:`ProposalFidelityAnchor`-anchored ref.
* ``reviewer_attested`` — :class:`ReviewerAttestedFidelityAnchor`-anchored ref.
* ``standard`` — DEC-015 standard technique, no anchor.
* ``no_op_baseline`` — additive baseline contract body (no
  :class:`TechniqueRef` exists for it; populated by the methodology
  compiler at emission time for entries with ``technique_id is None`` +
  ``is_baseline=True``). Never appears on :class:`TechniqueRef` itself —
  only on :class:`TechniqueContractBody`.
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


_REF_CONTEXT_KINDS: frozenset[str] = frozenset(
    {"paper_extract", "proposal", "reviewer_attested", "standard"}
)


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
    context_kind: ContextKind
    """Explicit grounding flavor (``agent_refactor/upstream_requirements §1``).
    Required at construction; producers (planner / ingestion subagent /
    paper-ingestion projection / fixture authors) set it explicitly. Must
    be one of the four ``TechniqueRef`` variants — ``no_op_baseline`` is
    a contract-body-only value and never appears here (the
    :meth:`_validate_context_kind` validator rejects it).
    """

    @model_validator(mode="after")
    def _validate_context_kind(self) -> TechniqueRef:
        """Consistency-check ``context_kind`` against the anchor / standard mapping.

        Producers set ``context_kind`` explicitly; this validator rejects
        anything inconsistent with :data:`anchor_implied_context_kind`.
        ``no_op_baseline`` is rejected here because it is a contract-body
        variant only (additive baselines never have a :class:`TechniqueRef`).
        """
        if self.context_kind not in _REF_CONTEXT_KINDS:
            raise ValueError(
                f"context_kind {self.context_kind!r} is not valid on TechniqueRef; "
                f"allowed values are {sorted(_REF_CONTEXT_KINDS)} "
                "(no_op_baseline is a TechniqueContractBody-only variant)"
            )
        implied = anchor_implied_context_kind(self.fidelity_anchor, self.standard)
        if implied is not None and self.context_kind != implied:
            raise ValueError(
                f"context_kind {self.context_kind!r} disagrees with "
                f"fidelity_anchor / standard mapping ({implied!r}); "
                "see upstream_requirements §1 for the canonical mapping"
            )
        return self


def anchor_implied_context_kind(
    anchor: FidelityAnchor | None, standard: bool
) -> ContextKind | None:
    """Map an anchor + standard flag onto the implied :data:`ContextKind`.

    Returns ``None`` when no mapping applies (non-standard + anchor-less,
    which is rejected by Task 1.5 verification downstream). Used by the
    :class:`TechniqueRef` validator's consistency check and by callers
    that synthesize a ``context_kind`` from anchor / standard inputs.
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
