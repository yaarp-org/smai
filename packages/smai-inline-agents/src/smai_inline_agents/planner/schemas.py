"""Producer-side :class:`TechniqueDescription` schema (planner-refactor Step 2).

Replaces today's loosely-typed ``dict | str | None`` body that
:meth:`smai_cli.runtime.ProposalsService.submit` accepts. Closes
``agent_refactor/upstream_requirements.md`` §2 (typed technique
description) and is the producer-side surface for §1 (``context_kind``
on the contract).

Both producers emit this shape:

* The new planner agent at the end of phase 3 (``architectural_decisions.md §2``)
  for novel-technique proposals (``context_kind == "proposal"``).
* The ingestion subagent at the end of paper extraction
  (``architectural_decisions.md §9``) for paper-derived techniques
  (``context_kind == "paper_extract"``).

Consumers:

* The methodology compiler (reads ``context_kind`` — the description body
  itself stays agent-side per DEC-029).
* The implementer agent at bundle-construction time (the per-context
  ``GroundingContext`` variants source their content here).
* The proposal-pipeline writeback at
  ``proposals/{proposal_id}/technique_description.json``.

See ``~/projects/Yaarp/designs/smai/planner_refactor/design_notes/technique_description_schema.md``
for the full design rationale: the hybrid asymmetric paraphrase-vs-verbatim
split (§2), the sentinel pattern (§3), per-``context_kind`` enforcement
(§4), structural validation rules (§5), and the migration-dropped (per D10)
schema-version story (§6).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Source-grounding primitives (location + excerpt).
# ---------------------------------------------------------------------------


class SourceLocationSection(BaseModel):
    """A location pinned to a numbered or named section of the source."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["section"] = "section"
    section_id: str = Field(min_length=1, max_length=120)
    paragraph_index: int | None = Field(default=None, ge=0)


class SourceLocationEquation(BaseModel):
    """A location pinned to a labelled equation."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["equation"] = "equation"
    equation_label: str = Field(min_length=1, max_length=80)


class SourceLocationPage(BaseModel):
    """A location pinned to a page number when section-level pinning isn't possible."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["page"] = "page"
    page_number: int = Field(ge=1)


class SourceLocationInline(BaseModel):
    """Escape hatch: producer has the excerpt but cannot pin a section/page.

    Use when the excerpt comes from an abstract, figure caption, blog post, or
    other source where section/page coordinates don't apply or aren't
    recoverable. Producers should prefer a concrete location when possible;
    this variant is the honest alternative to hallucinating a section number.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["inline"] = "inline"
    note: str | None = Field(default=None, max_length=300)


SourceLocation = Annotated[
    SourceLocationSection | SourceLocationEquation | SourceLocationPage | SourceLocationInline,
    Field(discriminator="kind"),
]


class SourceExcerpt(BaseModel):
    """A verbatim quotation from the producer's source.

    SciReplicate-Bench's "maintain exact text and LaTeX formatting, do not
    paraphrase" discipline applies to :attr:`text`. Paraphrase / synthesis
    lives in the parent field's ``summary``; this field is the un-rewritten
    quote.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000)
    location: SourceLocation

    @model_validator(mode="after")
    def _text_must_be_nonblank(self) -> SourceExcerpt:
        if not self.text.strip():
            raise ValueError("SourceExcerpt.text must be non-blank after .strip()")
        return self


# ---------------------------------------------------------------------------
# Hybrid sub-schemas (paraphrased canonical form + verbatim excerpts).
# ---------------------------------------------------------------------------


class AlgorithmSpec(BaseModel):
    """The technique's algorithmic core (hybrid: paraphrased + verbatim)."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=50, max_length=2500)
    pseudocode: str | None = Field(default=None, max_length=4000)
    source_excerpts: list[SourceExcerpt] = Field(default_factory=list[SourceExcerpt], max_length=10)


class Hyperparameter(BaseModel):
    """One hyperparameter the technique exposes or fixes."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    summary: str = Field(min_length=10, max_length=300)
    value: str | None = None
    range_or_search: str | None = None
    default_for_smai: str | None = None
    source_excerpt: SourceExcerpt | None = None


class LossSpec(BaseModel):
    """A custom loss function the technique introduces."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=50, max_length=1500)
    formula: str | None = Field(default=None, max_length=2000)
    source_excerpts: list[SourceExcerpt] = Field(default_factory=list[SourceExcerpt], max_length=5)


class TrainingRecipe(BaseModel):
    """The training-time recipe the technique requires or recommends.

    All sub-fields are ``str | None`` because papers report optimizer /
    schedule / batch / epochs as ranges, sweep specs, or qualitative
    descriptors more often than as fixed integers; premature numeric
    typing here would force the producer to invent precision it doesn't
    have.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=50, max_length=2500)
    optimizer: str | None = Field(default=None, max_length=200)
    schedule: str | None = Field(default=None, max_length=300)
    batch_size: str | None = Field(default=None, max_length=120)
    epochs: str | None = Field(default=None, max_length=120)
    source_excerpts: list[SourceExcerpt] = Field(default_factory=list[SourceExcerpt], max_length=8)


# ---------------------------------------------------------------------------
# Empirical-context sub-schemas (paraphrased only).
# ---------------------------------------------------------------------------


class DatasetReference(BaseModel):
    """A dataset the source paper / proposal references as in-scope."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    split: str | None = Field(default=None, max_length=80)
    notes: str | None = Field(default=None, max_length=500)


# ---------------------------------------------------------------------------
# Confidence / sentinel discipline (design note §3).
# ---------------------------------------------------------------------------


class ConfidenceFlag(BaseModel):
    """Producer-authored annotation flagging a per-field uncertainty.

    Severity levels (design note §3):

    * ``unknown`` — producer could not find the value; field is ``None``.
    * ``uncertain`` — producer has a value but flags low confidence.
    * ``conflicting`` — sources disagree; producer picks one and flags it.

    The :meth:`TechniqueDescription._none_requires_flag` validator
    enforces that a ``None`` field (other than ones structurally forbidden
    by ``context_kind``) is matched by a ``ConfidenceFlag`` whose
    ``field_path`` points at it.
    """

    model_config = ConfigDict(extra="forbid")

    field_path: str = Field(
        min_length=2,
        max_length=200,
        pattern=r"^/[A-Za-z0-9_]+(/[A-Za-z0-9_]+|/[0-9]+)*$",
    )
    severity: Literal["unknown", "uncertain", "conflicting"]
    note: str = Field(min_length=10, max_length=500)


# ---------------------------------------------------------------------------
# TechniqueDescription (the top-level schema).
# ---------------------------------------------------------------------------


class TechniqueDescription(BaseModel):
    """Typed description of a technique (planner-refactor Step 2 schema).

    Emitted by the planner (phase 3) and by the ingestion subagent, and
    persisted at ``proposals/{proposal_id}/technique_description.json``
    (proposals) or
    ``papers/{arxiv_id}/techniques/{technique_id}/technique_description.json``
    (paper extracts).

    The :class:`TechniqueContract` does NOT embed this schema; the
    contract carries ``context_kind`` only and the methodology compiler
    stays narrow (``02-dsl-and-contracts.md §7.5`` + DEC-029). The
    implementer agent reads the typed description at
    bundle-construction time.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Identity and prose (paraphrased only) ---

    name: str = Field(
        min_length=2,
        max_length=80,
        pattern=r"^[a-z][a-z0-9_-]*$",
    )
    summary: str = Field(min_length=50, max_length=600)
    motivation: str = Field(min_length=50, max_length=1500)
    problem_setting: str = Field(min_length=50, max_length=1500)
    limitations: str | None = Field(default=None, max_length=1500)

    # --- Algorithmic core (hybrid: paraphrased + verbatim excerpts) ---

    algorithm: AlgorithmSpec
    hyperparameters: list[Hyperparameter] = Field(
        default_factory=list[Hyperparameter], max_length=50
    )
    loss_function: LossSpec | None = None
    training_recipe: TrainingRecipe | None = None

    # --- Empirical context (paraphrased only) ---

    datasets: list[DatasetReference] = Field(default_factory=list[DatasetReference], max_length=20)
    evaluation_protocol: str | None = Field(default=None, max_length=2000)
    prerequisites: list[str] = Field(default_factory=list[str], max_length=20)
    baselines_compared_against: list[str] = Field(default_factory=list[str], max_length=20)

    # --- Producer-grounding metadata ---

    context_kind: Literal["paper_extract", "proposal", "reviewer_attested", "standard"]
    confidence_flags: list[ConfidenceFlag] = Field(
        default_factory=list[ConfidenceFlag], max_length=30
    )
    producer_notes: str | None = Field(default=None, max_length=2000)

    # --- Provenance ---

    source_arxiv_id: str | None = Field(default=None, max_length=40)
    source_proposal_id: str | None = Field(default=None, max_length=80)
    source_reviewer: str | None = Field(default=None, max_length=120)

    schema_version: Literal["1"] = "1"

    # --- Validators ---

    @model_validator(mode="after")
    def _enforce_context_kind(self) -> TechniqueDescription:
        """Per-variant enforcement rules (design note §4)."""
        flagged_paths = {f.field_path for f in self.confidence_flags}

        if self.context_kind == "paper_extract":
            if self.source_arxiv_id is None:
                raise ValueError("paper_extract requires source_arxiv_id")
            if self.source_proposal_id is not None:
                raise ValueError("paper_extract forbids source_proposal_id")
            if self.source_reviewer is not None:
                raise ValueError("paper_extract forbids source_reviewer")
            if not self.algorithm.source_excerpts:
                raise ValueError("paper_extract requires algorithm.source_excerpts to be non-empty")
            for i, hp in enumerate(self.hyperparameters):
                if hp.value is not None and hp.source_excerpt is None:
                    raise ValueError(
                        f"paper_extract requires hyperparameters[{i}].source_excerpt "
                        f"when hyperparameters[{i}].value is not None"
                    )
            if self.loss_function is not None and not self.loss_function.source_excerpts:
                raise ValueError(
                    "paper_extract requires loss_function.source_excerpts to be non-empty"
                )
            if self.training_recipe is not None and not self.training_recipe.source_excerpts:
                raise ValueError(
                    "paper_extract requires training_recipe.source_excerpts to be non-empty"
                )
            if self.limitations is None and "/limitations" not in flagged_paths:
                raise ValueError(
                    "paper_extract requires limitations to be set OR flagged "
                    "(extraction must affirmatively address it)"
                )

        elif self.context_kind == "proposal":
            if self.source_proposal_id is None:
                raise ValueError("proposal requires source_proposal_id")
            if self.source_arxiv_id is not None:
                raise ValueError("proposal forbids source_arxiv_id")
            if self.source_reviewer is not None:
                raise ValueError("proposal forbids source_reviewer")
            if self.algorithm.source_excerpts:
                raise ValueError("proposal forbids algorithm.source_excerpts (no paper to quote)")
            for i, hp in enumerate(self.hyperparameters):
                if hp.source_excerpt is not None:
                    raise ValueError(f"proposal forbids hyperparameters[{i}].source_excerpt")
            if self.loss_function is not None and self.loss_function.source_excerpts:
                raise ValueError("proposal forbids loss_function.source_excerpts")
            if self.training_recipe is not None and self.training_recipe.source_excerpts:
                raise ValueError("proposal forbids training_recipe.source_excerpts")

        elif self.context_kind == "reviewer_attested":
            if self.source_reviewer is None:
                raise ValueError("reviewer_attested requires source_reviewer")
            if self.source_arxiv_id is not None:
                raise ValueError("reviewer_attested forbids source_arxiv_id")
            if self.source_proposal_id is not None:
                raise ValueError("reviewer_attested forbids source_proposal_id")
            if not self.algorithm.source_excerpts and "/algorithm" not in flagged_paths:
                raise ValueError(
                    "reviewer_attested requires algorithm.source_excerpts to be "
                    "non-empty OR a /algorithm ConfidenceFlag"
                )

        else:  # context_kind == "standard"
            if self.source_arxiv_id is not None:
                raise ValueError("standard forbids source_arxiv_id")
            if self.source_proposal_id is not None:
                raise ValueError("standard forbids source_proposal_id")
            if self.source_reviewer is not None:
                raise ValueError("standard forbids source_reviewer")

        return self

    @model_validator(mode="after")
    def _none_requires_flag(self) -> TechniqueDescription:
        """A ``None`` for a field that isn't structurally forbidden by
        ``context_kind`` must be matched by a :class:`ConfidenceFlag`
        entry whose ``field_path`` points at it.

        Standard techniques (DEC-015) legitimately leave most fields
        empty and are exempt. ``paper_extract``'s stricter rule on
        ``/limitations`` is handled by :meth:`_enforce_context_kind`
        above; this validator skips it for that variant.
        """
        if self.context_kind == "standard":
            return self

        flagged_paths = {f.field_path for f in self.confidence_flags}
        checked: list[tuple[str, object]] = [
            ("/loss_function", self.loss_function),
            ("/training_recipe", self.training_recipe),
            ("/evaluation_protocol", self.evaluation_protocol),
        ]
        if self.context_kind != "paper_extract":
            checked.append(("/limitations", self.limitations))
        for path, value in checked:
            if value is None and path not in flagged_paths:
                raise ValueError(
                    f"{path} is None but no ConfidenceFlag references it; "
                    "producer must either fill the field or annotate the gap."
                )
        return self


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
