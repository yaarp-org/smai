"""Tests for the planner-refactor Step 2 :class:`TechniqueDescription` schema.

Per ``planner_refactor/design_notes/technique_description_schema.md``:

* Round-trip for all 4 ``context_kind`` variants (paper_extract,
  proposal, reviewer_attested, standard).
* Per-``context_kind`` ``model_validator`` rejection cases (§4).
* Sentinel pattern: ``None`` + ``ConfidenceFlag`` (§3).
* Structural validation rules (§5).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from smai_inline_agents.planner import (
    AlgorithmSpec,
    ConfidenceFlag,
    DatasetReference,
    Hyperparameter,
    LossSpec,
    SourceExcerpt,
    SourceLocationInline,
    SourceLocationPage,
    SourceLocationSection,
    TechniqueDescription,
    TrainingRecipe,
)

# === Round-trip per context_kind ============================================


def _paper_extract_example() -> TechniqueDescription:
    return TechniqueDescription(
        name="cutout",
        summary=(
            "Cutout masks random square patches of input images with zeros "
            "during training, encouraging the network to use broader context."
        ),
        motivation=(
            "Random crop and flip leave small high-contrast regions intact, "
            "letting models over-rely on local features; cutout forces broader "
            "feature use."
        ),
        problem_setting=(
            "Image classification with CNNs; demonstrated on CIFAR-10 / CIFAR-100 / SVHN."
        ),
        limitations=(
            "Mask size requires tuning per dataset; aggressive cutout can "
            "remove the class-discriminative region on small objects."
        ),
        algorithm=AlgorithmSpec(
            summary=(
                "Apply a square mask of side length L at a uniformly random "
                "location during training; mask values are zero; L is tuned."
            ),
            source_excerpts=[
                SourceExcerpt(
                    text=(
                        "We apply the cutout augmentation by randomly masking "
                        "out square regions of input during training."
                    ),
                    location=SourceLocationSection(section_id="3.1"),
                ),
            ],
        ),
        hyperparameters=[
            Hyperparameter(
                name="cutout_size",
                summary="Side length L of the square mask in pixels.",
                value="16",
                default_for_smai="16",
                source_excerpt=SourceExcerpt(
                    text="For CIFAR-10, we use a cutout size of 16x16 pixels.",
                    location=SourceLocationSection(section_id="4.1"),
                ),
            ),
        ],
        context_kind="paper_extract",
        confidence_flags=[
            ConfidenceFlag(
                field_path="/loss_function",
                severity="unknown",
                note="Cutout does not introduce a custom loss; field None.",
            ),
            ConfidenceFlag(
                field_path="/training_recipe",
                severity="unknown",
                note="Paper does not constrain the training recipe.",
            ),
            ConfidenceFlag(
                field_path="/evaluation_protocol",
                severity="unknown",
                note="Paper does not pin an evaluation protocol; uses standard test acc.",
            ),
        ],
        source_arxiv_id="1708.04552",
    )


def _proposal_example() -> TechniqueDescription:
    return TechniqueDescription(
        name="hybrid_loss_warmup",
        summary=(
            "Hybrid loss warmup blends cross-entropy with knowledge "
            "distillation across training epochs, annealing linearly."
        ),
        motivation=(
            "Pure cross-entropy and pure distillation both have failure modes; "
            "a scheduled blend may capture benefits of both."
        ),
        problem_setting=(
            "Supervised image classification with an available teacher model; "
            "student is smaller and trained from scratch."
        ),
        algorithm=AlgorithmSpec(
            summary=(
                "Total loss at epoch e of E: L = (1 - e/E) CE + (e/E) KL; "
                "linear schedule, no LR coupling."
            ),
        ),
        context_kind="proposal",
        confidence_flags=[
            ConfidenceFlag(
                field_path="/limitations",
                severity="unknown",
                note="Planner has not yet evaluated failure modes.",
            ),
            ConfidenceFlag(
                field_path="/loss_function",
                severity="unknown",
                note="No custom loss introduced beyond the schedule blend.",
            ),
            ConfidenceFlag(
                field_path="/training_recipe",
                severity="unknown",
                note="Training recipe unconstrained by this proposal.",
            ),
            ConfidenceFlag(
                field_path="/evaluation_protocol",
                severity="unknown",
                note="To be determined at experiment-spec phase.",
            ),
        ],
        source_proposal_id="prop-2026-001",
    )


def _reviewer_attested_example() -> TechniqueDescription:
    return TechniqueDescription(
        name="reviewer_attested_widget",
        summary=(
            "Reviewer-attested technique with an inline spec source quoted "
            "from a private notebook the reviewer verified."
        ),
        motivation=(
            "Reviewer-attested techniques cover internal knowledge not yet "
            "published; this is the minimum valid round-trip case."
        ),
        problem_setting=(
            "Any setting that admits the reviewer's spec; reviewer is "
            "responsible for downstream fidelity."
        ),
        algorithm=AlgorithmSpec(
            summary=(
                "Reviewer-attested algorithm; full spec lives in the inline "
                "excerpt below per ReviewerAttestedFidelityAnchor convention."
            ),
            source_excerpts=[
                SourceExcerpt(
                    text="See reviewer spec notebook 42.",
                    location=SourceLocationInline(note="Reviewer-attested; spec not in any paper."),
                ),
            ],
        ),
        context_kind="reviewer_attested",
        confidence_flags=[
            ConfidenceFlag(
                field_path="/limitations",
                severity="unknown",
                note="Reviewer did not document limitations.",
            ),
            ConfidenceFlag(
                field_path="/loss_function",
                severity="unknown",
                note="No custom loss.",
            ),
            ConfidenceFlag(
                field_path="/training_recipe",
                severity="unknown",
                note="No training recipe constraint.",
            ),
            ConfidenceFlag(
                field_path="/evaluation_protocol",
                severity="unknown",
                note="Reviewer did not specify an evaluation protocol.",
            ),
        ],
        source_reviewer="bob",
    )


def _standard_example() -> TechniqueDescription:
    return TechniqueDescription(
        name="random_horizontal_flip",
        summary=(
            "Mirrors each input image left-to-right with probability 0.5; "
            "the standard horizontal-flip augmentation."
        ),
        motivation=(
            "Horizontal symmetry is a near-universal property of natural "
            "images; flipping doubles effective training-set size at zero cost."
        ),
        problem_setting=(
            "Image classification / detection / segmentation where the task "
            "labels are invariant under horizontal mirroring."
        ),
        algorithm=AlgorithmSpec(
            summary=(
                "Sample u ~ Uniform(0,1) per image per epoch; if u<p (default "
                "0.5), flip horizontally. Available as RandomHorizontalFlip."
            ),
        ),
        hyperparameters=[
            Hyperparameter(
                name="p",
                summary="Probability of applying the flip on a given image.",
                value="0.5",
                default_for_smai="0.5",
            ),
        ],
        context_kind="standard",
    )


@pytest.mark.parametrize(
    "factory",
    [_paper_extract_example, _proposal_example, _reviewer_attested_example, _standard_example],
)
def test_round_trip_through_pydantic(
    factory: object,
) -> None:
    """Each ``context_kind`` variant round-trips byte-equal through Pydantic."""
    td = factory()  # type: ignore[operator]
    payload = td.model_dump(mode="json")
    parsed = TechniqueDescription.model_validate(payload)
    assert parsed == td


# === Per-context_kind validator rejection cases =============================


def test_paper_extract_requires_source_arxiv_id() -> None:
    with pytest.raises(ValidationError, match="paper_extract requires source_arxiv_id"):
        TechniqueDescription(
            name="xx",
            summary="x" * 50,
            motivation="x" * 50,
            problem_setting="x" * 50,
            algorithm=AlgorithmSpec(
                summary="x" * 50,
                source_excerpts=[
                    SourceExcerpt(
                        text="quote",
                        location=SourceLocationSection(section_id="3"),
                    )
                ],
            ),
            context_kind="paper_extract",
            source_arxiv_id=None,
            limitations="limit" * 11,
        )


def test_paper_extract_requires_algorithm_excerpts() -> None:
    with pytest.raises(ValidationError, match="algorithm.source_excerpts to be non-empty"):
        TechniqueDescription(
            name="xx",
            summary="x" * 50,
            motivation="x" * 50,
            problem_setting="x" * 50,
            algorithm=AlgorithmSpec(summary="x" * 50, source_excerpts=[]),
            context_kind="paper_extract",
            source_arxiv_id="1708.04552",
            limitations="limit" * 11,
        )


def test_paper_extract_requires_per_hyperparameter_excerpt() -> None:
    with pytest.raises(ValidationError, match="hyperparameters\\[0\\].source_excerpt"):
        TechniqueDescription(
            name="xx",
            summary="x" * 50,
            motivation="x" * 50,
            problem_setting="x" * 50,
            algorithm=AlgorithmSpec(
                summary="x" * 50,
                source_excerpts=[
                    SourceExcerpt(
                        text="q",
                        location=SourceLocationSection(section_id="3"),
                    )
                ],
            ),
            hyperparameters=[
                Hyperparameter(name="lr", summary="learning rate", value="0.1"),
            ],
            context_kind="paper_extract",
            source_arxiv_id="1708.04552",
            limitations="limit" * 11,
        )


def test_paper_extract_requires_limitations_or_flag() -> None:
    with pytest.raises(
        ValidationError,
        match="paper_extract requires limitations to be set OR flagged",
    ):
        TechniqueDescription(
            name="xx",
            summary="x" * 50,
            motivation="x" * 50,
            problem_setting="x" * 50,
            algorithm=AlgorithmSpec(
                summary="x" * 50,
                source_excerpts=[
                    SourceExcerpt(
                        text="q",
                        location=SourceLocationSection(section_id="3"),
                    )
                ],
            ),
            context_kind="paper_extract",
            source_arxiv_id="1708.04552",
            limitations=None,
        )


def test_proposal_forbids_algorithm_excerpts() -> None:
    with pytest.raises(
        ValidationError,
        match="proposal forbids algorithm.source_excerpts",
    ):
        TechniqueDescription(
            name="xx",
            summary="x" * 50,
            motivation="x" * 50,
            problem_setting="x" * 50,
            algorithm=AlgorithmSpec(
                summary="x" * 50,
                source_excerpts=[
                    SourceExcerpt(
                        text="q",
                        location=SourceLocationSection(section_id="3"),
                    )
                ],
            ),
            context_kind="proposal",
            source_proposal_id="p1",
            confidence_flags=[
                ConfidenceFlag(
                    field_path="/limitations",
                    severity="unknown",
                    note="field unset for fixture",
                ),
                ConfidenceFlag(
                    field_path="/loss_function",
                    severity="unknown",
                    note="field unset for fixture",
                ),
                ConfidenceFlag(
                    field_path="/training_recipe",
                    severity="unknown",
                    note="field unset for fixture",
                ),
                ConfidenceFlag(
                    field_path="/evaluation_protocol",
                    severity="unknown",
                    note="field unset for fixture",
                ),
            ],
        )


def test_proposal_requires_source_proposal_id() -> None:
    with pytest.raises(ValidationError, match="proposal requires source_proposal_id"):
        TechniqueDescription(
            name="xx",
            summary="x" * 50,
            motivation="x" * 50,
            problem_setting="x" * 50,
            algorithm=AlgorithmSpec(summary="x" * 50),
            context_kind="proposal",
            source_proposal_id=None,
            confidence_flags=[
                ConfidenceFlag(
                    field_path=f"/{p}",
                    severity="unknown",
                    note="field unset for fixture",
                )
                for p in (
                    "limitations",
                    "loss_function",
                    "training_recipe",
                    "evaluation_protocol",
                )
            ],
        )


def test_reviewer_attested_requires_algorithm_excerpts_or_flag() -> None:
    with pytest.raises(
        ValidationError,
        match="reviewer_attested requires algorithm.source_excerpts",
    ):
        TechniqueDescription(
            name="xx",
            summary="x" * 50,
            motivation="x" * 50,
            problem_setting="x" * 50,
            algorithm=AlgorithmSpec(summary="x" * 50),
            context_kind="reviewer_attested",
            source_reviewer="bob",
            confidence_flags=[
                ConfidenceFlag(
                    field_path=f"/{p}",
                    severity="unknown",
                    note="field unset for fixture",
                )
                for p in (
                    "limitations",
                    "loss_function",
                    "training_recipe",
                    "evaluation_protocol",
                )
            ],
        )


def test_standard_forbids_provenance_pointers() -> None:
    with pytest.raises(ValidationError, match="standard forbids source_arxiv_id"):
        TechniqueDescription(
            name="xx",
            summary="x" * 50,
            motivation="x" * 50,
            problem_setting="x" * 50,
            algorithm=AlgorithmSpec(summary="x" * 50),
            context_kind="standard",
            source_arxiv_id="1708.04552",
        )


# === Sentinel pattern (§3) ==================================================


def test_none_field_requires_confidence_flag() -> None:
    """A ``None`` for a non-forbidden field must be matched by a flag."""
    with pytest.raises(
        ValidationError,
        match="/loss_function is None but no ConfidenceFlag",
    ):
        TechniqueDescription(
            name="xx",
            summary="x" * 50,
            motivation="x" * 50,
            problem_setting="x" * 50,
            algorithm=AlgorithmSpec(summary="x" * 50),
            context_kind="proposal",
            source_proposal_id="p1",
            loss_function=None,
            # /training_recipe and /evaluation_protocol are flagged; /loss_function is not.
            confidence_flags=[
                ConfidenceFlag(
                    field_path="/limitations",
                    severity="unknown",
                    note="field unset for fixture",
                ),
                ConfidenceFlag(
                    field_path="/training_recipe",
                    severity="unknown",
                    note="field unset for fixture",
                ),
                ConfidenceFlag(
                    field_path="/evaluation_protocol",
                    severity="unknown",
                    note="field unset for fixture",
                ),
            ],
        )


def test_standard_skips_none_requires_flag_check() -> None:
    """DEC-015 standard techniques legitimately leave most fields empty."""
    td = _standard_example()
    assert td.loss_function is None
    assert td.training_recipe is None
    assert td.evaluation_protocol is None
    assert td.confidence_flags == []


# === Source-excerpt validation ==============================================


def test_source_excerpt_blank_text_rejected() -> None:
    with pytest.raises(ValidationError, match="non-blank"):
        SourceExcerpt(
            text="   ",
            location=SourceLocationSection(section_id="3"),
        )


def test_source_location_discriminator_round_trip() -> None:
    """Discriminated union picks the right variant on rehydration."""
    excerpts = [
        SourceExcerpt(
            text="section quote",
            location=SourceLocationSection(section_id="3.2"),
        ),
        SourceExcerpt(
            text="page quote",
            location=SourceLocationPage(page_number=7),
        ),
        SourceExcerpt(
            text="inline quote",
            location=SourceLocationInline(note="From an abstract."),
        ),
    ]
    for excerpt in excerpts:
        payload = excerpt.model_dump(mode="json")
        parsed = SourceExcerpt.model_validate(payload)
        assert parsed == excerpt


# === Structural validation rules (§5) =======================================


def test_name_slug_pattern_rejected_for_uppercase() -> None:
    with pytest.raises(ValidationError):
        TechniqueDescription(
            name="Cutout",  # uppercase not allowed
            summary="x" * 50,
            motivation="x" * 50,
            problem_setting="x" * 50,
            algorithm=AlgorithmSpec(summary="x" * 50),
            context_kind="standard",
        )


def test_hyperparameter_name_must_be_python_identifier() -> None:
    with pytest.raises(ValidationError):
        Hyperparameter(name="My-Param", summary="learning rate")


def test_paper_extract_full_shape_with_loss_and_training_recipe() -> None:
    """Exercise the full hybrid shape with loss + training recipe excerpts."""
    td = TechniqueDescription(
        name="hybrid_paper_technique",
        summary=(
            "Paper-derived technique exercising every hybrid field at once "
            "for the schema round-trip test."
        ),
        motivation=(
            "Need a fixture covering AlgorithmSpec + LossSpec + TrainingRecipe "
            "with verbatim source excerpts to validate the full hybrid path."
        ),
        problem_setting=(
            "Synthetic; fixture targets schema completeness, not a real ML problem setting."
        ),
        limitations="Mock limitations for the round-trip fixture.",
        algorithm=AlgorithmSpec(
            summary=(
                "Mock algorithm summary; satisfies AlgorithmSpec's minimum "
                "length rule for the round-trip fixture."
            ),
            source_excerpts=[
                SourceExcerpt(
                    text="Algorithm quote from paper.",
                    location=SourceLocationSection(section_id="3"),
                )
            ],
        ),
        loss_function=LossSpec(
            summary=(
                "Mock loss summary; satisfies LossSpec's minimum length rule "
                "for the round-trip fixture."
            ),
            formula="L = CE(p, y)",
            source_excerpts=[
                SourceExcerpt(
                    text="Loss quote from paper.",
                    location=SourceLocationSection(section_id="4"),
                )
            ],
        ),
        training_recipe=TrainingRecipe(
            summary=(
                "Mock training recipe summary; satisfies TrainingRecipe's "
                "minimum length rule for the round-trip fixture."
            ),
            optimizer="AdamW",
            schedule="cosine",
            batch_size="128",
            epochs="100",
            source_excerpts=[
                SourceExcerpt(
                    text="Training recipe quote.",
                    location=SourceLocationSection(section_id="5"),
                )
            ],
        ),
        datasets=[DatasetReference(name="CIFAR-10")],
        context_kind="paper_extract",
        confidence_flags=[
            ConfidenceFlag(
                field_path="/evaluation_protocol",
                severity="unknown",
                note="Paper does not specify an evaluation protocol.",
            ),
        ],
        source_arxiv_id="9999.99999",
    )
    payload = td.model_dump(mode="json")
    parsed = TechniqueDescription.model_validate(payload)
    assert parsed == td
    assert parsed.loss_function is not None
    assert parsed.training_recipe is not None
