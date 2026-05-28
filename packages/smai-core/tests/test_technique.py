"""Tests for ``TechniqueRef`` and the ``FidelityAnchor`` discriminated union."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from smai_core import (
    FidelityAnchorAdapter,
    PaperFidelityAnchor,
    ProposalFidelityAnchor,
    ReviewerAttestedFidelityAnchor,
    TechniqueRef,
)


def test_paper_anchor_validates() -> None:
    anchor = PaperFidelityAnchor(doi="10.1145/123", arxiv_id="1234.5678", title="ResNet")
    assert anchor.kind == "paper"


def test_proposal_anchor_validates() -> None:
    anchor = ProposalFidelityAnchor(proposal_id="prop_42", submitted_by="alice")
    assert anchor.kind == "proposal"


def test_reviewer_attested_anchor_validates() -> None:
    anchor = ReviewerAttestedFidelityAnchor(
        spec_text="Use Adam with lr=1e-3, beta1=0.9", attested_by="bob"
    )
    assert anchor.kind == "reviewer_attested"


def test_anchor_round_trip_paper() -> None:
    anchor = PaperFidelityAnchor(doi="10.1145/abc")
    payload = anchor.model_dump(mode="json")
    assert PaperFidelityAnchor.model_validate(payload) == anchor


def test_anchor_round_trip_proposal() -> None:
    anchor = ProposalFidelityAnchor(proposal_id="prop_99")
    payload = anchor.model_dump(mode="json")
    assert ProposalFidelityAnchor.model_validate(payload) == anchor


def test_anchor_round_trip_reviewer() -> None:
    anchor = ReviewerAttestedFidelityAnchor(spec_text="canonical spec")
    payload = anchor.model_dump(mode="json")
    assert ReviewerAttestedFidelityAnchor.model_validate(payload) == anchor


def test_anchor_discriminator_routes_paper() -> None:
    parsed = FidelityAnchorAdapter.validate_python({"kind": "paper", "doi": "10.1/x"})
    assert isinstance(parsed, PaperFidelityAnchor)


def test_anchor_discriminator_routes_proposal() -> None:
    parsed = FidelityAnchorAdapter.validate_python({"kind": "proposal", "proposal_id": "p1"})
    assert isinstance(parsed, ProposalFidelityAnchor)


def test_anchor_discriminator_routes_reviewer() -> None:
    parsed = FidelityAnchorAdapter.validate_python(
        {"kind": "reviewer_attested", "spec_text": "spec"}
    )
    assert isinstance(parsed, ReviewerAttestedFidelityAnchor)


def test_anchor_discriminator_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        FidelityAnchorAdapter.validate_python({"kind": "twitter_thread"})


def test_technique_ref_standard_validates() -> None:
    ref = TechniqueRef(
        id="tech_resnet50",
        name="ResNet-50",
        description="Residual network with 50 layers",
        category="architecture",
        compatible_factor_types=["substitutive"],
        standard=True,
        affects_extension_points=["model"],
        context_kind="standard",
    )
    assert ref.standard is True
    assert ref.fidelity_anchor is None


def test_technique_ref_with_paper_anchor_round_trip() -> None:
    ref = TechniqueRef(
        id="tech_cutout",
        name="Cutout",
        description="Random patch occlusion augmentation",
        category="augmentation",
        compatible_factor_types=["additive"],
        standard=False,
        fidelity_anchor=PaperFidelityAnchor(doi="10.5555/cutout"),
        affects_extension_points=["augment"],
        implies_controlled=["architecture", "optimization"],
        context_kind="paper_extract",
    )
    payload = ref.model_dump(mode="json")
    parsed = TechniqueRef.model_validate(payload)
    assert parsed == ref
    assert isinstance(parsed.fidelity_anchor, PaperFidelityAnchor)


def test_technique_ref_with_reviewer_anchor_round_trip() -> None:
    ref = TechniqueRef(
        id="tech_novel",
        name="Novel",
        description="A novel technique attested by reviewer",
        category="other",
        compatible_factor_types=["additive", "substitutive"],
        standard=False,
        fidelity_anchor=ReviewerAttestedFidelityAnchor(spec_text="see notebook 42"),
        affects_extension_points=[],
        context_kind="reviewer_attested",
    )
    payload = ref.model_dump(mode="json")
    parsed = TechniqueRef.model_validate(payload)
    assert parsed == ref


def test_technique_ref_rejects_unknown_factor_type() -> None:
    with pytest.raises(ValidationError):
        TechniqueRef.model_validate(
            {
                "id": "x",
                "name": "x",
                "description": "x",
                "category": "other",
                "compatible_factor_types": ["pipeline"],
                "affects_extension_points": [],
                "context_kind": "standard",
                "standard": True,
            }
        )
