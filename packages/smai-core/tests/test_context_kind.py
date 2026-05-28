"""Tests for the planner-refactor Step 2 ``context_kind`` discriminator.

Covers:

* :class:`TechniqueRef.context_kind` required + consistency-check rules
  (``upstream_requirements §1`` mapping).
* Methodology compiler reads :attr:`TechniqueRef.context_kind`
  directly and copies onto :class:`TechniqueContractBody.context_kind`
  — no inference branch at compile time.
* Additive baselines (no :class:`TechniqueRef`) get the synthesized
  ``no_op_baseline`` body discriminator from the compiler, so consumers
  never branch on ``context_kind is None``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from smai_core import (
    PaperFidelityAnchor,
    ProposalFidelityAnchor,
    ReviewerAttestedFidelityAnchor,
    TechniqueRef,
)

# === TechniqueRef.context_kind required + consistency rules ================


def test_paper_anchored_ref_accepts_paper_extract() -> None:
    ref = TechniqueRef(
        id="t",
        name="t",
        description="d",
        category="c",
        compatible_factor_types=["additive"],
        standard=False,
        fidelity_anchor=PaperFidelityAnchor(doi="10.x"),
        affects_extension_points=[],
        context_kind="paper_extract",
    )
    assert ref.context_kind == "paper_extract"


def test_proposal_anchored_ref_accepts_proposal() -> None:
    ref = TechniqueRef(
        id="t",
        name="t",
        description="d",
        category="c",
        compatible_factor_types=["additive"],
        standard=False,
        fidelity_anchor=ProposalFidelityAnchor(proposal_id="p1"),
        affects_extension_points=[],
        context_kind="proposal",
    )
    assert ref.context_kind == "proposal"


def test_reviewer_attested_ref_accepts_reviewer_attested() -> None:
    ref = TechniqueRef(
        id="t",
        name="t",
        description="d",
        category="c",
        compatible_factor_types=["additive"],
        standard=False,
        fidelity_anchor=ReviewerAttestedFidelityAnchor(spec_text="see notebook"),
        affects_extension_points=[],
        context_kind="reviewer_attested",
    )
    assert ref.context_kind == "reviewer_attested"


def test_standard_anchorless_ref_accepts_standard() -> None:
    ref = TechniqueRef(
        id="t",
        name="t",
        description="d",
        category="c",
        compatible_factor_types=["substitutive"],
        standard=True,
        affects_extension_points=[],
        context_kind="standard",
    )
    assert ref.context_kind == "standard"


def test_explicit_context_kind_round_trips() -> None:
    """Producer can set context_kind explicitly; round-trip preserves it."""
    ref = TechniqueRef(
        id="t",
        name="t",
        description="d",
        category="c",
        compatible_factor_types=["additive"],
        standard=False,
        fidelity_anchor=PaperFidelityAnchor(doi="10.x"),
        affects_extension_points=[],
        context_kind="paper_extract",
    )
    payload = ref.model_dump(mode="json")
    parsed = TechniqueRef.model_validate(payload)
    assert parsed.context_kind == "paper_extract"
    assert parsed == ref


def test_missing_context_kind_raises() -> None:
    """``context_kind`` is required (D10: no legacy NULL rows, planner cannot
    infer via construct-time backfill)."""
    with pytest.raises(ValidationError) as excinfo:
        TechniqueRef.model_validate(
            {
                "id": "t",
                "name": "t",
                "description": "d",
                "category": "c",
                "compatible_factor_types": ["additive"],
                "standard": True,
                "affects_extension_points": [],
            }
        )
    assert "context_kind" in str(excinfo.value)


def test_no_op_baseline_rejected_on_ref() -> None:
    """``no_op_baseline`` is a TechniqueContractBody-only variant; rejected
    on TechniqueRef because additive baselines have no ref at all."""
    with pytest.raises(ValidationError, match="no_op_baseline"):
        TechniqueRef(
            id="t",
            name="t",
            description="d",
            category="c",
            compatible_factor_types=["additive"],
            standard=True,
            affects_extension_points=[],
            context_kind="no_op_baseline",  # type: ignore[arg-type]
        )


def test_paper_anchor_mismatch_with_proposal_context_kind_raises() -> None:
    """Paper anchor + ``context_kind="proposal"`` is inconsistent; reject."""
    with pytest.raises(ValidationError, match="disagrees with"):
        TechniqueRef(
            id="t",
            name="t",
            description="d",
            category="c",
            compatible_factor_types=["additive"],
            standard=False,
            fidelity_anchor=PaperFidelityAnchor(doi="10.x"),
            affects_extension_points=[],
            context_kind="proposal",
        )


def test_proposal_anchor_mismatch_with_paper_extract_context_kind_raises() -> None:
    """Proposal anchor + ``context_kind="paper_extract"`` is inconsistent; reject."""
    with pytest.raises(ValidationError, match="disagrees with"):
        TechniqueRef(
            id="t",
            name="t",
            description="d",
            category="c",
            compatible_factor_types=["additive"],
            standard=False,
            fidelity_anchor=ProposalFidelityAnchor(proposal_id="p1"),
            affects_extension_points=[],
            context_kind="paper_extract",
        )


def test_reviewer_anchor_mismatch_with_standard_context_kind_raises() -> None:
    """Reviewer anchor + ``context_kind="standard"`` is inconsistent; reject."""
    with pytest.raises(ValidationError, match="disagrees with"):
        TechniqueRef(
            id="t",
            name="t",
            description="d",
            category="c",
            compatible_factor_types=["additive"],
            standard=False,
            fidelity_anchor=ReviewerAttestedFidelityAnchor(spec_text="see"),
            affects_extension_points=[],
            context_kind="standard",
        )


# === Compiler integration: context_kind reads directly =====================


def test_compiler_copies_context_kind_onto_contract() -> None:
    """The methodology compiler reads ``ref.context_kind`` and copies it
    to ``TechniqueContractBody.context_kind`` — no inference branch.
    Additive baselines (no technique_id) get the synthesized
    ``no_op_baseline`` variant so consumers never branch on None."""
    from _emit_helpers import compile_experiment_fixture  # type: ignore[import-not-found]

    art_set, _ = compile_experiment_fixture("cutout_on_cifar10")
    contracts_by_entry = {c.body.entry_id: c for c in art_set.technique_contracts}
    # cutout_on_cifar10 has a standard cutout technique + an additive baseline.
    assert "cutout_treatment" in contracts_by_entry
    treatment = contracts_by_entry["cutout_treatment"].body
    assert treatment.technique_id is not None
    assert treatment.context_kind == "standard", (
        f"cutout treatment is a DEC-015 standard technique; expected "
        f"context_kind='standard', got {treatment.context_kind!r}"
    )
    # Additive baseline: no technique_id → compiler synthesizes "no_op_baseline".
    baseline = contracts_by_entry["no_aug_baseline"].body
    assert baseline.technique_id is None
    assert baseline.context_kind == "no_op_baseline", (
        f"additive baseline expected context_kind='no_op_baseline', got {baseline.context_kind!r}"
    )


def test_compiler_propagates_paper_extract_context_kind() -> None:
    """A TechniqueRef with a PaperFidelityAnchor → ``"paper_extract"``
    on the contract body. Exercises the compiler's
    ``ref.context_kind → body.context_kind`` projection by swapping
    the cutout fixture's technique registry for a paper-anchored ref.
    """
    from _emit_helpers import compile_experiment_fixture  # type: ignore[import-not-found]
    from _verification_helpers import fixture_registries  # type: ignore[import-not-found]

    base = fixture_registries()
    # Replace the cutout technique with a paper-anchored variant; same
    # name + extension points so the fixture's compile path is otherwise
    # unchanged. Round-trip through model_validate to copy all defaults.
    original = base.technique_registry["tech_cutout"]
    paper_ref = TechniqueRef.model_validate(
        {
            **original.model_dump(mode="python"),
            "standard": False,
            "fidelity_anchor": PaperFidelityAnchor(
                doi="10.5555/cutout", arxiv_id="1708.04552"
            ).model_dump(mode="python"),
            "context_kind": "paper_extract",
        }
    )
    swapped_registries = base.model_copy(
        update={"technique_registry": {**base.technique_registry, "tech_cutout": paper_ref}}
    )

    art_set, _ = compile_experiment_fixture("cutout_on_cifar10", registries=swapped_registries)
    contracts_by_entry = {c.body.entry_id: c for c in art_set.technique_contracts}
    treatment = contracts_by_entry["cutout_treatment"].body
    assert treatment.fidelity_anchor is not None
    assert treatment.fidelity_anchor.kind == "paper"
    assert treatment.context_kind == "paper_extract"
