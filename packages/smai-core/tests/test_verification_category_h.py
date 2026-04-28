"""Per-rule tests for Category H — FactorModel cross-CG checks (§5.9)."""

from __future__ import annotations

from _verification_helpers import (
    ControlledConditions,
    FactorModel,
    basic_conditions,
    codes,
    fixture_registries,
    force_set,
    make_experiment,
)
from smai_core.verification.category_h_factor_model_cross_cg import (
    factor_model_bidirectional_references_consistent,
    factor_model_cross_cg_factor_vs_control,
    factor_model_factor_name_uniqueness,
    factor_model_shared_conditions_match,
)


def _trio_factor_model():  # type: ignore[no-untyped-def]
    """Two CGs that share both a FactorModel and the typed core conditions."""
    fm = FactorModel(
        id="fm_test",
        research_question="Why?",
        shared_conditions=basic_conditions(),
        comparison_group_ids=["cg_a", "cg_b"],
    )
    cg_a = make_experiment(
        cg_id="cg_a",
        controlled_conditions=basic_conditions(),
        factor_model_id="fm_test",
    )
    cg_b = make_experiment(
        cg_id="cg_b",
        controlled_conditions=basic_conditions(),
        factor_model_id="fm_test",
    )
    return fm, [cg_a, cg_b]


# factor_model.bidirectional_references_consistent -------------------------


def test_bidirectional_passes_when_aligned() -> None:
    fm, cgs = _trio_factor_model()
    assert factor_model_bidirectional_references_consistent(fm, cgs, fixture_registries()) == []


def test_bidirectional_fails_when_id_unknown() -> None:
    fm, cgs = _trio_factor_model()
    force_set(fm, "comparison_group_ids", ["cg_a", "cg_unknown"])
    assert "factor_model.bidirectional_references_consistent" in codes(
        factor_model_bidirectional_references_consistent(fm, cgs, fixture_registries())
    )


def test_bidirectional_fails_when_cg_factor_model_id_mismatched() -> None:
    fm, cgs = _trio_factor_model()
    force_set(cgs[1], "factor_model_id", "fm_other")
    assert "factor_model.bidirectional_references_consistent" in codes(
        factor_model_bidirectional_references_consistent(fm, cgs, fixture_registries())
    )


# factor_model.shared_conditions_match -------------------------------------


def test_shared_conditions_match_passes_when_consistent() -> None:
    fm, cgs = _trio_factor_model()
    assert factor_model_shared_conditions_match(fm, cgs, fixture_registries()) == []


def test_shared_conditions_match_fails_when_value_diverges() -> None:
    fm, cgs = _trio_factor_model()
    cc = basic_conditions()
    cc.dataset["name"] = "imagenet"  # diverges from FactorModel.shared
    force_set(cgs[1], "controlled_conditions", cc)
    assert "factor_model.shared_conditions_match" in codes(
        factor_model_shared_conditions_match(fm, cgs, fixture_registries())
    )


def test_shared_conditions_match_skipped_when_no_shared() -> None:
    fm, cgs = _trio_factor_model()
    force_set(fm, "shared_conditions", None)
    assert factor_model_shared_conditions_match(fm, cgs, fixture_registries()) == []


# factor_model.factor_name_uniqueness --------------------------------------


def test_factor_name_uniqueness_passes_when_distinct() -> None:
    fm, cgs = _trio_factor_model()
    force_set(cgs[1].factors[0], "name", "optimizer_choice")
    findings = factor_model_factor_name_uniqueness(fm, cgs, fixture_registries())
    assert findings == []


def test_factor_name_uniqueness_warns_on_duplicate() -> None:
    fm, cgs = _trio_factor_model()
    findings = factor_model_factor_name_uniqueness(fm, cgs, fixture_registries())
    assert "factor_model.factor_name_uniqueness" in codes(findings)
    assert all(f.severity == "warning" for f in findings)


# factor_model.cross_cg_factor_vs_control ---------------------------------


def test_cross_cg_advisory_does_not_fire_when_no_overlap() -> None:
    fm, cgs = _trio_factor_model()
    force_set(cgs[1].factors[0], "name", "optimizer_choice")
    findings = factor_model_cross_cg_factor_vs_control(fm, cgs, fixture_registries())
    assert findings == []


def test_cross_cg_advisory_fires_on_value_mismatch() -> None:
    fm, cgs = _trio_factor_model()
    force_set(cgs[1].factors[0], "name", "optimizer_choice")
    # cg_a's factor is "architecture"; baseline level name is "VGG-16".
    # cg_b plants "architecture" in its controlled_conditions with a non-matching value.
    cc = ControlledConditions.model_validate(
        {
            "dataset": {"name": "cifar10", "split": "standard"},
            "optimization": {"optimizer": "adamw", "lr": 0.001, "epochs": 100},
            "seeds": [1, 2, 3, 4, 5],
            "architecture": "ResNet-50",  # not "VGG-16"
        }
    )
    force_set(cgs[1], "controlled_conditions", cc)
    findings = factor_model_cross_cg_factor_vs_control(fm, cgs, fixture_registries())
    assert "factor_model.cross_cg_factor_vs_control" in codes(findings)
    assert all(f.severity == "advisory" for f in findings)
