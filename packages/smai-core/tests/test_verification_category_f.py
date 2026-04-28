"""Per-rule tests for Category F — Technique compatibility (§5.7)."""

from __future__ import annotations

from _verification_helpers import (
    Entry,
    Factor,
    Level,
    PaperFidelityAnchor,
    codes,
    fixture_registries,
    fixture_technique_registry,
    force_set,
    make_experiment,
    standard_technique,
)
from smai_core.verification.category_f_technique_compatibility import (
    TECHNIQUE_CATEGORY_V1_CLOSED_SET,
    technique_category_in_closed_set,
    technique_factor_type_compatible,
    technique_fidelity_anchor_present_or_standard,
    technique_id_registered,
    technique_params_validate,
)

# technique.id_registered ---------------------------------------------------


def test_id_registered_passes_for_known_techniques() -> None:
    experiment = make_experiment()
    assert technique_id_registered(experiment, fixture_registries()) == []


def test_id_registered_fails_for_unknown_technique() -> None:
    experiment = make_experiment()
    force_set(experiment.entries[1].level, "technique_id", "tech_nope")
    assert "technique.id_registered" in codes(
        technique_id_registered(experiment, fixture_registries())
    )


# technique.factor_type_compatible -----------------------------------------


def test_factor_type_compatible_passes_for_substitutive_arch() -> None:
    experiment = make_experiment()
    assert technique_factor_type_compatible(experiment, fixture_registries()) == []


def test_factor_type_compatible_fails_when_only_additive() -> None:
    """``tech_cutout`` is additive-only; using it under substitutive fails."""
    experiment = make_experiment()
    force_set(experiment.entries[1].level, "technique_id", "tech_cutout")
    assert "technique.factor_type_compatible" in codes(
        technique_factor_type_compatible(experiment, fixture_registries())
    )


# technique.params_validate (and friends) ----------------------------------


def test_params_validate_passes_for_valid_cutout_params() -> None:
    experiment = make_experiment(
        factor=Factor(name="aug", type="additive", description="x"),
        entries=[
            Entry(
                id="b",
                is_baseline=True,
                level=Level(factor="aug", name="absent"),
            ),
            Entry(
                id="t",
                is_baseline=False,
                level=Level(
                    factor="aug",
                    name="cutout",
                    technique_id="tech_cutout",
                    technique_params={"patch_size": 16},
                ),
            ),
        ],
    )
    assert technique_params_validate(experiment, fixture_registries()) == []


def test_params_validate_fails_for_value_out_of_range() -> None:
    experiment = make_experiment(
        factor=Factor(name="aug", type="additive", description="x"),
        entries=[
            Entry(
                id="b",
                is_baseline=True,
                level=Level(factor="aug", name="absent"),
            ),
            Entry(
                id="t",
                is_baseline=False,
                level=Level(
                    factor="aug",
                    name="cutout",
                    technique_id="tech_cutout",
                    technique_params={"patch_size": -1},  # below minimum:1
                ),
            ),
        ],
    )
    assert "technique.params_validate" in codes(
        technique_params_validate(experiment, fixture_registries())
    )


def test_params_validate_emits_params_missing_when_required_field_absent() -> None:
    experiment = make_experiment(
        factor=Factor(name="aug", type="additive", description="x"),
        entries=[
            Entry(
                id="b",
                is_baseline=True,
                level=Level(factor="aug", name="absent"),
            ),
            Entry(
                id="t",
                is_baseline=False,
                level=Level(
                    factor="aug",
                    name="cutout",
                    technique_id="tech_cutout",
                    technique_params=None,  # required patch_size missing
                ),
            ),
        ],
    )
    assert "technique.params_missing" in codes(
        technique_params_validate(experiment, fixture_registries())
    )


def test_params_validate_emits_params_not_accepted_when_no_schema() -> None:
    """Use ``tech_alibi`` (no parameter_schema) with stray params."""
    registries = fixture_registries()
    experiment = make_experiment(
        factor=Factor(name="position_strategy", type="substitutive", description="x"),
        entries=[
            Entry(
                id="b",
                is_baseline=True,
                level=Level(
                    factor="position_strategy",
                    name="none",
                    technique_id="tech_no_position_embedding",
                ),
            ),
            Entry(
                id="t",
                is_baseline=False,
                level=Level(
                    factor="position_strategy",
                    name="alibi",
                    technique_id="tech_alibi",
                    technique_params={"unexpected": "value"},
                ),
            ),
        ],
    )
    assert "technique.params_not_accepted" in codes(
        technique_params_validate(experiment, registries)
    )


# technique.fidelity_anchor_present_or_standard ---------------------------


def test_fidelity_anchor_passes_for_standard_techniques() -> None:
    experiment = make_experiment()
    assert technique_fidelity_anchor_present_or_standard(experiment, fixture_registries()) == []


def test_fidelity_anchor_passes_for_paper_anchor_on_non_standard() -> None:
    techs = fixture_technique_registry()
    techs["tech_resnet50_cifar10"] = standard_technique(
        "tech_resnet50_cifar10",
        category="architecture",
        standard=False,
        fidelity_anchor=PaperFidelityAnchor(doi="10.1234/abc"),
    )
    registries = fixture_registries().model_copy(update={"technique_registry": techs})
    experiment = make_experiment()
    assert technique_fidelity_anchor_present_or_standard(experiment, registries) == []


def test_fidelity_anchor_fails_for_non_standard_no_anchor() -> None:
    techs = fixture_technique_registry()
    techs["tech_resnet50_cifar10"] = standard_technique(
        "tech_resnet50_cifar10",
        category="architecture",
        standard=False,
        fidelity_anchor=None,
    )
    registries = fixture_registries().model_copy(update={"technique_registry": techs})
    experiment = make_experiment()
    assert "technique.fidelity_anchor_present_or_standard" in codes(
        technique_fidelity_anchor_present_or_standard(experiment, registries)
    )


# technique.category_in_closed_set -----------------------------------------


def test_category_closed_set_passes_for_architecture() -> None:
    experiment = make_experiment()
    assert technique_category_in_closed_set(experiment, fixture_registries()) == []


def test_category_closed_set_fails_for_unknown_category() -> None:
    techs = fixture_technique_registry()
    techs["tech_resnet50_cifar10"] = standard_technique(
        "tech_resnet50_cifar10",
        category="architecturee",  # typo
    )
    registries = fixture_registries().model_copy(update={"technique_registry": techs})
    experiment = make_experiment()
    assert "technique.category_in_closed_set" in codes(
        technique_category_in_closed_set(experiment, registries)
    )


def test_v1_closed_set_includes_expected_categories() -> None:
    expected = {
        "architecture",
        "augmentation",
        "optimizer",
        "loss",
        "regularization",
        "normalization",
        "activation",
        "pruning",
        "fine_tuning",
        "position_embedding",
        "self_supervision",
        "distillation",
        "schedule",
        "callback",
        "other",
    }
    assert TECHNIQUE_CATEGORY_V1_CLOSED_SET == expected
