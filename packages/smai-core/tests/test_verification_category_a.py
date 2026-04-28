"""Per-rule tests for Category A — Factor structure (§5.2)."""

from __future__ import annotations

from _verification_helpers import (
    Entry,
    Factor,
    Level,
    codes,
    fixture_registries,
    force_set,
    make_experiment,
)
from smai_core.verification.category_a_factor_structure import (
    factor_has_entries,
    factor_plugin_validate,
    factor_type_registered,
    factor_unique_name_within_experiment,
)


def test_factor_type_registered_passes_for_substitutive() -> None:
    experiment = make_experiment()
    assert factor_type_registered(experiment, fixture_registries()) == []


def test_factor_type_registered_fails_for_unknown_type() -> None:
    experiment = make_experiment()
    # Bypass the entity-level Literal by post-hoc patching — Pydantic forbids
    # construction with a non-Literal value, but assignment is permissible
    # since Factor is mutable in v1.
    force_set(experiment.factors[0], "type", "pipeline")
    assert "factor.type_registered" in codes(
        factor_type_registered(experiment, fixture_registries())
    )


def test_factor_unique_name_passes_for_single_factor() -> None:
    experiment = make_experiment()
    assert factor_unique_name_within_experiment(experiment, fixture_registries()) == []


def test_factor_unique_name_fails_when_two_factors_share_name() -> None:
    experiment = make_experiment()
    duplicate = Factor(
        name=experiment.factors[0].name,
        type="additive",
        description="dup",
    )
    # Bypass Pydantic Field(max_length=1) for this negative test.
    force_set(experiment, "factors", [experiment.factors[0], duplicate])
    assert "factor.unique_name_within_experiment" in codes(
        factor_unique_name_within_experiment(experiment, fixture_registries())
    )


def test_factor_has_entries_passes_with_two_entries() -> None:
    experiment = make_experiment()
    assert factor_has_entries(experiment, fixture_registries()) == []


def test_factor_has_entries_fails_with_one_entry() -> None:
    experiment = make_experiment()
    # Drop one entry by reaching into __dict__ since the entity is frozen-ish.
    force_set(experiment, "entries", [experiment.entries[0]])
    assert "factor.has_entries" in codes(factor_has_entries(experiment, fixture_registries()))


def test_factor_plugin_validate_passes_on_valid_substitutive() -> None:
    experiment = make_experiment()
    assert factor_plugin_validate(experiment, fixture_registries()) == []


def test_factor_plugin_validate_surfaces_substitutive_baseline_count() -> None:
    experiment = make_experiment(
        entries=[
            Entry(
                id="a",
                is_baseline=False,
                level=Level(
                    factor="architecture",
                    name="VGG-16",
                    technique_id="tech_vgg16_cifar10",
                ),
            ),
            Entry(
                id="b",
                is_baseline=False,
                level=Level(
                    factor="architecture",
                    name="ResNet-50",
                    technique_id="tech_resnet50_cifar10",
                ),
            ),
        ]
    )
    assert "substitutive.baseline_count" in codes(
        factor_plugin_validate(experiment, fixture_registries())
    )


def test_factor_plugin_validate_surfaces_additive_treatment_must_have_technique() -> None:
    experiment = make_experiment(
        factor=Factor(name="dropout", type="additive", description="x"),
        entries=[
            Entry(
                id="baseline",
                is_baseline=True,
                level=Level(factor="dropout", name="absent"),
            ),
            Entry(
                id="treatment",
                is_baseline=False,
                level=Level(factor="dropout", name="present"),  # null technique_id
            ),
        ],
    )
    assert "additive.treatment_must_have_technique" in codes(
        factor_plugin_validate(experiment, fixture_registries())
    )


def test_factor_plugin_validate_no_op_when_plugin_unloaded() -> None:
    """If the factor's type isn't loaded, plugin dispatch is a no-op —
    ``factor.type_registered`` reports the absence separately."""
    experiment = make_experiment()
    # Drop the registry's plugins; dispatch returns no findings.
    registries = fixture_registries().model_copy(update={"factor_type_plugins": {}})
    assert factor_plugin_validate(experiment, registries) == []
