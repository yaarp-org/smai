"""Per-rule tests for Category C — Controlled conditions (§5.4)."""

from __future__ import annotations

from _verification_helpers import (
    basic_conditions,
    basic_validation,
    codes,
    fixture_registries,
    make_experiment,
)
from smai_core.verification.category_c_controlled_conditions import (
    controls_required_fields_present,
    controls_seeds_count_matches_required,
    controls_seeds_unique,
    controls_technique_implies_required_present,
)

# controls.required_fields_present ------------------------------------------


def test_required_fields_present_passes() -> None:
    experiment = make_experiment()
    assert controls_required_fields_present(experiment, fixture_registries()) == []


def test_required_fields_present_fails_on_missing_dataset_name() -> None:
    cc = basic_conditions()
    cc.dataset.pop("name", None)
    cc.dataset["alias"] = "anonymous"  # keep the dict non-empty so the value-check fires
    experiment = make_experiment(controlled_conditions=cc)
    assert "controls.required_fields_present" in codes(
        controls_required_fields_present(experiment, fixture_registries())
    )


def test_required_fields_present_fails_on_missing_optimizer() -> None:
    cc = basic_conditions()
    cc.optimization.pop("optimizer", None)
    experiment = make_experiment(controlled_conditions=cc)
    assert "controls.required_fields_present" in codes(
        controls_required_fields_present(experiment, fixture_registries())
    )


def test_required_fields_present_fails_on_empty_seeds() -> None:
    experiment = make_experiment(
        controlled_conditions=basic_conditions(seeds=[]),
        validation=basic_validation(seed_count_required=1),
    )
    assert "controls.required_fields_present" in codes(
        controls_required_fields_present(experiment, fixture_registries())
    )


# controls.seeds_count_matches_required -------------------------------------


def test_seeds_count_passes_when_enough() -> None:
    experiment = make_experiment(
        controlled_conditions=basic_conditions(seeds=[1, 2, 3]),
        validation=basic_validation(seed_count_required=3),
    )
    assert controls_seeds_count_matches_required(experiment, fixture_registries()) == []


def test_seeds_count_fails_when_too_few() -> None:
    experiment = make_experiment(
        controlled_conditions=basic_conditions(seeds=[1, 2]),
        validation=basic_validation(seed_count_required=5),
    )
    assert "controls.seeds_count_matches_required" in codes(
        controls_seeds_count_matches_required(experiment, fixture_registries())
    )


# controls.seeds_unique -----------------------------------------------------


def test_seeds_unique_passes_when_distinct() -> None:
    experiment = make_experiment()
    assert controls_seeds_unique(experiment, fixture_registries()) == []


def test_seeds_unique_fails_on_duplicate_values() -> None:
    experiment = make_experiment(
        controlled_conditions=basic_conditions(seeds=[1, 1, 2]),
        validation=basic_validation(seed_count_required=2),
    )
    assert "controls.seeds_unique" in codes(controls_seeds_unique(experiment, fixture_registries()))


# controls.technique_implies_required_present ------------------------------


def test_technique_implies_passes_when_implied_field_present() -> None:
    """tech_vgg16_cifar10 implies 'dataset' which is in core conditions."""
    experiment = make_experiment()
    assert controls_technique_implies_required_present(experiment, fixture_registries()) == []


def test_technique_implies_fails_when_implied_missing() -> None:
    """Plant a technique that implies 'architecture' and use it without that field."""
    registries = fixture_registries()
    # tech_cutout implies "architecture"; use it on an additive factor without
    # an architecture field in controlled_conditions.
    cc = basic_conditions(extra=None)  # no architecture key
    from smai_core import Entry, Factor, Level

    experiment = make_experiment(
        factor=Factor(name="aug", type="additive", description="x"),
        entries=[
            Entry(id="b", is_baseline=True, level=Level(factor="aug", name="absent")),
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
        controlled_conditions=cc,
    )
    assert "controls.technique_implies_required_present" in codes(
        controls_technique_implies_required_present(experiment, registries)
    )
