"""Per-rule tests for Category B — Entry / factor compatibility (§5.3)."""

from __future__ import annotations

from _verification_helpers import (
    Entry,
    Level,
    codes,
    fixture_registries,
    force_set,
    make_experiment,
)
from smai_core.verification.category_b_entry_factor_compatibility import (
    entry_at_least_two,
    entry_level_factor_matches,
    entry_level_name_unique,
    entry_no_duplicate_techniques_with_same_params,
    entry_unique_id,
)

# entry.unique_id ------------------------------------------------------------


def test_entry_unique_id_passes_when_distinct() -> None:
    experiment = make_experiment()
    assert entry_unique_id(experiment, fixture_registries()) == []


def test_entry_unique_id_fails_on_duplicates() -> None:
    experiment = make_experiment()
    force_set(experiment.entries[1], "id", experiment.entries[0].id)
    assert "entry.unique_id" in codes(entry_unique_id(experiment, fixture_registries()))


# entry.level_factor_matches ------------------------------------------------


def test_entry_level_factor_matches_passes() -> None:
    experiment = make_experiment()
    assert entry_level_factor_matches(experiment, fixture_registries()) == []


def test_entry_level_factor_matches_fails_when_typo() -> None:
    experiment = make_experiment()
    force_set(experiment.entries[1].level, "factor", "architectures")  # typo
    assert "entry.level_factor_matches" in codes(
        entry_level_factor_matches(experiment, fixture_registries())
    )


# entry.level_name_unique ---------------------------------------------------


def test_entry_level_name_unique_passes() -> None:
    experiment = make_experiment()
    assert entry_level_name_unique(experiment, fixture_registries()) == []


def test_entry_level_name_unique_fails_on_duplicate_names() -> None:
    experiment = make_experiment()
    force_set(experiment.entries[1].level, "name", experiment.entries[0].level.name)
    assert "entry.level_name_unique" in codes(
        entry_level_name_unique(experiment, fixture_registries())
    )


# entry.at_least_two ---------------------------------------------------------


def test_entry_at_least_two_passes_with_two_entries() -> None:
    experiment = make_experiment()
    assert entry_at_least_two(experiment, fixture_registries()) == []


def test_entry_at_least_two_fails_with_one_entry() -> None:
    experiment = make_experiment()
    force_set(experiment, "entries", [experiment.entries[0]])
    assert "entry.at_least_two" in codes(entry_at_least_two(experiment, fixture_registries()))


# entry.no_duplicate_techniques_with_same_params ----------------------------


def test_no_duplicates_passes_for_distinct_treatments() -> None:
    experiment = make_experiment()
    assert entry_no_duplicate_techniques_with_same_params(experiment, fixture_registries()) == []


def test_no_duplicates_warns_on_byte_equal_treatments() -> None:
    experiment = make_experiment(
        entries=[
            Entry(
                id="baseline",
                is_baseline=True,
                level=Level(
                    factor="architecture",
                    name="VGG-16",
                    technique_id="tech_vgg16_cifar10",
                ),
            ),
            Entry(
                id="t1",
                is_baseline=False,
                level=Level(
                    factor="architecture",
                    name="ResNet-50",
                    technique_id="tech_resnet50_cifar10",
                ),
            ),
            Entry(
                id="t2",
                is_baseline=False,
                level=Level(
                    factor="architecture",
                    name="ResNet-50-dup",
                    technique_id="tech_resnet50_cifar10",
                ),
            ),
        ]
    )
    findings = entry_no_duplicate_techniques_with_same_params(experiment, fixture_registries())
    assert "entry.no_duplicate_techniques_with_same_params" in codes(findings)
    assert all(f.severity == "warning" for f in findings)
