"""Per-rule tests for Category E — Validation criteria soundness (§5.6)."""

from __future__ import annotations

from _verification_helpers import (
    Entry,
    Factor,
    Level,
    NumericValue,
    TrendCheck,
    basic_validation,
    codes,
    fixture_registries,
    force_set,
    make_experiment,
)
from smai_core.verification.category_e_validation_criteria_soundness import (
    validation_aggregation_method_known,
    validation_baseline_entry_id_resolves,
    validation_comparison_rule_well_formed,
    validation_seed_count_recommended_minimum,
    validation_seed_count_required_positive,
    validation_threshold_sign_matches_direction,
    validation_trend_check_applicability,
)

# validation.baseline_entry_id_resolves -------------------------------------


def test_baseline_entry_id_resolves_passes_with_one_baseline() -> None:
    experiment = make_experiment()
    assert validation_baseline_entry_id_resolves(experiment, fixture_registries()) == []


def test_baseline_entry_id_resolves_fails_with_zero_baselines() -> None:
    experiment = make_experiment()
    for entry in experiment.entries:
        force_set(entry, "is_baseline", False)
    assert "validation.baseline_entry_id_resolves" in codes(
        validation_baseline_entry_id_resolves(experiment, fixture_registries())
    )


# validation.threshold_sign_matches_direction ------------------------------


def test_threshold_zero_allowed() -> None:
    experiment = make_experiment(validation=basic_validation(threshold=0.0))
    assert validation_threshold_sign_matches_direction(experiment, fixture_registries()) == []


def test_threshold_negative_rejected() -> None:
    experiment = make_experiment(validation=basic_validation(threshold=-0.01))
    assert "validation.threshold_sign_matches_direction" in codes(
        validation_threshold_sign_matches_direction(experiment, fixture_registries())
    )


# validation.aggregation_method_known --------------------------------------


def test_aggregation_method_known_passes_for_mean() -> None:
    experiment = make_experiment()
    assert validation_aggregation_method_known(experiment, fixture_registries()) == []


def test_aggregation_method_known_fails_when_dropped_from_registry() -> None:
    experiment = make_experiment()
    registries = fixture_registries().model_copy(
        update={"aggregation_rule_registry": {"median": object()}}  # mean dropped
    )
    assert "validation.aggregation_method_known" in codes(
        validation_aggregation_method_known(experiment, registries)
    )


# validation.comparison_rule_well_formed -----------------------------------


def test_comparison_rule_well_formed_passes_for_baseline_no_target() -> None:
    experiment = make_experiment()
    assert validation_comparison_rule_well_formed(experiment, fixture_registries()) == []


def test_comparison_rule_well_formed_fails_for_baseline_with_target_value() -> None:
    experiment = make_experiment(
        validation=basic_validation(rule="compare_to_baseline", target_value=0.85)
    )
    assert "validation.comparison_rule_well_formed" in codes(
        validation_comparison_rule_well_formed(experiment, fixture_registries())
    )


def test_comparison_rule_well_formed_passes_for_target_with_value() -> None:
    experiment = make_experiment(
        validation=basic_validation(rule="compare_to_target", target_value=0.85)
    )
    assert validation_comparison_rule_well_formed(experiment, fixture_registries()) == []


# validation.seed_count_required_positive ----------------------------------


def test_seed_count_positive_passes_for_three() -> None:
    experiment = make_experiment(validation=basic_validation(seed_count_required=3))
    assert validation_seed_count_required_positive(experiment, fixture_registries()) == []


def test_seed_count_positive_fails_for_zero() -> None:
    experiment = make_experiment(validation=basic_validation(seed_count_required=0))
    assert "validation.seed_count_required_positive" in codes(
        validation_seed_count_required_positive(experiment, fixture_registries())
    )


# validation.seed_count_recommended_minimum -------------------------------


def test_seed_count_recommended_passes_for_three() -> None:
    experiment = make_experiment(validation=basic_validation(seed_count_required=3))
    assert validation_seed_count_recommended_minimum(experiment, fixture_registries()) == []


def test_seed_count_recommended_warns_for_one() -> None:
    experiment = make_experiment(
        validation=basic_validation(seed_count_required=1),
    )
    findings = validation_seed_count_recommended_minimum(experiment, fixture_registries())
    assert "validation.seed_count_recommended_minimum" in codes(findings)
    assert all(f.severity == "warning" for f in findings)


# validation.trend_check_applicability -------------------------------------


def test_trend_check_applicability_passes_when_no_trend_check() -> None:
    experiment = make_experiment()
    assert validation_trend_check_applicability(experiment, fixture_registries()) == []


def test_trend_check_applicability_fails_when_no_value_on_treatments() -> None:
    experiment = make_experiment(
        validation=basic_validation(trend_check=TrendCheck()),
    )
    assert "validation.trend_check_applicability" in codes(
        validation_trend_check_applicability(experiment, fixture_registries())
    )


def test_trend_check_applicability_fails_on_kind_mismatch() -> None:
    experiment = make_experiment(
        factor=Factor(name="dropout", type="additive", description="x"),
        entries=[
            Entry(
                id="b",
                is_baseline=True,
                level=Level(factor="dropout", name="absent"),
            ),
            Entry(
                id="t1",
                is_baseline=False,
                level=Level(
                    factor="dropout",
                    name="p05",
                    technique_id="tech_cutout",
                    technique_params={"patch_size": 4},
                    value=NumericValue(value=0.5, kind="continuous"),
                ),
            ),
            Entry(
                id="t2",
                is_baseline=False,
                level=Level(
                    factor="dropout",
                    name="p10",
                    technique_id="tech_cutout",
                    technique_params={"patch_size": 8},
                    value=NumericValue(value=1, kind="ordinal"),
                ),
            ),
        ],
        validation=basic_validation(trend_check=TrendCheck()),
    )
    assert "validation.trend_check_applicability" in codes(
        validation_trend_check_applicability(experiment, fixture_registries())
    )
