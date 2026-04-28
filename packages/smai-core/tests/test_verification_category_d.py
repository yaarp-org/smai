"""Per-rule tests for Category D — Metric well-formedness (§5.5)."""

from __future__ import annotations

from _verification_helpers import (
    AtomicMetricRef,
    basic_validation,
    codes,
    fixture_registries,
    make_experiment,
)
from smai_core.entities.metric import ParametricMetricRef
from smai_core.verification.category_d_metric_well_formedness import (
    metric_atomic_ref_registered,
    metric_cost_metric_in_optional_telemetry,
    metric_direction_matches_registry,
    metric_optional_telemetry_no_overlap_with_required,
    metric_optional_telemetry_well_formed,
    metric_parametric_family_registered,
    metric_parametric_required_parameters_present,
    metric_parametric_value_in_seen_set,
)

# metric.atomic_ref_registered ----------------------------------------------


def test_atomic_ref_registered_passes_for_accuracy() -> None:
    experiment = make_experiment()
    assert metric_atomic_ref_registered(experiment, fixture_registries()) == []


def test_atomic_ref_registered_fails_on_typo() -> None:
    experiment = make_experiment(
        validation=basic_validation(metric=AtomicMetricRef(ref="acuracy")),
    )
    assert "metric.atomic_ref_registered" in codes(
        metric_atomic_ref_registered(experiment, fixture_registries())
    )


# metric.parametric_family_registered ---------------------------------------


def test_parametric_family_passes_for_top_k() -> None:
    experiment = make_experiment(
        validation=basic_validation(
            metric=ParametricMetricRef(family="top_k_accuracy", parameters={"k": 1})
        ),
    )
    assert metric_parametric_family_registered(experiment, fixture_registries()) == []


def test_parametric_family_fails_on_unknown_family() -> None:
    experiment = make_experiment(
        validation=basic_validation(
            metric=ParametricMetricRef(family="bogus_family", parameters={"k": 1})
        ),
    )
    assert "metric.parametric_family_registered" in codes(
        metric_parametric_family_registered(experiment, fixture_registries())
    )


# metric.parametric_required_parameters_present ----------------------------


def test_required_parameters_passes_when_all_present() -> None:
    experiment = make_experiment(
        validation=basic_validation(
            metric=ParametricMetricRef(family="top_k_accuracy", parameters={"k": 1})
        ),
    )
    assert metric_parametric_required_parameters_present(experiment, fixture_registries()) == []


def test_required_parameters_fails_when_missing() -> None:
    experiment = make_experiment(
        validation=basic_validation(
            metric=ParametricMetricRef(family="top_k_accuracy", parameters={})
        ),
    )
    assert "metric.parametric_required_parameters_present" in codes(
        metric_parametric_required_parameters_present(experiment, fixture_registries())
    )


# metric.parametric_value_in_seen_set --------------------------------------


def test_value_in_seen_set_passes_for_top1() -> None:
    experiment = make_experiment(
        validation=basic_validation(
            metric=ParametricMetricRef(family="top_k_accuracy", parameters={"k": 1})
        ),
    )
    assert metric_parametric_value_in_seen_set(experiment, fixture_registries()) == []


def test_value_in_seen_set_fails_for_unseen_k() -> None:
    experiment = make_experiment(
        validation=basic_validation(
            metric=ParametricMetricRef(family="top_k_accuracy", parameters={"k": 7})
        ),
    )
    assert "metric.parametric_value_in_seen_set" in codes(
        metric_parametric_value_in_seen_set(experiment, fixture_registries())
    )


# metric.direction_matches_registry ----------------------------------------


def test_direction_matches_passes_when_consistent() -> None:
    experiment = make_experiment()  # accuracy + higher_is_better
    assert metric_direction_matches_registry(experiment, fixture_registries()) == []


def test_direction_matches_fails_when_contradicts_registry() -> None:
    experiment = make_experiment(
        validation=basic_validation(
            metric=AtomicMetricRef(ref="accuracy"),
            direction="lower_is_better",
        ),
    )
    assert "metric.direction_matches_registry" in codes(
        metric_direction_matches_registry(experiment, fixture_registries())
    )


def test_direction_matches_passes_when_registry_ambiguous() -> None:
    experiment = make_experiment(
        validation=basic_validation(
            metric=AtomicMetricRef(ref="latency"),
            direction="lower_is_better",
        ),
    )
    assert metric_direction_matches_registry(experiment, fixture_registries()) == []


# metric.cost_metric_in_optional_telemetry ----------------------------------


def test_cost_metric_warning_does_not_fire_for_quality_metric() -> None:
    experiment = make_experiment()
    assert metric_cost_metric_in_optional_telemetry(experiment, fixture_registries()) == []


def test_cost_metric_warning_fires_for_cost_verdict() -> None:
    experiment = make_experiment(
        validation=basic_validation(
            metric=AtomicMetricRef(ref="flops"),
            direction="lower_is_better",
        ),
    )
    findings = metric_cost_metric_in_optional_telemetry(experiment, fixture_registries())
    assert "metric.cost_metric_in_optional_telemetry" in codes(findings)
    assert all(f.severity == "warning" for f in findings)


# metric.optional_telemetry_well_formed -------------------------------------


def test_optional_telemetry_well_formed_passes_for_registered_atomic() -> None:
    experiment = make_experiment(
        validation=basic_validation(optional_telemetry=[AtomicMetricRef(ref="latency")]),
    )
    assert metric_optional_telemetry_well_formed(experiment, fixture_registries()) == []


def test_optional_telemetry_well_formed_fails_for_unregistered_atomic() -> None:
    experiment = make_experiment(
        validation=basic_validation(optional_telemetry=[AtomicMetricRef(ref="bogus_metric")]),
    )
    assert "metric.optional_telemetry_well_formed" in codes(
        metric_optional_telemetry_well_formed(experiment, fixture_registries())
    )


# metric.optional_telemetry_no_overlap_with_required ------------------------


def test_optional_telemetry_overlap_passes_when_distinct() -> None:
    experiment = make_experiment(
        validation=basic_validation(
            metric=AtomicMetricRef(ref="accuracy"),
            optional_telemetry=[AtomicMetricRef(ref="latency")],
        ),
    )
    assert (
        metric_optional_telemetry_no_overlap_with_required(experiment, fixture_registries()) == []
    )


def test_optional_telemetry_overlap_warns_on_byte_equal() -> None:
    experiment = make_experiment(
        validation=basic_validation(
            metric=AtomicMetricRef(ref="accuracy"),
            optional_telemetry=[AtomicMetricRef(ref="accuracy")],
        ),
    )
    findings = metric_optional_telemetry_no_overlap_with_required(experiment, fixture_registries())
    assert "metric.optional_telemetry_no_overlap_with_required" in codes(findings)
    assert all(f.severity == "warning" for f in findings)
