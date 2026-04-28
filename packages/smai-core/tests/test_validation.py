"""Tests for the validation surface entities."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from smai_core import (
    AggregationRule,
    AtomicMetricRef,
    ComparisonRule,
    ParametricMetricRef,
    TrendCheck,
    ValidationCriteria,
)


def test_aggregation_mean() -> None:
    agg = AggregationRule(method="mean")
    assert agg.method == "mean"


def test_aggregation_median() -> None:
    agg = AggregationRule(method="median")
    assert agg.method == "median"


def test_aggregation_rejects_unknown() -> None:
    with pytest.raises(ValidationError):
        AggregationRule.model_validate({"method": "p95"})


def test_aggregation_round_trip() -> None:
    agg = AggregationRule(method="mean")
    assert AggregationRule.model_validate(agg.model_dump(mode="json")) == agg


def test_comparison_baseline_artifact_mode_validates() -> None:
    rule = ComparisonRule(
        rule="compare_to_baseline", threshold=0.01, baseline_entry_id="entry_baseline"
    )
    assert rule.baseline_entry_id == "entry_baseline"


def test_comparison_baseline_dsl_mode_rejects_user_supplied_id() -> None:
    with pytest.raises(ValidationError):
        ComparisonRule.model_validate(
            {
                "rule": "compare_to_baseline",
                "threshold": 0.01,
                "baseline_entry_id": "entry_baseline",
            },
            context={"smai_mode": "dsl"},
        )


def test_comparison_baseline_dsl_mode_validates_without_id() -> None:
    rule = ComparisonRule.model_validate(
        {"rule": "compare_to_baseline", "threshold": 0.01},
        context={"smai_mode": "dsl"},
    )
    assert rule.baseline_entry_id is None


def test_comparison_baseline_artifact_mode_rejects_missing_id() -> None:
    with pytest.raises(ValidationError):
        ComparisonRule.model_validate(
            {"rule": "compare_to_baseline", "threshold": 0.01},
            context={"smai_mode": "artifact"},
        )


def test_comparison_target_validates() -> None:
    rule = ComparisonRule(rule="compare_to_target", threshold=0.0, target_value=0.95)
    assert rule.target_value == 0.95


def test_comparison_target_rejects_missing_value() -> None:
    with pytest.raises(ValidationError):
        ComparisonRule.model_validate({"rule": "compare_to_target", "threshold": 0.0})


def test_comparison_round_trip() -> None:
    rule = ComparisonRule(rule="compare_to_baseline", threshold=0.01, baseline_entry_id="entry_b")
    payload = rule.model_dump(mode="json")
    assert ComparisonRule.model_validate(payload) == rule


def test_trend_check_default() -> None:
    tc = TrendCheck()
    assert tc.expected_direction == "no_expectation"
    assert tc.alert_on_violation is False


def test_trend_check_explicit() -> None:
    tc = TrendCheck(expected_direction="monotonic_increasing", alert_on_violation=True)
    assert tc.expected_direction == "monotonic_increasing"


def test_trend_check_round_trip() -> None:
    tc = TrendCheck(expected_direction="monotonic_decreasing")
    assert TrendCheck.model_validate(tc.model_dump(mode="json")) == tc


def test_validation_criteria_atomic_metric_validates() -> None:
    vc = ValidationCriteria(
        metric=AtomicMetricRef(ref="accuracy"),
        direction="higher_is_better",
        aggregation=AggregationRule(method="mean"),
        comparison=ComparisonRule(
            rule="compare_to_baseline", threshold=0.01, baseline_entry_id="entry_b"
        ),
        seed_count_required=3,
        rationale="Standard accuracy delta over baseline.",
    )
    assert isinstance(vc.metric, AtomicMetricRef)


def test_validation_criteria_parametric_metric_round_trip() -> None:
    vc = ValidationCriteria(
        metric=ParametricMetricRef(family="top_k_accuracy", parameters={"k": 5}),
        direction="higher_is_better",
        aggregation=AggregationRule(method="median"),
        comparison=ComparisonRule(rule="compare_to_target", threshold=0.0, target_value=0.9),
        seed_count_required=5,
        optional_telemetry=[AtomicMetricRef(ref="latency")],
    )
    payload = vc.model_dump(mode="json")
    parsed = ValidationCriteria.model_validate(payload)
    assert parsed == vc
    assert isinstance(parsed.metric, ParametricMetricRef)
    assert parsed.optional_telemetry is not None
    assert isinstance(parsed.optional_telemetry[0], AtomicMetricRef)
