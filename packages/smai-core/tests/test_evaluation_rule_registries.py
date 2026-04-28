"""Tests for the aggregation + comparison rule registries (Task 1.4).

Behavioral specs come from ``designs/smai/06-mechanical-evaluation.md`` §6.
The full evaluator (Task 1.7) exercises the rules end-to-end against
``RawMetrics``; here we cover the registries' shape, lookup, and per-rule
arithmetic.
"""

from __future__ import annotations

import math

import pytest
from smai_core import (
    AGGREGATION_RULE_REGISTRY,
    COMPARISON_RULE_REGISTRY,
    AggregationFunction,
    AggregationRuleNotRegistered,
    ComparisonFunction,
    ComparisonRuleNotRegistered,
    get_aggregation_rule,
    get_comparison_rule,
)

# `pytest.approx` is convenient but its return type confuses pyright strict
# mode (``reportUnknownMemberType``); a small ``math.isclose`` helper keeps
# the tests strict-clean.
_TOL = 1e-9


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=_TOL)


# --- aggregation registry ---------------------------------------------------


def test_aggregation_registry_exposes_v1_closed_set() -> None:
    assert set(AGGREGATION_RULE_REGISTRY) == {"mean", "median"}


def test_aggregation_rules_satisfy_protocol() -> None:
    for rule in AGGREGATION_RULE_REGISTRY.values():
        assert isinstance(rule, AggregationFunction)


def test_aggregation_rule_names_match_registry_keys() -> None:
    for key, rule in AGGREGATION_RULE_REGISTRY.items():
        assert rule.name == key


def test_get_aggregation_rule_returns_registered_rule() -> None:
    assert get_aggregation_rule("mean") is AGGREGATION_RULE_REGISTRY["mean"]
    assert get_aggregation_rule("median") is AGGREGATION_RULE_REGISTRY["median"]


def test_get_aggregation_rule_raises_for_unknown_name() -> None:
    with pytest.raises(AggregationRuleNotRegistered) as excinfo:
        get_aggregation_rule("trimmed_mean")
    assert "trimmed_mean" in str(excinfo.value)


def test_aggregation_rule_not_registered_is_key_error_subclass() -> None:
    assert issubclass(AggregationRuleNotRegistered, KeyError)


def test_mean_aggregation_matches_definition() -> None:
    rule = get_aggregation_rule("mean")
    assert _close(rule.aggregate([1.0, 2.0, 3.0, 4.0]), 2.5)


def test_median_aggregation_odd_length() -> None:
    rule = get_aggregation_rule("median")
    assert _close(rule.aggregate([1.0, 5.0, 3.0]), 3.0)


def test_median_aggregation_even_length_uses_two_middle_mean() -> None:
    rule = get_aggregation_rule("median")
    assert _close(rule.aggregate([1.0, 2.0, 3.0, 4.0]), 2.5)


def test_aggregation_rules_reject_empty_list() -> None:
    for rule in AGGREGATION_RULE_REGISTRY.values():
        with pytest.raises(ValueError):
            rule.aggregate([])


# --- comparison registry ----------------------------------------------------


def test_comparison_registry_exposes_v1_closed_set() -> None:
    assert set(COMPARISON_RULE_REGISTRY) == {"compare_to_baseline", "compare_to_target"}


def test_comparison_rules_satisfy_protocol() -> None:
    for rule in COMPARISON_RULE_REGISTRY.values():
        assert isinstance(rule, ComparisonFunction)


def test_comparison_rule_names_match_registry_keys() -> None:
    for key, rule in COMPARISON_RULE_REGISTRY.items():
        assert rule.name == key


def test_get_comparison_rule_returns_registered_rule() -> None:
    assert (
        get_comparison_rule("compare_to_baseline")
        is COMPARISON_RULE_REGISTRY["compare_to_baseline"]
    )
    assert get_comparison_rule("compare_to_target") is COMPARISON_RULE_REGISTRY["compare_to_target"]


def test_get_comparison_rule_raises_for_unknown_name() -> None:
    with pytest.raises(ComparisonRuleNotRegistered) as excinfo:
        get_comparison_rule("compare_to_paired_baseline")
    assert "compare_to_paired_baseline" in str(excinfo.value)


def test_comparison_rule_not_registered_is_key_error_subclass() -> None:
    assert issubclass(ComparisonRuleNotRegistered, KeyError)


def test_compute_delta_higher_is_better_returns_treatment_minus_baseline() -> None:
    rule = get_comparison_rule("compare_to_baseline")
    delta = rule.compute_delta(0.81, 0.80, "higher_is_better")
    assert _close(delta, 0.01)


def test_compute_delta_lower_is_better_sign_flips() -> None:
    rule = get_comparison_rule("compare_to_baseline")
    # Treatment (lower-better) of 0.10 versus baseline 0.12 — improvement → +0.02.
    delta = rule.compute_delta(0.10, 0.12, "lower_is_better")
    assert _close(delta, 0.02)


def test_compare_to_target_uses_same_arithmetic_as_baseline() -> None:
    target_rule = get_comparison_rule("compare_to_target")
    baseline_rule = get_comparison_rule("compare_to_baseline")
    target_delta = target_rule.compute_delta(0.81, 0.80, "higher_is_better")
    baseline_delta = baseline_rule.compute_delta(0.81, 0.80, "higher_is_better")
    assert _close(target_delta, baseline_delta)


def test_passed_threshold_returns_true_when_delta_meets_threshold() -> None:
    rule = get_comparison_rule("compare_to_baseline")
    assert rule.passed_threshold(0.01, 0.01, "higher_is_better") is True
    assert rule.passed_threshold(0.02, 0.01, "higher_is_better") is True


def test_passed_threshold_returns_false_when_delta_below_threshold() -> None:
    rule = get_comparison_rule("compare_to_baseline")
    assert rule.passed_threshold(0.005, 0.01, "higher_is_better") is False


def test_passed_threshold_zero_allows_ties() -> None:
    """Per 02-dsl-and-contracts.md §5.6: threshold=0 admits ties."""
    rule = get_comparison_rule("compare_to_baseline")
    assert rule.passed_threshold(0.0, 0.0, "higher_is_better") is True


def test_compute_delta_rejects_unknown_direction() -> None:
    rule = get_comparison_rule("compare_to_baseline")
    with pytest.raises(ValueError):
        rule.compute_delta(1.0, 1.0, "ambiguous")


def test_passed_threshold_rejects_unknown_direction() -> None:
    rule = get_comparison_rule("compare_to_baseline")
    with pytest.raises(ValueError):
        rule.passed_threshold(0.1, 0.05, "ambiguous")
