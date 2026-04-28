"""Carry-forward tests for emit-time engine behaviors.

* Cost-tagged metric auto-include + dedup warning advisory (§5.5 row 10).
* Trend-check auto-enable (§5.6 row 8).
* ``ComparisonRule.baseline_entry_id`` compiler fill (§2.5).
"""

from __future__ import annotations

import pytest
from _emit_helpers import (  # type: ignore[import-not-found]
    compile_experiment_fixture,
    load_fixture_document,
)
from _verification_helpers import (  # type: ignore[import-not-found]
    basic_validation,
    fixture_registries,
    make_experiment,
)
from smai_core import (
    AtomicMetricRef,
    ParametricMetricRef,
    TrendCheck,
    emit_artifacts,
    metric_ref_to_runtime_key,
    verify,
)
from smai_core.artifacts._emit import _cost_auto_include


def test_cost_auto_include_covers_atomic_cost_entries() -> None:
    refs = _cost_auto_include(fixture_registries())
    keys = {metric_ref_to_runtime_key(r) for r in refs}
    assert {"params", "flops", "latency"}.issubset(keys)


def test_cost_auto_include_expands_parametric_compute_cost() -> None:
    refs = _cost_auto_include(fixture_registries())
    family_keys = {
        metric_ref_to_runtime_key(r)
        for r in refs
        if isinstance(r, ParametricMetricRef) and r.family == "compute_cost"
    }
    # one ref per parameter_values_seen entry; registry ships 4 values.
    assert len(family_keys) == 4
    assert any("training time" in k for k in family_keys)


def test_optional_telemetry_populated_when_user_declares_nothing() -> None:
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    telem = art_set.harness_contract.body.optional_telemetry
    keys = {metric_ref_to_runtime_key(r) for r in telem}
    assert {"params", "flops", "latency"}.issubset(keys)


def test_optional_telemetry_sorted_by_runtime_key() -> None:
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    keys = [metric_ref_to_runtime_key(r) for r in art_set.harness_contract.body.optional_telemetry]
    assert keys == sorted(keys)


def test_user_declared_optional_telemetry_dedupes_with_warning() -> None:
    """User declares ``params`` (already auto-included) → dedupe + warning."""
    registries = fixture_registries()
    experiment = make_experiment(
        validation=basic_validation(
            optional_telemetry=[AtomicMetricRef(ref="params")],
        ),
    )
    verified = verify(experiment, registries)
    art_set, report = emit_artifacts(verified, registries)
    # Telemetry contains "params" exactly once
    keys = [metric_ref_to_runtime_key(r) for r in art_set.harness_contract.body.optional_telemetry]
    assert keys.count("params") == 1
    # Warning surfaced
    assert any(
        w.code == "metric.optional_telemetry_redundant_user_declaration" for w in report.warnings
    )


def test_user_declared_non_redundant_telemetry_added_no_warning() -> None:
    """User declares a distinct atomic metric → folded in, no advisory."""
    registries = fixture_registries()
    experiment = make_experiment(
        validation=basic_validation(
            optional_telemetry=[AtomicMetricRef(ref="perplexity")],
        ),
    )
    verified = verify(experiment, registries)
    art_set, report = emit_artifacts(verified, registries)
    keys = {metric_ref_to_runtime_key(r) for r in art_set.harness_contract.body.optional_telemetry}
    assert "perplexity" in keys
    assert {"params", "flops", "latency"}.issubset(keys)
    assert all(
        w.code != "metric.optional_telemetry_redundant_user_declaration" for w in report.warnings
    )


def test_trend_check_auto_enables_when_all_treatments_have_consistent_value() -> None:
    art_set, _ = compile_experiment_fixture("pruning_sparsity_sweep")
    tc = art_set.validation_config.body.trend_check
    assert tc is not None
    assert tc.expected_direction == "no_expectation"
    assert tc.alert_on_violation is False


def test_trend_check_not_enabled_when_no_numeric_values_present() -> None:
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    assert art_set.validation_config.body.trend_check is None


def test_trend_check_user_declaration_preserved() -> None:
    """User-declared trend_check is honored (auto-enable does not override)."""
    from _emit_helpers import compile_experiment_fixture as _cef  # noqa: F401

    # Pruning sweep has explicit numeric values + user-declared
    # ``trend_check`` semantics suit "monotonic decreasing"; we stub a
    # registry where the user explicitly sets the trend_check on the same
    # fixture-shaped experiment.
    from smai_core import (
        AggregationRule,
        AtomicMetricRef,
        ComparisonRule,
        ControlledConditions,
        Entry,
        ExperimentDefinition,
        Factor,
        Level,
        NumericValue,
        TechniqueParams,
        ValidationCriteria,
    )

    registries = fixture_registries()
    factor = Factor(
        name="pruning_sparsity",
        type="additive",
        description="d",
    )
    entries = [
        Entry(
            id="dense_baseline",
            is_baseline=True,
            level=Level(factor="pruning_sparsity", name="dense", technique_id=None),
        ),
        Entry(
            id="sparsity_50",
            is_baseline=False,
            level=Level(
                factor="pruning_sparsity",
                name="half",
                technique_id="tech_rigl",
                technique_params=TechniqueParams({"sparsity": 0.5}),
                value=NumericValue(value=0.5, kind="continuous", min=0.0, max=1.0),
            ),
        ),
        Entry(
            id="sparsity_90",
            is_baseline=False,
            level=Level(
                factor="pruning_sparsity",
                name="ninety",
                technique_id="tech_rigl",
                technique_params=TechniqueParams({"sparsity": 0.9}),
                value=NumericValue(value=0.9, kind="continuous", min=0.0, max=1.0),
            ),
        ),
    ]
    validation = ValidationCriteria.model_validate(
        {
            "metric": AtomicMetricRef(ref="accuracy").model_dump(),
            "direction": "higher_is_better",
            "aggregation": AggregationRule(method="mean").model_dump(),
            "comparison": {
                "rule": "compare_to_baseline",
                "threshold": 0.01,
            },
            "seed_count_required": 5,
            "trend_check": TrendCheck(
                expected_direction="monotonic_decreasing", alert_on_violation=True
            ).model_dump(),
        },
        context={"smai_mode": "dsl"},
    )
    controlled = ControlledConditions.model_validate(
        {
            "dataset": {"name": "cifar10"},
            "optimization": {"optimizer": "adamw"},
            "seeds": [1, 2, 3, 4, 5],
            "architecture": "resnet50",
        }
    )
    experiment = ExperimentDefinition(
        id="cg_user_trend",
        hypothesis="t",
        factor_model_id=None,
        factors=[factor],
        controlled_conditions=controlled,
        entries=entries,
        validation=validation,
    )
    # Avoid unused-import noise from `ComparisonRule`.
    _ = ComparisonRule
    verified = verify(experiment, registries)
    art_set, _ = emit_artifacts(verified, registries)
    tc = art_set.validation_config.body.trend_check
    assert tc is not None
    assert tc.expected_direction == "monotonic_decreasing"
    assert tc.alert_on_violation is True


@pytest.mark.parametrize(
    "fixture, expected",
    [
        ("resnet50_vs_vgg16_cifar10", "vgg16_reference"),
        ("cutout_on_cifar10", "no_aug_baseline"),
        ("pruning_sparsity_sweep", "dense_baseline"),
        ("position_embeddings_wikitext103", "no_pe_reference"),
    ],
)
def test_baseline_entry_id_filled(fixture: str, expected: str) -> None:
    art_set, _ = compile_experiment_fixture(fixture)
    assert art_set.validation_config.body.comparison.baseline_entry_id == expected


def test_compare_to_target_does_not_fill_baseline_entry_id() -> None:
    registries = fixture_registries()
    experiment = make_experiment(
        validation=basic_validation(rule="compare_to_target", target_value=0.95),
    )
    verified = verify(experiment, registries)
    art_set, _ = emit_artifacts(verified, registries)
    comparison = art_set.validation_config.body.comparison
    assert comparison.rule == "compare_to_target"
    assert comparison.target_value == 0.95
    assert comparison.baseline_entry_id is None


def test_factor_model_documents_round_trip_through_compile_experiment() -> None:
    """Cross-check: ``compile_experiment(FactorModelDocument)`` returns a dict."""
    from smai_core import compile_experiment

    doc = load_fixture_document("factor_model_resnet50_imagenet")
    sets = compile_experiment(doc, fixture_registries())  # type: ignore[arg-type]
    assert isinstance(sets, dict)
    assert len(sets) == 3
