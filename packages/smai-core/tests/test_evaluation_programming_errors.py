"""Programming-error cases for the mechanical evaluator.

Per ``designs/smai/06-mechanical-evaluation.md`` §7.1: structural input
violations raise :class:`RawMetricsShapeError` rather than silently
returning ``inconclusive``. The qualitative split between "the experiment
didn't go well" (returns ``inconclusive``) and "the integrator wired up
the call wrong" (raises) is load-bearing — collapsing the two would erode
the diagnostic value of ``inconclusive``.
"""

from __future__ import annotations

import pytest
from _emit_helpers import compile_experiment_fixture  # type: ignore[import-not-found]
from _evaluation_helpers import (  # type: ignore[import-not-found]
    constant_seed_outcomes,
    make_raw_metrics,
    metric_key_for,
)
from smai_core import (
    EntryMetrics,
    RawMetrics,
    RawMetricsShapeError,
    SeedRunOutcome,
    evaluate,
)


def test_extra_unknown_entry_raises() -> None:
    """Per §7: ``RawMetrics`` references entries not in the experiment."""
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    seeds = (42, 1337, 2024, 9999, 55)
    rm = make_raw_metrics(
        {
            "vgg16_reference": constant_seed_outcomes(seeds, 0.85, metric_key),
            "resnet50_treatment": constant_seed_outcomes(seeds, 0.92, metric_key),
            "phantom_entry": constant_seed_outcomes(seeds, 0.99, metric_key),
        }
    )
    entries = list(art_set.experiment_plan.body.entries)
    with pytest.raises(RawMetricsShapeError) as excinfo:
        evaluate(config, rm, entries=entries)
    assert "phantom_entry" in str(excinfo.value)


def test_missing_entry_id_raises_when_entries_supplied() -> None:
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    seeds = (42, 1337, 2024, 9999, 55)
    rm = make_raw_metrics(
        {
            "vgg16_reference": constant_seed_outcomes(seeds, 0.85, metric_key),
        }
    )
    entries = list(art_set.experiment_plan.body.entries)
    with pytest.raises(RawMetricsShapeError) as excinfo:
        evaluate(config, rm, entries=entries)
    assert "resnet50_treatment" in str(excinfo.value)


def test_missing_baseline_entry_raises() -> None:
    """Per §7: baseline entry must be present for compare_to_baseline."""
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    seeds = (42, 1337, 2024, 9999, 55)
    rm = make_raw_metrics(
        {
            "resnet50_treatment": constant_seed_outcomes(seeds, 0.92, metric_key),
        }
    )
    with pytest.raises(RawMetricsShapeError) as excinfo:
        evaluate(config, rm)
    assert "vgg16_reference" in str(excinfo.value)
    assert "baseline" in str(excinfo.value)


def test_missing_required_metric_key_raises() -> None:
    """Per §7: required metric runtime key must be present in completed seeds."""
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    config = art_set.validation_config
    rm = RawMetrics(
        by_entry={
            "vgg16_reference": EntryMetrics(
                entry_id="vgg16_reference",
                seed_outcomes={
                    42: SeedRunOutcome(completed=True, required={"unrelated_metric": 0.85}),
                },
            ),
            "resnet50_treatment": EntryMetrics(
                entry_id="resnet50_treatment",
                seed_outcomes={
                    42: SeedRunOutcome(completed=True, required={"unrelated_metric": 0.92}),
                },
            ),
        }
    )
    with pytest.raises(RawMetricsShapeError) as excinfo:
        evaluate(config, rm)
    assert "top_k_accuracy__k=1" in str(excinfo.value)


def test_completed_with_required_none_raises() -> None:
    """Per §7: completed=True implies non-None ``required``."""
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    rm = RawMetrics(
        by_entry={
            "vgg16_reference": EntryMetrics(
                entry_id="vgg16_reference",
                seed_outcomes={
                    42: SeedRunOutcome(completed=True, required=None),
                },
            ),
            "resnet50_treatment": EntryMetrics(
                entry_id="resnet50_treatment",
                seed_outcomes={
                    42: SeedRunOutcome(completed=True, required={metric_key: 0.92}),
                },
            ),
        }
    )
    with pytest.raises(RawMetricsShapeError) as excinfo:
        evaluate(config, rm)
    assert "required is None" in str(excinfo.value)


def test_required_metric_value_wrong_type_raises() -> None:
    """A string where a numeric is expected is a programmer error."""
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    rm = RawMetrics.model_validate(
        {
            "by_entry": {
                "vgg16_reference": {
                    "entry_id": "vgg16_reference",
                    "seed_outcomes": {
                        "42": {
                            "completed": True,
                            # str-shaped value coerced through the union;
                            # caught by the evaluator's shape check.
                            "required": {metric_key: 1},
                        },
                    },
                },
                "resnet50_treatment": {
                    "entry_id": "resnet50_treatment",
                    "seed_outcomes": {
                        "42": {
                            "completed": True,
                            "required": {metric_key: 1},
                        },
                    },
                },
            }
        }
    )
    # Manually inject a non-numeric value bypassing Pydantic's coercion to
    # exercise the evaluator's defensive type check (Pydantic rejects most
    # shapes; this confirms the evaluator's belt-and-suspenders check fires
    # if a Tier B integrator builds the dict directly).
    rm.by_entry["vgg16_reference"].seed_outcomes[42].required = {metric_key: "not a number"}  # type: ignore[dict-item]
    with pytest.raises(RawMetricsShapeError) as excinfo:
        evaluate(config, rm)
    assert "non-numeric" in str(excinfo.value)


def test_compare_to_baseline_without_baseline_id_raises() -> None:
    """If compiler somehow emitted compare_to_baseline with null baseline_entry_id."""
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    config = art_set.validation_config
    # Bypass artifact-mode validation by mutating the locked field directly
    # — this simulates a malformed config reaching the evaluator.
    object.__setattr__(config.body.comparison, "baseline_entry_id", None)
    metric_key = metric_key_for(config)
    seeds = (42, 1337, 2024, 9999, 55)
    rm = make_raw_metrics(
        {
            "vgg16_reference": constant_seed_outcomes(seeds, 0.85, metric_key),
            "resnet50_treatment": constant_seed_outcomes(seeds, 0.92, metric_key),
        }
    )
    with pytest.raises(RawMetricsShapeError) as excinfo:
        evaluate(config, rm)
    assert "baseline_entry_id" in str(excinfo.value)


def test_empty_entries_argument_raises() -> None:
    """Empty ``entries`` is treated as a programming error, not optional-passthrough."""
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    seeds = (42, 1337, 2024, 9999, 55)
    rm = make_raw_metrics(
        {
            "vgg16_reference": constant_seed_outcomes(seeds, 0.85, metric_key),
            "resnet50_treatment": constant_seed_outcomes(seeds, 0.92, metric_key),
        }
    )
    with pytest.raises(RawMetricsShapeError):
        evaluate(config, rm, entries=[])
