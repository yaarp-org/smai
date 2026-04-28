"""``inconclusive`` and degraded-experiment cases for the evaluator.

Per ``designs/smai/06-mechanical-evaluation.md`` §7. The evaluator is total
over well-formed inputs: experiments that didn't complete enough seeds, or
where NaN/Inf appeared in metric values, return a structured
``inconclusive`` verdict with a populated ``failure_reason`` rather than
raising.
"""

from __future__ import annotations

import math

from _emit_helpers import compile_experiment_fixture  # type: ignore[import-not-found]
from _evaluation_helpers import (  # type: ignore[import-not-found]
    constant_seed_outcomes,
    failed_seed_outcomes,
    make_raw_metrics,
    merge_outcomes,
    metric_key_for,
)
from smai_core import (
    EntryMetrics,
    RawMetrics,
    SeedRunOutcome,
    evaluate,
)

# --- insufficient completed seeds ------------------------------------------


def test_baseline_below_required_seeds_is_inconclusive() -> None:
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    rm = make_raw_metrics(
        {
            "vgg16_reference": merge_outcomes(
                constant_seed_outcomes((42, 1337), 0.85, metric_key),
                failed_seed_outcomes((2024, 9999, 55), reason="OOM at epoch 47"),
            ),
            "resnet50_treatment": constant_seed_outcomes(
                (42, 1337, 2024, 9999, 55), 0.92, metric_key
            ),
        }
    )
    result = evaluate(config, rm)
    assert result.verdict.result == "inconclusive"
    assert result.verdict.treatment_outcomes == []
    assert result.verdict_context.failure_reason is not None
    assert "vgg16_reference" in result.verdict_context.failure_reason


def test_treatment_below_required_seeds_is_inconclusive() -> None:
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    rm = make_raw_metrics(
        {
            "vgg16_reference": constant_seed_outcomes((42, 1337, 2024, 9999, 55), 0.85, metric_key),
            "resnet50_treatment": merge_outcomes(
                constant_seed_outcomes((42, 1337, 2024), 0.92, metric_key),
                failed_seed_outcomes((9999, 55), reason="NaN gradient"),
            ),
        }
    )
    result = evaluate(config, rm)
    assert result.verdict.result == "inconclusive"
    assert result.verdict_context.failure_reason is not None
    assert "resnet50_treatment" in result.verdict_context.failure_reason


def test_inconclusive_records_failed_seed_anomalies() -> None:
    """Per §7: failed seeds appear as ``seed_failed`` anomalies."""
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    rm = make_raw_metrics(
        {
            "vgg16_reference": merge_outcomes(
                constant_seed_outcomes((42,), 0.85, metric_key),
                failed_seed_outcomes((1337, 2024, 9999, 55), reason="timeout"),
            ),
            "resnet50_treatment": constant_seed_outcomes(
                (42, 1337, 2024, 9999, 55), 0.92, metric_key
            ),
        }
    )
    result = evaluate(config, rm)
    seed_failed = [a for a in result.verdict_context.anomalies if a.kind == "seed_failed"]
    assert {a.seed for a in seed_failed} == {1337, 2024, 9999, 55}
    assert all(a.entry_id == "vgg16_reference" for a in seed_failed)


# --- all seeds failed for one entry (degraded but not inconclusive) -------


def test_all_seeds_failed_for_treatment_excludes_treatment_with_anomaly() -> None:
    """Per §7: all-failed treatment is excluded; remaining treatments are evaluated."""
    art_set, _ = compile_experiment_fixture("pruning_sparsity_sweep")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    seeds = (42, 1337, 2024, 9999, 55)
    rm = make_raw_metrics(
        {
            "dense_baseline": constant_seed_outcomes(seeds, 0.93, metric_key),
            "sparsity_50": constant_seed_outcomes(seeds, 0.95, metric_key),
            "sparsity_70": constant_seed_outcomes(seeds, 0.88, metric_key),
            "sparsity_90": constant_seed_outcomes(seeds, 0.80, metric_key),
            "sparsity_99": failed_seed_outcomes(seeds, reason="OOM"),
        }
    )
    result = evaluate(config, rm)
    # sparsity_50 still passes, so verdict is `pass`.
    assert result.verdict.result == "pass"
    treatment_ids = {t.entry_id for t in result.verdict.treatment_outcomes}
    assert "sparsity_99" not in treatment_ids
    all_failed = [a for a in result.verdict_context.anomalies if a.kind == "all_seeds_failed"]
    assert any(a.entry_id == "sparsity_99" for a in all_failed)


def test_all_seeds_failed_for_baseline_is_inconclusive() -> None:
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    seeds = (42, 1337, 2024, 9999, 55)
    rm = make_raw_metrics(
        {
            "vgg16_reference": failed_seed_outcomes(seeds, reason="OOM at epoch 47"),
            "resnet50_treatment": constant_seed_outcomes(seeds, 0.92, metric_key),
        }
    )
    result = evaluate(config, rm)
    assert result.verdict.result == "inconclusive"
    assert result.verdict_context.failure_reason is not None
    assert "vgg16_reference" in result.verdict_context.failure_reason


def test_all_seeds_failed_for_all_entries_is_inconclusive() -> None:
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    config = art_set.validation_config
    seeds = (42, 1337, 2024, 9999, 55)
    rm = make_raw_metrics(
        {
            "vgg16_reference": failed_seed_outcomes(seeds, reason="OOM"),
            "resnet50_treatment": failed_seed_outcomes(seeds, reason="OOM"),
        }
    )
    result = evaluate(config, rm)
    assert result.verdict.result == "inconclusive"
    assert result.verdict.treatment_outcomes == []


# --- NaN handling ----------------------------------------------------------


def test_nan_value_is_filtered_and_logged_as_anomaly() -> None:
    """Per §7: NaN values flagged as ``nan_value`` and excluded from aggregate."""
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    rm = RawMetrics(
        by_entry={
            "vgg16_reference": EntryMetrics(
                entry_id="vgg16_reference",
                seed_outcomes={
                    42: SeedRunOutcome(completed=True, required={metric_key: 0.85}),
                    1337: SeedRunOutcome(completed=True, required={metric_key: 0.84}),
                    2024: SeedRunOutcome(completed=True, required={metric_key: 0.86}),
                    9999: SeedRunOutcome(completed=True, required={metric_key: 0.85}),
                    55: SeedRunOutcome(completed=True, required={metric_key: 0.84}),
                },
            ),
            "resnet50_treatment": EntryMetrics(
                entry_id="resnet50_treatment",
                seed_outcomes={
                    42: SeedRunOutcome(completed=True, required={metric_key: 0.92}),
                    1337: SeedRunOutcome(completed=True, required={metric_key: float("nan")}),
                    2024: SeedRunOutcome(completed=True, required={metric_key: 0.93}),
                    9999: SeedRunOutcome(completed=True, required={metric_key: 0.91}),
                    55: SeedRunOutcome(completed=True, required={metric_key: 0.92}),
                },
            ),
        }
    )
    result = evaluate(config, rm)
    # NaN excluded → treatment has 4 seeds (one short of seed_count_required=5) → inconclusive.
    assert result.verdict.result == "inconclusive"
    nan_anoms = [a for a in result.verdict_context.anomalies if a.kind == "nan_value"]
    assert len(nan_anoms) == 1
    assert nan_anoms[0].entry_id == "resnet50_treatment"
    assert nan_anoms[0].seed == 1337


def test_inf_value_is_filtered_and_logged_as_nan_anomaly() -> None:
    """Per §7: ``Inf`` is treated like NaN for the verdict path."""
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    rm = RawMetrics(
        by_entry={
            "vgg16_reference": EntryMetrics(
                entry_id="vgg16_reference",
                seed_outcomes={
                    seed: SeedRunOutcome(completed=True, required={metric_key: 0.85})
                    for seed in (42, 1337, 2024, 9999, 55)
                },
            ),
            "resnet50_treatment": EntryMetrics(
                entry_id="resnet50_treatment",
                seed_outcomes={
                    42: SeedRunOutcome(completed=True, required={metric_key: 0.92}),
                    1337: SeedRunOutcome(completed=True, required={metric_key: math.inf}),
                    2024: SeedRunOutcome(completed=True, required={metric_key: 0.93}),
                    9999: SeedRunOutcome(completed=True, required={metric_key: 0.91}),
                    55: SeedRunOutcome(completed=True, required={metric_key: 0.92}),
                },
            ),
        }
    )
    result = evaluate(config, rm)
    nan_anoms = [a for a in result.verdict_context.anomalies if a.kind == "nan_value"]
    assert any(a.seed == 1337 and a.entry_id == "resnet50_treatment" for a in nan_anoms)


# --- failure_reason content ------------------------------------------------


def test_inconclusive_failure_reason_includes_per_seed_detail() -> None:
    """Per §5.5: ``failure_reason`` inlines runtime failure strings."""
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    rm = make_raw_metrics(
        {
            "vgg16_reference": merge_outcomes(
                constant_seed_outcomes((42, 1337), 0.85, metric_key),
                {
                    2024: SeedRunOutcome(completed=False, failure_reason="OOM at step 1850"),
                    9999: SeedRunOutcome(completed=False, failure_reason="timeout at 4h"),
                    55: SeedRunOutcome(completed=False, failure_reason="NaN gradient"),
                },
            ),
            "resnet50_treatment": constant_seed_outcomes(
                (42, 1337, 2024, 9999, 55), 0.92, metric_key
            ),
        }
    )
    result = evaluate(config, rm)
    fr = result.verdict_context.failure_reason
    assert fr is not None
    assert "OOM" in fr or "timeout" in fr or "NaN" in fr
    assert "2 of 5" in fr


def test_pass_case_has_no_failure_reason() -> None:
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
    result = evaluate(config, rm)
    assert result.verdict.result == "pass"
    assert result.verdict_context.failure_reason is None
