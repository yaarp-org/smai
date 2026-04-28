"""Trend-observation and multi-treatment tests.

Per ``designs/smai/06-mechanical-evaluation.md`` §4.2 (multi-treatment
verdict semantics) and §5.4 (trend observation). Trend observation is
informational — it never affects the boolean verdict (DEC-031 sub-decision
#7); it grounds narrative writeup only.
"""

from __future__ import annotations

from _emit_helpers import compile_experiment_fixture  # type: ignore[import-not-found]
from _evaluation_helpers import (  # type: ignore[import-not-found]
    constant_seed_outcomes,
    make_raw_metrics,
    metric_key_for,
)
from smai_core import evaluate

# --- Multi-treatment semantics (§4.2) ------------------------------------


def test_multi_treatment_pass_iff_at_least_one_passes() -> None:
    art_set, _ = compile_experiment_fixture("pruning_sparsity_sweep")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    seeds = (42, 1337, 2024, 9999, 55)
    # threshold = 0.01; baseline = 0.93
    rm = make_raw_metrics(
        {
            "dense_baseline": constant_seed_outcomes(seeds, 0.93, metric_key),
            "sparsity_50": constant_seed_outcomes(seeds, 0.95, metric_key),  # +0.02 PASSES
            "sparsity_70": constant_seed_outcomes(seeds, 0.85, metric_key),  # FAILS
            "sparsity_90": constant_seed_outcomes(seeds, 0.70, metric_key),  # FAILS
            "sparsity_99": constant_seed_outcomes(seeds, 0.55, metric_key),  # FAILS
        }
    )
    result = evaluate(config, rm)
    assert result.verdict.result == "pass"
    passed = [t for t in result.verdict.treatment_outcomes if t.passed_threshold]
    failed = [t for t in result.verdict.treatment_outcomes if not t.passed_threshold]
    assert {t.entry_id for t in passed} == {"sparsity_50"}
    assert {t.entry_id for t in failed} == {"sparsity_70", "sparsity_90", "sparsity_99"}


def test_multi_treatment_fail_when_all_treatments_below_threshold() -> None:
    art_set, _ = compile_experiment_fixture("position_embeddings_wikitext103")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    seeds = (42, 1337, 2024, 9999, 55)
    # lower_is_better, threshold 0.5; baseline = 50.0
    # All treatments at 49.9: delta = +0.1 < 0.5 → all FAIL.
    rm = make_raw_metrics(
        {
            "no_pe_reference": constant_seed_outcomes(seeds, 50.0, metric_key),
            "absolute_pe": constant_seed_outcomes(seeds, 49.9, metric_key),
            "rope": constant_seed_outcomes(seeds, 49.95, metric_key),
            "alibi": constant_seed_outcomes(seeds, 49.8, metric_key),
        }
    )
    result = evaluate(config, rm)
    assert result.verdict.result == "fail"
    assert all(not t.passed_threshold for t in result.verdict.treatment_outcomes)


def test_multi_treatment_per_treatment_outcome_carries_threshold_and_margin() -> None:
    """Per §4: each TreatmentOutcome restates threshold and reports margin."""
    art_set, _ = compile_experiment_fixture("pruning_sparsity_sweep")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    seeds = (42, 1337, 2024, 9999, 55)
    rm = make_raw_metrics(
        {
            "dense_baseline": constant_seed_outcomes(seeds, 0.93, metric_key),
            "sparsity_50": constant_seed_outcomes(seeds, 0.95, metric_key),
            "sparsity_70": constant_seed_outcomes(seeds, 0.85, metric_key),
            "sparsity_90": constant_seed_outcomes(seeds, 0.70, metric_key),
            "sparsity_99": constant_seed_outcomes(seeds, 0.55, metric_key),
        }
    )
    result = evaluate(config, rm)
    # threshold restated on every outcome.
    assert {t.threshold for t in result.verdict.treatment_outcomes} == {0.01}
    # tolerance_margin = delta - threshold; positive iff passed.
    for t in result.verdict.treatment_outcomes:
        assert (t.tolerance_margin >= 0) == t.passed_threshold


def test_multi_treatment_delta_summaries_present_per_treatment() -> None:
    art_set, _ = compile_experiment_fixture("pruning_sparsity_sweep")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    seeds = (42, 1337, 2024, 9999, 55)
    rm = make_raw_metrics(
        {
            "dense_baseline": constant_seed_outcomes(seeds, 0.93, metric_key),
            "sparsity_50": constant_seed_outcomes(seeds, 0.95, metric_key),
            "sparsity_70": constant_seed_outcomes(seeds, 0.85, metric_key),
            "sparsity_90": constant_seed_outcomes(seeds, 0.70, metric_key),
            "sparsity_99": constant_seed_outcomes(seeds, 0.55, metric_key),
        }
    )
    result = evaluate(config, rm)
    treatment_ids = {ds.treatment_entry_id for ds in result.verdict_context.delta_summaries}
    assert treatment_ids == {"sparsity_50", "sparsity_70", "sparsity_90", "sparsity_99"}


# --- Trend observation (§5.4) -------------------------------------------


def test_trend_check_unset_yields_no_observation() -> None:
    """Per §5.4: ``trend_check`` not populated → ``trend_observation == None``."""
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    config = art_set.validation_config
    assert config.body.trend_check is None
    metric_key = metric_key_for(config)
    seeds = (42, 1337, 2024, 9999, 55)
    rm = make_raw_metrics(
        {
            "vgg16_reference": constant_seed_outcomes(seeds, 0.85, metric_key),
            "resnet50_treatment": constant_seed_outcomes(seeds, 0.92, metric_key),
        }
    )
    result = evaluate(config, rm)
    assert result.verdict_context.trend_observation is None


def test_pruning_sweep_auto_enables_trend_check_at_emission() -> None:
    """Per Task 1.6: trend_check auto-enabled when entries have NumericValue."""
    art_set, _ = compile_experiment_fixture("pruning_sparsity_sweep")
    assert art_set.validation_config.body.trend_check is not None


def test_trend_observation_requires_entries_argument() -> None:
    """Without ``entries`` we cannot order by NumericValue → observation is None."""
    art_set, _ = compile_experiment_fixture("pruning_sparsity_sweep")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    seeds = (42, 1337, 2024, 9999, 55)
    rm = make_raw_metrics(
        {
            "dense_baseline": constant_seed_outcomes(seeds, 0.93, metric_key),
            "sparsity_50": constant_seed_outcomes(seeds, 0.92, metric_key),
            "sparsity_70": constant_seed_outcomes(seeds, 0.88, metric_key),
            "sparsity_90": constant_seed_outcomes(seeds, 0.80, metric_key),
            "sparsity_99": constant_seed_outcomes(seeds, 0.55, metric_key),
        }
    )
    # No entries supplied — no trend observation even though trend_check is set.
    result = evaluate(config, rm)
    assert result.verdict_context.trend_observation is None


def test_trend_observation_populates_when_entries_supplied() -> None:
    art_set, _ = compile_experiment_fixture("pruning_sparsity_sweep")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    seeds = (42, 1337, 2024, 9999, 55)
    rm = make_raw_metrics(
        {
            "dense_baseline": constant_seed_outcomes(seeds, 0.93, metric_key),
            "sparsity_50": constant_seed_outcomes(seeds, 0.92, metric_key),
            "sparsity_70": constant_seed_outcomes(seeds, 0.88, metric_key),
            "sparsity_90": constant_seed_outcomes(seeds, 0.80, metric_key),
            "sparsity_99": constant_seed_outcomes(seeds, 0.55, metric_key),
        }
    )
    entries = list(art_set.experiment_plan.body.entries)
    result = evaluate(config, rm, entries=entries)
    obs = result.verdict_context.trend_observation
    assert obs is not None
    # Aggregated values: 0.92, 0.88, 0.80, 0.55 — strictly decreasing.
    assert obs.observed_direction == "monotonic_decreasing"
    # trend_check.expected_direction defaults to "no_expectation" per Task 1.6.
    assert obs.expected_direction == "no_expectation"
    assert obs.consistent is True
    assert obs.inversion_count == 0


def test_trend_observation_does_not_change_verdict() -> None:
    """Per §5.4: trend observation is informational only."""
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
            "sparsity_99": constant_seed_outcomes(seeds, 0.55, metric_key),
        }
    )
    entries = list(art_set.experiment_plan.body.entries)
    a = evaluate(config, rm)
    b = evaluate(config, rm, entries=entries)
    assert a.verdict.result == b.verdict.result == "pass"
