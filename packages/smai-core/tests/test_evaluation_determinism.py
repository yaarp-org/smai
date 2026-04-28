"""Property-based determinism tests for the mechanical evaluator.

Per ``designs/smai/06-mechanical-evaluation.md`` §1 + §9.3: ``evaluate``
is a pure function — same inputs always produce byte-equal outputs.
Stable-float-ordering and lexicographic key sorting in canonical-JSON
serialization make this a property, not "usually the same."

Tests:

* For any well-formed ``RawMetrics``, two ``evaluate`` calls produce
  byte-equal canonical JSON.
* Permuting the seed-outcome dict order does not change the verdict.
* Permuting the per-entry dict order does not change the verdict.
"""

from __future__ import annotations

from typing import Final

from _emit_helpers import compile_experiment_fixture  # type: ignore[import-not-found]
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from smai_core import (
    EntryMetrics,
    RawMetrics,
    SeedRunOutcome,
    canonical_json,
    evaluate,
    metric_ref_to_runtime_key,
)

_SEEDS: Final[tuple[int, ...]] = (42, 1337, 2024, 9999, 55)


def _scalar_value() -> st.SearchStrategy[float]:
    """Bounded finite floats — keeps the property meaningful for accuracy-shaped metrics."""
    return st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def _build_raw_metrics(metric_key: str, baseline: float, treatment: float) -> RawMetrics:
    return RawMetrics(
        by_entry={
            "vgg16_reference": EntryMetrics(
                entry_id="vgg16_reference",
                seed_outcomes={
                    s: SeedRunOutcome(completed=True, required={metric_key: baseline})
                    for s in _SEEDS
                },
            ),
            "resnet50_treatment": EntryMetrics(
                entry_id="resnet50_treatment",
                seed_outcomes={
                    s: SeedRunOutcome(completed=True, required={metric_key: treatment})
                    for s in _SEEDS
                },
            ),
        }
    )


@given(
    baseline=_scalar_value(),
    treatment=_scalar_value(),
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_evaluate_is_pure(baseline: float, treatment: float) -> None:
    """Per §1: same inputs always produce byte-equal outputs."""
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    config = art_set.validation_config
    metric_key = metric_ref_to_runtime_key(config.body.metric)
    rm = _build_raw_metrics(metric_key, baseline, treatment)
    a = evaluate(config, rm)
    b = evaluate(config, rm)
    assert canonical_json(a) == canonical_json(b)


@given(
    baseline=_scalar_value(),
    treatment=_scalar_value(),
    seed=st.integers(min_value=0, max_value=10**6),
)
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_seed_outcome_dict_order_does_not_change_verdict(
    baseline: float, treatment: float, seed: int
) -> None:
    """Per §9.3: aggregations sort by seed before reducing."""
    import random

    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    config = art_set.validation_config
    metric_key = metric_ref_to_runtime_key(config.body.metric)
    canonical = _build_raw_metrics(metric_key, baseline, treatment)
    rng = random.Random(seed)
    shuffled_baseline_seeds = list(canonical.by_entry["vgg16_reference"].seed_outcomes.items())
    rng.shuffle(shuffled_baseline_seeds)
    shuffled_treatment_seeds = list(canonical.by_entry["resnet50_treatment"].seed_outcomes.items())
    rng.shuffle(shuffled_treatment_seeds)
    shuffled = RawMetrics(
        by_entry={
            "vgg16_reference": EntryMetrics(
                entry_id="vgg16_reference",
                seed_outcomes=dict(shuffled_baseline_seeds),
            ),
            "resnet50_treatment": EntryMetrics(
                entry_id="resnet50_treatment",
                seed_outcomes=dict(shuffled_treatment_seeds),
            ),
        }
    )
    a = evaluate(config, canonical)
    b = evaluate(config, shuffled)
    assert canonical_json(a) == canonical_json(b)


@given(
    baseline=_scalar_value(),
    treatment=_scalar_value(),
)
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_by_entry_dict_order_does_not_change_verdict(baseline: float, treatment: float) -> None:
    """Per §9.3: per-entry iteration sorts by entry_id before comparing."""
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    config = art_set.validation_config
    metric_key = metric_ref_to_runtime_key(config.body.metric)
    canonical = _build_raw_metrics(metric_key, baseline, treatment)
    reversed_order = RawMetrics(
        by_entry={
            "resnet50_treatment": canonical.by_entry["resnet50_treatment"],
            "vgg16_reference": canonical.by_entry["vgg16_reference"],
        }
    )
    a = evaluate(config, canonical)
    b = evaluate(config, reversed_order)
    assert canonical_json(a) == canonical_json(b)
