"""Per-fixture integration tests for the mechanical evaluator.

Per ``designs/smai/06-mechanical-evaluation.md`` §1 + §4 + §7. For each of
the four worked-example fixtures (Tier B end-to-end), we hand-craft a
``RawMetrics`` that should produce a known ``pass`` verdict and one that
should produce ``fail``. Round-trips through ``compile_experiment``
(Task 1.6) → ``evaluate(...)``. Pinned ``EvaluationResult`` content hashes
on the canonical "pass" cases catch silent verdict-path drift.
"""

from __future__ import annotations

import math

import pytest
from _emit_helpers import (  # type: ignore[import-not-found]
    EXPERIMENT_FIXTURES,
    compile_experiment_fixture,
    load_fixture_document,
)
from _evaluation_helpers import (  # type: ignore[import-not-found]
    constant_seed_outcomes,
    make_raw_metrics,
    metric_key_for,
    varied_seed_outcomes,
)
from smai_core import (
    EVALUATOR_VERSION,
    ExperimentDocument,
    artifact_hash,
    evaluate,
)


def _close(a: float, b: float, *, tol: float = 1e-9) -> bool:
    return math.isclose(a, b, abs_tol=tol)


# --- per-fixture pass + fail cases ----------------------------------------

# (fixture_name, baseline_id, treatment_id, baseline_value_pass,
#  treatment_value_pass, baseline_value_fail, treatment_value_fail, threshold)
_PASS_FAIL_PAIRS: tuple[tuple[str, str, str, float, float, float, float, float], ...] = (
    # ResNet50 vs VGG16: higher_is_better, threshold 0.005
    (
        "resnet50_vs_vgg16_cifar10",
        "vgg16_reference",
        "resnet50_treatment",
        0.85,
        0.92,
        0.85,
        0.852,
        0.005,
    ),
    # Cutout: higher_is_better, threshold 0.003
    ("cutout_on_cifar10", "no_aug_baseline", "cutout_treatment", 0.90, 0.95, 0.90, 0.901, 0.003),
    # Position embeddings: lower_is_better (perplexity), threshold 0.5
    ("position_embeddings_wikitext103", "no_pe_reference", "rope", 50.0, 30.0, 50.0, 49.9, 0.5),
)


@pytest.mark.parametrize(
    "fixture,baseline_id,treatment_id,b_pass,t_pass,_b_fail,_t_fail,_threshold",
    _PASS_FAIL_PAIRS,
)
def test_fixture_pass_case(
    fixture: str,
    baseline_id: str,
    treatment_id: str,
    b_pass: float,
    t_pass: float,
    _b_fail: float,
    _t_fail: float,
    _threshold: float,
) -> None:
    art_set, _ = compile_experiment_fixture(fixture)
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    seeds = (42, 1337, 2024, 9999, 55)
    rm = make_raw_metrics(
        {
            baseline_id: constant_seed_outcomes(seeds, b_pass, metric_key),
            treatment_id: constant_seed_outcomes(seeds, t_pass, metric_key),
        }
    )
    result = evaluate(config, rm)
    assert result.verdict.result == "pass", result.verdict.result
    [outcome] = [t for t in result.verdict.treatment_outcomes if t.entry_id == treatment_id]
    assert outcome.passed_threshold is True


@pytest.mark.parametrize(
    "fixture,baseline_id,treatment_id,_b_pass,_t_pass,b_fail,t_fail,_threshold",
    _PASS_FAIL_PAIRS,
)
def test_fixture_fail_case(
    fixture: str,
    baseline_id: str,
    treatment_id: str,
    _b_pass: float,
    _t_pass: float,
    b_fail: float,
    t_fail: float,
    _threshold: float,
) -> None:
    art_set, _ = compile_experiment_fixture(fixture)
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    seeds = (42, 1337, 2024, 9999, 55)
    rm = make_raw_metrics(
        {
            baseline_id: constant_seed_outcomes(seeds, b_fail, metric_key),
            treatment_id: constant_seed_outcomes(seeds, t_fail, metric_key),
        }
    )
    result = evaluate(config, rm)
    assert result.verdict.result == "fail", result.verdict.result
    [outcome] = [t for t in result.verdict.treatment_outcomes if t.entry_id == treatment_id]
    assert outcome.passed_threshold is False


def test_pruning_sweep_pass_with_one_passing_treatment() -> None:
    """Multi-treatment fixture (1 baseline + 4 treatments). Pass iff ≥1 wins."""
    art_set, _ = compile_experiment_fixture("pruning_sparsity_sweep")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    seeds = (42, 1337, 2024, 9999, 55)
    rm = make_raw_metrics(
        {
            "dense_baseline": constant_seed_outcomes(seeds, 0.93, metric_key),
            # Only sparsity_50 beats threshold; the others fall below.
            "sparsity_50": constant_seed_outcomes(seeds, 0.95, metric_key),
            "sparsity_70": constant_seed_outcomes(seeds, 0.88, metric_key),
            "sparsity_90": constant_seed_outcomes(seeds, 0.80, metric_key),
            "sparsity_99": constant_seed_outcomes(seeds, 0.55, metric_key),
        }
    )
    result = evaluate(config, rm)
    assert result.verdict.result == "pass"
    passed_count = sum(1 for t in result.verdict.treatment_outcomes if t.passed_threshold)
    assert passed_count == 1


def test_pruning_sweep_fail_when_no_treatment_passes() -> None:
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
    result = evaluate(config, rm)
    assert result.verdict.result == "fail"
    assert all(not t.passed_threshold for t in result.verdict.treatment_outcomes)


# --- evaluator output structure --------------------------------------------


def test_verdict_carries_validation_config_hash() -> None:
    """Per §4: verdict ties itself to the input ``ValidationConfig`` hash."""
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
    assert result.verdict.validation_config_hash == config.envelope.content_hash
    assert result.verdict.compiler_version == config.envelope.compiler_version
    assert result.verdict.evaluator_version == EVALUATOR_VERSION


def test_per_entry_aggregate_includes_baseline_and_treatments() -> None:
    """Per §4: per_entry_aggregate covers every entry; treatment_outcomes excludes baseline."""
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
    result = evaluate(config, rm)
    entries = {p.entry_id for p in result.verdict.per_entry_aggregate}
    assert entries == {
        "dense_baseline",
        "sparsity_50",
        "sparsity_70",
        "sparsity_90",
        "sparsity_99",
    }
    treatments = {t.entry_id for t in result.verdict.treatment_outcomes}
    assert "dense_baseline" not in treatments
    assert treatments == {"sparsity_50", "sparsity_70", "sparsity_90", "sparsity_99"}


def test_aggregation_uses_locked_method_mean() -> None:
    art_set, _ = compile_experiment_fixture("resnet50_vs_vgg16_cifar10")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    rm = make_raw_metrics(
        {
            "vgg16_reference": varied_seed_outcomes(
                {42: 0.84, 1337: 0.85, 2024: 0.86, 9999: 0.83, 55: 0.87}, metric_key
            ),
            "resnet50_treatment": varied_seed_outcomes(
                {42: 0.91, 1337: 0.92, 2024: 0.90, 9999: 0.93, 55: 0.91}, metric_key
            ),
        }
    )
    result = evaluate(config, rm)
    bagg = next(p for p in result.verdict.per_entry_aggregate if p.entry_id == "vgg16_reference")
    tagg = next(p for p in result.verdict.per_entry_aggregate if p.entry_id == "resnet50_treatment")
    assert _close(bagg.aggregated_metric_value, 0.85, tol=1e-9)
    assert _close(tagg.aggregated_metric_value, 0.914, tol=1e-9)


def test_lower_is_better_sign_flips_delta() -> None:
    """Perplexity fixture: treatment with lower value should pass (positive delta)."""
    art_set, _ = compile_experiment_fixture("position_embeddings_wikitext103")
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    seeds = (42, 1337, 2024, 9999, 55)
    rm = make_raw_metrics(
        {
            "no_pe_reference": constant_seed_outcomes(seeds, 50.0, metric_key),
            "absolute_pe": constant_seed_outcomes(seeds, 47.0, metric_key),
            "rope": constant_seed_outcomes(seeds, 45.0, metric_key),
            "alibi": constant_seed_outcomes(seeds, 46.0, metric_key),
        }
    )
    result = evaluate(config, rm)
    rope = next(t for t in result.verdict.treatment_outcomes if t.entry_id == "rope")
    # baseline=50, treatment=45, lower_is_better → signed delta = -(45-50) = +5
    assert _close(rope.delta_from_baseline, 5.0, tol=1e-9)
    assert rope.passed_threshold is True


# --- determinism ----------------------------------------------------------


def test_evaluator_is_deterministic_within_run() -> None:
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
    a = evaluate(config, rm)
    b = evaluate(config, rm)
    assert a.model_dump_json() == b.model_dump_json()


# --- golden EvaluationResult hash regression ------------------------------

# Pinned content hashes for the canonical "pass" cases per fixture. These
# hashes pin both the verdict and verdict-context structure; drift means
# verdict-path output changed and should be reviewed before updating these.
GOLDEN_RESULT_HASHES: dict[str, str] = {
    "resnet50_vs_vgg16_cifar10": (
        "4ac3c3043fecd7d3db28cf18e718da9f244e120192dc5204d7dbdba3da065647"
    ),
    "cutout_on_cifar10": ("ce7a170c0364d34ea465c0acb15c50b51f7eee397697f28c39d685ccb2a5b8f4"),
    "pruning_sparsity_sweep": ("7ad3f7fb2385cdfc97389b7352b0d7f5c687ce4bd7c3088a47e06d3fa1262ab1"),
    "position_embeddings_wikitext103": (
        "248b4c5a80707466a5488472ee932d415f9f674881ed7944d878db2255173230"
    ),
}


def _canonical_pass_raw_metrics(fixture: str) -> tuple[object, object]:
    """Build the canonical pass-case (config, raw_metrics) pair for ``fixture``."""
    art_set, _ = compile_experiment_fixture(fixture)
    config = art_set.validation_config
    metric_key = metric_key_for(config)
    seeds = (42, 1337, 2024, 9999, 55)
    if fixture == "resnet50_vs_vgg16_cifar10":
        rm = make_raw_metrics(
            {
                "vgg16_reference": constant_seed_outcomes(seeds, 0.85, metric_key),
                "resnet50_treatment": constant_seed_outcomes(seeds, 0.92, metric_key),
            }
        )
    elif fixture == "cutout_on_cifar10":
        rm = make_raw_metrics(
            {
                "no_aug_baseline": constant_seed_outcomes(seeds, 0.90, metric_key),
                "cutout_treatment": constant_seed_outcomes(seeds, 0.95, metric_key),
            }
        )
    elif fixture == "pruning_sparsity_sweep":
        rm = make_raw_metrics(
            {
                "dense_baseline": constant_seed_outcomes(seeds, 0.93, metric_key),
                "sparsity_50": constant_seed_outcomes(seeds, 0.95, metric_key),
                "sparsity_70": constant_seed_outcomes(seeds, 0.88, metric_key),
                "sparsity_90": constant_seed_outcomes(seeds, 0.80, metric_key),
                "sparsity_99": constant_seed_outcomes(seeds, 0.55, metric_key),
            }
        )
    elif fixture == "position_embeddings_wikitext103":
        rm = make_raw_metrics(
            {
                "no_pe_reference": constant_seed_outcomes(seeds, 50.0, metric_key),
                "absolute_pe": constant_seed_outcomes(seeds, 47.0, metric_key),
                "rope": constant_seed_outcomes(seeds, 45.0, metric_key),
                "alibi": constant_seed_outcomes(seeds, 46.0, metric_key),
            }
        )
    else:
        raise ValueError(f"unknown fixture {fixture}")
    return config, rm


@pytest.mark.parametrize("name", EXPERIMENT_FIXTURES)
def test_evaluation_result_golden_hash(name: str) -> None:
    """Pinned ``EvaluationResult`` hash per fixture's canonical pass case."""
    config, rm = _canonical_pass_raw_metrics(name)
    result = evaluate(config, rm)  # type: ignore[arg-type]
    actual = artifact_hash(result)
    expected = GOLDEN_RESULT_HASHES[name]
    assert expected != "", (
        f"GOLDEN_RESULT_HASHES[{name!r}] not pinned yet; current hash is {actual}"
    )
    assert actual == expected, (
        f"{name} EvaluationResult hash drift: expected {expected} got {actual}"
    )


def test_pass_case_uses_first_fixture_metric_key_correctly() -> None:
    """Per §3: ``parametric`` family encodes as ``family__k=1`` in v1."""
    doc = load_fixture_document("resnet50_vs_vgg16_cifar10")
    assert isinstance(doc, ExperimentDocument)
    config, rm = _canonical_pass_raw_metrics("resnet50_vs_vgg16_cifar10")
    result = evaluate(config, rm)  # type: ignore[arg-type]
    # The runtime key for ``top_k_accuracy(k=1)`` should be present in
    # per_seed_values for every completed seed.
    for entry_id, by_seed in result.verdict_context.per_seed_values.items():
        for seed, key_to_value in by_seed.items():
            assert "top_k_accuracy__k=1" in key_to_value, (
                f"{entry_id}/{seed}: missing canonical metric key"
            )
