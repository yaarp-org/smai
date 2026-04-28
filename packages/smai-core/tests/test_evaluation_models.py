"""Schema tests for the evaluator's input/output Pydantic models.

Per ``designs/smai/06-mechanical-evaluation.md`` §2 (``RawMetrics`` and
friends), §4 (``Verdict``), §5 (``VerdictContext``). Each model validates
a hand-crafted instance, round-trips through JSON, and rejects malformed
input.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError
from smai_core import (
    Anomaly,
    CostSummary,
    EntryMetrics,
    EntryStatistics,
    EvaluationResult,
    PerEntryAggregate,
    RawMetrics,
    ReproducibilityMetadata,
    SeedRunOutcome,
    TreatmentDeltaSummary,
    TreatmentOutcome,
    TrendObservation,
    Verdict,
    VerdictContext,
    raw_metrics_canonical_form,
    raw_metrics_canonical_hash,
)


def _completed(value: float) -> SeedRunOutcome:
    return SeedRunOutcome(completed=True, required={"accuracy": value})


def _raw_metrics_simple() -> RawMetrics:
    return RawMetrics(
        by_entry={
            "baseline": EntryMetrics(
                entry_id="baseline",
                seed_outcomes={1: _completed(0.50), 2: _completed(0.51)},
            ),
            "treatment": EntryMetrics(
                entry_id="treatment",
                seed_outcomes={1: _completed(0.80), 2: _completed(0.82)},
            ),
        }
    )


# --- RawMetrics + EntryMetrics + SeedRunOutcome ----------------------------


def test_seed_run_outcome_requires_explicit_completed() -> None:
    with pytest.raises(PydanticValidationError):
        SeedRunOutcome.model_validate({"required": {"accuracy": 0.9}})


def test_seed_run_outcome_round_trips_via_json() -> None:
    so = SeedRunOutcome(
        completed=True, required={"accuracy": 0.9}, optional={"params": 138_000_000}
    )
    revived = SeedRunOutcome.model_validate_json(so.model_dump_json())
    assert revived == so


def test_seed_run_outcome_failure_path_round_trip() -> None:
    so = SeedRunOutcome(completed=False, failure_reason="OOM at step 1850")
    revived = SeedRunOutcome.model_validate_json(so.model_dump_json())
    assert revived == so


def test_entry_metrics_extra_field_rejected() -> None:
    with pytest.raises(PydanticValidationError):
        EntryMetrics.model_validate({"entry_id": "x", "seed_outcomes": {}, "rogue_field": True})


def test_raw_metrics_round_trip() -> None:
    rm = _raw_metrics_simple()
    revived = RawMetrics.model_validate_json(rm.model_dump_json())
    assert revived == rm


def test_raw_metrics_canonical_form_excludes_failure_reason_and_optional() -> None:
    """Per §2.3: canonical form is the verdict-bearing subset only."""
    rm_a = RawMetrics(
        by_entry={
            "x": EntryMetrics(
                entry_id="x",
                seed_outcomes={
                    1: SeedRunOutcome(
                        completed=True,
                        required={"accuracy": 0.5},
                        optional={"params": 100_000_000},
                        failure_reason=None,
                    ),
                },
            ),
        }
    )
    rm_b = RawMetrics(
        by_entry={
            "x": EntryMetrics(
                entry_id="x",
                seed_outcomes={
                    1: SeedRunOutcome(
                        completed=True,
                        required={"accuracy": 0.5},
                        optional={"params": 999},  # different telemetry
                        failure_reason="ignored",
                    ),
                },
            ),
        }
    )
    assert raw_metrics_canonical_form(rm_a) == raw_metrics_canonical_form(rm_b)
    assert raw_metrics_canonical_hash(rm_a) == raw_metrics_canonical_hash(rm_b)


def test_raw_metrics_canonical_hash_distinguishes_required_values() -> None:
    rm_a = _raw_metrics_simple()
    rm_b = RawMetrics(
        by_entry={
            "baseline": EntryMetrics(
                entry_id="baseline",
                seed_outcomes={1: _completed(0.50), 2: _completed(0.99)},  # changed
            ),
            "treatment": EntryMetrics(
                entry_id="treatment",
                seed_outcomes={1: _completed(0.80), 2: _completed(0.82)},
            ),
        }
    )
    assert raw_metrics_canonical_hash(rm_a) != raw_metrics_canonical_hash(rm_b)


def test_raw_metrics_canonical_hash_is_byte_stable() -> None:
    rm = _raw_metrics_simple()
    assert raw_metrics_canonical_hash(rm) == raw_metrics_canonical_hash(rm)


def test_raw_metrics_canonical_hash_absorbs_low_order_noise() -> None:
    """Per §2.3: float64 → 1e-9 stable rounding absorbs low-order noise."""
    rm_a = RawMetrics(
        by_entry={
            "x": EntryMetrics(
                entry_id="x",
                seed_outcomes={1: SeedRunOutcome(completed=True, required={"v": 0.123456789})},
            )
        }
    )
    rm_b = RawMetrics(
        by_entry={
            "x": EntryMetrics(
                entry_id="x",
                seed_outcomes={
                    1: SeedRunOutcome(completed=True, required={"v": 0.123456789 + 1e-12})
                },
            )
        }
    )
    assert raw_metrics_canonical_hash(rm_a) == raw_metrics_canonical_hash(rm_b)


# --- Verdict + helpers -----------------------------------------------------


def test_verdict_minimum_round_trip() -> None:
    v = Verdict(
        result="pass",
        treatment_outcomes=[
            TreatmentOutcome(
                entry_id="t",
                aggregated_metric_value=0.91,
                delta_from_baseline=0.10,
                threshold=0.005,
                passed_threshold=True,
                tolerance_margin=0.095,
            ),
        ],
        per_entry_aggregate=[
            PerEntryAggregate(
                entry_id="b",
                is_baseline=True,
                aggregated_metric_value=0.81,
                seeds_completed=5,
                seeds_required=5,
                completed_fraction=1.0,
            ),
            PerEntryAggregate(
                entry_id="t",
                is_baseline=False,
                aggregated_metric_value=0.91,
                seeds_completed=5,
                seeds_required=5,
                completed_fraction=1.0,
            ),
        ],
        validation_config_hash="a" * 64,
        raw_metrics_hash="b" * 64,
        compiler_version="0.1.0",
        evaluator_version="0.1.0",
    )
    revived = Verdict.model_validate_json(v.model_dump_json())
    assert revived == v


def test_verdict_rejects_unknown_result_state() -> None:
    """Per §4.1: result is exactly pass | fail | inconclusive."""
    with pytest.raises(PydanticValidationError):
        Verdict.model_validate(
            {
                "result": "weak_pass",
                "treatment_outcomes": [],
                "per_entry_aggregate": [],
                "validation_config_hash": "x",
                "raw_metrics_hash": "x",
                "compiler_version": "0.1.0",
                "evaluator_version": "0.1.0",
            }
        )


def test_verdict_extra_field_rejected() -> None:
    """Per §4.3: locked surface — no agent commentary, no t-test results."""
    with pytest.raises(PydanticValidationError):
        Verdict.model_validate(
            {
                "result": "pass",
                "treatment_outcomes": [],
                "per_entry_aggregate": [],
                "validation_config_hash": "x",
                "raw_metrics_hash": "x",
                "compiler_version": "0.1.0",
                "evaluator_version": "0.1.0",
                "p_value": 0.001,  # forbidden
            }
        )


# --- VerdictContext + helpers ----------------------------------------------


def _stub_repro() -> ReproducibilityMetadata:
    return ReproducibilityMetadata(
        validation_config_hash="x",
        harness_contract_hash="",
        experiment_plan_hash="",
        technique_contract_hashes={},
        schema_version=1,
        compiler_version="0.1.0",
        evaluator_version="0.1.0",
        registry_hashes={},
    )


def test_verdict_context_round_trip() -> None:
    vc = VerdictContext(
        per_seed_values={"b": {1: {"accuracy": 0.5}}},
        statistical_summary={
            "b": EntryStatistics(
                entry_id="b",
                metric_name="accuracy",
                n=1,
                mean=0.5,
                std=0.0,
                median=0.5,
                ci_95=(0.5, 0.5),
                min=0.5,
                max=0.5,
            ),
        },
        delta_summaries=[],
        cost_telemetry=None,
        anomalies=[],
        trend_observation=None,
        reproducibility=_stub_repro(),
        failure_reason=None,
    )
    revived = VerdictContext.model_validate_json(vc.model_dump_json())
    assert revived == vc


def test_anomaly_kind_locked_to_v1_set() -> None:
    """Per §5.3 / §5.5: closed enum of anomaly kinds."""
    with pytest.raises(PydanticValidationError):
        Anomaly.model_validate(
            {"kind": "mysterious_drift", "entry_id": "e", "seed": None, "detail": "x"}
        )


def test_trend_observation_locked_directions() -> None:
    """Per §5.4: closed enum of expected/observed directions."""
    with pytest.raises(PydanticValidationError):
        TrendObservation.model_validate(
            {
                "expected_direction": "monotonic_increasing",
                "observed_direction": "downward",  # forbidden
                "consistent": False,
                "inversion_count": 1,
            }
        )


def test_cost_summary_default_ratio_is_none() -> None:
    cs = CostSummary(entry_id="t", metric_name="params", mean_value=23_528_522.0)
    assert cs.ratio_to_baseline is None


def test_treatment_delta_summary_optional_ci() -> None:
    """Per §5: ``delta_ci_95`` and ``delta_std`` optional when N < 2 either side."""
    tds = TreatmentDeltaSummary(
        treatment_entry_id="t",
        metric_name="accuracy",
        delta_mean=0.10,
        delta_std=None,
        delta_ci_95=None,
        relative_delta=0.20,
    )
    revived = TreatmentDeltaSummary.model_validate_json(tds.model_dump_json())
    assert revived == tds


def test_evaluation_result_round_trip() -> None:
    """``EvaluationResult`` composes the verdict + context (§1)."""
    er = EvaluationResult(
        verdict=Verdict(
            result="inconclusive",
            treatment_outcomes=[],
            per_entry_aggregate=[],
            validation_config_hash="x",
            raw_metrics_hash="y",
            compiler_version="0.1.0",
            evaluator_version="0.1.0",
        ),
        verdict_context=VerdictContext(
            per_seed_values={},
            statistical_summary={},
            delta_summaries=[],
            cost_telemetry=None,
            anomalies=[],
            trend_observation=None,
            reproducibility=_stub_repro(),
            failure_reason="all baseline seeds failed",
        ),
    )
    revived = EvaluationResult.model_validate_json(er.model_dump_json())
    assert revived == er
