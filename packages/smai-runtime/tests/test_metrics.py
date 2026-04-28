"""Metric-emission contract tests (§10).

The shape this module writes must round-trip through ``smai_core.evaluate``
when assembled into a ``RawMetrics`` instance — that is the load-bearing
property of the runtime / methodology boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from smai_core import (
    AggregationRule,
    AtomicMetricRef,
    ComparisonRule,
    EntryMetrics,
    HarnessContract,
    RawMetrics,
    SeedRunOutcome,
    ValidationConfig,
    ValidationConfigBody,
    evaluate,
)
from smai_runtime import (
    METRICS_FILENAME,
    MetricsContractError,
    build_seed_run_outcome,
    filter_to_known_keys,
    optional_runtime_keys,
    required_runtime_keys,
    validate_metrics_dict,
    write_metrics,
)


def test_validate_required_keys_pass(additive_harness_contract: HarnessContract) -> None:
    validate_metrics_dict({"accuracy": 0.9, "loss": 0.5}, additive_harness_contract)


def test_validate_required_keys_fail(additive_harness_contract: HarnessContract) -> None:
    with pytest.raises(MetricsContractError) as exc:
        validate_metrics_dict({"loss": 0.5}, additive_harness_contract)
    assert "accuracy" in exc.value.missing_keys


def test_required_and_optional_runtime_keys(additive_harness_contract: HarnessContract) -> None:
    assert required_runtime_keys(additive_harness_contract) == ["accuracy"]
    assert optional_runtime_keys(additive_harness_contract) == ["loss"]


def test_filter_to_known_keys_drops_unknown(additive_harness_contract: HarnessContract) -> None:
    out = filter_to_known_keys(
        {"accuracy": 0.9, "loss": 0.5, "rogue": 1.2},
        additive_harness_contract,
    )
    assert out == {"accuracy": 0.9, "loss": 0.5}


def test_write_metrics_roundtrip(
    tmp_path: Path,
    additive_harness_contract: HarnessContract,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    target = write_metrics(
        workspace,
        {"accuracy": 0.93, "loss": 0.12},
        additive_harness_contract,
    )
    assert target == workspace / METRICS_FILENAME
    payload: Any = json.loads(target.read_text())
    assert payload == {"accuracy": 0.93, "loss": 0.12}


def test_build_seed_run_outcome_split(additive_harness_contract: HarnessContract) -> None:
    outcome = build_seed_run_outcome(
        {"accuracy": 0.93, "loss": 0.5, "ignored": 1.0},
        additive_harness_contract,
    )
    assert outcome.completed is True
    assert outcome.required == {"accuracy": 0.93}
    assert outcome.optional == {"loss": 0.5}


def test_build_seed_run_outcome_failed(additive_harness_contract: HarnessContract) -> None:
    outcome = build_seed_run_outcome(
        {},
        additive_harness_contract,
        completed=False,
        failure_reason="OOM",
    )
    assert outcome.completed is False
    assert outcome.required is None
    assert outcome.failure_reason == "OOM"


def _validation_config_for(harness: HarnessContract) -> ValidationConfig:
    """Tiny ValidationConfig consuming RawMetrics so we can call evaluate()."""
    from smai_core import ArtifactEnvelope

    body = ValidationConfigBody(
        parent_experiment_hash=harness.envelope.content_hash,
        metric=AtomicMetricRef(ref="accuracy"),
        direction="higher_is_better",
        aggregation=AggregationRule(method="mean"),
        comparison=ComparisonRule(rule="compare_to_baseline", threshold=0.0, baseline_entry_id="b"),
        seed_count_required=2,
    )
    env = ArtifactEnvelope(
        artifact_kind="validation_config",
        artifact_id="vc",
        schema_version=1,
        compiler_version="0.1.0",
        parent_experiment_id="exp",
        registry_hashes={},
        surface_map={},
        content_hash="vc_hash",
    )
    return ValidationConfig(envelope=env, body=body)


def test_metrics_round_trip_through_smai_core_evaluate(
    additive_harness_contract: HarnessContract,
    tmp_path: Path,
) -> None:
    """End-to-end: write metrics, build SeedRunOutcomes, assemble RawMetrics,
    pass to ``smai_core.evaluate`` and assert a verdict comes out.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()

    seed_outcomes_b: dict[int, SeedRunOutcome] = {}
    seed_outcomes_t: dict[int, SeedRunOutcome] = {}
    for seed in (1, 2, 3):
        write_metrics(
            workspace,
            {"accuracy": 0.80, "loss": 0.6},
            additive_harness_contract,
        )
        b_metrics = json.loads((workspace / METRICS_FILENAME).read_text())
        t_metrics = {"accuracy": 0.85, "loss": 0.5}

        seed_outcomes_b[seed] = build_seed_run_outcome(b_metrics, additive_harness_contract)
        seed_outcomes_t[seed] = build_seed_run_outcome(t_metrics, additive_harness_contract)

    raw = RawMetrics(
        by_entry={
            "b": EntryMetrics(entry_id="b", seed_outcomes=seed_outcomes_b),
            "t": EntryMetrics(entry_id="t", seed_outcomes=seed_outcomes_t),
        }
    )

    vc = _validation_config_for(additive_harness_contract)
    result = evaluate(vc, raw)
    # Deterministic verdict: higher_is_better, threshold 0.0, t > b → pass.
    assert result.verdict.result in {"pass", "fail", "inconclusive"}
    # Specifically: the harness contract has 3 seeds, all completed, t mean
    # is 0.85, b mean is 0.80, threshold 0.0 → pass.
    assert result.verdict.result == "pass"
