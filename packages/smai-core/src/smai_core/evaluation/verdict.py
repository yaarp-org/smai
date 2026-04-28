""":class:`Verdict`, :class:`VerdictContext`, :class:`EvaluationResult`.

Per ``designs/smai/06-mechanical-evaluation.md`` §4 (Verdict — locked output)
and §5 (VerdictContext — deterministic narrative-grounding). The two are
co-emitted; their union is :class:`EvaluationResult`.

Per §4.3, :class:`Verdict` deliberately omits statistical-test results,
effect-size CIs, per-seed numbers, and any narrative text. Those fields
either do not exist (statistical tests) or live on :class:`VerdictContext`.

Per §5.1, :class:`VerdictContext` is the *deterministic* side of the
narrative-grounding split — pure aggregation of ``RawMetrics`` plus the
locked :class:`ValidationConfig`. The interpretive side (English narrative)
is the contextual evaluator agent's job in the pipeline layer; it is not
implemented in ``smai-core``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class PerEntryAggregate(BaseModel):
    """Per-entry aggregate metric value plus seed-completion accounting (§4)."""

    model_config = ConfigDict(extra="forbid")

    entry_id: str
    is_baseline: bool
    aggregated_metric_value: float
    seeds_completed: int
    seeds_required: int
    completed_fraction: float


class TreatmentOutcome(BaseModel):
    """Per-treatment delta-from-baseline view (§4)."""

    model_config = ConfigDict(extra="forbid")

    entry_id: str
    aggregated_metric_value: float
    delta_from_baseline: float
    threshold: float
    passed_threshold: bool
    tolerance_margin: float


class Verdict(BaseModel):
    """The locked verdict (§4).

    ``result`` is one of three states (§4.1): ``pass`` (≥1 treatment beat
    baseline by threshold), ``fail`` (no treatment beat threshold under
    sufficient seeds), ``inconclusive`` (insufficient completed seeds, or
    NaN cascade per §7). Per §4.3, this artifact deliberately does NOT
    contain statistical-test results, effect-size CIs, per-seed numbers, or
    narrative text — those live on :class:`VerdictContext`.
    """

    model_config = ConfigDict(extra="forbid")

    result: Literal["pass", "fail", "inconclusive"]
    treatment_outcomes: list[TreatmentOutcome]
    per_entry_aggregate: list[PerEntryAggregate]
    validation_config_hash: str
    raw_metrics_hash: str
    compiler_version: str
    evaluator_version: str


class EntryStatistics(BaseModel):
    """Per-entry statistical summary over completed seeds (§5)."""

    model_config = ConfigDict(extra="forbid")

    entry_id: str
    metric_name: str
    n: int
    mean: float
    std: float
    median: float
    ci_95: tuple[float, float]
    min: float
    max: float


class TreatmentDeltaSummary(BaseModel):
    """Per-treatment delta summary with CI when seed counts permit (§5)."""

    model_config = ConfigDict(extra="forbid")

    treatment_entry_id: str
    metric_name: str
    delta_mean: float
    delta_std: float | None
    delta_ci_95: tuple[float, float] | None
    relative_delta: float


class CostSummary(BaseModel):
    """Per-entry cost-metric summary (§5.2).

    Populated when ``HarnessContract.body.optional_telemetry`` includes
    cost-tagged metrics (``category: "cost"``). ``ratio_to_baseline`` is
    populated when the baseline entry also has a value and that value is
    non-zero — otherwise ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    entry_id: str
    metric_name: str
    mean_value: float
    ratio_to_baseline: float | None = None


class Anomaly(BaseModel):
    """A structured anomaly flagged in :class:`VerdictContext` (§5.3)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "seed_failed",
        "nan_value",
        "outlier",
        "all_seeds_failed",
        "trend_violation",
    ]
    entry_id: str
    seed: int | None
    detail: str


class TrendObservation(BaseModel):
    """Trend observation when ``ValidationConfig.body.trend_check`` is set (§5.4).

    Trend observations DO NOT affect the verdict; they ground narrative
    writeup only (DEC-031 sub-decision #7).
    """

    model_config = ConfigDict(extra="forbid")

    expected_direction: Literal[
        "monotonic_increasing",
        "monotonic_decreasing",
        "no_expectation",
    ]
    observed_direction: Literal[
        "monotonic_increasing",
        "monotonic_decreasing",
        "non_monotonic",
    ]
    consistent: bool
    inversion_count: int


class ReproducibilityMetadata(BaseModel):
    """Provenance bundle for replay (§5).

    The evaluator can fill ``validation_config_hash``, ``compiler_version``,
    ``schema_version``, ``evaluator_version`` from the
    :class:`ValidationConfig` envelope. ``harness_contract_hash``,
    ``experiment_plan_hash``, ``technique_contract_hashes``, and
    ``registry_hashes`` are populated when the caller supplies them via the
    optional :class:`smai_core.evaluation.ReproducibilityInputs` pack;
    otherwise they default to empty.
    """

    model_config = ConfigDict(extra="forbid")

    validation_config_hash: str
    harness_contract_hash: str
    experiment_plan_hash: str
    technique_contract_hashes: dict[str, str]
    schema_version: int
    compiler_version: str
    evaluator_version: str
    registry_hashes: dict[str, str]


class VerdictContext(BaseModel):
    """Deterministic narrative-grounding bundle (§5).

    ``per_seed_values``: ``by_entry`` → seed → metric_name → value, filtered
    to completed seeds.
    ``statistical_summary``: per-entry mean / std / CI95 / min / max.
    ``delta_summaries``: per-treatment vs. baseline deltas.
    ``cost_telemetry``: cost-metric summaries when present.
    ``anomalies``: failed seeds, NaN values, outliers, trend violations.
    ``trend_observation``: present iff
    ``ValidationConfig.body.trend_check`` is populated.
    ``reproducibility``: provenance bundle for replay.
    ``failure_reason``: structured English; populated iff
    ``verdict.result == "inconclusive"``.

    None of these fields affect the boolean verdict.
    """

    model_config = ConfigDict(extra="forbid")

    per_seed_values: dict[str, dict[int, dict[str, float | int]]]
    statistical_summary: dict[str, EntryStatistics]
    delta_summaries: list[TreatmentDeltaSummary]
    cost_telemetry: dict[str, CostSummary] | None
    anomalies: list[Anomaly]
    trend_observation: TrendObservation | None
    reproducibility: ReproducibilityMetadata
    failure_reason: str | None


class EvaluationResult(BaseModel):
    """Co-emitted :class:`Verdict` + :class:`VerdictContext` (§1)."""

    model_config = ConfigDict(extra="forbid")

    verdict: Verdict
    verdict_context: VerdictContext
