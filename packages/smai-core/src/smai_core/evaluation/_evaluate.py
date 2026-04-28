""":func:`evaluate` — pure deterministic projection from
``(ValidationConfig, RawMetrics)`` to :class:`EvaluationResult`.

Per ``designs/smai/06-mechanical-evaluation.md`` §1, §4–§7. No I/O, no LLM,
no global state. Same inputs always produce byte-equal outputs.

Failure-mode summary (§7):

* All seeds completed (or sufficient successful subset) → ``pass`` / ``fail``.
* Insufficient completed seeds for any entry → ``inconclusive``.
* All seeds of a treatment failed → exclude treatment, log
  ``all_seeds_failed`` anomaly, proceed with remaining treatments.
* All seeds of the baseline failed (compare_to_baseline) →
  ``inconclusive``.
* ``RawMetrics`` references entries not in the verdict config or omits the
  required metric key → :class:`RawMetricsShapeError` (programming error
  per §7.1; raises rather than silently returning ``inconclusive``).

Verdict semantics for multi-treatment experiments per §4.2: ``pass`` iff at
least one treatment beats baseline by threshold; ``fail`` otherwise.
"""

from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING, Final, cast

from smai_core.evaluation.aggregation import get_aggregation_rule
from smai_core.evaluation.comparison import get_comparison_rule
from smai_core.evaluation.raw_metrics import (
    EntryMetrics,
    RawMetrics,
    raw_metrics_canonical_hash,
)
from smai_core.evaluation.verdict import (
    Anomaly,
    CostSummary,
    EntryStatistics,
    EvaluationResult,
    PerEntryAggregate,
    ReproducibilityMetadata,
    TreatmentDeltaSummary,
    TreatmentOutcome,
    TrendObservation,
    Verdict,
    VerdictContext,
)

# Deferred imports to avoid the circular load triggered by the
# ``_runtime_keys → entities/__init__ → registries → evaluation`` chain.
# At module-import time (which can be re-entered partway through
# ``_runtime_keys`` initialization), the symbols below are not yet defined
# in their source modules; resolving them lazily inside :func:`evaluate`
# (and helpers) is safe because by then those modules have completed
# initialization.
if TYPE_CHECKING:
    from smai_core.artifacts.validation_config import ValidationConfig
    from smai_core.entities.factor import Entry

EVALUATOR_VERSION: Final[str] = "0.1.0"
"""Semver of the mechanical evaluator (per §4 + §9.1).

Stamped into :class:`Verdict.evaluator_version` and
:class:`ReproducibilityMetadata.evaluator_version`. Bumped per the §9.1
versioning rules: major on aggregation/comparison semantic changes; minor
on backward-compatible field additions; patch on bug fixes (which are
explicitly allowed to change verdict outputs on re-evaluation).
"""

_CI95_Z: Final[float] = 1.959963984540054
"""Two-sided normal 97.5th-percentile multiplier for ~95% CI."""


class RawMetricsShapeError(Exception):
    """Raised when ``RawMetrics`` violates the locked-config input contract.

    Per §7 + §7.1. The two qualitatively different failure shapes —
    "experiment didn't go well" (``inconclusive``) vs. "integrator wired
    up the call wrong" (raises) — are deliberately separated. Naming this
    exception lets integrators catch it as a stable type rather than parse
    message strings.

    Triggers:

    * ``RawMetrics`` references entries not present in
      ``ValidationConfig.body.comparison.baseline_entry_id``'s baseline
      (or in the supplied ``entries`` set, if provided).
    * The baseline entry is absent from ``raw_metrics.by_entry`` for a
      ``compare_to_baseline`` rule.
    * A completed seed lacks the required metric runtime key in
      :attr:`SeedRunOutcome.required`.
    * A required metric value is the wrong type (non-numeric).
    """


# ---------------------------------------------------------------------------
# Input-shape validation (§7.1)
# ---------------------------------------------------------------------------


def _validate_input_shape(
    config: ValidationConfig,
    raw_metrics: RawMetrics,
    entries: list[Entry] | None,
) -> str:
    """Validate ``raw_metrics`` against ``config`` + optional entries.

    Returns the runtime metric key for ``config.body.metric``. Raises
    :class:`RawMetricsShapeError` for any structural violation.
    """
    from smai_core._runtime_keys import metric_ref_to_runtime_key

    body = config.body
    metric_key = metric_ref_to_runtime_key(body.metric)

    if entries is not None:
        valid_entry_ids = {entry.id for entry in entries}
        if not valid_entry_ids:
            raise RawMetricsShapeError(
                "entries argument was an empty list; expected at least one Entry"
            )
        unknown = sorted(set(raw_metrics.by_entry) - valid_entry_ids)
        if unknown:
            raise RawMetricsShapeError(
                f"raw_metrics references entry IDs not in the experiment: {unknown}"
            )
        missing = sorted(valid_entry_ids - set(raw_metrics.by_entry))
        if missing:
            raise RawMetricsShapeError(
                f"raw_metrics is missing entry IDs declared in entries: {missing}"
            )

    if body.comparison.rule == "compare_to_baseline":
        baseline_id = body.comparison.baseline_entry_id
        if baseline_id is None:
            raise RawMetricsShapeError(
                "compare_to_baseline ValidationConfig has no baseline_entry_id "
                "set; this should have been filled by the compiler at emission"
            )
        if baseline_id not in raw_metrics.by_entry:
            raise RawMetricsShapeError(
                f"raw_metrics is missing the baseline entry "
                f"{baseline_id!r} required by compare_to_baseline"
            )

    for entry_id, em in raw_metrics.by_entry.items():
        for seed, outcome in em.seed_outcomes.items():
            if not outcome.completed:
                continue
            if outcome.required is None:
                raise RawMetricsShapeError(
                    f"entry {entry_id!r} seed {seed} reports completed=True but required is None"
                )
            if metric_key not in outcome.required:
                raise RawMetricsShapeError(
                    f"entry {entry_id!r} seed {seed} is missing required metric key {metric_key!r}"
                )
            # Defensive runtime check: a Tier B integrator that mutates
            # ``required`` after Pydantic validation can land a non-numeric
            # value here; the static type is ``float | int`` but we cannot
            # rely on that at evaluator-call time. The ``cast`` widens for
            # the isinstance check; pyright would otherwise call it
            # tautological.
            checked = cast("object", outcome.required[metric_key])
            if not isinstance(checked, (int, float)) or isinstance(checked, bool):
                raise RawMetricsShapeError(
                    f"entry {entry_id!r} seed {seed} required metric "
                    f"{metric_key!r} has non-numeric value {checked!r}"
                )

    return metric_key


# ---------------------------------------------------------------------------
# Per-entry numeric helpers
# ---------------------------------------------------------------------------


def _completed_seed_values(
    em: EntryMetrics, metric_key: str
) -> tuple[dict[int, float], list[Anomaly]]:
    """Return ``(seed -> metric_value, anomalies)`` for ``em``.

    NaN / Inf values are filtered out and reported as ``nan_value``
    anomalies (§7); failed seeds are reported as ``seed_failed``
    anomalies. The returned dict is keyed by seed value and contains only
    finite-numeric completed seeds.
    """
    values: dict[int, float] = {}
    anomalies: list[Anomaly] = []
    for seed in sorted(em.seed_outcomes):
        outcome = em.seed_outcomes[seed]
        if not outcome.completed:
            anomalies.append(
                Anomaly(
                    kind="seed_failed",
                    entry_id=em.entry_id,
                    seed=seed,
                    detail=outcome.failure_reason or "completed=False; no failure_reason supplied",
                )
            )
            continue
        assert outcome.required is not None  # guarded by _validate_input_shape
        raw = outcome.required[metric_key]
        f = float(raw)
        if math.isnan(f) or math.isinf(f):
            anomalies.append(
                Anomaly(
                    kind="nan_value",
                    entry_id=em.entry_id,
                    seed=seed,
                    detail=f"required[{metric_key!r}] is non-finite ({raw!r})",
                )
            )
            continue
        values[seed] = f
    return values, anomalies


def _outlier_anomalies(
    entry_id: str,
    seed_values: dict[int, float],
    aggregation_method: str,
) -> list[Anomaly]:
    """Flag outliers per §5.3 (3σ for mean, 3·MAD for median)."""
    if len(seed_values) < 3:
        return []
    sorted_seeds = sorted(seed_values)
    values = [seed_values[s] for s in sorted_seeds]
    out: list[Anomaly] = []
    if aggregation_method == "median":
        med = statistics.median(values)
        deviations = [abs(v - med) for v in values]
        mad = statistics.median(deviations)
        if mad == 0.0:
            return []
        for seed, value in zip(sorted_seeds, values, strict=True):
            if abs(value - med) > 3.0 * mad:
                out.append(
                    Anomaly(
                        kind="outlier",
                        entry_id=entry_id,
                        seed=seed,
                        detail=(
                            f"value {value!r} deviates by "
                            f"{abs(value - med):.6g} from median {med!r} "
                            f"(>3·MAD={3.0 * mad:.6g})"
                        ),
                    )
                )
        return out
    mean_v = statistics.fmean(values)
    if len(values) < 2:
        return out
    sd = statistics.stdev(values)
    if sd == 0.0:
        return []
    for seed, value in zip(sorted_seeds, values, strict=True):
        if abs(value - mean_v) > 3.0 * sd:
            out.append(
                Anomaly(
                    kind="outlier",
                    entry_id=entry_id,
                    seed=seed,
                    detail=(
                        f"value {value!r} deviates by "
                        f"{abs(value - mean_v):.6g} from mean {mean_v!r} "
                        f"(>3σ={3.0 * sd:.6g})"
                    ),
                )
            )
    return out


def _entry_statistics(
    entry_id: str,
    metric_key: str,
    seed_values: dict[int, float],
) -> EntryStatistics:
    """Compute :class:`EntryStatistics` over ``seed_values``."""
    sorted_seeds = sorted(seed_values)
    values = [seed_values[s] for s in sorted_seeds]
    n = len(values)
    if n == 0:
        return EntryStatistics(
            entry_id=entry_id,
            metric_name=metric_key,
            n=0,
            mean=0.0,
            std=0.0,
            median=0.0,
            ci_95=(0.0, 0.0),
            min=0.0,
            max=0.0,
        )
    mean_v = statistics.fmean(values)
    std_v = statistics.stdev(values) if n >= 2 else 0.0
    median_v = statistics.median(values)
    if n >= 2:
        half_width = _CI95_Z * std_v / math.sqrt(n)
        ci = (mean_v - half_width, mean_v + half_width)
    else:
        ci = (mean_v, mean_v)
    return EntryStatistics(
        entry_id=entry_id,
        metric_name=metric_key,
        n=n,
        mean=mean_v,
        std=std_v,
        median=median_v,
        ci_95=ci,
        min=min(values),
        max=max(values),
    )


def _delta_summary(
    treatment_id: str,
    metric_key: str,
    treatment_values: dict[int, float],
    baseline_values: dict[int, float],
    direction: str,
) -> TreatmentDeltaSummary:
    """Per-treatment vs. baseline summary (§5)."""
    t_vals = [treatment_values[s] for s in sorted(treatment_values)]
    b_vals = [baseline_values[s] for s in sorted(baseline_values)]
    t_mean = statistics.fmean(t_vals)
    b_mean = statistics.fmean(b_vals)
    raw_delta = t_mean - b_mean
    sign = 1.0 if direction == "higher_is_better" else -1.0
    delta_mean = sign * raw_delta
    relative = (raw_delta / b_mean) if b_mean != 0.0 else float("inf")
    if len(t_vals) >= 2 and len(b_vals) >= 2:
        var_t = statistics.variance(t_vals) / len(t_vals)
        var_b = statistics.variance(b_vals) / len(b_vals)
        delta_std: float | None = math.sqrt(var_t + var_b)
        half_width = _CI95_Z * delta_std
        delta_ci: tuple[float, float] | None = (
            delta_mean - half_width,
            delta_mean + half_width,
        )
    else:
        delta_std = None
        delta_ci = None
    return TreatmentDeltaSummary(
        treatment_entry_id=treatment_id,
        metric_name=metric_key,
        delta_mean=delta_mean,
        delta_std=delta_std,
        delta_ci_95=delta_ci,
        relative_delta=relative,
    )


# ---------------------------------------------------------------------------
# Cost telemetry (§5.2)
# ---------------------------------------------------------------------------


def _scalar(v: object) -> float | None:
    """Return ``v`` as a finite ``float`` or ``None`` when non-scalar / invalid."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    return None


def _cost_telemetry(
    raw_metrics: RawMetrics,
    metric_key: str,
    baseline_id: str | None,
) -> dict[str, CostSummary] | None:
    """Summarize ``optional`` telemetry into per-(entry, key) :class:`CostSummary`s.

    The evaluator does not see the metric registry directly; per §5.2, the
    upstream emitter only puts cost-tagged metrics into
    ``HarnessContract.body.optional_telemetry``, so by the time the
    evaluator runs, every scalar key in ``SeedRunOutcome.optional`` is a
    cost metric. Non-scalar telemetry (lists, dicts) is skipped — only
    scalars summarize cleanly.

    The required-metric runtime key is excluded so a runtime that
    accidentally co-puts the verdict metric in ``optional`` doesn't get
    redundantly summarized.

    Per-entry baseline ratios are populated when the baseline has a
    non-zero mean for the same key.

    Returns ``None`` when no cost-shaped telemetry was found anywhere.
    """
    means_by_entry: dict[str, dict[str, float]] = {}
    for entry_id in sorted(raw_metrics.by_entry):
        em = raw_metrics.by_entry[entry_id]
        per_key: dict[str, list[float]] = {}
        for seed in sorted(em.seed_outcomes):
            outcome = em.seed_outcomes[seed]
            if not outcome.completed or outcome.optional is None:
                continue
            for key, value in outcome.optional.items():
                if key == metric_key:
                    continue
                scalar = _scalar(value)
                if scalar is None:
                    continue
                per_key.setdefault(key, []).append(scalar)
        if per_key:
            means_by_entry[entry_id] = {k: statistics.fmean(v) for k, v in per_key.items()}

    if not means_by_entry:
        return None

    baseline_means: dict[str, float] = (
        means_by_entry.get(baseline_id, {}) if baseline_id is not None else {}
    )

    out: dict[str, CostSummary] = {}
    for entry_id in sorted(means_by_entry):
        for key in sorted(means_by_entry[entry_id]):
            mean_v = means_by_entry[entry_id][key]
            ratio: float | None
            if entry_id == baseline_id:
                ratio = None
            else:
                bm = baseline_means.get(key)
                if bm is None or bm == 0.0:
                    ratio = None
                else:
                    ratio = mean_v / bm
            out[f"{entry_id}::{key}"] = CostSummary(
                entry_id=entry_id,
                metric_name=key,
                mean_value=mean_v,
                ratio_to_baseline=ratio,
            )
    return out


# ---------------------------------------------------------------------------
# Trend observation (§5.4)
# ---------------------------------------------------------------------------


def _compute_trend_observation(
    config: ValidationConfig,
    entries: list[Entry] | None,
    aggregates: dict[str, float],
) -> tuple[TrendObservation | None, list[Anomaly]]:
    """Return ``(observation, anomalies)`` for a populated ``trend_check``.

    Returns ``(None, [])`` when no observation can be computed — either
    ``trend_check`` is unset, ``entries`` weren't supplied, or the
    non-baseline entries lack consistent-``kind`` ``NumericValue``s. Per
    §5.4, this is informational; it never affects the verdict bit.
    """
    trend = config.body.trend_check
    if trend is None:
        return None, []
    if entries is None:
        return None, []
    non_baseline_with_value: list[tuple[float, str]] = []
    kinds: set[str] = set()
    for entry in entries:
        if entry.is_baseline:
            continue
        if entry.level.value is None:
            return None, []
        kinds.add(entry.level.value.kind)
        non_baseline_with_value.append((float(entry.level.value.value), entry.id))
    if not non_baseline_with_value or len(kinds) != 1:
        return None, []
    non_baseline_with_value.sort(key=lambda pair: (pair[0], pair[1]))
    aggregated_path: list[float] = []
    for _, entry_id in non_baseline_with_value:
        if entry_id not in aggregates:
            return None, []
        aggregated_path.append(aggregates[entry_id])
    inversions = 0
    for prev, curr in zip(aggregated_path, aggregated_path[1:], strict=False):
        if curr == prev:
            continue
        if trend.expected_direction == "monotonic_increasing" and curr < prev:
            inversions += 1
        elif trend.expected_direction == "monotonic_decreasing" and curr > prev:
            inversions += 1
    diffs = [curr - prev for prev, curr in zip(aggregated_path, aggregated_path[1:], strict=False)]
    if not diffs:
        observed: str = "non_monotonic"
    elif all(d > 0 for d in diffs):
        observed = "monotonic_increasing"
    elif all(d < 0 for d in diffs):
        observed = "monotonic_decreasing"
    elif all(d >= 0 for d in diffs):
        observed = "monotonic_increasing"
    elif all(d <= 0 for d in diffs):
        observed = "monotonic_decreasing"
    else:
        observed = "non_monotonic"
    if trend.expected_direction == "no_expectation":
        consistent = True
        inversions = 0
    else:
        consistent = inversions == 0 and observed == trend.expected_direction
    obs = TrendObservation(
        expected_direction=trend.expected_direction,
        observed_direction=observed,  # type: ignore[arg-type]
        consistent=consistent,
        inversion_count=inversions,
    )
    anomalies: list[Anomaly] = []
    if trend.alert_on_violation and not consistent:
        anomalies.append(
            Anomaly(
                kind="trend_violation",
                entry_id="",
                seed=None,
                detail=(
                    f"observed direction {observed!r} (inversion_count={inversions}) "
                    f"differs from expected {trend.expected_direction!r}"
                ),
            )
        )
    return obs, anomalies


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _build_failure_reason(
    insufficient: list[tuple[str, int, int, list[Anomaly]]],
    seed_count_required: int,
) -> str:
    """Render the ``failure_reason`` string for an inconclusive verdict."""
    parts: list[str] = []
    for entry_id, completed, _required, ans in insufficient:
        seed_failures = [a for a in ans if a.kind in ("seed_failed", "nan_value")]
        bullets: list[str] = []
        for a in seed_failures:
            bullets.append(f"seed {a.seed}: {a.detail}")
        bullets_text = "; ".join(bullets) if bullets else "no per-seed detail"
        parts.append(
            f"Entry {entry_id!r}: {completed} of {seed_count_required} required "
            f"seeds completed ({bullets_text})"
        )
    return ". ".join(parts) + "."


def evaluate(
    config: ValidationConfig,
    raw_metrics: RawMetrics,
    *,
    entries: list[Entry] | None = None,
) -> EvaluationResult:
    """Pure projection from ``(ValidationConfig, RawMetrics)`` to verdict.

    See module docstring + §1, §4–§7. Returns a structured
    :class:`EvaluationResult`; raises :class:`RawMetricsShapeError` on
    structural input violations.

    The optional ``entries`` keyword exists to enable §5.4 trend
    observation: trend computation requires entry-level
    :class:`smai_core.entities.numeric.NumericValue` data which is not
    carried on :class:`ValidationConfig`. Tier B integrators that want
    trend observation pass their experiment's entries in; Tier A
    pipelines wire this through from the loaded
    :class:`smai_core.artifacts.ExperimentPlan`. Omitting ``entries``
    leaves :attr:`VerdictContext.trend_observation` as ``None`` even when
    ``config.body.trend_check`` is populated. (Divergence from the
    canonical signature in §1; defensible because the canonical signature
    cannot satisfy §5.4 in isolation.)
    """
    metric_key = _validate_input_shape(config, raw_metrics, entries)
    body = config.body
    aggregator = get_aggregation_rule(body.aggregation.method)
    comparator = get_comparison_rule(body.comparison.rule)
    seed_count_required = body.seed_count_required
    direction = body.direction
    threshold = body.comparison.threshold

    baseline_id: str | None
    target_value: float | None
    if body.comparison.rule == "compare_to_baseline":
        baseline_id = body.comparison.baseline_entry_id
        target_value = None
    else:
        baseline_id = None
        target_value = body.comparison.target_value

    sorted_entry_ids = sorted(raw_metrics.by_entry)

    per_seed_values: dict[str, dict[int, dict[str, float | int]]] = {}
    completed_values: dict[str, dict[int, float]] = {}
    completed_anomalies: list[Anomaly] = []
    outlier_anomalies: list[Anomaly] = []
    all_failed: list[str] = []

    for entry_id in sorted_entry_ids:
        em = raw_metrics.by_entry[entry_id]
        seed_vals, ans = _completed_seed_values(em, metric_key)
        completed_anomalies.extend(ans)
        completed_values[entry_id] = seed_vals
        outlier_anomalies.extend(_outlier_anomalies(entry_id, seed_vals, body.aggregation.method))
        psv: dict[int, dict[str, float | int]] = {}
        for seed in sorted(seed_vals):
            outcome = em.seed_outcomes[seed]
            assert outcome.required is not None
            psv[seed] = dict(outcome.required)
        per_seed_values[entry_id] = psv
        if not seed_vals and any(s in em.seed_outcomes for s in em.seed_outcomes):
            all_failed.append(entry_id)

    aggregates: dict[str, float] = {}
    for entry_id, seed_vals in completed_values.items():
        if not seed_vals:
            continue
        sorted_seeds = sorted(seed_vals)
        ordered = [seed_vals[s] for s in sorted_seeds]
        aggregates[entry_id] = aggregator.aggregate(ordered)

    all_seeds_failed_anomalies: list[Anomaly] = []
    insufficient: list[tuple[str, int, int, list[Anomaly]]] = []
    for entry_id in sorted_entry_ids:
        completed = len(completed_values[entry_id])
        if completed == 0 and entry_id in all_failed:
            all_seeds_failed_anomalies.append(
                Anomaly(
                    kind="all_seeds_failed",
                    entry_id=entry_id,
                    seed=None,
                    detail=(
                        f"0 of {seed_count_required} required seeds completed "
                        f"for entry {entry_id!r}"
                    ),
                )
            )
        elif completed < seed_count_required:
            entry_anomalies = [a for a in completed_anomalies if a.entry_id == entry_id]
            insufficient.append((entry_id, completed, seed_count_required, entry_anomalies))

    baseline_inconclusive = False
    if baseline_id is not None:
        baseline_completed = len(completed_values.get(baseline_id, {}))
        if baseline_completed < seed_count_required:
            baseline_inconclusive = True

    treatment_ids = [
        eid
        for eid in sorted_entry_ids
        if (baseline_id is None or eid != baseline_id)
        and eid not in {a.entry_id for a in all_seeds_failed_anomalies}
    ]

    treatment_outcomes: list[TreatmentOutcome] = []
    delta_summaries: list[TreatmentDeltaSummary] = []
    if (
        not baseline_inconclusive
        and not insufficient
        and (baseline_id is None or baseline_id in aggregates)
    ):
        if baseline_id is not None:
            comparand = aggregates[baseline_id]
        else:
            assert target_value is not None
            comparand = target_value
        for tid in treatment_ids:
            if tid not in aggregates:
                continue
            t_agg = aggregates[tid]
            delta = comparator.compute_delta(t_agg, comparand, direction)
            passed = comparator.passed_threshold(delta, threshold, direction)
            treatment_outcomes.append(
                TreatmentOutcome(
                    entry_id=tid,
                    aggregated_metric_value=t_agg,
                    delta_from_baseline=delta,
                    threshold=threshold,
                    passed_threshold=passed,
                    tolerance_margin=delta - threshold,
                )
            )
            if baseline_id is not None and baseline_id in completed_values:
                delta_summaries.append(
                    _delta_summary(
                        tid,
                        metric_key,
                        completed_values[tid],
                        completed_values[baseline_id],
                        direction,
                    )
                )

    per_entry_aggregate: list[PerEntryAggregate] = []
    for entry_id in sorted_entry_ids:
        seed_vals = completed_values[entry_id]
        is_baseline_entry = entry_id == baseline_id
        agg_value = aggregates.get(entry_id, 0.0)
        completed = len(seed_vals)
        total_seeds = len(raw_metrics.by_entry[entry_id].seed_outcomes)
        completed_fraction = completed / total_seeds if total_seeds > 0 else 0.0
        per_entry_aggregate.append(
            PerEntryAggregate(
                entry_id=entry_id,
                is_baseline=is_baseline_entry,
                aggregated_metric_value=agg_value,
                seeds_completed=completed,
                seeds_required=seed_count_required,
                completed_fraction=completed_fraction,
            )
        )

    failure_reason: str | None = None
    if baseline_inconclusive:
        result: str = "inconclusive"
        baseline_completed = len(completed_values.get(baseline_id or "", {}))
        baseline_anomalies = [a for a in completed_anomalies if a.entry_id == baseline_id]
        failure_reason = _build_failure_reason(
            [(baseline_id or "", baseline_completed, seed_count_required, baseline_anomalies)],
            seed_count_required,
        )
    elif insufficient:
        result = "inconclusive"
        failure_reason = _build_failure_reason(insufficient, seed_count_required)
    elif not treatment_outcomes:
        result = "inconclusive"
        if all_seeds_failed_anomalies:
            failed_ids = sorted({a.entry_id for a in all_seeds_failed_anomalies})
            failure_reason = (
                f"All non-baseline entries had every seed fail: {failed_ids}; "
                f"no treatment outcomes computable."
            )
        else:
            failure_reason = "No treatments to evaluate against the baseline."
    else:
        any_passed = any(t.passed_threshold for t in treatment_outcomes)
        result = "pass" if any_passed else "fail"

    statistical_summary: dict[str, EntryStatistics] = {
        entry_id: _entry_statistics(entry_id, metric_key, completed_values[entry_id])
        for entry_id in sorted_entry_ids
    }

    cost_telemetry = _cost_telemetry(raw_metrics, metric_key, baseline_id)

    trend_observation, trend_anomalies = _compute_trend_observation(config, entries, aggregates)

    anomalies: list[Anomaly] = []
    anomalies.extend(completed_anomalies)
    anomalies.extend(outlier_anomalies)
    anomalies.extend(all_seeds_failed_anomalies)
    anomalies.extend(trend_anomalies)
    anomalies.sort(key=lambda a: (a.entry_id, a.kind, a.seed if a.seed is not None else -1))

    reproducibility = ReproducibilityMetadata(
        validation_config_hash=config.envelope.content_hash,
        harness_contract_hash="",
        experiment_plan_hash="",
        technique_contract_hashes={},
        schema_version=config.envelope.schema_version,
        compiler_version=config.envelope.compiler_version,
        evaluator_version=EVALUATOR_VERSION,
        registry_hashes=dict(config.envelope.registry_hashes),
    )

    verdict = Verdict(
        result=result,  # type: ignore[arg-type]
        treatment_outcomes=treatment_outcomes,
        per_entry_aggregate=per_entry_aggregate,
        validation_config_hash=config.envelope.content_hash,
        raw_metrics_hash=raw_metrics_canonical_hash(raw_metrics),
        compiler_version=config.envelope.compiler_version,
        evaluator_version=EVALUATOR_VERSION,
    )
    verdict_context = VerdictContext(
        per_seed_values=per_seed_values,
        statistical_summary=statistical_summary,
        delta_summaries=delta_summaries,
        cost_telemetry=cost_telemetry,
        anomalies=anomalies,
        trend_observation=trend_observation,
        reproducibility=reproducibility,
        failure_reason=failure_reason,
    )
    return EvaluationResult(verdict=verdict, verdict_context=verdict_context)


__all__ = [
    "EVALUATOR_VERSION",
    "RawMetricsShapeError",
    "evaluate",
]
