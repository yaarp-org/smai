"""Mechanical evaluation surface — registries + evaluator.

Per ``designs/smai/06-mechanical-evaluation.md``: aggregation and comparison
rule registries (Task 1.4), :class:`RawMetrics` input shape and the locked
:class:`Verdict` / :class:`VerdictContext` outputs (Task 1.7), plus the
pure :func:`evaluate` entry point. The registries are static, closed in v1
(per DEC-031); the evaluator dispatches via them and adds nothing
non-deterministic to the verdict path.
"""

from smai_core.evaluation._evaluate import (
    EVALUATOR_VERSION,
    RawMetricsShapeError,
    evaluate,
)
from smai_core.evaluation.aggregation import (
    AGGREGATION_RULE_REGISTRY,
    AggregationFunction,
    AggregationRuleNotRegistered,
    get_aggregation_rule,
)
from smai_core.evaluation.comparison import (
    COMPARISON_RULE_REGISTRY,
    ComparisonFunction,
    ComparisonRuleNotRegistered,
    get_comparison_rule,
)
from smai_core.evaluation.raw_metrics import (
    EntryMetrics,
    RawMetrics,
    SeedRunOutcome,
    raw_metrics_canonical_form,
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

__all__ = [
    "AGGREGATION_RULE_REGISTRY",
    "COMPARISON_RULE_REGISTRY",
    "EVALUATOR_VERSION",
    "AggregationFunction",
    "AggregationRuleNotRegistered",
    "Anomaly",
    "ComparisonFunction",
    "ComparisonRuleNotRegistered",
    "CostSummary",
    "EntryMetrics",
    "EntryStatistics",
    "EvaluationResult",
    "PerEntryAggregate",
    "RawMetrics",
    "RawMetricsShapeError",
    "ReproducibilityMetadata",
    "SeedRunOutcome",
    "TreatmentDeltaSummary",
    "TreatmentOutcome",
    "TrendObservation",
    "Verdict",
    "VerdictContext",
    "evaluate",
    "get_aggregation_rule",
    "get_comparison_rule",
    "raw_metrics_canonical_form",
    "raw_metrics_canonical_hash",
]
