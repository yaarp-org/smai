"""Mechanical evaluation surface — rule registries and (later) the evaluator.

Per ``designs/smai/06-mechanical-evaluation.md``: the aggregation and
comparison rule registries live here, named per ``§6``. Task 1.4 ships the
registries (``mean`` / ``median``; ``compare_to_baseline`` /
``compare_to_target``); Task 1.7 will add the ``evaluate(...)`` entry point
that consumes them. The registries are static, closed in v1 (per DEC-031),
and built from the ``Protocol`` shapes the canonical doc specifies.
"""

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

__all__ = [
    "AGGREGATION_RULE_REGISTRY",
    "COMPARISON_RULE_REGISTRY",
    "AggregationFunction",
    "AggregationRuleNotRegistered",
    "ComparisonFunction",
    "ComparisonRuleNotRegistered",
    "get_aggregation_rule",
    "get_comparison_rule",
]
