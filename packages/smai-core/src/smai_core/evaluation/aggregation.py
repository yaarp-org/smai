"""Aggregation rule registry — ``mean`` / ``median``.

Per ``designs/smai/06-mechanical-evaluation.md`` §6: closed v1 registry of
per-entry seed-dimension aggregators. Each rule is keyed by its ``name`` and
satisfies the :class:`AggregationFunction` Protocol below.

Reduction semantics follow the §6 spec verbatim:

- ``mean``:   ``sum(values) / len(values)``
- ``median``: middle element of the sorted list (mean of the two middle
  elements when the list has even length).

Both raise :class:`ValueError` on an empty input list — the evaluator (Task
1.7) is responsible for screening empty / insufficient-seed cases out via
``inconclusive`` semantics before calling these primitives.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AggregationFunction(Protocol):
    """Protocol satisfied by every aggregation rule in the v1 closed set."""

    name: str
    """Canonical name; the key the rule is registered under."""

    def aggregate(self, values: list[float]) -> float:
        """Reduce a list of per-seed metric values to a single float."""
        ...


class AggregationRuleNotRegistered(KeyError):
    """Raised by :func:`get_aggregation_rule` when ``name`` is not registered.

    The evaluator (Task 1.7) and verification rules (Task 1.5) typically work
    against the registry directly; this helper exists for runtime / Tier B
    consumers that prefer a raising lookup.
    """


class _MeanAggregation:
    name: str = "mean"

    def aggregate(self, values: list[float]) -> float:
        if not values:
            raise ValueError("mean.aggregate received empty values list")
        return sum(values) / len(values)


class _MedianAggregation:
    name: str = "median"

    def aggregate(self, values: list[float]) -> float:
        if not values:
            raise ValueError("median.aggregate received empty values list")
        ordered = sorted(values)
        n = len(ordered)
        mid = n // 2
        if n % 2 == 1:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0


AGGREGATION_RULE_REGISTRY: dict[str, AggregationFunction] = {
    "mean": _MeanAggregation(),
    "median": _MedianAggregation(),
}
"""The v1 closed set, keyed by ``AggregationFunction.name``.

Closed per DEC-031 sub-decision #1's reasoning extended to evaluator
primitives — opening custom aggregations would let an outer system silently
substitute "mean that drops the worst seed", eroding the locked-metric
guarantee. vN extension would carry audit metadata (``custom_aggregation``,
``aggregation_code_hash``) on the verdict.
"""


def get_aggregation_rule(name: str) -> AggregationFunction:
    """Look up an aggregation rule by name; raise on miss."""
    rule = AGGREGATION_RULE_REGISTRY.get(name)
    if rule is None:
        raise AggregationRuleNotRegistered(
            f"aggregation rule {name!r} is not registered; v1 closed set is "
            f"{sorted(AGGREGATION_RULE_REGISTRY)}"
        )
    return rule
