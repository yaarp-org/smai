"""``Registries`` — the bag of registries the compiler operates against.

Per ``designs/smai/01-data-model.md`` §3.8 and ``02-dsl-and-contracts.md``
§4. The compiler hashes each registry at startup and stamps the hashes onto
every emitted artifact's envelope so verdicts can be replayed against the
same registries.

``factor_type_plugins`` is keyed by ``plugin.name`` and stores instances
satisfying the ``FactorTypePlugin`` Protocol (``02-dsl-and-contracts.md``
§3). The Protocol is imported from ``smai_core.factor_types`` rather than
``smai_core.entities`` to avoid a circular import — the Protocol module
itself keeps its references back to entity types under ``TYPE_CHECKING``.

In addition to the three registries the canonical ``02 §4`` snippet
enumerates (technique, metric, factor-type), v1 also bundles the closed
aggregation and comparison rule registries from
``06-mechanical-evaluation.md`` §6 onto :class:`Registries` so that
``load_default_registries()`` returns one fully-loaded handle for both the
compiler path (Tasks 1.5 / 1.6) and the evaluator path (Task 1.7). Both
default to the static registries shipped under ``smai_core.evaluation``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from smai_core.entities.metric import MetricRegistry
from smai_core.entities.technique import TechniqueRef
from smai_core.evaluation.aggregation import (
    AGGREGATION_RULE_REGISTRY,
    AggregationFunction,
)
from smai_core.evaluation.comparison import (
    COMPARISON_RULE_REGISTRY,
    ComparisonFunction,
)
from smai_core.factor_types._protocol import FactorTypePlugin


def _default_aggregation_rules() -> dict[str, AggregationFunction]:
    return dict(AGGREGATION_RULE_REGISTRY)


def _default_comparison_rules() -> dict[str, ComparisonFunction]:
    return dict(COMPARISON_RULE_REGISTRY)


class Registries(BaseModel):
    """The bag of registries passed to factor-type plugins and to the compiler."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    technique_registry: dict[str, TechniqueRef]
    metric_registry: MetricRegistry
    factor_type_plugins: dict[str, FactorTypePlugin]
    aggregation_rule_registry: dict[str, AggregationFunction] = Field(
        default_factory=_default_aggregation_rules
    )
    comparison_rule_registry: dict[str, ComparisonFunction] = Field(
        default_factory=_default_comparison_rules
    )
