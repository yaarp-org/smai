"""Default-:class:`Registries` factory.

``load_default_registries()`` is the canonical "give me a working registry
set" entry point that downstream tasks (1.5 verification, 1.6 contract
emission, 1.7 evaluator) and Tier B integrators consume. It bundles:

- the closed v1 metric registry (loaded from the JSON shipped under
  ``smai_core.data``);
- the closed v1 aggregation rule registry (``mean`` / ``median``);
- the closed v1 comparison rule registry (``compare_to_baseline`` /
  ``compare_to_target``);
- the v1 built-in factor-type plugins (``additive``, ``substitutive``)
  discovered via the ``smai.factor_types`` entry-point group;
- an empty technique registry — the v1 starter set + paper-ingestion /
  proposal outputs are populated by pipeline-layer code (per
  ``02-dsl-and-contracts.md`` §4.1) and merged in before invoking the
  compiler.

Per the canonical doc's frozen-after-load contract
(``02-dsl-and-contracts.md`` §4.2), callers should treat the returned
:class:`Registries` as read-only and supply technique entries via a fresh
:class:`Registries` constructed from this one's parts (or, in tests, via
direct mutation — Pydantic ``model_copy(update=...)``).
"""

from __future__ import annotations

from smai_core.data import load_metric_registry
from smai_core.entities.registries import Registries
from smai_core.entities.technique import TechniqueRef
from smai_core.evaluation.aggregation import AGGREGATION_RULE_REGISTRY
from smai_core.evaluation.comparison import COMPARISON_RULE_REGISTRY
from smai_core.factor_types._loader import load_builtin_factor_type_plugins


def load_default_registries(
    *,
    technique_registry: dict[str, TechniqueRef] | None = None,
) -> Registries:
    """Construct a :class:`Registries` populated with v1 defaults.

    ``technique_registry`` defaults to an empty dict; callers populate it
    from the v1 starter set and/or paper-ingestion outputs before passing
    the registry into the compiler. Per ``02-dsl-and-contracts.md`` §4.1,
    the technique registry is *input* to the methodology layer, not state
    owned by it.

    The returned object should be treated as read-only — the compiler hashes
    its registries and stamps the hashes onto every emitted artifact's
    envelope (``02-dsl-and-contracts.md`` §4.2).
    """
    return Registries(
        technique_registry=technique_registry if technique_registry is not None else {},
        metric_registry=load_metric_registry(),
        factor_type_plugins=load_builtin_factor_type_plugins(),
        aggregation_rule_registry=dict(AGGREGATION_RULE_REGISTRY),
        comparison_rule_registry=dict(COMPARISON_RULE_REGISTRY),
    )
