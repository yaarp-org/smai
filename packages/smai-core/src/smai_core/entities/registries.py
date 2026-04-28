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
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from smai_core.entities.metric import MetricRegistry
from smai_core.entities.technique import TechniqueRef
from smai_core.factor_types._protocol import FactorTypePlugin


class Registries(BaseModel):
    """The bag of registries passed to factor-type plugins and to the compiler."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    technique_registry: dict[str, TechniqueRef]
    metric_registry: MetricRegistry
    factor_type_plugins: dict[str, FactorTypePlugin]
