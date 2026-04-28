"""``Registries`` — the bag of registries the compiler operates against.

Per ``designs/smai/01-data-model.md`` §3.8. The compiler hashes each registry
at startup and stamps the hashes onto every emitted artifact's envelope so
verdicts can be replayed against the same registries.

The ``factor_type_plugins`` field is typed as ``dict[str, Any]`` in this task
because the ``FactorTypePlugin`` Protocol is Task 1.3's deliverable
(``02-dsl-and-contracts.md`` §3). When Task 1.3 lands, this field's value
type tightens to the Protocol class. The shape of the field — a name-keyed
dict — is stable.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from smai_core.entities.metric import MetricRegistry
from smai_core.entities.technique import TechniqueRef


class Registries(BaseModel):
    """The bag of registries passed to factor-type plugins and to the compiler."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    technique_registry: dict[str, TechniqueRef]
    metric_registry: MetricRegistry
    factor_type_plugins: dict[str, Any]
