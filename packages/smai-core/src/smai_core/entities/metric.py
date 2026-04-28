"""Metric references and the loaded-form metric registry.

Per ``designs/smai/01-data-model.md`` §3.1.2 (``MetricRef`` discriminated
union) and §3.8.1 (``MetricRegistry`` shape). Closed registry per DEC-031
sub-decision #1.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class AtomicMetricRef(BaseModel):
    """Reference to a registered atomic metric (e.g., ``accuracy``)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["atomic"] = "atomic"
    ref: str


class ParametricMetricRef(BaseModel):
    """Reference to a registered parametric family (e.g., ``top_k_accuracy``)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["parametric"] = "parametric"
    family: str
    parameters: dict[str, str | int | float]


MetricRef = Annotated[
    AtomicMetricRef | ParametricMetricRef,
    Field(discriminator="kind"),
]
"""Discriminated union over the closed metric registry's two reference shapes.

Atomic: a canonical name from the registered atomic set.
Parametric: a family name plus parameter values from that family's
``parameter_values_seen`` set.
"""

MetricRefAdapter: TypeAdapter[MetricRef] = TypeAdapter(MetricRef)
"""``TypeAdapter`` for the union; use to validate raw dicts into ``MetricRef``."""


class AtomicMetricEntry(BaseModel):
    """One row in the loaded-form atomic metric registry (§3.8.1)."""

    model_config = ConfigDict(extra="forbid")

    canonical_name: str
    aliases: list[str]
    task_types: list[str]
    direction: Literal["higher_is_better", "lower_is_better", "ambiguous"]
    input_shape: str
    frequency: int
    category: Literal["quality", "cost"]
    notes: str | None = None


class ParametricFamily(BaseModel):
    """One row in the loaded-form parametric metric registry (§3.8.1).

    Single-parameter families declare ``parameter`` as ``str``; multi-parameter
    families declare ``parameter`` as ``list[str]``. ``parameter_values_seen``
    follows the same shape: a flat list for single-parameter families, or a
    dict keyed by parameter name for multi-parameter families.

    ``direction`` is a single Literal for the family's uniform direction in
    the common case; for families whose direction varies by parameter value
    (e.g., ``min_traj_error_at_k`` per ``dev/smai_metric_registry_v1.md``), a
    dict keyed by parameter value.
    """

    model_config = ConfigDict(extra="forbid")

    family_name: str
    parameter: str | list[str]
    parameter_values_seen: list[str | int | float] | dict[str, list[str | int | float]]
    parameter_type: str
    task_types: list[str]
    direction: (
        Literal["higher_is_better", "lower_is_better", "ambiguous"]
        | dict[str, Literal["higher_is_better", "lower_is_better"]]
    )
    frequency: int
    category: Literal["quality", "cost"]
    notes: str | None = None


class MetricRegistry(BaseModel):
    """Loaded-form metric registry: atomic entries + parametric families.

    Keyed by ``canonical_name`` and ``family_name`` respectively. The on-disk
    YAML/JSON shape is a ``smai-core`` implementation detail; loader code (Task
    1.4) maps from that to this shape and enforces the registry-policy
    invariant (no ``=`` or ``__`` in canonical names, family names, or
    parameter values) per ``01-data-model.md`` §3.8.1.
    """

    model_config = ConfigDict(extra="forbid")

    atomic: dict[str, AtomicMetricEntry]
    parametric: dict[str, ParametricFamily]

    def get_atomic(self, name: str) -> AtomicMetricEntry | None:
        """Return the atomic entry by canonical name, or ``None`` if absent."""
        return self.atomic.get(name)

    def get_family(self, family_name: str) -> ParametricFamily | None:
        """Return the parametric family by name, or ``None`` if absent."""
        return self.parametric.get(family_name)
