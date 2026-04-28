""":class:`HarnessContract` — what the harness must expose.

Per ``02-dsl-and-contracts.md`` §7.4 (post-Decision-1 narrowing): factor +
flattened controls + required/optional metrics + no-go zones. Per-key
extension-point declarations are gone; that information is produced
post-build by the pipeline-layer harness builder agent as a
``HarnessAPIManifest`` (§10).
"""

from __future__ import annotations

from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict

from smai_core.artifacts._envelope import ArtifactEnvelope
from smai_core.entities.factor import Factor
from smai_core.entities.metric import MetricRef

FixedVariableValue: TypeAlias = str | int | float | bool | list[Any] | dict[str, Any]
"""JSON-shaped admissible value types for :class:`FixedVariable` (§7.4)."""


class FixedVariable(BaseModel):
    """A single ceteris-paribus value lifted out of ``ControlledConditions`` (§7.4).

    ``path`` locates the variable in the harness config by JSONPath
    (``optimization.learning_rate``). ``type_hint`` is informational only —
    it documents the value's Python type for harness-side configuration
    code.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    value: FixedVariableValue
    type_hint: str | None = None


class HarnessContractBody(BaseModel):
    """Body fields of :class:`HarnessContract` (§7.4)."""

    model_config = ConfigDict(extra="forbid")

    parent_experiment_hash: str
    factor: Factor
    seeds: list[int]
    fixed_variables: list[FixedVariable]
    required_metrics: list[MetricRef]
    optional_telemetry: list[MetricRef]
    no_go_zones: list[str]


class HarnessContract(BaseModel):
    """The harness's compiled contract (§7.4)."""

    model_config = ConfigDict(extra="forbid")

    envelope: ArtifactEnvelope
    body: HarnessContractBody
