"""Validation surface entities.

Per ``designs/smai/01-data-model.md`` §3.6: ``AggregationRule``,
``ComparisonRule``, ``TrendCheck``, ``ValidationCriteria``.
"""

from __future__ import annotations

from typing import Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator

from smai_core.entities.metric import MetricRef


class AggregationRule(BaseModel):
    """Per-entry aggregation over the seed dimension. Closed in v1."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["mean", "median"]


class ComparisonRule(BaseModel):
    """Comparison rule with one compiler-fill field (``baseline_entry_id``).

    Single class for both DSL input and emitted artifact, with a context-keyed
    model validator (per §4.8). Callers pass ``context={"smai_mode": "dsl"}``
    to ``model_validate`` at DSL ingestion time and
    ``context={"smai_mode": "artifact"}`` when re-validating an emitted
    ``ValidationConfig``. The default mode (no context) is ``"artifact"`` —
    the gate that fires after the compiler has filled the field.

    DSL gate: ``baseline_entry_id`` MUST be ``None`` (the user identifies the
    baseline via ``Entry.is_baseline=True``).
    Artifact gate: ``baseline_entry_id`` MUST be non-``None`` when
    ``rule == "compare_to_baseline"``.

    ``target_value`` is required when ``rule == "compare_to_target"`` in both
    modes — it is structural, not compiler-fill.
    """

    model_config = ConfigDict(extra="forbid")

    rule: Literal["compare_to_baseline", "compare_to_target"]
    threshold: float
    baseline_entry_id: str | None = None
    target_value: float | None = None

    @model_validator(mode="after")
    def _gate_compiler_fill_fields(self, info: ValidationInfo) -> Self:
        ctx = cast(dict[str, Any] | None, info.context)
        mode: str = "artifact"
        if ctx is not None:
            raw_mode = ctx.get("smai_mode")
            if isinstance(raw_mode, str):
                mode = raw_mode
        if mode == "dsl" and self.baseline_entry_id is not None:
            raise ValueError("comparison.baseline_entry_id_user_supplied")
        if (
            mode == "artifact"
            and self.rule == "compare_to_baseline"
            and self.baseline_entry_id is None
        ):
            raise ValueError("comparison.baseline_entry_id_missing")
        if self.rule == "compare_to_target" and self.target_value is None:
            raise ValueError("comparison.target_value_missing")
        return self


class TrendCheck(BaseModel):
    """Optional verdict-context check for ordinal/continuous treatment axes.

    Auto-enabled by the compiler (DEC-031 sub-decision #7) when all
    non-baseline entries carry a ``value: NumericValue`` with consistent
    ``kind``. Result is informational — does NOT affect the boolean verdict,
    only the ``verdict_context`` bundle. Defaults match what auto-inference
    emits (conservative).
    """

    model_config = ConfigDict(extra="forbid")

    expected_direction: Literal[
        "monotonic_increasing",
        "monotonic_decreasing",
        "no_expectation",
    ] = "no_expectation"
    alert_on_violation: bool = False


class ValidationCriteria(BaseModel):
    """Full validation surface for a CG.

    ``direction`` is always required even when the registry has it (§4.5).
    ``rationale`` is recorded surface — free-text justification consumed by
    the metric-appropriateness fuzzy review gate (pipeline layer), not by the
    verdict path. ``optional_telemetry`` is unioned with the registry's
    auto-included cost metrics at compile time per the hybrid scheme in §4.10.
    """

    model_config = ConfigDict(extra="forbid")

    metric: MetricRef
    direction: Literal["higher_is_better", "lower_is_better"]
    aggregation: AggregationRule
    comparison: ComparisonRule
    seed_count_required: int
    rationale: str | None = None
    trend_check: TrendCheck | None = None
    optional_telemetry: list[MetricRef] | None = None
