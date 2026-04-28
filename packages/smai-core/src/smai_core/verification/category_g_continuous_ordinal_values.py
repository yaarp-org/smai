"""Category G — Continuous/ordinal value rules (§5.8).

When a Level carries a ``NumericValue``, its declared range is honored; when
multiple Levels in a factor carry values, they don't mix kinds (or units, by
warning); and when the technique's parameter schema declares a numeric
parameter that conceptually maps to ``value.value``, an advisory check
nudges the author to keep the level value and the technique param in sync.
"""

from __future__ import annotations

from typing import Any

from smai_core.entities.experiment import ExperimentDefinition
from smai_core.entities.registries import Registries
from smai_core.entities.validation_report import ValidationError


def value_in_declared_range(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``value.in_declared_range`` (error)."""
    findings: list[ValidationError] = []
    for entry in experiment.entries:
        value = entry.level.value
        if value is None:
            continue
        if value.min is not None and value.value < value.min:
            findings.append(
                ValidationError(
                    code="value.in_declared_range",
                    message=(
                        f"Entry '{entry.id}' level.value.value={value.value} is below "
                        f"declared min={value.min}."
                    ),
                    location=f"entries[id={entry.id}].level.value",
                )
            )
        if value.max is not None and value.value > value.max:
            findings.append(
                ValidationError(
                    code="value.in_declared_range",
                    message=(
                        f"Entry '{entry.id}' level.value.value={value.value} is above "
                        f"declared max={value.max}."
                    ),
                    location=f"entries[id={entry.id}].level.value",
                )
            )
    return findings


def value_kind_consistency_within_factor(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``value.kind_consistency_within_factor`` (error).

    All Levels within a factor that carry a ``NumericValue`` agree on
    ``kind`` — don't mix ordinal and continuous within one factor.
    """
    findings: list[ValidationError] = []
    kinds: dict[str, list[str]] = {}
    for entry in experiment.entries:
        if entry.level.value is None:
            continue
        kinds.setdefault(entry.level.value.kind, []).append(entry.id)
    if len(kinds) > 1:
        findings.append(
            ValidationError(
                code="value.kind_consistency_within_factor",
                message=(
                    f"Factor mixes value.kind values across entries: "
                    f"{ {k: sorted(v) for k, v in kinds.items()}!r}. All "
                    f"Levels with NumericValue must share the same kind."
                ),
                location="entries[*].level.value.kind",
            )
        )
    return findings


def value_unit_consistency_within_factor(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``value.unit_consistency_within_factor`` (warning).

    Mixing fractions and percentages (etc.) within one factor is unusual; warn.
    Entries without a value, or with ``unit=None``, are ignored — declaring
    no unit at all is a separate stylistic choice.
    """
    findings: list[ValidationError] = []
    units: dict[str, list[str]] = {}
    for entry in experiment.entries:
        value = entry.level.value
        if value is None or value.unit is None:
            continue
        units.setdefault(value.unit, []).append(entry.id)
    if len(units) > 1:
        findings.append(
            ValidationError(
                code="value.unit_consistency_within_factor",
                message=(
                    f"Factor mixes value.unit declarations across entries: "
                    f"{ {u: sorted(v) for u, v in units.items()}!r}. Mismatched "
                    f"units are unusual — confirm that fractions/percentages "
                    f"weren't accidentally mixed."
                ),
                location="entries[*].level.value.unit",
                severity="warning",
            )
        )
    return findings


def _technique_single_numeric_param(parameter_schema: dict[str, Any] | None) -> str | None:
    """If the schema admits exactly one numeric (number/integer) property, return its name."""
    if not parameter_schema:
        return None
    properties_obj = parameter_schema.get("properties")
    if not isinstance(properties_obj, dict):
        return None
    numeric_keys: list[str] = []
    for key, prop in properties_obj.items():  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(prop, dict) or not isinstance(key, str):
            continue
        prop_type = prop.get("type")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if prop_type in ("number", "integer"):
            numeric_keys.append(key)
    return numeric_keys[0] if len(numeric_keys) == 1 else None


def value_technique_params_alignment(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``value.technique_params_alignment`` (advisory).

    Heuristic: if the technique declares exactly one numeric parameter, treat
    that parameter as the conceptual mate of ``Level.value.value``; advise
    when the level provides both and they disagree. v1 informational; vN may
    harden to warning per §5.8.
    """
    findings: list[ValidationError] = []
    techniques = registries.technique_registry
    for entry in experiment.entries:
        value = entry.level.value
        params = entry.level.technique_params
        tech_id = entry.level.technique_id
        if value is None or params is None or tech_id is None:
            continue
        technique = techniques.get(tech_id)
        if technique is None:
            continue
        param_name = _technique_single_numeric_param(technique.parameter_schema)
        if param_name is None or param_name not in params:
            continue
        param_value = params[param_name]
        if not isinstance(param_value, (int, float)):
            continue
        if float(param_value) != float(value.value):
            findings.append(
                ValidationError(
                    code="value.technique_params_alignment",
                    message=(
                        f"Entry '{entry.id}' has level.value={value.value} but "
                        f"technique_params[{param_name!r}]={param_value!r}; the "
                        f"single numeric technique param conceptually mirrors the "
                        f"level value but they disagree."
                    ),
                    location=f"entries[id={entry.id}]",
                    severity="advisory",
                )
            )
    return findings


CATEGORY_G_RULES = (
    value_in_declared_range,
    value_kind_consistency_within_factor,
    value_unit_consistency_within_factor,
    value_technique_params_alignment,
)
"""Category G rule registration order; matches §5.8."""
