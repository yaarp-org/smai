"""Category D — Metric well-formedness (§5.5).

Verdict metric and optional-telemetry metrics resolve against the closed v1
metric registry (DEC-031 sub-decision #1). Atomic refs match a registered
canonical name; parametric refs match a registered family name and supply
every parameter the family declares with a value in the family's
``parameter_values_seen`` set.

``metric.direction_explicit`` (§5.5 row 5) is enforced by Pydantic on
``ValidationCriteria.direction: Literal["higher_is_better", "lower_is_better"]``
— it cannot fail at Pass 2, so it is not a Pass-2 callable per §5.5's "already
enforced by the schema; restated here for completeness" annotation.

``metric.optional_telemetry_auto_inclusion`` is engine behavior (compiler
auto-includes cost-tagged metrics; §5.5 row 10) and is not a rule.
"""

from __future__ import annotations

from smai_core.entities.experiment import ExperimentDefinition
from smai_core.entities.metric import (
    AtomicMetricRef,
    MetricRef,
    ParametricFamily,
    ParametricMetricRef,
)
from smai_core.entities.registries import Registries
from smai_core.entities.validation_report import ValidationError


def _all_metric_refs(experiment: ExperimentDefinition) -> list[tuple[str, MetricRef]]:
    """Return ``(location, ref)`` pairs for every MetricRef on the validation surface."""
    refs: list[tuple[str, MetricRef]] = [("validation.metric", experiment.validation.metric)]
    if experiment.validation.optional_telemetry:
        for idx, ref in enumerate(experiment.validation.optional_telemetry):
            refs.append((f"validation.optional_telemetry[{idx}]", ref))
    return refs


def _family_required_params(family: ParametricFamily) -> list[str]:
    return [family.parameter] if isinstance(family.parameter, str) else list(family.parameter)


def _direction_for_parametric(family: ParametricFamily, ref: ParametricMetricRef) -> str | None:
    """Resolve a parametric ref's effective registry direction.

    - Family-level Literal direction → return that string.
    - Family-level per-value dict → look up via each of the ref's parameter
      values; the first match wins. If no value matches, treat as ambiguous
      (returns ``"ambiguous"``) so the rule layer accepts any explicit
      direction declaration.
    """
    direction = family.direction
    if isinstance(direction, str):
        return direction
    for value in ref.parameters.values():
        key = str(value)
        if key in direction:
            return direction[key]
    return "ambiguous"


def metric_atomic_ref_registered(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``metric.atomic_ref_registered`` (error)."""
    findings: list[ValidationError] = []
    for location, ref in _all_metric_refs(experiment):
        if isinstance(ref, AtomicMetricRef):
            if registries.metric_registry.get_atomic(ref.ref) is None:
                findings.append(
                    ValidationError(
                        code="metric.atomic_ref_registered",
                        message=(f"Atomic metric {ref.ref!r} is not in the v1 metric registry."),
                        location=location,
                    )
                )
    return findings


def metric_parametric_family_registered(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``metric.parametric_family_registered`` (error)."""
    findings: list[ValidationError] = []
    for location, ref in _all_metric_refs(experiment):
        if isinstance(ref, ParametricMetricRef):
            if registries.metric_registry.get_family(ref.family) is None:
                findings.append(
                    ValidationError(
                        code="metric.parametric_family_registered",
                        message=(
                            f"Parametric family {ref.family!r} is not in the v1 metric registry."
                        ),
                        location=location,
                    )
                )
    return findings


def metric_parametric_required_parameters_present(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``metric.parametric_required_parameters_present`` (error)."""
    findings: list[ValidationError] = []
    for location, ref in _all_metric_refs(experiment):
        if isinstance(ref, ParametricMetricRef):
            family = registries.metric_registry.get_family(ref.family)
            if family is None:
                continue
            required = _family_required_params(family)
            missing = [p for p in required if p not in ref.parameters]
            if missing:
                findings.append(
                    ValidationError(
                        code="metric.parametric_required_parameters_present",
                        message=(
                            f"Parametric family {ref.family!r} requires parameters "
                            f"{required!r}; missing {missing!r}."
                        ),
                        location=location,
                    )
                )
    return findings


def metric_parametric_value_in_seen_set(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``metric.parametric_value_in_seen_set`` (error). Strict v1 per §6."""
    findings: list[ValidationError] = []
    for location, ref in _all_metric_refs(experiment):
        if not isinstance(ref, ParametricMetricRef):
            continue
        family = registries.metric_registry.get_family(ref.family)
        if family is None:
            continue
        seen = family.parameter_values_seen
        required = _family_required_params(family)
        if isinstance(seen, dict):
            for param_name in required:
                if param_name not in ref.parameters:
                    continue
                value = ref.parameters[param_name]
                if value not in seen.get(param_name, []):
                    findings.append(
                        ValidationError(
                            code="metric.parametric_value_in_seen_set",
                            message=(
                                f"Parametric family {ref.family!r} parameter "
                                f"{param_name!r}={value!r} is not in the v1 "
                                f"parameter_values_seen list "
                                f"{seen.get(param_name, [])!r}."
                            ),
                            location=location,
                        )
                    )
        else:
            param_name = required[0]
            if param_name in ref.parameters:
                value = ref.parameters[param_name]
                if value not in seen:
                    findings.append(
                        ValidationError(
                            code="metric.parametric_value_in_seen_set",
                            message=(
                                f"Parametric family {ref.family!r} value "
                                f"{param_name!r}={value!r} is not in the v1 "
                                f"parameter_values_seen list {seen!r}."
                            ),
                            location=location,
                        )
                    )
    return findings


def metric_direction_matches_registry(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``metric.direction_matches_registry`` (error).

    Applies to the verdict metric only — ``optional_telemetry`` carries no
    direction declaration. ``ambiguous`` registry direction (e.g., ``score``,
    ``latency``) accepts any explicit direction.
    """
    findings: list[ValidationError] = []
    ref = experiment.validation.metric
    declared = experiment.validation.direction
    registry_direction: str | None = None
    if isinstance(ref, AtomicMetricRef):
        entry = registries.metric_registry.get_atomic(ref.ref)
        if entry is not None:
            registry_direction = entry.direction
    else:
        family = registries.metric_registry.get_family(ref.family)
        if family is not None:
            registry_direction = _direction_for_parametric(family, ref)
    if registry_direction is None or registry_direction == "ambiguous":
        return findings
    if registry_direction != declared:
        findings.append(
            ValidationError(
                code="metric.direction_matches_registry",
                message=(
                    f"Validation direction {declared!r} disagrees with the v1 "
                    f"registry direction {registry_direction!r} for this metric."
                ),
                location="validation.direction",
            )
        )
    return findings


def metric_cost_metric_in_optional_telemetry(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``metric.cost_metric_in_optional_telemetry`` (warning).

    The verdict metric should not be a registry-tagged ``cost`` metric — those
    (``params``, ``flops``, ``latency``, ``compute_cost.*``) are typically
    Pareto co-axes, not headline metrics.
    """
    findings: list[ValidationError] = []
    ref = experiment.validation.metric
    category: str | None = None
    if isinstance(ref, AtomicMetricRef):
        entry = registries.metric_registry.get_atomic(ref.ref)
        if entry is not None:
            category = entry.category
    else:
        family = registries.metric_registry.get_family(ref.family)
        if family is not None:
            category = family.category
    if category == "cost":
        findings.append(
            ValidationError(
                code="metric.cost_metric_in_optional_telemetry",
                message=(
                    "Verdict metric is registry-tagged 'cost'; cost metrics are "
                    "typically Pareto co-axes rather than primary verdict metrics."
                ),
                location="validation.metric",
                severity="warning",
            )
        )
    return findings


def metric_optional_telemetry_well_formed(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``metric.optional_telemetry_well_formed`` (error).

    Each user-declared optional-telemetry ``MetricRef`` is registered (atomic
    canonical name or parametric family + parameters). Coverage overlaps with
    the per-ref-kind rules above (which already iterate the optional-telemetry
    list); this rule emits a *summary* finding so the user-facing report is
    clear that the optional_telemetry list as a whole is malformed even when
    only the registry-membership sub-check catches it.
    """
    findings: list[ValidationError] = []
    optional = experiment.validation.optional_telemetry or []
    for idx, ref in enumerate(optional):
        location = f"validation.optional_telemetry[{idx}]"
        if isinstance(ref, AtomicMetricRef):
            if registries.metric_registry.get_atomic(ref.ref) is None:
                findings.append(
                    ValidationError(
                        code="metric.optional_telemetry_well_formed",
                        message=(
                            f"optional_telemetry entry {ref.ref!r} is not a "
                            f"registered atomic metric."
                        ),
                        location=location,
                    )
                )
        else:
            family = registries.metric_registry.get_family(ref.family)
            if family is None:
                findings.append(
                    ValidationError(
                        code="metric.optional_telemetry_well_formed",
                        message=(
                            f"optional_telemetry entry uses unregistered parametric "
                            f"family {ref.family!r}."
                        ),
                        location=location,
                    )
                )
                continue
            required = _family_required_params(family)
            missing = [p for p in required if p not in ref.parameters]
            if missing:
                findings.append(
                    ValidationError(
                        code="metric.optional_telemetry_well_formed",
                        message=(
                            f"optional_telemetry entry parametric family "
                            f"{ref.family!r} is missing parameter(s) {missing!r}."
                        ),
                        location=location,
                    )
                )
    return findings


def metric_optional_telemetry_no_overlap_with_required(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``metric.optional_telemetry_no_overlap_with_required`` (warning).

    A user-declared ``optional_telemetry`` entry that is byte-equal to the
    verdict metric is redundant — the verdict metric is already in
    ``HarnessContract.required_metrics``.
    """
    findings: list[ValidationError] = []
    optional = experiment.validation.optional_telemetry or []
    if not optional:
        return findings
    verdict = experiment.validation.metric
    verdict_dump = verdict.model_dump()
    for idx, ref in enumerate(optional):
        if ref.model_dump() == verdict_dump:
            findings.append(
                ValidationError(
                    code="metric.optional_telemetry_no_overlap_with_required",
                    message=(
                        "optional_telemetry entry is byte-equal to the verdict "
                        "metric; this is redundant (the verdict metric is already "
                        "in required_metrics)."
                    ),
                    location=f"validation.optional_telemetry[{idx}]",
                    severity="warning",
                )
            )
    return findings


CATEGORY_D_RULES = (
    metric_atomic_ref_registered,
    metric_parametric_family_registered,
    metric_parametric_required_parameters_present,
    metric_parametric_value_in_seen_set,
    metric_direction_matches_registry,
    metric_cost_metric_in_optional_telemetry,
    metric_optional_telemetry_well_formed,
    metric_optional_telemetry_no_overlap_with_required,
)
"""Category D rule registration order; matches §5.5 (sans the schema-enforced
``metric.direction_explicit`` and the engine-behavior
``metric.optional_telemetry_auto_inclusion``)."""
