"""Category A — Factor structure (§5.2).

Each factor is well-formed: type is registered, names are unique within the
experiment, the factor has at least 2 entries, and the factor-type plugin's
unified ``validate(...)`` method (§3.1, §5.11) reports no errors.

The Pass-1 ``factor.length_constraint_v1`` rule (factors length ∈ [1,1]) is
enforced by ``ExperimentDefinition.factors``'s Pydantic ``Field`` constraints
and is NOT a Pass-2 rule per §5.2.
"""

from __future__ import annotations

from smai_core.entities.experiment import ExperimentDefinition
from smai_core.entities.registries import Registries
from smai_core.entities.validation_report import ValidationError


def factor_type_registered(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``factor.type_registered`` (error). Factor's ``type`` matches a loaded plugin."""
    findings: list[ValidationError] = []
    plugins = registries.factor_type_plugins
    for factor in experiment.factors:
        if factor.type not in plugins:
            findings.append(
                ValidationError(
                    code="factor.type_registered",
                    message=(
                        f"Factor '{factor.name}' declares type {factor.type!r} "
                        f"which is not a loaded factor-type plugin "
                        f"(loaded: {sorted(plugins) if plugins else '<none>'})."
                    ),
                    location=f"factors[name={factor.name}]",
                )
            )
    return findings


def factor_unique_name_within_experiment(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``factor.unique_name_within_experiment`` (error). Factor names are unique."""
    findings: list[ValidationError] = []
    seen: dict[str, int] = {}
    for factor in experiment.factors:
        seen[factor.name] = seen.get(factor.name, 0) + 1
    for name, count in seen.items():
        if count > 1:
            findings.append(
                ValidationError(
                    code="factor.unique_name_within_experiment",
                    message=(
                        f"Factor name '{name}' appears {count} times; factor names "
                        f"must be unique within an experiment."
                    ),
                    location=f"factors[name={name}]",
                )
            )
    return findings


def factor_has_entries(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``factor.has_entries`` (error). Factor has ≥2 entries.

    The plugin contract also enforces this; the rule exists at the verifier
    level so a missing/unloaded plugin does not silently skip the check.
    """
    findings: list[ValidationError] = []
    if len(experiment.entries) < 2:
        factor_name = experiment.factors[0].name if experiment.factors else "<unknown>"
        findings.append(
            ValidationError(
                code="factor.has_entries",
                message=(
                    f"Factor '{factor_name}' has {len(experiment.entries)} entries; needs ≥2."
                ),
                location=f"factors[name={factor_name}]",
            )
        )
    return findings


def factor_plugin_validate(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``factor.plugin_validate`` (error).

    Dispatches to ``factor_type_plugin.validate(experiment, registries)``
    (per §3.1, §5.11) and surfaces every plugin finding with its plugin-namespaced
    code intact (the plugin's ``code`` becomes a sub-error per §5.11).

    If the factor's type is not loaded, this rule is a no-op — the
    ``factor.type_registered`` rule reports that case separately so the
    verifier doesn't double-report.
    """
    findings: list[ValidationError] = []
    plugins = registries.factor_type_plugins
    for factor in experiment.factors:
        plugin = plugins.get(factor.type)
        if plugin is None:
            continue
        findings.extend(plugin.validate(experiment, registries))
    return findings


CATEGORY_A_RULES = (
    factor_type_registered,
    factor_unique_name_within_experiment,
    factor_has_entries,
    factor_plugin_validate,
)
"""Category A rule registration order; matches §5.2's table order."""
