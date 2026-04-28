"""Category C — Controlled conditions (§5.4).

The CG's controlled conditions are well-formed: required fields are present,
the seed list is long enough and distinct, and every entry's technique's
``implies_controlled`` declarations are honored.

``controls.factor_not_in_controls`` (§5.4 row 4) is plugin-delegated — its
finding flows through Category A's ``factor.plugin_validate`` dispatch. The
``controls.no_extra_fields`` row in §5.4 is intentionally not a rule
(``extra="allow"`` admits domain-specific top-level fields by design).

All ``controls.*`` rules look up fields via
``ControlledConditions.has_field(name)`` so the rule logic is uniform across
declared core fields, Pydantic ``extra="allow"`` extras, and
``additional_fixed`` keys (§5.4 lookup convention).
"""

from __future__ import annotations

from smai_core.entities.experiment import ExperimentDefinition
from smai_core.entities.registries import Registries
from smai_core.entities.validation_report import ValidationError


def controls_required_fields_present(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``controls.required_fields_present`` (error).

    The global v1 requirements are ``dataset.name``, ``optimization.optimizer``,
    and a non-empty ``seeds`` list (§5.4 row 1). ``dataset`` and ``optimization``
    are typed as dicts on the entity; we check the inner key. ``seeds`` is a
    declared list field — its non-emptiness is the structural requirement.
    """
    findings: list[ValidationError] = []
    cc = experiment.controlled_conditions

    if "name" not in cc.dataset or not cc.dataset["name"]:
        findings.append(
            ValidationError(
                code="controls.required_fields_present",
                message="controlled_conditions.dataset.name is required and non-empty.",
                location="controlled_conditions.dataset.name",
            )
        )
    if "optimizer" not in cc.optimization or not cc.optimization["optimizer"]:
        findings.append(
            ValidationError(
                code="controls.required_fields_present",
                message=("controlled_conditions.optimization.optimizer is required and non-empty."),
                location="controlled_conditions.optimization.optimizer",
            )
        )
    if not cc.seeds:
        findings.append(
            ValidationError(
                code="controls.required_fields_present",
                message="controlled_conditions.seeds must be a non-empty list.",
                location="controlled_conditions.seeds",
            )
        )
    return findings


def controls_seeds_count_matches_required(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``controls.seeds_count_matches_required`` (error).

    ``len(controlled_conditions.seeds) >= validation.seed_count_required`` —
    the user-supplied seed list must cover at least the verdict's minimum.
    """
    findings: list[ValidationError] = []
    cc = experiment.controlled_conditions
    required = experiment.validation.seed_count_required
    if len(cc.seeds) < required:
        findings.append(
            ValidationError(
                code="controls.seeds_count_matches_required",
                message=(
                    f"controlled_conditions.seeds has {len(cc.seeds)} value(s); "
                    f"validation.seed_count_required is {required}."
                ),
                location="controlled_conditions.seeds",
            )
        )
    return findings


def controls_seeds_unique(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``controls.seeds_unique`` (error). Seed values are distinct."""
    findings: list[ValidationError] = []
    seeds = experiment.controlled_conditions.seeds
    if len(set(seeds)) != len(seeds):
        seen: dict[int, int] = {}
        for seed in seeds:
            seen[seed] = seen.get(seed, 0) + 1
        duplicates = sorted(s for s, count in seen.items() if count > 1)
        findings.append(
            ValidationError(
                code="controls.seeds_unique",
                message=(
                    f"controlled_conditions.seeds has duplicate value(s) "
                    f"{duplicates!r}; all seed values must be distinct."
                ),
                location="controlled_conditions.seeds",
            )
        )
    return findings


def controls_technique_implies_required_present(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``controls.technique_implies_required_present`` (error).

    For every entry, every field declared in the entry's technique's
    ``implies_controlled`` list must be present in ``controlled_conditions``
    (checked via ``has_field()`` to cover all three structural locations).
    Entries whose ``technique_id`` is null (additive baseline) or whose
    technique is not in the registry contribute no implied requirements —
    technique-not-registered is reported separately by ``technique.id_registered``.
    """
    findings: list[ValidationError] = []
    techniques = registries.technique_registry
    cc = experiment.controlled_conditions
    for entry in experiment.entries:
        tech_id = entry.level.technique_id
        if tech_id is None:
            continue
        technique = techniques.get(tech_id)
        if technique is None:
            continue
        for implied in technique.implies_controlled:
            if not cc.has_field(implied):
                findings.append(
                    ValidationError(
                        code="controls.technique_implies_required_present",
                        message=(
                            f"Entry '{entry.id}' uses technique '{tech_id}' which "
                            f"requires controlled field '{implied}'; not present in "
                            f"controlled_conditions."
                        ),
                        location=f"entries[id={entry.id}]",
                    )
                )
    return findings


CATEGORY_C_RULES = (
    controls_required_fields_present,
    controls_seeds_count_matches_required,
    controls_seeds_unique,
    controls_technique_implies_required_present,
)
"""Category C rule registration order. ``controls.factor_not_in_controls`` is
plugin-delegated; ``controls.no_extra_fields`` is not a rule (§5.4)."""
