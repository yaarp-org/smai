"""``substitutive`` factor-type plugin — choice among alternatives.

Per ``designs/smai/02-dsl-and-contracts.md`` §3.3. Baseline = a reference
level (e.g., VGG). Every entry references a non-null ``technique_id`` —
'absent' levels are not admitted (use ``additive`` for null-control
semantics). The factor's name (and the shared category of the entries'
techniques, by v1 convention) must NOT appear in
``controlled_conditions``: a field cannot be both varied and held constant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from smai_core.entities.validation_report import ValidationError

if TYPE_CHECKING:
    from smai_core.entities.experiment import ExperimentDefinition
    from smai_core.entities.registries import Registries


class SubstitutivePlugin:
    """Choice among alternatives. Baseline = reference level."""

    name: str = "substitutive"
    description: str = "Choice among alternatives. Baseline = reference level."

    def validate(
        self,
        experiment: ExperimentDefinition,
        registries: Registries,
    ) -> list[ValidationError]:
        errors: list[ValidationError] = []
        factor = experiment.factors[0]
        entries = experiment.entries
        techniques = registries.technique_registry

        # --- factor structural checks ---
        if len(entries) < 2:
            errors.append(
                ValidationError(
                    code="substitutive.factor_too_few_entries",
                    message=(
                        f"Substitutive factor '{factor.name}' has {len(entries)} entries; "
                        f"needs >=2 (reference + >=1 alternative)."
                    ),
                    location=f"factors[name={factor.name}]",
                )
            )

        # --- entry-level checks ---
        baseline_entries = [e for e in entries if e.is_baseline]
        if len(baseline_entries) != 1:
            errors.append(
                ValidationError(
                    code="substitutive.baseline_count",
                    message=(
                        f"Substitutive factor '{factor.name}' must have exactly 1 baseline "
                        f"entry; found {len(baseline_entries)}."
                    ),
                    location=f"factors[name={factor.name}]",
                )
            )
        for entry in entries:
            if entry.level.technique_id is None:
                errors.append(
                    ValidationError(
                        code="substitutive.all_techniques_required",
                        message=(
                            f"Substitutive entry '{entry.id}' has technique_id=null; "
                            f"substitutive factors don't admit 'absent' levels (use "
                            f"additive for null-control semantics)."
                        ),
                        location=f"entries[id={entry.id}]",
                    )
                )

        # --- controlled_conditions checks ---
        # The factor's name (and, by v1 convention, the shared category name
        # of the entries' techniques) must NOT appear in controlled_conditions.
        forbidden_names: set[str] = {factor.name}
        techs_in_use = [
            techniques[entry.level.technique_id]
            for entry in entries
            if entry.level.technique_id is not None and entry.level.technique_id in techniques
        ]
        if techs_in_use:
            shared_categories = {tech.category for tech in techs_in_use}
            if len(shared_categories) == 1:
                forbidden_names.update(shared_categories)
        for forbidden in forbidden_names:
            if experiment.controlled_conditions.has_field(forbidden):
                errors.append(
                    ValidationError(
                        code="substitutive.factor_not_in_controls",
                        message=(
                            f"Field '{forbidden}' is both the factor and a controlled "
                            f"condition; this is contradictory."
                        ),
                        location=f"controlled_conditions.{forbidden}",
                    )
                )

        return errors


plugin: SubstitutivePlugin = SubstitutivePlugin()
"""Module-level singleton referenced by the entry-point registration."""
