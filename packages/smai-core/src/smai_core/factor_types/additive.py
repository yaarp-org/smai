"""``additive`` factor-type plugin — presence/absence of a component.

Per ``designs/smai/02-dsl-and-contracts.md`` §3.2. Baseline = absent
(``technique_id`` null on the baseline entry's level, per DEC-013); each
non-baseline entry must reference a non-null ``technique_id``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from smai_core.entities.validation_report import ValidationError

if TYPE_CHECKING:
    from smai_core.entities.experiment import ExperimentDefinition
    from smai_core.entities.registries import Registries


class AdditivePlugin:
    """Presence/absence of a component. Baseline = absent."""

    name: str = "additive"
    description: str = "Presence/absence of a component. Baseline = absent."

    def validate(
        self,
        experiment: ExperimentDefinition,
        registries: Registries,
    ) -> list[ValidationError]:
        errors: list[ValidationError] = []
        factor = experiment.factors[0]
        entries = experiment.entries

        # --- factor structural checks ---
        if len(entries) < 2:
            errors.append(
                ValidationError(
                    code="additive.factor_too_few_entries",
                    message=(
                        f"Additive factor '{factor.name}' has {len(entries)} entries; "
                        f"needs >=2 (baseline + >=1 treatment)."
                    ),
                    location=f"factors[name={factor.name}]",
                )
            )

        # --- entry-level checks ---
        baseline_entries = [e for e in entries if e.is_baseline]
        if len(baseline_entries) != 1:
            errors.append(
                ValidationError(
                    code="additive.baseline_count",
                    message=(
                        f"Additive factor '{factor.name}' must have exactly 1 baseline "
                        f"entry; found {len(baseline_entries)}."
                    ),
                    location=f"factors[name={factor.name}]",
                )
            )
        for entry in entries:
            if entry.is_baseline and entry.level.technique_id is not None:
                errors.append(
                    ValidationError(
                        code="additive.baseline_must_be_null_technique",
                        message=(
                            f"Additive baseline entry '{entry.id}' has a level with "
                            f"non-null technique_id; baseline must be 'absent' "
                            f"(technique_id null)."
                        ),
                        location=f"entries[id={entry.id}]",
                    )
                )
            if not entry.is_baseline and entry.level.technique_id is None:
                errors.append(
                    ValidationError(
                        code="additive.treatment_must_have_technique",
                        message=(
                            f"Additive non-baseline entry '{entry.id}' has a level with "
                            f"null technique_id; only the baseline can be 'absent'."
                        ),
                        location=f"entries[id={entry.id}]",
                    )
                )

        # --- controlled_conditions checks ---
        # No additive-specific required/forbidden fields in v1; the global
        # rule controls.technique_implies_required_present (Task 1.5) handles
        # technique-category-specific requirements.

        return errors


plugin: AdditivePlugin = AdditivePlugin()
"""Module-level singleton referenced by the entry-point registration."""
