"""Category B — Entry / factor compatibility (§5.3).

Each entry is a well-formed arm of the comparison: id is unique, the inline
level points at the CG's factor, level names don't collide, the CG has at
least two entries, and duplicate treatments don't sneak in under different
ids.

``entry.plugin_validate`` (per §5.3 row 4) is *descriptive* of the entry-level
subset of the unified plugin dispatch — those sub-errors flow through
``factor.plugin_validate`` in Category A (§5.11), so this module does not
re-dispatch the plugin.
"""

from __future__ import annotations

import json

from smai_core.entities.experiment import ExperimentDefinition
from smai_core.entities.factor import Entry
from smai_core.entities.registries import Registries
from smai_core.entities.validation_report import ValidationError


def entry_unique_id(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``entry.unique_id`` (error). Entry ids are distinct within the CG."""
    findings: list[ValidationError] = []
    seen: dict[str, int] = {}
    for entry in experiment.entries:
        seen[entry.id] = seen.get(entry.id, 0) + 1
    for entry_id, count in seen.items():
        if count > 1:
            findings.append(
                ValidationError(
                    code="entry.unique_id",
                    message=(
                        f"Entry id '{entry_id}' appears {count} times; entry ids "
                        f"must be unique within an experiment."
                    ),
                    location=f"entries[id={entry_id}]",
                )
            )
    return findings


def entry_level_factor_matches(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``entry.level_factor_matches`` (error). ``entry.level.factor`` matches the CG factor."""
    findings: list[ValidationError] = []
    if not experiment.factors:
        return findings
    factor_name = experiment.factors[0].name
    for entry in experiment.entries:
        if entry.level.factor != factor_name:
            findings.append(
                ValidationError(
                    code="entry.level_factor_matches",
                    message=(
                        f"Entry '{entry.id}' has level.factor={entry.level.factor!r} "
                        f"but the CG's factor name is {factor_name!r}."
                    ),
                    location=f"entries[id={entry.id}].level.factor",
                )
            )
    return findings


def entry_level_name_unique(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``entry.level_name_unique`` (error). Level names are distinct within the CG."""
    findings: list[ValidationError] = []
    seen: dict[str, list[str]] = {}
    for entry in experiment.entries:
        seen.setdefault(entry.level.name, []).append(entry.id)
    for level_name, entry_ids in seen.items():
        if len(entry_ids) > 1:
            findings.append(
                ValidationError(
                    code="entry.level_name_unique",
                    message=(
                        f"Level name '{level_name}' is shared by entries "
                        f"{sorted(entry_ids)!r}; level names must be unique."
                    ),
                    location=f"entries[level.name={level_name}]",
                )
            )
    return findings


def entry_at_least_two(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``entry.at_least_two`` (error). A CG has ≥2 entries (baseline + ≥1 treatment)."""
    findings: list[ValidationError] = []
    if len(experiment.entries) < 2:
        findings.append(
            ValidationError(
                code="entry.at_least_two",
                message=(
                    f"CG has {len(experiment.entries)} entries; needs ≥2 "
                    f"(one baseline + ≥1 treatment)."
                ),
                location="entries",
            )
        )
    return findings


def _params_signature(entry: Entry) -> str:
    """Stable JSON signature of ``(technique_id, technique_params, value)``.

    Used for the duplicate-detection rule. ``json.dumps(..., sort_keys=True)``
    canonicalizes parameter dict ordering. The full level shape (including
    ``value``) is hashed so a duplicate technique with the same params but a
    different ordinal/continuous value is NOT flagged — those represent
    distinct sweep points.
    """
    level = entry.level
    params: dict[str, object] = dict(level.technique_params or {})
    value = (
        None
        if level.value is None
        else {
            "value": level.value.value,
            "kind": level.value.kind,
            "unit": level.value.unit,
            "min": level.value.min,
            "max": level.value.max,
        }
    )
    return json.dumps(
        {"technique_id": level.technique_id, "params": params, "value": value},
        sort_keys=True,
    )


def entry_no_duplicate_techniques_with_same_params(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``entry.no_duplicate_techniques_with_same_params`` (warning).

    Two non-baseline entries that reference the same technique with byte-equal
    params and byte-equal value are almost certainly a mistake — the same
    treatment is being run under two ids. Skipping baselines lets users
    explicitly include a "no-op control" treatment without flagging.
    """
    findings: list[ValidationError] = []
    seen: dict[str, str] = {}
    for entry in experiment.entries:
        if entry.is_baseline or entry.level.technique_id is None:
            continue
        sig = _params_signature(entry)
        if sig in seen:
            findings.append(
                ValidationError(
                    code="entry.no_duplicate_techniques_with_same_params",
                    message=(
                        f"Entry '{entry.id}' duplicates entry '{seen[sig]}' — same "
                        f"technique_id, technique_params, and value."
                    ),
                    location=f"entries[id={entry.id}]",
                    severity="warning",
                )
            )
        else:
            seen[sig] = entry.id
    return findings


CATEGORY_B_RULES = (
    entry_unique_id,
    entry_level_factor_matches,
    entry_level_name_unique,
    entry_at_least_two,
    entry_no_duplicate_techniques_with_same_params,
)
"""Category B rule registration order; matches §5.3 (sans the descriptive
``entry.plugin_validate`` row, which folds into Category A's
``factor.plugin_validate`` per §5.11)."""
