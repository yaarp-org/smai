"""Category F — Technique compatibility (§5.7).

Each entry's technique resolves in the registry, declares the right factor
type compatibility, validates its per-entry parameters against the technique's
JSON Schema, and carries a fidelity anchor unless ``standard=True`` per
DEC-032.

Adds the closed-enum check on ``TechniqueRef.category`` (Task 1.5
carry-forward from Task 1.1; the v1 closed set is enumerated in
``01-data-model.md`` §3.2 and is verified here rather than via Pydantic
``Literal`` so registry extensions in vN are additive data, not schema
changes).

The dropped ``technique.affects_extension_points_appropriate`` rule (§5.7
trailing paragraph, removed per DEC-031 / Decision 1) is intentionally absent
— harness-side wiring lives in pipeline-layer review gates against the
``HarnessAPIManifest``.
"""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from smai_core.entities.experiment import ExperimentDefinition
from smai_core.entities.registries import Registries
from smai_core.entities.validation_report import ValidationError

TECHNIQUE_CATEGORY_V1_CLOSED_SET: frozenset[str] = frozenset(
    {
        "architecture",
        "augmentation",
        "optimizer",
        "loss",
        "regularization",
        "normalization",
        "activation",
        "pruning",
        "fine_tuning",
        "position_embedding",
        "self_supervision",
        "distillation",
        "schedule",
        "callback",
        "other",
    }
)
"""Closed v1 enum for ``TechniqueRef.category`` per ``01-data-model.md`` §3.2.

Stored as data so the rule layer enforces it (rather than as a Pydantic
``Literal`` on ``TechniqueRef.category``). Extending this set in vN is an
additive registry change; Tier B integrators that care can subclass /
shadow this constant if they need a different vocabulary."""


def technique_id_registered(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``technique.id_registered`` (error)."""
    findings: list[ValidationError] = []
    techniques = registries.technique_registry
    for entry in experiment.entries:
        tech_id = entry.level.technique_id
        if tech_id is None:
            continue
        if tech_id not in techniques:
            findings.append(
                ValidationError(
                    code="technique.id_registered",
                    message=(
                        f"Entry '{entry.id}' references technique_id={tech_id!r} "
                        f"which is not in the technique registry."
                    ),
                    location=f"entries[id={entry.id}].level.technique_id",
                )
            )
    return findings


def technique_factor_type_compatible(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``technique.factor_type_compatible`` (error).

    Each entry's technique declares ``compatible_factor_types`` containing the
    experiment's factor type. Skips entries whose technique is unregistered —
    ``technique.id_registered`` reports those separately.
    """
    findings: list[ValidationError] = []
    if not experiment.factors:
        return findings
    factor_type = experiment.factors[0].type
    techniques = registries.technique_registry
    for entry in experiment.entries:
        tech_id = entry.level.technique_id
        if tech_id is None:
            continue
        technique = techniques.get(tech_id)
        if technique is None:
            continue
        if factor_type not in technique.compatible_factor_types:
            findings.append(
                ValidationError(
                    code="technique.factor_type_compatible",
                    message=(
                        f"Entry '{entry.id}' uses technique '{tech_id}' which is "
                        f"not compatible with factor type {factor_type!r} "
                        f"(declared compatible_factor_types: "
                        f"{technique.compatible_factor_types!r})."
                    ),
                    location=f"entries[id={entry.id}].level.technique_id",
                )
            )
    return findings


def _validate_against_schema(instance: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Run JSON Schema validation; return error messages (empty if valid)."""
    try:
        validator = Draft202012Validator(schema)
    except SchemaError as exc:
        return [f"technique parameter_schema is itself invalid: {exc.message}"]
    messages: list[str] = []
    for err in validator.iter_errors(instance):  # pyright: ignore[reportUnknownMemberType]
        message_obj: Any = err.message
        messages.append(str(message_obj))
    return messages


def technique_params_validate(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``technique.params_validate`` (error).

    Three sub-codes per §5.7 row 3:

    - ``technique.params_not_accepted`` — params are non-null but the
      technique declares no parameter schema.
    - ``technique.params_missing`` — schema declares required fields but
      params are null.
    - ``technique.params_validate`` — JSON Schema validation against
      ``parameter_schema`` failed (the structural case).
    """
    findings: list[ValidationError] = []
    techniques = registries.technique_registry
    for entry in experiment.entries:
        tech_id = entry.level.technique_id
        if tech_id is None:
            continue
        technique = techniques.get(tech_id)
        if technique is None:
            continue
        params = entry.level.technique_params
        schema = technique.parameter_schema

        if params is not None and schema is None:
            findings.append(
                ValidationError(
                    code="technique.params_not_accepted",
                    message=(
                        f"Entry '{entry.id}' supplies technique_params for "
                        f"technique '{tech_id}' which declares no parameter_schema."
                    ),
                    location=f"entries[id={entry.id}].level.technique_params",
                )
            )
            continue
        if schema is None:
            continue
        required_raw: Any = schema.get("required") or []
        required: list[str] = []
        if isinstance(required_raw, list):
            for item in required_raw:  # pyright: ignore[reportUnknownVariableType]
                required.append(str(item))  # pyright: ignore[reportUnknownArgumentType]
        if params is None:
            if required:
                findings.append(
                    ValidationError(
                        code="technique.params_missing",
                        message=(
                            f"Entry '{entry.id}' has no technique_params but "
                            f"technique '{tech_id}' requires {required!r}."
                        ),
                        location=f"entries[id={entry.id}].level.technique_params",
                    )
                )
            continue
        errors = _validate_against_schema(dict(params), schema)
        for err in errors:
            findings.append(
                ValidationError(
                    code="technique.params_validate",
                    message=(
                        f"Entry '{entry.id}' technique_params failed schema "
                        f"validation for technique '{tech_id}': {err}"
                    ),
                    location=f"entries[id={entry.id}].level.technique_params",
                )
            )
    return findings


def technique_fidelity_anchor_present_or_standard(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``technique.fidelity_anchor_present_or_standard`` (error).

    Per DEC-015 + DEC-032: ``standard=True`` techniques don't need a
    ``fidelity_anchor``; ``standard=False`` techniques must have one. The
    anchor may be of any kind admitted by the discriminated union (``paper``,
    ``proposal``, ``reviewer_attested``); the rule does not constrain which.
    """
    findings: list[ValidationError] = []
    techniques = registries.technique_registry
    seen: set[str] = set()
    for entry in experiment.entries:
        tech_id = entry.level.technique_id
        if tech_id is None or tech_id in seen:
            continue
        seen.add(tech_id)
        technique = techniques.get(tech_id)
        if technique is None:
            continue
        if not technique.standard and technique.fidelity_anchor is None:
            findings.append(
                ValidationError(
                    code="technique.fidelity_anchor_present_or_standard",
                    message=(
                        f"Technique '{tech_id}' has standard=False but no "
                        f"fidelity_anchor; non-standard techniques must declare "
                        f"a paper / proposal / reviewer_attested anchor."
                    ),
                    location=f"technique_registry[{tech_id}]",
                )
            )
    return findings


def technique_category_in_closed_set(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``technique.category_in_closed_set`` (error).

    Closes the v1 ``TechniqueRef.category`` enum at the rule layer
    (``01-data-model.md`` §3.2). Resolves the Task 1.1 carry-forward — the
    entity uses ``str`` rather than ``Literal[...]`` so registry extensions
    in vN are additive data, not schema changes.

    Reports the technique once even when referenced by multiple entries.
    """
    findings: list[ValidationError] = []
    techniques = registries.technique_registry
    seen: set[str] = set()
    for entry in experiment.entries:
        tech_id = entry.level.technique_id
        if tech_id is None or tech_id in seen:
            continue
        seen.add(tech_id)
        technique = techniques.get(tech_id)
        if technique is None:
            continue
        if technique.category not in TECHNIQUE_CATEGORY_V1_CLOSED_SET:
            findings.append(
                ValidationError(
                    code="technique.category_in_closed_set",
                    message=(
                        f"Technique '{tech_id}' has category={technique.category!r} "
                        f"which is not in the v1 closed set "
                        f"{sorted(TECHNIQUE_CATEGORY_V1_CLOSED_SET)!r}."
                    ),
                    location=f"technique_registry[{tech_id}].category",
                )
            )
    return findings


CATEGORY_F_RULES = (
    technique_id_registered,
    technique_factor_type_compatible,
    technique_params_validate,
    technique_fidelity_anchor_present_or_standard,
    technique_category_in_closed_set,
)
"""Category F rule registration order; matches §5.7 plus the
``technique.category_in_closed_set`` rule resolving Task 1.1's carry-forward."""
