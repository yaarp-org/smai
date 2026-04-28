"""Category H — FactorModel cross-CG checks (§5.9).

These rules operate on a ``FactorModel`` plus its comprising
``ExperimentDefinition`` list, not on a single CG. They are dispatched by
``verify(...)`` only when the input document is a ``FactorModelDocument``;
they NEVER run on a stand-alone CG.

``factor_model.id_unique`` (§5.9 row 1) is intentionally not a Pass-2 rule in
v1 — it requires *multiple* loaded ``FactorModel`` objects to compare against,
which the v1 single-document ``verify(...)`` entry point doesn't see. The
check belongs upstream of ``smai-core`` (Tier B integrators / pipeline-layer
``MetadataStore``); reopen if/when ``verify(...)`` accepts a registry of
documents.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

from smai_core.entities.experiment import ExperimentDefinition, FactorModel
from smai_core.entities.registries import Registries
from smai_core.entities.validation_report import ValidationError

FactorModelRuleCallable: TypeAlias = Callable[
    [FactorModel, list[ExperimentDefinition], Registries], list[ValidationError]
]
"""Signature for Category H rules.

Different shape from per-CG rules (``RuleCallable``): receives the FactorModel
plus the list of comprising experiments. The verifier dispatches both
families, partitioned by document kind, and combines the findings into one
:class:`~smai_core.entities.validation_report.ValidationReport`.
"""


def factor_model_bidirectional_references_consistent(
    factor_model: FactorModel,
    experiments: list[ExperimentDefinition],
    registries: Registries,
) -> list[ValidationError]:
    """``factor_model.bidirectional_references_consistent`` (error).

    Each id in ``factor_model.comparison_group_ids`` is the id of one of the
    comprising experiments, AND each comprising experiment's
    ``factor_model_id`` equals ``factor_model.id``. Both directions are
    checked so a typo in either side surfaces.
    """
    findings: list[ValidationError] = []
    by_id: dict[str, ExperimentDefinition] = {exp.id: exp for exp in experiments}
    for cg_id in factor_model.comparison_group_ids:
        if cg_id not in by_id:
            findings.append(
                ValidationError(
                    code="factor_model.bidirectional_references_consistent",
                    message=(
                        f"FactorModel '{factor_model.id}' lists comparison_group_id "
                        f"'{cg_id}' which is not among the loaded experiments "
                        f"({sorted(by_id)!r})."
                    ),
                    location=(f"factor_model.comparison_group_ids[id={cg_id}]"),
                )
            )
    for exp in experiments:
        if exp.factor_model_id is None or exp.factor_model_id != factor_model.id:
            findings.append(
                ValidationError(
                    code="factor_model.bidirectional_references_consistent",
                    message=(
                        f"Experiment '{exp.id}' has factor_model_id="
                        f"{exp.factor_model_id!r} but is loaded under FactorModel "
                        f"'{factor_model.id}'."
                    ),
                    location=f"experiments[id={exp.id}].factor_model_id",
                )
            )
    return findings


def factor_model_shared_conditions_match(
    factor_model: FactorModel,
    experiments: list[ExperimentDefinition],
    registries: Registries,
) -> list[ValidationError]:
    """``factor_model.shared_conditions_match`` (error).

    For each comprising CG, every field present in
    ``factor_model.shared_conditions`` is also in
    ``CG.controlled_conditions`` and carries the same value. CGs may have
    additional fields beyond ``shared_conditions`` (those are fine); they
    must not contradict.
    """
    findings: list[ValidationError] = []
    shared = factor_model.shared_conditions
    if shared is None:
        return findings
    shared_dump = shared.model_dump(exclude_none=True)
    shared_dump.pop("additional_fixed", None)

    def _flatten(cc_dump: dict[str, Any]) -> dict[str, Any]:
        out = dict(cc_dump)
        af = out.pop("additional_fixed", None)
        if isinstance(af, dict):
            for k, v in af.items():  # pyright: ignore[reportUnknownVariableType]
                if isinstance(k, str):
                    out.setdefault(k, v)  # pyright: ignore[reportUnknownArgumentType]
        return out

    shared_flat = _flatten(shared_dump)
    for exp in experiments:
        cc_flat = _flatten(exp.controlled_conditions.model_dump(exclude_none=True))
        for field, expected in shared_flat.items():
            if field not in cc_flat:
                findings.append(
                    ValidationError(
                        code="factor_model.shared_conditions_match",
                        message=(
                            f"Experiment '{exp.id}' is missing shared_conditions "
                            f"field '{field}' from FactorModel '{factor_model.id}'."
                        ),
                        location=f"experiments[id={exp.id}].controlled_conditions",
                    )
                )
                continue
            if cc_flat[field] != expected:
                findings.append(
                    ValidationError(
                        code="factor_model.shared_conditions_match",
                        message=(
                            f"Experiment '{exp.id}' has controlled_conditions."
                            f"{field}={cc_flat[field]!r} but FactorModel "
                            f"'{factor_model.id}' shared_conditions.{field}="
                            f"{expected!r}."
                        ),
                        location=(f"experiments[id={exp.id}].controlled_conditions.{field}"),
                    )
                )
    return findings


def factor_model_factor_name_uniqueness(
    factor_model: FactorModel,
    experiments: list[ExperimentDefinition],
    registries: Registries,
) -> list[ValidationError]:
    """``factor_model.factor_name_uniqueness`` (warning).

    Within one FactorModel, no two CGs declare a factor with the same name.
    Catches accidental duplicate-factor experiments (the worked Example 4
    ResNet-50/ImageNet model has activation/normalization/augmentation —
    distinct factors per CG).
    """
    findings: list[ValidationError] = []
    by_factor: dict[str, list[str]] = {}
    for exp in experiments:
        for factor in exp.factors:
            by_factor.setdefault(factor.name, []).append(exp.id)
    for name, exp_ids in by_factor.items():
        if len(exp_ids) > 1:
            findings.append(
                ValidationError(
                    code="factor_model.factor_name_uniqueness",
                    message=(
                        f"Factor name '{name}' is shared by experiments "
                        f"{sorted(exp_ids)!r} within FactorModel "
                        f"'{factor_model.id}'."
                    ),
                    location=f"factor_model[{factor_model.id}]",
                    severity="warning",
                )
            )
    return findings


def factor_model_cross_cg_factor_vs_control(
    factor_model: FactorModel,
    experiments: list[ExperimentDefinition],
    registries: Registries,
) -> list[ValidationError]:
    """``factor_model.cross_cg_factor_vs_control`` (advisory).

    If CG_n's ``controlled_conditions`` carries a field X that is the factor
    in another CG_m within the same FactorModel, X's value in CG_n should
    match CG_m's reference baseline level (``Level.name`` of the entry with
    ``is_baseline=True``). Encourages "variations are studied at the
    reference architecture point" research-question integrity (§5.9 row 5).
    """
    findings: list[ValidationError] = []
    factor_to_baseline: dict[str, tuple[str, str | None]] = {}
    for exp in experiments:
        if not exp.factors:
            continue
        factor_name = exp.factors[0].name
        baseline = next((e for e in exp.entries if e.is_baseline), None)
        baseline_level_name = baseline.level.name if baseline is not None else None
        factor_to_baseline[factor_name] = (exp.id, baseline_level_name)

    for exp in experiments:
        cc = exp.controlled_conditions
        for factor_name, (other_exp_id, expected) in factor_to_baseline.items():
            if other_exp_id == exp.id:
                continue
            if not cc.has_field(factor_name) or expected is None:
                continue
            actual = cc.get_field(factor_name)
            if actual != expected:
                findings.append(
                    ValidationError(
                        code="factor_model.cross_cg_factor_vs_control",
                        message=(
                            f"Experiment '{exp.id}' holds '{factor_name}'={actual!r} "
                            f"as a controlled condition while experiment "
                            f"'{other_exp_id}' uses '{factor_name}' as a factor "
                            f"with baseline level '{expected}'. Variations are "
                            f"typically studied at the reference baseline point."
                        ),
                        location=(f"experiments[id={exp.id}].controlled_conditions.{factor_name}"),
                        severity="advisory",
                    )
                )
    return findings


CATEGORY_H_RULES: tuple[FactorModelRuleCallable, ...] = (
    factor_model_bidirectional_references_consistent,
    factor_model_shared_conditions_match,
    factor_model_factor_name_uniqueness,
    factor_model_cross_cg_factor_vs_control,
)
"""Category H rule registration order; matches §5.9 (sans
``factor_model.id_unique`` which requires cross-document state)."""
