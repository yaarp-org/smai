"""``verify(...)`` entry point — Pass-2 rule composition.

Per ``designs/smai/02-dsl-and-contracts.md`` §5. Runs the full Pass-2 rule
list deterministically (categories A → B → C → D → E → F → G → §5.10
pipeline-rejection, then Category H for FactorModel documents), partitions
findings by severity, and either returns a
:class:`VerifiedExperimentDefinition` (Pass-2 success) or raises
:class:`VerificationError` (one or more error-severity findings).

Determinism (§5.13): rule execution order is the file-order of category
``CATEGORY_*_RULES`` tuples; within each rule, finding emission order is
data-order (iteration over the input). Same input + same registries always
produces the same :class:`ValidationReport`.
"""

from __future__ import annotations

from typing import overload

from smai_core.dsl.document import (
    DslDocument,
    ExperimentDocument,
    FactorModelDocument,
)
from smai_core.entities.experiment import (
    ExperimentDefinition,
    FactorModel,
    VerifiedExperimentDefinition,
)
from smai_core.entities.registries import Registries
from smai_core.entities.validation_report import ValidationError, ValidationReport
from smai_core.verification._types import (
    RuleCallable,
    VerificationError,
)
from smai_core.verification.category_a_factor_structure import CATEGORY_A_RULES
from smai_core.verification.category_b_entry_factor_compatibility import (
    CATEGORY_B_RULES,
)
from smai_core.verification.category_c_controlled_conditions import (
    CATEGORY_C_RULES,
)
from smai_core.verification.category_d_metric_well_formedness import (
    CATEGORY_D_RULES,
)
from smai_core.verification.category_e_validation_criteria_soundness import (
    CATEGORY_E_RULES,
)
from smai_core.verification.category_f_technique_compatibility import (
    CATEGORY_F_RULES,
)
from smai_core.verification.category_g_continuous_ordinal_values import (
    CATEGORY_G_RULES,
)
from smai_core.verification.category_h_factor_model_cross_cg import (
    CATEGORY_H_RULES,
    FactorModelRuleCallable,
)
from smai_core.verification.category_pipeline_rejection import (
    PIPELINE_REJECTION_RULES,
)

PER_CG_RULES: tuple[RuleCallable, ...] = (
    *CATEGORY_A_RULES,
    *CATEGORY_B_RULES,
    *CATEGORY_C_RULES,
    *CATEGORY_D_RULES,
    *CATEGORY_E_RULES,
    *CATEGORY_F_RULES,
    *CATEGORY_G_RULES,
    *PIPELINE_REJECTION_RULES,
)
"""All Pass-2 rule callables that take ``(ExperimentDefinition, Registries)``.

Order is file-order across categories per §5.13's determinism guarantee.
The Category H rules are dispatched separately via
:data:`FACTOR_MODEL_RULES` because they take a different signature.
"""


FACTOR_MODEL_RULES: tuple[FactorModelRuleCallable, ...] = CATEGORY_H_RULES
"""Category H rule callables for FactorModel-level cross-CG checks."""


def _partition(findings: list[ValidationError]) -> ValidationReport:
    errors: list[ValidationError] = []
    warnings: list[ValidationError] = []
    advisories: list[ValidationError] = []
    for finding in findings:
        if finding.severity == "error":
            errors.append(finding)
        elif finding.severity == "warning":
            warnings.append(finding)
        else:
            advisories.append(finding)
    return ValidationReport(
        success=not errors,
        errors=errors,
        warnings=warnings,
        advisories=advisories,
    )


def _run_per_cg_rules(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    findings: list[ValidationError] = []
    for rule in PER_CG_RULES:
        findings.extend(rule(experiment, registries))
    return findings


def _run_factor_model_rules(
    factor_model: FactorModel,
    experiments: list[ExperimentDefinition],
    registries: Registries,
) -> list[ValidationError]:
    findings: list[ValidationError] = []
    for rule in FACTOR_MODEL_RULES:
        findings.extend(rule(factor_model, experiments, registries))
    return findings


def verify_experiment(
    experiment: ExperimentDefinition, registries: Registries
) -> VerifiedExperimentDefinition:
    """Run Pass-2 verification on a single CG.

    Returns :class:`VerifiedExperimentDefinition` (the marker subclass that is
    the type-system precondition for ``emit_artifacts(...)`` in Task 1.6) on
    success. Raises :class:`VerificationError` carrying the full
    :class:`ValidationReport` when one or more error-severity rules fired.
    Warnings and advisories do NOT raise; they are recorded on the report and
    surface alongside the success path through ``verify_to_report(...)``.
    """
    findings = _run_per_cg_rules(experiment, registries)
    report = _partition(findings)
    if not report.success:
        raise VerificationError(report)
    if isinstance(experiment, VerifiedExperimentDefinition):
        return experiment
    # Re-validate with DSL context: at verify-time ``baseline_entry_id`` is
    # still null (compiler-fill happens during emit_artifacts in Task 1.6).
    return VerifiedExperimentDefinition.model_validate(
        experiment.model_dump(), context={"smai_mode": "dsl"}
    )


def verify_to_report(document: DslDocument, registries: Registries) -> ValidationReport:
    """Run Pass-2 verification and return a :class:`ValidationReport` directly.

    Useful when callers want all findings (including warnings/advisories) and
    don't want exception-driven control flow. Dispatches on document kind:

    - :class:`ExperimentDocument` → per-CG rules only.
    - :class:`FactorModelDocument` → Category H rules on the wrapper plus
      per-CG rules on each comprising experiment, combined into one report.
    """
    if isinstance(document, ExperimentDocument):
        findings = _run_per_cg_rules(document.experiment, registries)
        return _partition(findings)
    findings: list[ValidationError] = []
    findings.extend(
        _run_factor_model_rules(document.factor_model, document.experiments, registries)
    )
    for experiment in document.experiments:
        findings.extend(_run_per_cg_rules(experiment, registries))
    return _partition(findings)


@overload
def verify(
    document: ExperimentDefinition, registries: Registries
) -> VerifiedExperimentDefinition: ...


@overload
def verify(
    document: ExperimentDocument, registries: Registries
) -> VerifiedExperimentDefinition: ...


@overload
def verify(
    document: FactorModelDocument, registries: Registries
) -> list[VerifiedExperimentDefinition]: ...


def verify(  # type: ignore[no-redef]
    document: ExperimentDefinition | DslDocument, registries: Registries
) -> VerifiedExperimentDefinition | list[VerifiedExperimentDefinition]:
    """Verify a CG, an :class:`ExperimentDocument`, or a FactorModelDocument.

    Returns:

    - :class:`VerifiedExperimentDefinition` for a single CG input.
    - ``list[VerifiedExperimentDefinition]`` for a FactorModelDocument
      (one per comprising CG, ordered by ``factor_model.comparison_group_ids``
      where possible, else input order).

    Raises :class:`VerificationError` on any error-severity finding.
    Warnings / advisories are silently recorded; use ``verify_to_report``
    when you need them.

    Idempotency: passing a :class:`VerifiedExperimentDefinition` re-verifies
    deterministically and returns it (or an equal :class:`VerifiedExperimentDefinition`).
    """
    if isinstance(document, ExperimentDocument):
        return verify_experiment(document.experiment, registries)
    if isinstance(document, FactorModelDocument):
        report = verify_to_report(document, registries)
        if not report.success:
            raise VerificationError(report)
        return [
            VerifiedExperimentDefinition.model_validate(
                exp.model_dump(), context={"smai_mode": "dsl"}
            )
            for exp in document.experiments
        ]
    return verify_experiment(document, registries)
