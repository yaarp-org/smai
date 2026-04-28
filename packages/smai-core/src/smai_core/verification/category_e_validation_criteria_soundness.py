"""Category E — Validation criteria soundness (§5.6).

The verdict-bearing surface (``ValidationCriteria``) holds together: the
comparison rule's compiler-fill prerequisite is satisfied, the threshold sign
agrees with the direction, the aggregation/comparison method names are in
the closed v1 registries, the seed count is positive (and not silently low),
and any ``trend_check`` declaration applies to entries that actually carry a
``NumericValue``.

``validation.trend_check_inferred`` is engine behavior (auto-enable on
ordinal/continuous entries; §5.6 row 8) and is not a rule.
"""

from __future__ import annotations

from smai_core.entities.experiment import ExperimentDefinition
from smai_core.entities.registries import Registries
from smai_core.entities.validation_report import ValidationError


def validation_baseline_entry_id_resolves(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``validation.baseline_entry_id_resolves`` (error).

    Pre-emission sanity check: when ``comparison.rule == "compare_to_baseline"``,
    exactly one entry carries ``is_baseline=True``. The compiler will fill
    ``baseline_entry_id`` at emission time from that entry; a 0/≥2-baseline
    document would have nothing to fill (or ambiguous fill). Plugin rules also
    enforce exactly-1 baseline; this rule documents the dependency from the
    rule layer in addition.
    """
    findings: list[ValidationError] = []
    if experiment.validation.comparison.rule != "compare_to_baseline":
        return findings
    baselines = [e for e in experiment.entries if e.is_baseline]
    if len(baselines) != 1:
        findings.append(
            ValidationError(
                code="validation.baseline_entry_id_resolves",
                message=(
                    f"comparison.rule='compare_to_baseline' requires exactly 1 "
                    f"entry with is_baseline=True; found {len(baselines)}. "
                    f"baseline_entry_id cannot be resolved at emission."
                ),
                location="validation.comparison",
            )
        )
    return findings


def validation_threshold_sign_matches_direction(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``validation.threshold_sign_matches_direction`` (error).

    Threshold < 0 is rejected; threshold == 0 is allowed (per §5.6: "Use
    compare_to_baseline with threshold=0.0 to allow ties; not an error").
    Direction is preserved on the artifact's ``direction`` field, so a
    positive threshold means "treatment beats baseline / target by ≥ threshold"
    in either direction.
    """
    findings: list[ValidationError] = []
    threshold = experiment.validation.comparison.threshold
    if threshold < 0:
        findings.append(
            ValidationError(
                code="validation.threshold_sign_matches_direction",
                message=(
                    f"validation.comparison.threshold={threshold} is negative; "
                    f"thresholds must be ≥ 0 (use 0.0 to allow ties)."
                ),
                location="validation.comparison.threshold",
            )
        )
    return findings


def validation_aggregation_method_known(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``validation.aggregation_method_known`` (error).

    The aggregation method name resolves in the closed ``mean``/``median``
    registry from ``06-mechanical-evaluation.md`` §6. Pydantic already locks
    the ``Literal`` to that set; the rule layer additionally checks the
    registry to defend against a registry that drops one of the v1 entries.
    """
    findings: list[ValidationError] = []
    method = experiment.validation.aggregation.method
    if method not in registries.aggregation_rule_registry:
        findings.append(
            ValidationError(
                code="validation.aggregation_method_known",
                message=(
                    f"Aggregation method {method!r} is not in the loaded "
                    f"aggregation registry (loaded: "
                    f"{sorted(registries.aggregation_rule_registry)!r})."
                ),
                location="validation.aggregation.method",
            )
        )
    return findings


def validation_comparison_rule_well_formed(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``validation.comparison_rule_well_formed`` (error).

    Three combined checks:

    - The comparison ``rule`` name is in the closed v1 comparison registry
      (defensive, parallel to ``aggregation_method_known``).
    - ``compare_to_target`` requires ``target_value`` (schema-enforced; this
      rule layer adds redundant defense for clarity in the report).
    - ``compare_to_baseline`` requires ``target_value`` to be absent (NOT
      schema-enforced — only the rule layer catches a stray ``target_value``
      paired with a baseline rule).
    """
    findings: list[ValidationError] = []
    rule = experiment.validation.comparison.rule
    target_value = experiment.validation.comparison.target_value

    if rule not in registries.comparison_rule_registry:
        findings.append(
            ValidationError(
                code="validation.comparison_rule_well_formed",
                message=(
                    f"Comparison rule {rule!r} is not in the loaded comparison "
                    f"registry (loaded: "
                    f"{sorted(registries.comparison_rule_registry)!r})."
                ),
                location="validation.comparison.rule",
            )
        )
    if rule == "compare_to_target" and target_value is None:
        findings.append(
            ValidationError(
                code="validation.comparison_rule_well_formed",
                message=(
                    "comparison.rule='compare_to_target' requires "
                    "comparison.target_value to be set."
                ),
                location="validation.comparison.target_value",
            )
        )
    if rule == "compare_to_baseline" and target_value is not None:
        findings.append(
            ValidationError(
                code="validation.comparison_rule_well_formed",
                message=(
                    f"comparison.rule='compare_to_baseline' must NOT carry a "
                    f"target_value (got {target_value!r}); target_value is only "
                    f"valid with compare_to_target."
                ),
                location="validation.comparison.target_value",
            )
        )
    return findings


def validation_seed_count_required_positive(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``validation.seed_count_required_positive`` (error)."""
    findings: list[ValidationError] = []
    n = experiment.validation.seed_count_required
    if n < 1:
        findings.append(
            ValidationError(
                code="validation.seed_count_required_positive",
                message=f"validation.seed_count_required={n} must be ≥ 1.",
                location="validation.seed_count_required",
            )
        )
    return findings


def validation_seed_count_recommended_minimum(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``validation.seed_count_recommended_minimum`` (warning).

    Non-blocking: some users want N=1 quick verdicts. ``< 3`` warns about
    statistical power without rejecting the document.
    """
    findings: list[ValidationError] = []
    n = experiment.validation.seed_count_required
    if n >= 1 and n < 3:
        findings.append(
            ValidationError(
                code="validation.seed_count_recommended_minimum",
                message=(
                    f"validation.seed_count_required={n} (< 3); statistical power "
                    f"is low. Consider ≥ 3 seeds for a robust verdict."
                ),
                location="validation.seed_count_required",
                severity="warning",
            )
        )
    return findings


def validation_trend_check_applicability(
    experiment: ExperimentDefinition, registries: Registries
) -> list[ValidationError]:
    """``validation.trend_check_applicability`` (error).

    If ``trend_check`` is explicitly declared, every non-baseline entry has
    a ``Level.value`` carrying a ``NumericValue`` with consistent ``kind``.
    (Trend doesn't make sense if entries don't have ordinal/continuous values.)
    """
    findings: list[ValidationError] = []
    if experiment.validation.trend_check is None:
        return findings
    non_baselines = [e for e in experiment.entries if not e.is_baseline]
    missing = [e.id for e in non_baselines if e.level.value is None]
    if missing:
        findings.append(
            ValidationError(
                code="validation.trend_check_applicability",
                message=(
                    f"trend_check is declared but non-baseline entries {missing!r} "
                    f"lack a Level.value. trend_check requires NumericValue on every "
                    f"non-baseline entry."
                ),
                location="validation.trend_check",
            )
        )
        return findings
    kinds = {e.level.value.kind for e in non_baselines if e.level.value is not None}
    if len(kinds) > 1:
        findings.append(
            ValidationError(
                code="validation.trend_check_applicability",
                message=(
                    f"trend_check is declared but non-baseline entries mix value "
                    f"kinds {sorted(kinds)!r}; trend_check requires a consistent "
                    f"kind."
                ),
                location="validation.trend_check",
            )
        )
    return findings


CATEGORY_E_RULES = (
    validation_baseline_entry_id_resolves,
    validation_threshold_sign_matches_direction,
    validation_aggregation_method_known,
    validation_comparison_rule_well_formed,
    validation_seed_count_required_positive,
    validation_seed_count_recommended_minimum,
    validation_trend_check_applicability,
)
"""Category E rule registration order; matches §5.6 (sans the engine-behavior
``validation.trend_check_inferred``)."""
