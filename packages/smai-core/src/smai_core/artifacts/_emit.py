"""Pure emission of the four contract artifacts from a verified experiment.

Per ``02-dsl-and-contracts.md`` §8.1 + §8.4. Sequential build:

    ExperimentPlan → HarnessContract → TechniqueContract[] → ValidationConfig

Each artifact's ``content_hash`` is computed before any artifact that needs to
reference it is built (parent-hash references thread through the DAG; no
cycles). Determinism (§8.2) rests on canonical-form serialization +
canonical ordering of set-semantic lists.

Carry-forwards from earlier tasks:

* §5.5 row 10 — auto-include cost-tagged metrics in ``optional_telemetry``;
  dedup by ``metric_ref_to_runtime_key``; emit warning advisory when a
  user declaration would be re-added by the auto-include set.
* §5.6 row 8 — auto-enable ``trend_check`` when all non-baseline entries
  carry consistent-``kind`` ``NumericValue``s.
* §2.5 — fill ``ComparisonRule.baseline_entry_id`` from the experiment's
  baseline entry.

This module is the single source of truth for the engine behaviors
the canonical doc flags as "(engine behavior, not a rule)" — they are not
verification rules, but they happen here, deterministically.
"""

from __future__ import annotations

from itertools import product
from typing import Any, cast

from smai_core._runtime_keys import metric_ref_to_runtime_key
from smai_core.artifacts._canonical import (
    freeze_with_hash,
    hash_registry_dict,
    hash_string_set,
)
from smai_core.artifacts._envelope import ArtifactEnvelope, ArtifactKind
from smai_core.artifacts._set import ContractArtifactSet
from smai_core.artifacts._surface_maps import (
    EXPERIMENT_PLAN_SURFACE_MAP,
    HARNESS_CONTRACT_SURFACE_MAP,
    TECHNIQUE_CONTRACT_SURFACE_MAP,
    VALIDATION_CONFIG_SURFACE_MAP,
)
from smai_core.artifacts.experiment_plan import ExperimentPlan, ExperimentPlanBody
from smai_core.artifacts.harness_contract import (
    FixedVariable,
    HarnessContract,
    HarnessContractBody,
)
from smai_core.artifacts.technique_contract import (
    TechniqueContract,
    TechniqueContractBody,
)
from smai_core.artifacts.validation_config import (
    ValidationConfig,
    ValidationConfigBody,
)
from smai_core.dsl.document import FactorModelDocument
from smai_core.entities.conditions import ControlledConditions
from smai_core.entities.experiment import VerifiedExperimentDefinition
from smai_core.entities.factor import Entry
from smai_core.entities.metric import (
    AtomicMetricRef,
    MetricRef,
    ParametricFamily,
    ParametricMetricRef,
)
from smai_core.entities.registries import Registries
from smai_core.entities.validation import (
    ComparisonRule,
    TrendCheck,
    ValidationCriteria,
)
from smai_core.entities.validation_report import ValidationError, ValidationReport

COMPILER_VERSION: str = "0.1.0"
"""Compiler semver stamped onto every artifact envelope (§7.1).

Bumped on incompatible compiler-output changes; matches the ``smai-core``
package version for v1 simplicity.
"""

ARTIFACT_SCHEMA_VERSION: int = 1
"""Schema version stamped onto every artifact envelope (§7.1).

Bumped on incompatible schema changes; v1 ships at ``1``.
"""

# Standard runtime no-go zones. Per §7.4.2 worked example.
_DEFAULT_NO_GO_ZONES: tuple[str, ...] = ("experiment.py", "techniques/__init__.py")


# ---------------------------------------------------------------------------
# Cost-tag auto-inclusion (§5.5 row 10)
# ---------------------------------------------------------------------------


def _enumerate_parametric_refs(family: ParametricFamily) -> list[ParametricMetricRef]:
    """Enumerate every concrete :class:`ParametricMetricRef` for ``family``.

    Single-parameter families produce one ref per element of
    ``parameter_values_seen``; multi-parameter families produce the cartesian
    product over each parameter's value set. The values come straight from
    the registry; the registry-policy invariant (§3.8.1) guarantees no value
    contains ``=`` or ``__`` so the runtime key encoding round-trips.
    """
    seen = family.parameter_values_seen
    if isinstance(family.parameter, str):
        # single-parameter; seen is list[str | int | float]
        if not isinstance(seen, list):
            return []
        return [
            ParametricMetricRef(family=family.family_name, parameters={family.parameter: v})
            for v in seen
        ]
    # multi-parameter; seen is dict[str, list[...]]
    if not isinstance(seen, dict):
        return []
    param_names = list(family.parameter)
    value_lists = [seen[p] for p in param_names if p in seen]
    if len(value_lists) != len(param_names):
        return []
    return [
        ParametricMetricRef(
            family=family.family_name,
            parameters={name: value for name, value in zip(param_names, combo, strict=True)},
        )
        for combo in product(*value_lists)
    ]


def _cost_auto_include(registries: Registries) -> list[MetricRef]:
    """Return every cost-tagged :class:`MetricRef` in canonical order.

    The canonical-union rule (§6.6) deduplicates by
    ``metric_ref_to_runtime_key``. The output here is the auto-included
    side; the user-declared side is folded in by the caller.
    """
    refs: list[MetricRef] = []
    for name, entry in registries.metric_registry.atomic.items():
        if entry.category == "cost":
            refs.append(AtomicMetricRef(ref=name))
    for _, family in registries.metric_registry.parametric.items():
        if family.category == "cost":
            refs.extend(_enumerate_parametric_refs(family))
    refs.sort(key=metric_ref_to_runtime_key)
    return refs


def _canonical_metric_union(
    primary: list[MetricRef],
    secondary: list[MetricRef],
) -> tuple[list[MetricRef], list[str]]:
    """Return ``(deduped_union, redundant_keys)`` per §6.6.

    Deduplicates by ``metric_ref_to_runtime_key``; ``primary`` wins on
    collision. ``redundant_keys`` is the list of runtime keys present in
    *both* inputs — used by the emitter to surface a warning advisory when
    user-declared ``optional_telemetry`` would be re-added by the
    auto-include set.
    """
    seen: set[str] = set()
    out: list[MetricRef] = []
    duplicates: list[str] = []
    primary_keys: set[str] = set()
    for ref in primary:
        key = metric_ref_to_runtime_key(ref)
        if key in seen:
            continue
        seen.add(key)
        primary_keys.add(key)
        out.append(ref)
    for ref in secondary:
        key = metric_ref_to_runtime_key(ref)
        if key in primary_keys:
            duplicates.append(key)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    out.sort(key=metric_ref_to_runtime_key)
    return out, duplicates


# ---------------------------------------------------------------------------
# Trend-check auto-enable (§5.6 row 8)
# ---------------------------------------------------------------------------


def _maybe_default_trend_check(
    experiment: VerifiedExperimentDefinition,
) -> TrendCheck | None:
    """Return the default :class:`TrendCheck` if §5.6 row 8 applies.

    Conditions: every non-baseline entry carries a ``NumericValue`` and all
    those values share the same ``kind``. The default is
    ``expected_direction="no_expectation"``, ``alert_on_violation=False`` —
    pure narrative grounding, no verdict-path effect.
    """
    non_baseline = [e for e in experiment.entries if not e.is_baseline]
    if not non_baseline:
        return None
    kinds: set[str] = set()
    for entry in non_baseline:
        if entry.level.value is None:
            return None
        kinds.add(entry.level.value.kind)
    if len(kinds) != 1:
        return None
    return TrendCheck(expected_direction="no_expectation", alert_on_violation=False)


# ---------------------------------------------------------------------------
# Controlled-conditions flattening for HarnessContract.fixed_variables
# ---------------------------------------------------------------------------


_PRIMITIVE_TYPE_HINTS: dict[type, str] = {
    bool: "bool",  # checked before int — bool subclasses int in Python
    int: "int",
    float: "float",
    str: "str",
}


def _type_hint(value: Any) -> str | None:
    """Infer an informational ``type_hint`` for a fixed variable."""
    for cls, hint in _PRIMITIVE_TYPE_HINTS.items():
        if isinstance(value, cls):
            return hint
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return None


def _flatten(prefix: str, value: Any) -> list[FixedVariable]:
    if isinstance(value, dict):
        out: list[FixedVariable] = []
        nested: dict[str, Any] = cast("dict[str, Any]", value)
        # Sort dict keys for determinism — canonical_json sorts at hash time
        # but FixedVariable list is also sorted by path before emission, so
        # local order doesn't matter; sorted() keeps debugging output tidy.
        for key in sorted(nested.keys()):
            out.extend(_flatten(f"{prefix}.{key}" if prefix else key, nested[key]))
        return out
    return [FixedVariable(path=prefix, value=value, type_hint=_type_hint(value))]


def _flatten_controlled_conditions(
    controlled: ControlledConditions,
) -> list[FixedVariable]:
    """Flatten ``controlled_conditions`` into ``HarnessContract.fixed_variables`` (§7.4).

    Excludes ``seeds`` (which is its own top-level field on
    :class:`HarnessContract`). ``additional_fixed`` is folded back into the
    top-level path namespace; ``extra="allow"`` fields land at top level
    too.
    """
    out: list[FixedVariable] = []
    out.extend(_flatten("dataset", controlled.dataset))
    out.extend(_flatten("optimization", controlled.optimization))
    if controlled.additional_fixed:
        for key in sorted(controlled.additional_fixed.keys()):
            out.extend(_flatten(key, controlled.additional_fixed[key]))
    extras = controlled.model_extra or {}
    for key in sorted(extras.keys()):
        out.extend(_flatten(key, extras[key]))
    out.sort(key=lambda v: v.path)
    return out


# ---------------------------------------------------------------------------
# Per-artifact builders
# ---------------------------------------------------------------------------


def _envelope(
    *,
    artifact_kind: ArtifactKind,
    artifact_id: str,
    parent_experiment_id: str,
    parent_factor_model_id: str | None,
    registry_hashes: dict[str, str],
    surface_map_template: dict[str, Any],
) -> ArtifactEnvelope:
    return ArtifactEnvelope(
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        schema_version=ARTIFACT_SCHEMA_VERSION,
        compiler_version=COMPILER_VERSION,
        parent_experiment_id=parent_experiment_id,
        parent_factor_model_id=parent_factor_model_id,
        registry_hashes=dict(registry_hashes),
        surface_map=dict(surface_map_template),
    )


def _canonicalize_validation(
    validation: ValidationCriteria, baseline_entry_id: str
) -> ValidationCriteria:
    """Return a canonicalized copy of ``validation``.

    Two transforms:

    * Per §2.5, fill ``ComparisonRule.baseline_entry_id`` at emission so
      both the embedded :class:`ValidationCriteria` on :class:`ExperimentPlan`
      and the standalone :class:`ValidationConfig` artifact carry the same
      filled value (artifact-mode re-validation succeeds).
    * Per §8.2, sort the user-declared ``optional_telemetry`` list by
      ``metric_ref_to_runtime_key`` so input order doesn't perturb the
      ``ExperimentPlan`` hash.
    """
    update: dict[str, Any] = {}
    user_rule = validation.comparison
    if user_rule.rule == "compare_to_baseline":
        filled_payload: dict[str, Any] = user_rule.model_dump()
        filled_payload["baseline_entry_id"] = baseline_entry_id
        update["comparison"] = ComparisonRule.model_validate(filled_payload)
    if validation.optional_telemetry is not None:
        update["optional_telemetry"] = sorted(
            validation.optional_telemetry, key=metric_ref_to_runtime_key
        )
    return validation.model_copy(update=update) if update else validation


def _build_experiment_plan(
    experiment: VerifiedExperimentDefinition,
    *,
    parent_factor_model_id: str | None,
    registry_hashes: dict[str, str],
) -> ExperimentPlan:
    sorted_entries = sorted(experiment.entries, key=lambda e: e.id)
    baseline_entry = next(e for e in experiment.entries if e.is_baseline)
    filled_validation = _canonicalize_validation(experiment.validation, baseline_entry.id)
    body = ExperimentPlanBody(
        hypothesis=experiment.hypothesis,
        factor_model_id=experiment.factor_model_id,
        factors=list(experiment.factors),
        controlled_conditions=experiment.controlled_conditions,
        entries=sorted_entries,
        validation=filled_validation,
    )
    plan = ExperimentPlan(
        envelope=_envelope(
            artifact_kind="experiment_plan",
            artifact_id=f"{experiment.id}__plan",
            parent_experiment_id=experiment.id,
            parent_factor_model_id=parent_factor_model_id,
            registry_hashes=registry_hashes,
            surface_map_template=EXPERIMENT_PLAN_SURFACE_MAP,
        ),
        body=body,
    )
    return freeze_with_hash(plan)


def _build_harness_contract(
    experiment: VerifiedExperimentDefinition,
    optional_telemetry: list[MetricRef],
    *,
    parent_factor_model_id: str | None,
    parent_experiment_hash: str,
    registry_hashes: dict[str, str],
) -> HarnessContract:
    fixed_variables = _flatten_controlled_conditions(experiment.controlled_conditions)
    required_metrics = sorted([experiment.validation.metric], key=metric_ref_to_runtime_key)
    no_go_zones = sorted(_DEFAULT_NO_GO_ZONES)
    body = HarnessContractBody(
        parent_experiment_hash=parent_experiment_hash,
        factor=experiment.factors[0],
        seeds=list(experiment.controlled_conditions.seeds),
        fixed_variables=fixed_variables,
        required_metrics=required_metrics,
        optional_telemetry=optional_telemetry,
        no_go_zones=no_go_zones,
        # Lift the experiment's compute hints onto the contract so the
        # run-dispatcher reads from the locked artifact, not from the
        # mutable RunRecord row (round-3 friction (C)).
        compute=experiment.controlled_conditions.compute.model_copy(),
    )
    contract = HarnessContract(
        envelope=_envelope(
            artifact_kind="harness_contract",
            artifact_id=f"{experiment.id}__harness",
            parent_experiment_id=experiment.id,
            parent_factor_model_id=parent_factor_model_id,
            registry_hashes=registry_hashes,
            surface_map_template=HARNESS_CONTRACT_SURFACE_MAP,
        ),
        body=body,
    )
    return freeze_with_hash(contract)


def _build_technique_contract(
    entry: Entry,
    experiment: VerifiedExperimentDefinition,
    registries: Registries,
    *,
    parent_factor_model_id: str | None,
    parent_experiment_hash: str,
    parent_harness_contract_hash: str,
    registry_hashes: dict[str, str],
) -> TechniqueContract:
    technique_id = entry.level.technique_id
    standard = False
    fidelity_anchor = None
    if technique_id is not None:
        ref = registries.technique_registry.get(technique_id)
        if ref is None:
            raise KeyError(
                f"technique {technique_id!r} referenced by entry {entry.id!r} "
                f"is not in the technique registry — verification should have "
                f"caught this; registry consistency violation"
            )
        standard = ref.standard
        fidelity_anchor = ref.fidelity_anchor
    body = TechniqueContractBody(
        entry_id=entry.id,
        parent_experiment_id=experiment.id,
        parent_experiment_hash=parent_experiment_hash,
        parent_harness_contract_hash=parent_harness_contract_hash,
        technique_id=technique_id,
        technique_params=entry.level.technique_params,
        level_value=entry.level.value,
        is_baseline=entry.is_baseline,
        fidelity_anchor=fidelity_anchor,
        standard=standard,
    )
    contract = TechniqueContract(
        envelope=_envelope(
            artifact_kind="technique_contract",
            artifact_id=f"{experiment.id}__technique__{entry.id}",
            parent_experiment_id=experiment.id,
            parent_factor_model_id=parent_factor_model_id,
            registry_hashes=registry_hashes,
            surface_map_template=TECHNIQUE_CONTRACT_SURFACE_MAP,
        ),
        body=body,
    )
    return freeze_with_hash(contract)


def _build_validation_config(
    experiment: VerifiedExperimentDefinition,
    *,
    parent_factor_model_id: str | None,
    parent_experiment_hash: str,
    registry_hashes: dict[str, str],
    trend_check: TrendCheck | None,
) -> ValidationConfig:
    validation: ValidationCriteria = experiment.validation
    baseline_entry = next(e for e in experiment.entries if e.is_baseline)
    filled_validation = _canonicalize_validation(validation, baseline_entry.id)
    body = ValidationConfigBody(
        parent_experiment_hash=parent_experiment_hash,
        metric=filled_validation.metric,
        direction=filled_validation.direction,
        aggregation=filled_validation.aggregation,
        comparison=filled_validation.comparison,
        seed_count_required=filled_validation.seed_count_required,
        rationale=filled_validation.rationale,
        trend_check=trend_check if trend_check is not None else validation.trend_check,
    )
    cfg = ValidationConfig(
        envelope=_envelope(
            artifact_kind="validation_config",
            artifact_id=f"{experiment.id}__validation",
            parent_experiment_id=experiment.id,
            parent_factor_model_id=parent_factor_model_id,
            registry_hashes=registry_hashes,
            surface_map_template=VALIDATION_CONFIG_SURFACE_MAP,
        ),
        body=body,
    )
    return freeze_with_hash(cfg)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _compute_registry_hashes(registries: Registries) -> dict[str, str]:
    """Hash each registry to a sha256 hex digest stamped onto every envelope."""
    metric_payload = registries.metric_registry.model_dump(mode="json")
    technique_payload = {
        tech_id: ref.model_dump(mode="json")
        for tech_id, ref in registries.technique_registry.items()
    }
    factor_type_names = list(registries.factor_type_plugins.keys())
    return {
        "factor_types": hash_string_set(factor_type_names),
        "metrics": hash_registry_dict(metric_payload),
        "techniques": hash_registry_dict(technique_payload),
    }


def emit_artifacts(
    experiment: VerifiedExperimentDefinition,
    registries: Registries,
    *,
    parent_factor_model_id: str | None = None,
) -> tuple[ContractArtifactSet, ValidationReport]:
    """Sequential emission of one CG's contract artifacts (§8.1).

    Returns the immutable :class:`ContractArtifactSet` plus a
    :class:`ValidationReport` carrying any warnings produced by emit-time
    engine behaviors (§5.5 row 10 cost-tag dedup; future engine warnings).

    Determinism: same input + same registries + same compiler version always
    produces byte-equal outputs (§8.2). Emission has no I/O, no LLM, no
    timestamps.
    """
    registry_hashes = _compute_registry_hashes(registries)

    plan = _build_experiment_plan(
        experiment,
        parent_factor_model_id=parent_factor_model_id,
        registry_hashes=registry_hashes,
    )

    auto_included = _cost_auto_include(registries)
    user_declared = list(experiment.validation.optional_telemetry or [])
    optional_telemetry, redundant_keys = _canonical_metric_union(auto_included, user_declared)

    warnings: list[ValidationError] = []
    for key in redundant_keys:
        warnings.append(
            ValidationError(
                code="metric.optional_telemetry_redundant_user_declaration",
                severity="warning",
                message=(
                    f"User-declared optional_telemetry entry {key!r} is "
                    f"already auto-included from the registry's cost-tagged "
                    f"set; the duplicate was deduped at emission time."
                ),
                location=f"validation.optional_telemetry[{key!r}]",
            )
        )

    harness = _build_harness_contract(
        experiment,
        optional_telemetry,
        parent_factor_model_id=parent_factor_model_id,
        parent_experiment_hash=plan.envelope.content_hash,
        registry_hashes=registry_hashes,
    )

    technique_contracts: list[TechniqueContract] = []
    for entry in sorted(experiment.entries, key=lambda e: e.id):
        technique_contracts.append(
            _build_technique_contract(
                entry,
                experiment,
                registries,
                parent_factor_model_id=parent_factor_model_id,
                parent_experiment_hash=plan.envelope.content_hash,
                parent_harness_contract_hash=harness.envelope.content_hash,
                registry_hashes=registry_hashes,
            )
        )

    auto_trend_check: TrendCheck | None = None
    if experiment.validation.trend_check is None:
        auto_trend_check = _maybe_default_trend_check(experiment)
    validation_config = _build_validation_config(
        experiment,
        parent_factor_model_id=parent_factor_model_id,
        parent_experiment_hash=plan.envelope.content_hash,
        registry_hashes=registry_hashes,
        trend_check=auto_trend_check,
    )

    artifact_set = ContractArtifactSet(
        experiment_plan=plan,
        harness_contract=harness,
        technique_contracts=technique_contracts,
        validation_config=validation_config,
    )
    report = ValidationReport(
        success=True,
        errors=[],
        warnings=warnings,
        advisories=[],
    )
    return artifact_set, report


def emit_factor_model_artifacts(
    document: FactorModelDocument,
    verified: list[VerifiedExperimentDefinition],
    registries: Registries,
) -> tuple[dict[str, ContractArtifactSet], ValidationReport]:
    """Emit one :class:`ContractArtifactSet` per child experiment (§8.4).

    The :class:`FactorModelDocument` itself emits no artifact (per DEC-031
    sub-decision #5). Each comprising CG's artifacts carry
    ``parent_factor_model_id`` in the envelope.

    The returned report unions per-CG warnings.
    """
    if len(verified) != len(document.experiments):
        raise ValueError(
            f"verified list length {len(verified)} does not match document "
            f"experiments length {len(document.experiments)}"
        )
    factor_model_id = document.factor_model.id
    sets: dict[str, ContractArtifactSet] = {}
    warnings: list[ValidationError] = []
    advisories: list[ValidationError] = []
    for exp in sorted(verified, key=lambda e: e.id):
        artifact_set, report = emit_artifacts(
            exp,
            registries,
            parent_factor_model_id=factor_model_id,
        )
        sets[exp.id] = artifact_set
        warnings.extend(report.warnings)
        advisories.extend(report.advisories)
    combined = ValidationReport(success=True, errors=[], warnings=warnings, advisories=advisories)
    return sets, combined
