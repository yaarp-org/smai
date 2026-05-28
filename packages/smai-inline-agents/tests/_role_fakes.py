"""Test fixture factories for the single-call role wrappers.

Builds minimal but valid :class:`HarnessContract`,
:class:`TechniqueContract`, and :class:`EvaluationResult` instances —
the inputs the role-shaped wrappers (:func:`run_code_review`,
:func:`run_contextual_evaluation`) consume. The factories accept
overrides so individual tests can customize the shape they care about
without restating every field.

Module name uses the ``_role_*`` prefix so it doesn't collide with the
existing ``_agent_fakes`` / ``_agent_helpers`` modules in the same
``tests/`` directory (the conftest puts the directory on ``sys.path``;
collisions across modules in the same dir would shadow each other).
"""

from __future__ import annotations

from smai_core.artifacts import (
    ArtifactEnvelope,
    HarnessContract,
    HarnessContractBody,
    TechniqueContract,
    TechniqueContractBody,
)
from smai_core.entities.factor import Factor
from smai_core.entities.metric import AtomicMetricRef
from smai_core.evaluation import (
    EvaluationResult,
    PerEntryAggregate,
    ReproducibilityMetadata,
    TreatmentOutcome,
    Verdict,
    VerdictContext,
)


def _envelope(kind: str = "harness_contract") -> ArtifactEnvelope:
    return ArtifactEnvelope(
        artifact_kind=kind,  # type: ignore[arg-type]
        artifact_id="exp__demo",
        schema_version=1,
        compiler_version="0.1.0",
        parent_experiment_id="exp__demo",
        parent_factor_model_id=None,
        registry_hashes={
            "factor_types": "f",
            "metrics": "m",
            "techniques": "t",
        },
        surface_map={},
    )


def make_harness_contract(
    *,
    factor_type: str = "additive",
    factor_name: str = "augmentation",
    factor_description: str = "image augmentation",
) -> HarnessContract:
    body = HarnessContractBody(
        parent_experiment_hash="parent-hash",
        factor=Factor(
            name=factor_name,
            type=factor_type,  # type: ignore[arg-type]
            description=factor_description,
        ),
        seeds=[1, 2, 3],
        fixed_variables=[],
        required_metrics=[AtomicMetricRef(ref="accuracy")],
        optional_telemetry=[],
        no_go_zones=["experiment.py"],
    )
    return HarnessContract(envelope=_envelope("harness_contract"), body=body)


def make_technique_contract(
    *,
    entry_id: str,
    technique_id: str | None,
    is_baseline: bool,
) -> TechniqueContract:
    # Additive baselines (no technique_id) carry the ``no_op_baseline``
    # body discriminator; technique-backed fakes use ``standard``.
    context_kind = "no_op_baseline" if technique_id is None else "standard"
    body = TechniqueContractBody(
        entry_id=entry_id,
        parent_experiment_id="exp__demo",
        parent_experiment_hash="parent-hash",
        parent_harness_contract_hash="harness-hash",
        technique_id=technique_id,
        technique_params=None,
        level_value=None,
        is_baseline=is_baseline,
        fidelity_anchor=None,
        standard=False,
        context_kind=context_kind,
    )
    return TechniqueContract(envelope=_envelope("technique_contract"), body=body)


def _stub_repro() -> ReproducibilityMetadata:
    return ReproducibilityMetadata(
        validation_config_hash="vc-hash",
        harness_contract_hash="hc-hash",
        experiment_plan_hash="ep-hash",
        technique_contract_hashes={},
        schema_version=1,
        compiler_version="0.1.0",
        evaluator_version="0.1.0",
        registry_hashes={},
    )


def make_evaluation_result(
    *,
    result: str = "pass",
    treatment_entry_ids: tuple[str, ...] = ("t1",),
    baseline_entry_id: str = "b",
) -> EvaluationResult:
    """A pass-shaped :class:`EvaluationResult` good enough for prompt rendering."""
    treatment_outcomes = [
        TreatmentOutcome(
            entry_id=tid,
            aggregated_metric_value=0.85,
            delta_from_baseline=0.05,
            threshold=0.01,
            passed_threshold=True,
            tolerance_margin=0.04,
        )
        for tid in treatment_entry_ids
    ]
    per_entry = [
        PerEntryAggregate(
            entry_id=baseline_entry_id,
            is_baseline=True,
            aggregated_metric_value=0.80,
            seeds_completed=3,
            seeds_required=3,
            completed_fraction=1.0,
        ),
        *[
            PerEntryAggregate(
                entry_id=tid,
                is_baseline=False,
                aggregated_metric_value=0.85,
                seeds_completed=3,
                seeds_required=3,
                completed_fraction=1.0,
            )
            for tid in treatment_entry_ids
        ],
    ]
    verdict = Verdict(
        result=result,  # type: ignore[arg-type]
        treatment_outcomes=treatment_outcomes,
        per_entry_aggregate=per_entry,
        validation_config_hash="vc-hash",
        raw_metrics_hash="rm-hash",
        compiler_version="0.1.0",
        evaluator_version="0.1.0",
    )
    context = VerdictContext(
        per_seed_values={},
        statistical_summary={},
        delta_summaries=[],
        cost_telemetry=None,
        anomalies=[],
        trend_observation=None,
        reproducibility=_stub_repro(),
        failure_reason=None,
    )
    return EvaluationResult(verdict=verdict, verdict_context=context)


__all__ = [
    "make_evaluation_result",
    "make_harness_contract",
    "make_technique_contract",
]
