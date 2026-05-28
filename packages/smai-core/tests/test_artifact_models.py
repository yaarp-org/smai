"""Schema and JSON-round-trip tests for the four contract artifact models.

Per ``02-dsl-and-contracts.md`` §7. Each artifact validates a hand-crafted
instance, round-trips through JSON, and rejects malformed input.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError
from smai_core import (
    AggregationRule,
    ArtifactEnvelope,
    AtomicMetricRef,
    ComparisonRule,
    ContractArtifactSet,
    ControlledConditions,
    Entry,
    ExperimentPlan,
    ExperimentPlanBody,
    Factor,
    FixedVariable,
    HarnessContract,
    HarnessContractBody,
    Level,
    NumericValue,
    PaperFidelityAnchor,
    TechniqueContract,
    TechniqueContractBody,
    ValidationConfig,
    ValidationConfigBody,
    ValidationCriteria,
)


def _envelope(kind: str = "experiment_plan") -> ArtifactEnvelope:
    return ArtifactEnvelope(
        artifact_kind=kind,  # type: ignore[arg-type]
        artifact_id="exp__plan",
        schema_version=1,
        compiler_version="0.1.0",
        parent_experiment_id="exp",
        parent_factor_model_id=None,
        registry_hashes={"factor_types": "x", "metrics": "y", "techniques": "z"},
        surface_map={"body.hypothesis": "recorded"},
    )


def _plan_body() -> ExperimentPlanBody:
    return ExperimentPlanBody(
        hypothesis="x",
        factor_model_id=None,
        factors=[Factor(name="architecture", type="substitutive", description="d")],
        controlled_conditions=ControlledConditions(
            dataset={"name": "cifar10"},
            optimization={"optimizer": "adamw", "lr": 0.001},
            seeds=[1, 2, 3],
        ),
        entries=[
            Entry(
                id="b",
                is_baseline=True,
                level=Level(factor="architecture", name="VGG", technique_id="t_vgg"),
            ),
            Entry(
                id="t",
                is_baseline=False,
                level=Level(factor="architecture", name="ResNet", technique_id="t_resnet"),
            ),
        ],
        validation=ValidationCriteria(
            metric=AtomicMetricRef(ref="accuracy"),
            direction="higher_is_better",
            aggregation=AggregationRule(method="mean"),
            comparison=ComparisonRule(
                rule="compare_to_baseline", threshold=0.01, baseline_entry_id="b"
            ),
            seed_count_required=3,
        ),
    )


def test_envelope_minimal_construction() -> None:
    env = _envelope()
    assert env.artifact_kind == "experiment_plan"
    assert env.content_hash == ""
    assert env.parent_factor_model_id is None


def test_envelope_extra_forbid() -> None:
    with pytest.raises(PydanticValidationError):
        ArtifactEnvelope.model_validate(
            {
                "artifact_kind": "experiment_plan",
                "artifact_id": "x",
                "schema_version": 1,
                "compiler_version": "0.1.0",
                "parent_experiment_id": "x",
                "registry_hashes": {},
                "surface_map": {},
                "rogue_field": True,
            }
        )


def test_envelope_artifact_kind_closed_set() -> None:
    with pytest.raises(PydanticValidationError):
        ArtifactEnvelope.model_validate(
            {
                "artifact_kind": "not_a_kind",
                "artifact_id": "x",
                "schema_version": 1,
                "compiler_version": "0.1.0",
                "parent_experiment_id": "x",
                "registry_hashes": {},
                "surface_map": {},
            }
        )


def test_experiment_plan_round_trip() -> None:
    plan = ExperimentPlan(envelope=_envelope("experiment_plan"), body=_plan_body())
    raw = plan.model_dump(mode="json")
    re = ExperimentPlan.model_validate(raw)
    assert re == plan


def test_harness_contract_round_trip() -> None:
    body = HarnessContractBody(
        parent_experiment_hash="abc",
        factor=Factor(name="architecture", type="substitutive", description="d"),
        seeds=[1, 2, 3],
        fixed_variables=[
            FixedVariable(path="dataset.name", value="cifar10", type_hint="str"),
        ],
        required_metrics=[AtomicMetricRef(ref="accuracy")],
        optional_telemetry=[AtomicMetricRef(ref="params")],
        no_go_zones=["experiment.py"],
    )
    contract = HarnessContract(envelope=_envelope("harness_contract"), body=body)
    raw = contract.model_dump(mode="json")
    re = HarnessContract.model_validate(raw)
    assert re == contract


def test_technique_contract_round_trip() -> None:
    body = TechniqueContractBody(
        entry_id="t",
        parent_experiment_id="exp",
        parent_experiment_hash="aa",
        parent_harness_contract_hash="bb",
        technique_id="t_resnet",
        technique_params=None,
        level_value=NumericValue(value=0.5, kind="continuous"),
        is_baseline=False,
        fidelity_anchor=PaperFidelityAnchor(doi="10.x", arxiv_id="1.2"),
        standard=False,
        context_kind="paper_extract",
    )
    contract = TechniqueContract(envelope=_envelope("technique_contract"), body=body)
    raw = contract.model_dump(mode="json")
    re = TechniqueContract.model_validate(raw)
    assert re == contract


def test_technique_contract_admits_null_technique() -> None:
    body = TechniqueContractBody(
        entry_id="b",
        parent_experiment_id="exp",
        parent_experiment_hash="aa",
        parent_harness_contract_hash="bb",
        technique_id=None,
        technique_params=None,
        level_value=None,
        is_baseline=True,
        fidelity_anchor=None,
        standard=False,
        context_kind="no_op_baseline",
    )
    contract = TechniqueContract(envelope=_envelope("technique_contract"), body=body)
    assert contract.body.technique_id is None
    assert contract.body.fidelity_anchor is None


def test_validation_config_requires_baseline_entry_id() -> None:
    """Per §7.6: artifact-mode validation rejects unfilled baseline_entry_id."""
    with pytest.raises(PydanticValidationError):
        ValidationConfigBody.model_validate(
            {
                "parent_experiment_hash": "x",
                "metric": {"kind": "atomic", "ref": "accuracy"},
                "direction": "higher_is_better",
                "aggregation": {"method": "mean"},
                "comparison": {
                    "rule": "compare_to_baseline",
                    "threshold": 0.01,
                    # baseline_entry_id absent — artifact mode requires it
                },
                "seed_count_required": 3,
            }
        )


def test_validation_config_round_trip() -> None:
    body = ValidationConfigBody(
        parent_experiment_hash="x",
        metric=AtomicMetricRef(ref="accuracy"),
        direction="higher_is_better",
        aggregation=AggregationRule(method="mean"),
        comparison=ComparisonRule(
            rule="compare_to_baseline", threshold=0.01, baseline_entry_id="b"
        ),
        seed_count_required=3,
        rationale=None,
        trend_check=None,
    )
    cfg = ValidationConfig(envelope=_envelope("validation_config"), body=body)
    raw = cfg.model_dump(mode="json")
    re = ValidationConfig.model_validate(raw)
    assert re == cfg


def test_contract_artifact_set_round_trip() -> None:
    plan = ExperimentPlan(envelope=_envelope("experiment_plan"), body=_plan_body())
    harness = HarnessContract(
        envelope=_envelope("harness_contract"),
        body=HarnessContractBody(
            parent_experiment_hash="x",
            factor=Factor(name="architecture", type="substitutive", description="d"),
            seeds=[1],
            fixed_variables=[],
            required_metrics=[AtomicMetricRef(ref="accuracy")],
            optional_telemetry=[],
            no_go_zones=[],
        ),
    )
    cfg = ValidationConfig(
        envelope=_envelope("validation_config"),
        body=ValidationConfigBody(
            parent_experiment_hash="x",
            metric=AtomicMetricRef(ref="accuracy"),
            direction="higher_is_better",
            aggregation=AggregationRule(method="mean"),
            comparison=ComparisonRule(
                rule="compare_to_baseline", threshold=0.01, baseline_entry_id="b"
            ),
            seed_count_required=3,
        ),
    )
    artifact_set = ContractArtifactSet(
        experiment_plan=plan,
        harness_contract=harness,
        technique_contracts=[],
        validation_config=cfg,
    )
    raw = artifact_set.model_dump(mode="json")
    re = ContractArtifactSet.model_validate(raw)
    assert re == artifact_set


def test_fixed_variable_admits_primitive_and_compound() -> None:
    FixedVariable(path="optimization.lr", value=0.001, type_hint="float")
    FixedVariable(path="opt.optimizer", value="adamw", type_hint="str")
    FixedVariable(path="opt.layers", value=[1, 2, 3], type_hint="list")
    FixedVariable(path="opt.cfg", value={"a": 1}, type_hint="dict")


def test_envelope_surface_map_constrains_values() -> None:
    with pytest.raises(PydanticValidationError):
        ArtifactEnvelope.model_validate(
            {
                "artifact_kind": "experiment_plan",
                "artifact_id": "x",
                "schema_version": 1,
                "compiler_version": "0.1.0",
                "parent_experiment_id": "x",
                "registry_hashes": {},
                "surface_map": {"body.hypothesis": "not_a_surface"},
            }
        )
