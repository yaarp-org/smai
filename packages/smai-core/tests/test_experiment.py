"""Tests for ``ExperimentDefinition``, ``VerifiedExperimentDefinition``, ``FactorModel``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from smai_core import (
    AggregationRule,
    AtomicMetricRef,
    ComparisonRule,
    ControlledConditions,
    Entry,
    ExperimentDefinition,
    Factor,
    FactorModel,
    Level,
    ValidationCriteria,
    VerifiedExperimentDefinition,
)


def _experiment() -> ExperimentDefinition:
    return ExperimentDefinition(
        id="cg_001",
        hypothesis="Cutout improves accuracy on CIFAR-10.",
        factors=[Factor(name="augmentation", type="additive", description="cutout on/off")],
        controlled_conditions=ControlledConditions(
            dataset={"name": "cifar10", "split": "train", "version": "v1"},
            optimization={"optimizer": "sgd", "lr": 0.1},
            seeds=[1, 2, 3],
        ),
        entries=[
            Entry(
                id="entry_baseline",
                is_baseline=True,
                level=Level(factor="augmentation", name="absent"),
            ),
            Entry(
                id="entry_cutout",
                is_baseline=False,
                level=Level(
                    factor="augmentation",
                    name="cutout_16",
                    technique_id="tech_cutout",
                    technique_params={"patch_size": 16},
                ),
            ),
        ],
        validation=ValidationCriteria(
            metric=AtomicMetricRef(ref="accuracy"),
            direction="higher_is_better",
            aggregation=AggregationRule(method="mean"),
            comparison=ComparisonRule(
                rule="compare_to_baseline",
                threshold=0.01,
                baseline_entry_id="entry_baseline",
            ),
            seed_count_required=3,
        ),
    )


def test_experiment_validates() -> None:
    exp = _experiment()
    assert exp.id == "cg_001"
    assert len(exp.factors) == 1


def test_experiment_factors_max_length_one_rejects_two() -> None:
    with pytest.raises(ValidationError):
        ExperimentDefinition.model_validate(
            {
                "id": "cg_002",
                "hypothesis": "two factors not allowed",
                "factors": [
                    {"name": "a", "type": "additive", "description": "x"},
                    {"name": "b", "type": "additive", "description": "y"},
                ],
                "controlled_conditions": {
                    "dataset": {"name": "x", "split": "y", "version": "z"},
                    "optimization": {},
                    "seeds": [1],
                },
                "entries": [],
                "validation": {
                    "metric": {"kind": "atomic", "ref": "accuracy"},
                    "direction": "higher_is_better",
                    "aggregation": {"method": "mean"},
                    "comparison": {
                        "rule": "compare_to_target",
                        "threshold": 0.0,
                        "target_value": 0.9,
                    },
                    "seed_count_required": 1,
                },
            }
        )


def test_experiment_factors_min_length_one_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        ExperimentDefinition.model_validate(
            {
                "id": "cg_003",
                "hypothesis": "no factors",
                "factors": [],
                "controlled_conditions": {
                    "dataset": {"name": "x", "split": "y", "version": "z"},
                    "optimization": {},
                    "seeds": [1],
                },
                "entries": [],
                "validation": {
                    "metric": {"kind": "atomic", "ref": "accuracy"},
                    "direction": "higher_is_better",
                    "aggregation": {"method": "mean"},
                    "comparison": {
                        "rule": "compare_to_target",
                        "threshold": 0.0,
                        "target_value": 0.9,
                    },
                    "seed_count_required": 1,
                },
            }
        )


def test_experiment_round_trip() -> None:
    exp = _experiment()
    payload = exp.model_dump(mode="json")
    assert ExperimentDefinition.model_validate(payload) == exp


def test_verified_subclass_promotes_from_dump() -> None:
    exp = _experiment()
    promoted = VerifiedExperimentDefinition.model_validate(exp.model_dump())
    assert isinstance(promoted, VerifiedExperimentDefinition)
    assert isinstance(promoted, ExperimentDefinition)
    assert promoted.id == exp.id


def test_factor_model_validates() -> None:
    fm = FactorModel(
        id="fm_001",
        research_question="Which augmentations help most on CIFAR-10?",
        comparison_group_ids=["cg_001", "cg_002"],
    )
    assert fm.shared_conditions is None
    assert fm.comparison_group_ids == ["cg_001", "cg_002"]


def test_factor_model_round_trip_with_shared_conditions() -> None:
    fm = FactorModel(
        id="fm_002",
        research_question="Q",
        shared_conditions=ControlledConditions(
            dataset={"name": "cifar10", "split": "train", "version": "v1"},
            optimization={"optimizer": "sgd"},
            seeds=[1, 2, 3],
        ),
        comparison_group_ids=["cg_a"],
    )
    payload = fm.model_dump(mode="json")
    assert FactorModel.model_validate(payload) == fm
