"""Tests for the top-level DSL discriminated union (``ExperimentDocument`` /
``FactorModelDocument``) and its programmatic construction surface.

Covers §2.2 of ``02-dsl-and-contracts.md``: discriminator dispatch, default
discriminator values for programmatic construction, ``extra="forbid"`` on
both wrappers, and round-trip through ``model_dump``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from smai_core import (
    AggregationRule,
    AtomicMetricRef,
    ComparisonRule,
    ControlledConditions,
    DslDocumentAdapter,
    Entry,
    ExperimentDefinition,
    ExperimentDocument,
    Factor,
    FactorModel,
    FactorModelDocument,
    Level,
    ValidationCriteria,
)


def _experiment(*, factor_model_id: str | None = None) -> ExperimentDefinition:
    return ExperimentDefinition(
        id="cg_001",
        hypothesis="Cutout improves accuracy on CIFAR-10.",
        factor_model_id=factor_model_id,
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
                    name="cutout",
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


def _factor_model() -> FactorModel:
    return FactorModel(
        id="fm_001",
        research_question="Which augmentations help most?",
        comparison_group_ids=["cg_001"],
    )


# --- public surface re-exports -------------------------------------------------


def test_public_surface_reexports_dsl_types() -> None:
    """``smai_core`` re-exports the DSL surface so consumers can ``from smai_core
    import DslDocumentAdapter`` etc."""
    import smai_core

    for name in (
        "DslDocument",
        "DslDocumentAdapter",
        "ExperimentDocument",
        "FactorModelDocument",
        "load_dsl_document_from_json",
    ):
        assert hasattr(smai_core, name), f"smai_core is missing re-export: {name}"


# --- programmatic construction --------------------------------------------------


def test_experiment_document_default_kind() -> None:
    """``kind`` defaults to ``"experiment"`` for ergonomic programmatic construction."""
    doc = ExperimentDocument(experiment=_experiment())
    assert doc.kind == "experiment"


def test_factor_model_document_default_kind() -> None:
    doc = FactorModelDocument(factor_model=_factor_model(), experiments=[_experiment()])
    assert doc.kind == "factor_model"


def test_experiment_document_round_trip() -> None:
    doc = ExperimentDocument(experiment=_experiment())
    payload = doc.model_dump(mode="json")
    parsed = DslDocumentAdapter.validate_python(payload)
    assert isinstance(parsed, ExperimentDocument)
    assert parsed == doc


def test_factor_model_document_round_trip() -> None:
    doc = FactorModelDocument(
        factor_model=_factor_model(),
        experiments=[_experiment(factor_model_id="fm_001")],
    )
    payload = doc.model_dump(mode="json")
    parsed = DslDocumentAdapter.validate_python(payload)
    assert isinstance(parsed, FactorModelDocument)
    assert parsed == doc


def test_experiment_document_forbids_extra_top_level_fields() -> None:
    with pytest.raises(ValidationError):
        ExperimentDocument.model_validate(
            {
                "kind": "experiment",
                "experiment": _experiment().model_dump(mode="json"),
                "extra_field": "not allowed",
            }
        )


def test_factor_model_document_forbids_extra_top_level_fields() -> None:
    with pytest.raises(ValidationError):
        FactorModelDocument.model_validate(
            {
                "kind": "factor_model",
                "factor_model": _factor_model().model_dump(mode="json"),
                "experiments": [_experiment(factor_model_id="fm_001").model_dump(mode="json")],
                "extra_field": "nope",
            }
        )


# --- discriminator dispatch -----------------------------------------------------


def test_discriminator_dispatches_experiment() -> None:
    payload = {"kind": "experiment", "experiment": _experiment().model_dump(mode="json")}
    doc = DslDocumentAdapter.validate_python(payload)
    assert isinstance(doc, ExperimentDocument)


def test_discriminator_dispatches_factor_model() -> None:
    payload = {
        "kind": "factor_model",
        "factor_model": _factor_model().model_dump(mode="json"),
        "experiments": [_experiment(factor_model_id="fm_001").model_dump(mode="json")],
    }
    doc = DslDocumentAdapter.validate_python(payload)
    assert isinstance(doc, FactorModelDocument)


def test_discriminator_unknown_kind_rejected() -> None:
    payload = {"kind": "schema", "experiment": _experiment().model_dump(mode="json")}
    with pytest.raises(ValidationError):
        DslDocumentAdapter.validate_python(payload)


def test_discriminator_missing_kind_rejected() -> None:
    """Pydantic's discriminated union requires the discriminator at validation time."""
    payload = {"experiment": _experiment().model_dump(mode="json")}
    with pytest.raises(ValidationError):
        DslDocumentAdapter.validate_python(payload)


def test_factor_model_document_requires_experiments() -> None:
    with pytest.raises(ValidationError):
        DslDocumentAdapter.validate_python(
            {
                "kind": "factor_model",
                "factor_model": _factor_model().model_dump(mode="json"),
            }
        )


def test_factor_model_document_admits_empty_experiments_list() -> None:
    """An empty ``experiments`` list is structurally valid; cross-CG consistency is
    a Pass-2 verifier concern (Task 1.5), not a Pass-1 schema concern."""
    doc = DslDocumentAdapter.validate_python(
        {
            "kind": "factor_model",
            "factor_model": _factor_model().model_dump(mode="json"),
            "experiments": [],
        }
    )
    assert isinstance(doc, FactorModelDocument)
    assert doc.experiments == []
