"""Tests for context-keyed DSL/artifact gating propagation.

The conditional-required-field gating mechanism (per §2.5 of
``02-dsl-and-contracts.md``) lives on inner models — currently only
``ComparisonRule.baseline_entry_id``. These tests verify that
``DslDocumentAdapter.validate_python(payload, context={"smai_mode": "dsl"})``
correctly propagates the context to the nested ``ComparisonRule`` so DSL-mode
gates fire, and that the artifact-mode (default) gate fires when the document
is re-validated post-emission.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from smai_core import DslDocumentAdapter, ExperimentDocument


def _experiment_payload(*, baseline_entry_id: str | None = None) -> dict[str, object]:
    """Minimal experiment-document payload with a ``compare_to_baseline`` rule.

    ``baseline_entry_id`` defaults to ``None`` — the DSL form. Tests can pass a
    string to simulate either an artifact form or an illegal user-supplied DSL
    form.
    """
    comparison: dict[str, object] = {"rule": "compare_to_baseline", "threshold": 0.01}
    if baseline_entry_id is not None:
        comparison["baseline_entry_id"] = baseline_entry_id
    return {
        "kind": "experiment",
        "experiment": {
            "id": "cg_x",
            "hypothesis": "h",
            "factor_model_id": None,
            "factors": [{"name": "aug", "type": "additive", "description": "d"}],
            "controlled_conditions": {
                "dataset": {"name": "cifar10", "split": "s", "version": "v"},
                "optimization": {"optimizer": "sgd"},
                "seeds": [1, 2, 3],
            },
            "entries": [
                {
                    "id": "entry_baseline",
                    "is_baseline": True,
                    "level": {"factor": "aug", "name": "absent"},
                },
                {
                    "id": "entry_treatment",
                    "is_baseline": False,
                    "level": {
                        "factor": "aug",
                        "name": "cutout",
                        "technique_id": "tech_cutout",
                    },
                },
            ],
            "validation": {
                "metric": {"kind": "atomic", "ref": "accuracy"},
                "direction": "higher_is_better",
                "aggregation": {"method": "mean"},
                "comparison": comparison,
                "seed_count_required": 3,
            },
        },
    }


def test_dsl_mode_baseline_no_id_succeeds() -> None:
    payload = _experiment_payload()  # baseline_entry_id absent
    doc = DslDocumentAdapter.validate_python(payload, context={"smai_mode": "dsl"})
    assert isinstance(doc, ExperimentDocument)
    assert doc.experiment.validation.comparison.baseline_entry_id is None


def test_dsl_mode_baseline_with_user_supplied_id_rejected() -> None:
    payload = _experiment_payload(baseline_entry_id="entry_baseline")
    with pytest.raises(ValidationError) as exc:
        DslDocumentAdapter.validate_python(payload, context={"smai_mode": "dsl"})
    assert "comparison.baseline_entry_id_user_supplied" in str(exc.value)


def test_artifact_mode_baseline_no_id_rejected() -> None:
    payload = _experiment_payload()  # baseline_entry_id absent
    with pytest.raises(ValidationError) as exc:
        DslDocumentAdapter.validate_python(payload, context={"smai_mode": "artifact"})
    assert "comparison.baseline_entry_id_missing" in str(exc.value)


def test_artifact_mode_baseline_with_id_succeeds() -> None:
    payload = _experiment_payload(baseline_entry_id="entry_baseline")
    doc = DslDocumentAdapter.validate_python(payload, context={"smai_mode": "artifact"})
    assert isinstance(doc, ExperimentDocument)
    assert doc.experiment.validation.comparison.baseline_entry_id == "entry_baseline"


def test_default_context_is_artifact_mode() -> None:
    """No explicit context → ComparisonRule's default ``artifact`` gate fires."""
    payload = _experiment_payload()
    with pytest.raises(ValidationError) as exc:
        DslDocumentAdapter.validate_python(payload)
    assert "comparison.baseline_entry_id_missing" in str(exc.value)


def test_compare_to_target_requires_target_value_in_dsl_mode() -> None:
    payload = _experiment_payload()
    payload["experiment"]["validation"]["comparison"] = {  # type: ignore[index]
        "rule": "compare_to_target",
        "threshold": 0.0,
    }
    with pytest.raises(ValidationError) as exc:
        DslDocumentAdapter.validate_python(payload, context={"smai_mode": "dsl"})
    assert "comparison.target_value_missing" in str(exc.value)


def test_compare_to_target_with_value_succeeds_in_dsl_mode() -> None:
    payload = _experiment_payload()
    payload["experiment"]["validation"]["comparison"] = {  # type: ignore[index]
        "rule": "compare_to_target",
        "threshold": 0.0,
        "target_value": 0.95,
    }
    doc = DslDocumentAdapter.validate_python(payload, context={"smai_mode": "dsl"})
    assert isinstance(doc, ExperimentDocument)
    assert doc.experiment.validation.comparison.target_value == 0.95


def test_factor_model_document_propagates_context_to_comprising_cgs() -> None:
    """The DSL gate fires on every comprising CG's ``ComparisonRule`` inside a
    ``FactorModelDocument`` — context propagates uniformly."""
    cg = _experiment_payload()["experiment"]
    payload = {
        "kind": "factor_model",
        "factor_model": {
            "id": "fm_x",
            "research_question": "q",
            "comparison_group_ids": ["cg_x"],
        },
        "experiments": [cg],
    }
    # Default (artifact) mode rejects: the comprising CG's compare_to_baseline
    # rule has no baseline_entry_id.
    with pytest.raises(ValidationError):
        DslDocumentAdapter.validate_python(payload)
    # DSL mode accepts: the user-facing form is valid.
    doc = DslDocumentAdapter.validate_python(payload, context={"smai_mode": "dsl"})
    assert doc.experiments[0].validation.comparison.baseline_entry_id is None  # type: ignore[union-attr]


def test_no_context_dict_treated_as_default_mode() -> None:
    """Passing an empty context dict (no ``smai_mode`` key) falls back to the
    default ``artifact`` mode — the validator treats absence as default."""
    payload = _experiment_payload()
    with pytest.raises(ValidationError) as exc:
        DslDocumentAdapter.validate_python(payload, context={})
    assert "comparison.baseline_entry_id_missing" in str(exc.value)
