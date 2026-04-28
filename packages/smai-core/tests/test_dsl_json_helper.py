"""Tests for ``load_dsl_document_from_json``.

The helper bundles ``json.loads`` + ``DslDocumentAdapter.validate_python`` with
``smai_mode="dsl"`` context. Stdlib-only — no YAML helper because pyyaml is not
on the ``smai-core`` runtime dep allowlist (DEC-029).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from smai_core import (
    DslDocumentAdapter,
    ExperimentDocument,
    FactorModelDocument,
    load_dsl_document_from_json,
)

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "experiments"


def _yaml_to_json_text(name: str) -> str:
    payload = yaml.safe_load((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    return json.dumps(payload)


def test_load_from_string() -> None:
    text = _yaml_to_json_text("resnet50_vs_vgg16_cifar10.yaml")
    doc = load_dsl_document_from_json(text)
    assert isinstance(doc, ExperimentDocument)
    assert doc.experiment.id == "arch_resnet_vs_vgg_cifar10"


def test_load_from_path(tmp_path: Path) -> None:
    text = _yaml_to_json_text("cutout_on_cifar10.yaml")
    target = tmp_path / "experiment.json"
    target.write_text(text, encoding="utf-8")
    doc = load_dsl_document_from_json(target)
    assert isinstance(doc, ExperimentDocument)
    assert doc.experiment.id == "aug_cutout_vs_none_cifar10"


def test_load_factor_model_from_string() -> None:
    text = _yaml_to_json_text("factor_model_resnet50_imagenet.yaml")
    doc = load_dsl_document_from_json(text)
    assert isinstance(doc, FactorModelDocument)
    assert doc.factor_model.id == "resnet50_imagenet_arch_improvements_2024"


def test_round_trip_through_helper() -> None:
    """Helper output round-trips: parse → dump → parse again equals original."""
    text = _yaml_to_json_text("position_embeddings_wikitext103.yaml")
    doc = load_dsl_document_from_json(text)
    redumped = json.dumps(DslDocumentAdapter.dump_python(doc, mode="json"))
    re_parsed = load_dsl_document_from_json(redumped)
    assert re_parsed == doc


def test_helper_applies_dsl_mode() -> None:
    """A JSON document with ``compare_to_baseline`` and no ``baseline_entry_id``
    must succeed via the helper (which applies DSL mode). Without the helper's
    context, the default artifact gate would reject it."""
    text = _yaml_to_json_text("resnet50_vs_vgg16_cifar10.yaml")
    doc = load_dsl_document_from_json(text)
    assert isinstance(doc, ExperimentDocument)
    assert doc.experiment.validation.comparison.baseline_entry_id is None


def test_helper_rejects_user_supplied_baseline_entry_id() -> None:
    """The helper applies DSL mode, so a user-supplied ``baseline_entry_id`` is
    rejected (DSL gate)."""
    payload = {
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
                    "id": "b",
                    "is_baseline": True,
                    "level": {"factor": "aug", "name": "absent"},
                },
                {
                    "id": "t",
                    "is_baseline": False,
                    "level": {"factor": "aug", "name": "cutout", "technique_id": "tech_cutout"},
                },
            ],
            "validation": {
                "metric": {"kind": "atomic", "ref": "accuracy"},
                "direction": "higher_is_better",
                "aggregation": {"method": "mean"},
                "comparison": {
                    "rule": "compare_to_baseline",
                    "threshold": 0.01,
                    "baseline_entry_id": "b",  # forbidden in DSL mode
                },
                "seed_count_required": 3,
            },
        },
    }
    with pytest.raises(ValidationError) as exc:
        load_dsl_document_from_json(json.dumps(payload))
    assert "comparison.baseline_entry_id_user_supplied" in str(exc.value)


def test_helper_rejects_invalid_kind() -> None:
    text = json.dumps({"kind": "schema", "experiment": {}})
    with pytest.raises(ValidationError):
        load_dsl_document_from_json(text)


def test_helper_rejects_invalid_json() -> None:
    with pytest.raises(json.JSONDecodeError):
        load_dsl_document_from_json("{not valid json}")


def test_helper_path_must_exist(tmp_path: Path) -> None:
    target = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_dsl_document_from_json(target)
