"""Round-trip tests for the worked-example YAML fixtures.

The five fixtures under ``tests/fixtures/experiments/`` are the canonical
input set consumed by Tasks 1.5 / 1.6 / 1.7 and the Phase 2 smoke test
(2.D3) per ``implementation_plan.md`` §3.2 / Task 1.2. Each fixture is the
transcription of one worked example from
``Yaarp/dev/smai_design/02-example-experiments.md``, wrapped in the v2 DSL
document discriminator (``kind: experiment`` or ``kind: factor_model``)
per ``02-dsl-and-contracts.md`` §2.2.

YAML loading is caller-side per §2.1: ``smai-core`` does not import pyyaml
(the dep allowlist forbids it under DEC-029). Test code may use pyyaml as
a workspace-level dev dep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from smai_core import (
    DslDocumentAdapter,
    ExperimentDocument,
    FactorModelDocument,
)

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "experiments"

EXPERIMENT_FIXTURES = (
    "resnet50_vs_vgg16_cifar10.yaml",
    "cutout_on_cifar10.yaml",
    "pruning_sparsity_sweep.yaml",
    "position_embeddings_wikitext103.yaml",
)
FACTOR_MODEL_FIXTURES = ("factor_model_resnet50_imagenet.yaml",)
ALL_FIXTURES = EXPERIMENT_FIXTURES + FACTOR_MODEL_FIXTURES


def _load_yaml(name: str) -> dict[str, Any]:
    path = FIXTURE_DIR / name
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict), f"fixture {name} must be a YAML mapping"
    return cast(dict[str, Any], raw)


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_fixture_file_exists(name: str) -> None:
    assert (FIXTURE_DIR / name).is_file()


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_fixture_yaml_round_trip(name: str) -> None:
    payload = _load_yaml(name)
    doc = DslDocumentAdapter.validate_python(payload, context={"smai_mode": "dsl"})
    redumped = DslDocumentAdapter.dump_python(doc, mode="json")
    re_parsed = DslDocumentAdapter.validate_python(redumped, context={"smai_mode": "dsl"})
    assert re_parsed == doc


@pytest.mark.parametrize("name", EXPERIMENT_FIXTURES)
def test_experiment_fixtures_dispatch_experiment_document(name: str) -> None:
    payload = _load_yaml(name)
    doc = DslDocumentAdapter.validate_python(payload, context={"smai_mode": "dsl"})
    assert isinstance(doc, ExperimentDocument)
    # The CG's user-facing baseline_entry_id must be None at DSL ingestion.
    assert doc.experiment.validation.comparison.baseline_entry_id is None


@pytest.mark.parametrize("name", FACTOR_MODEL_FIXTURES)
def test_factor_model_fixtures_dispatch_factor_model_document(name: str) -> None:
    payload = _load_yaml(name)
    doc = DslDocumentAdapter.validate_python(payload, context={"smai_mode": "dsl"})
    assert isinstance(doc, FactorModelDocument)
    assert doc.factor_model.id
    assert len(doc.experiments) >= 1


def test_resnet_vs_vgg_substitutive_shape() -> None:
    doc = DslDocumentAdapter.validate_python(
        _load_yaml("resnet50_vs_vgg16_cifar10.yaml"),
        context={"smai_mode": "dsl"},
    )
    assert isinstance(doc, ExperimentDocument)
    exp = doc.experiment
    assert exp.factors[0].type == "substitutive"
    assert all(e.level.technique_id is not None for e in exp.entries)
    assert exp.validation.optional_telemetry is not None
    assert len(exp.validation.optional_telemetry) == 2


def test_cutout_additive_baseline_has_null_technique() -> None:
    doc = DslDocumentAdapter.validate_python(
        _load_yaml("cutout_on_cifar10.yaml"),
        context={"smai_mode": "dsl"},
    )
    assert isinstance(doc, ExperimentDocument)
    exp = doc.experiment
    assert exp.factors[0].type == "additive"
    baselines = [e for e in exp.entries if e.is_baseline]
    assert len(baselines) == 1
    assert baselines[0].level.technique_id is None
    treatments = [e for e in exp.entries if not e.is_baseline]
    assert all(e.level.technique_id is not None for e in treatments)


def test_pruning_sweep_has_numeric_values_on_treatment_levels() -> None:
    doc = DslDocumentAdapter.validate_python(
        _load_yaml("pruning_sparsity_sweep.yaml"),
        context={"smai_mode": "dsl"},
    )
    assert isinstance(doc, ExperimentDocument)
    treatments = [e for e in doc.experiment.entries if not e.is_baseline]
    assert len(treatments) == 4
    for entry in treatments:
        assert entry.level.value is not None
        assert entry.level.value.kind == "continuous"


def test_position_embeddings_uses_lower_is_better_perplexity() -> None:
    doc = DslDocumentAdapter.validate_python(
        _load_yaml("position_embeddings_wikitext103.yaml"),
        context={"smai_mode": "dsl"},
    )
    assert isinstance(doc, ExperimentDocument)
    assert doc.experiment.validation.direction == "lower_is_better"
    assert len(doc.experiment.entries) == 4


def test_factor_model_groups_three_cgs() -> None:
    doc = DslDocumentAdapter.validate_python(
        _load_yaml("factor_model_resnet50_imagenet.yaml"),
        context={"smai_mode": "dsl"},
    )
    assert isinstance(doc, FactorModelDocument)
    assert doc.factor_model.comparison_group_ids == [
        "cg_activation_function",
        "cg_normalization_layer",
        "cg_training_augmentation",
    ]
    assert {e.id for e in doc.experiments} == set(doc.factor_model.comparison_group_ids)
    for exp in doc.experiments:
        assert exp.factor_model_id == doc.factor_model.id


def test_factor_model_extras_admitted_on_controlled_conditions() -> None:
    """The mixup CG holds ``activation_function`` and ``normalization_layer``
    constant via ``ControlledConditions``'s ``extra="allow"`` surface."""
    doc = DslDocumentAdapter.validate_python(
        _load_yaml("factor_model_resnet50_imagenet.yaml"),
        context={"smai_mode": "dsl"},
    )
    assert isinstance(doc, FactorModelDocument)
    mixup = next(e for e in doc.experiments if e.id == "cg_training_augmentation")
    assert mixup.controlled_conditions.has_field("activation_function")
    assert mixup.controlled_conditions.get_field("activation_function") == "relu"
    assert mixup.controlled_conditions.get_field("normalization_layer") == "batchnorm"
