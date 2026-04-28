"""Shared helpers for Task 1.6 contract-emission tests.

Loads each YAML fixture, supplies the fixture technique registry from
``_verification_helpers``, and exposes a single ``compile_fixture`` entry
point that returns the artifact set + emission report for assertions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml
from _verification_helpers import fixture_registries  # type: ignore[import-not-found]
from smai_core import (
    DslDocumentAdapter,
    ExperimentDocument,
    FactorModelDocument,
    Registries,
    VerifiedExperimentDefinition,
    emit_artifacts,
    emit_factor_model_artifacts,
    verify,
)
from smai_core.artifacts._set import ContractArtifactSet
from smai_core.entities.validation_report import ValidationReport

FIXTURE_DIR: Path = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "experiments"

EXPERIMENT_FIXTURES: tuple[str, ...] = (
    "resnet50_vs_vgg16_cifar10",
    "cutout_on_cifar10",
    "pruning_sparsity_sweep",
    "position_embeddings_wikitext103",
)
FACTOR_MODEL_FIXTURES: tuple[str, ...] = ("factor_model_resnet50_imagenet",)
ALL_FIXTURES: tuple[str, ...] = EXPERIMENT_FIXTURES + FACTOR_MODEL_FIXTURES


def load_fixture_payload(name: str) -> dict[str, Any]:
    """Load a YAML fixture as a raw dict (caller-side YAML parsing per §2.1)."""
    raw = yaml.safe_load((FIXTURE_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    assert isinstance(raw, dict), f"fixture {name} must be a YAML mapping"
    return cast(dict[str, Any], raw)


def load_fixture_document(name: str) -> ExperimentDocument | FactorModelDocument:
    """Validate a fixture into a typed DSL document."""
    return DslDocumentAdapter.validate_python(
        load_fixture_payload(name), context={"smai_mode": "dsl"}
    )


def compile_experiment_fixture(
    name: str, registries: Registries | None = None
) -> tuple[ContractArtifactSet, ValidationReport]:
    """Verify + emit one experiment fixture; return ``(set, report)``."""
    doc = load_fixture_document(name)
    assert isinstance(doc, ExperimentDocument)
    reg = registries if registries is not None else fixture_registries()
    verified = verify(doc, reg)
    assert isinstance(verified, VerifiedExperimentDefinition)
    return emit_artifacts(verified, reg)


def compile_factor_model_fixture(
    name: str, registries: Registries | None = None
) -> tuple[dict[str, ContractArtifactSet], ValidationReport]:
    """Verify + emit one factor-model fixture; return ``(dict[id->set], report)``."""
    doc = load_fixture_document(name)
    assert isinstance(doc, FactorModelDocument)
    reg = registries if registries is not None else fixture_registries()
    verified = verify(doc, reg)
    assert isinstance(verified, list)
    return emit_factor_model_artifacts(doc, verified, reg)


__all__ = [
    "ALL_FIXTURES",
    "EXPERIMENT_FIXTURES",
    "FACTOR_MODEL_FIXTURES",
    "FIXTURE_DIR",
    "compile_experiment_fixture",
    "compile_factor_model_fixture",
    "load_fixture_document",
    "load_fixture_payload",
]
