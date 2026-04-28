"""Shared fixtures for smai-runtime tests.

Builds canonical ``HarnessContract``, ``TechniqueContract``, and
``HarnessAPIManifest`` instances with their hash chain settled so test
cases can drop them straight into a workspace and exercise the runtime.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest
from smai_core import (
    ArtifactEnvelope,
    AtomicMetricRef,
    Factor,
    FixedVariable,
    HarnessContract,
    HarnessContractBody,
    TechniqueContract,
    TechniqueContractBody,
    freeze_with_hash,
)
from smai_runtime import (
    HarnessAPIManifest,
    HarnessExtensionPoint,
    freeze_manifest,
)


def _envelope(kind: str, artifact_id: str) -> ArtifactEnvelope:
    return ArtifactEnvelope(
        artifact_kind=kind,  # type: ignore[arg-type]
        artifact_id=artifact_id,
        schema_version=1,
        compiler_version="0.1.0",
        parent_experiment_id="exp",
        parent_factor_model_id=None,
        registry_hashes={"factor_types": "x", "metrics": "y", "techniques": "z"},
        surface_map={},
    )


def make_additive_harness_contract(
    *,
    seeds: Iterable[int] = (1, 2, 3),
    required_metrics: Iterable[str] = ("accuracy",),
    optional_telemetry: Iterable[str] = ("loss",),
    no_go_zones: Iterable[str] = ("experiment.py", "techniques/__init__.py"),
) -> HarnessContract:
    body = HarnessContractBody(
        parent_experiment_hash="exp_hash",
        factor=Factor(name="augmentation", type="additive", description="aug"),
        seeds=list(seeds),
        fixed_variables=[
            FixedVariable(path="dataset.name", value="cifar10", type_hint="str"),
            FixedVariable(path="optimization.lr", value=0.001, type_hint="float"),
        ],
        required_metrics=[AtomicMetricRef(ref=m) for m in required_metrics],
        optional_telemetry=[AtomicMetricRef(ref=m) for m in optional_telemetry],
        no_go_zones=list(no_go_zones),
    )
    return freeze_with_hash(
        HarnessContract(envelope=_envelope("harness_contract", "exp__hc"), body=body)
    )


def make_substitutive_harness_contract(
    *,
    seeds: Iterable[int] = (1, 2, 3),
    required_metrics: Iterable[str] = ("accuracy",),
) -> HarnessContract:
    body = HarnessContractBody(
        parent_experiment_hash="exp_hash",
        factor=Factor(name="architecture", type="substitutive", description="arch"),
        seeds=list(seeds),
        fixed_variables=[
            FixedVariable(path="dataset.name", value="cifar10", type_hint="str"),
        ],
        required_metrics=[AtomicMetricRef(ref=m) for m in required_metrics],
        optional_telemetry=[],
        no_go_zones=["experiment.py", "techniques/__init__.py"],
    )
    return freeze_with_hash(
        HarnessContract(envelope=_envelope("harness_contract", "sub__hc"), body=body)
    )


def make_technique_contract(
    *,
    parent_harness_contract_hash: str,
    entry_id: str = "t1",
    technique_id: str | None = "tech_abc",
    is_baseline: bool = False,
) -> TechniqueContract:
    body = TechniqueContractBody(
        entry_id=entry_id,
        parent_experiment_id="exp",
        parent_experiment_hash="exp_hash",
        parent_harness_contract_hash=parent_harness_contract_hash,
        technique_id=technique_id,
        technique_params={"alpha": 0.1},
        level_value=None,
        is_baseline=is_baseline,
        fidelity_anchor=None,
        standard=True,
    )
    return freeze_with_hash(
        TechniqueContract(envelope=_envelope("technique_contract", f"{entry_id}__tc"), body=body)
    )


def make_additive_manifest(
    *,
    parent_harness_contract_hash: str,
    runtime_template_version: str = "1.0.0",
) -> HarnessAPIManifest:
    return freeze_manifest(
        HarnessAPIManifest(
            extension_points=[
                HarnessExtensionPoint(
                    key="train_transforms",
                    type_signature="list[Callable]",
                    purpose="extra training transforms appended to the harness pipeline",
                    optional=True,
                    integration_pattern="append",
                ),
                HarnessExtensionPoint(
                    key="callbacks",
                    type_signature="list[Callable]",
                    purpose="extra training-loop callbacks",
                    optional=True,
                    integration_pattern="append",
                ),
            ],
            integration_pattern_summary="augmentation-only",
            harness_version_hash="harness_v_hash_abc",
            parent_harness_contract_hash=parent_harness_contract_hash,
            manifest_schema_version=1,
            runtime_template_version=runtime_template_version,
        )
    )


def make_substitutive_manifest(
    *,
    parent_harness_contract_hash: str,
    runtime_template_version: str = "1.0.0",
) -> HarnessAPIManifest:
    return freeze_manifest(
        HarnessAPIManifest(
            extension_points=[
                HarnessExtensionPoint(
                    key="model_wrapper",
                    type_signature="Callable[[nn.Module], nn.Module]",
                    purpose="builds the model architecture for this entry",
                    optional=False,
                    integration_pattern="replace",
                ),
            ],
            integration_pattern_summary="architecture replacement",
            harness_version_hash="sub_harness_v_hash",
            parent_harness_contract_hash=parent_harness_contract_hash,
            manifest_schema_version=1,
            runtime_template_version=runtime_template_version,
        )
    )


@pytest.fixture
def additive_harness_contract() -> HarnessContract:
    return make_additive_harness_contract()


@pytest.fixture
def additive_technique_contract(additive_harness_contract: HarnessContract) -> TechniqueContract:
    return make_technique_contract(
        parent_harness_contract_hash=additive_harness_contract.envelope.content_hash,
        is_baseline=False,
    )


@pytest.fixture
def additive_baseline_technique_contract(
    additive_harness_contract: HarnessContract,
) -> TechniqueContract:
    return make_technique_contract(
        parent_harness_contract_hash=additive_harness_contract.envelope.content_hash,
        entry_id="b",
        technique_id=None,
        is_baseline=True,
    )


@pytest.fixture
def additive_manifest(additive_harness_contract: HarnessContract) -> HarnessAPIManifest:
    return make_additive_manifest(
        parent_harness_contract_hash=additive_harness_contract.envelope.content_hash,
    )


@pytest.fixture
def substitutive_harness_contract() -> HarnessContract:
    return make_substitutive_harness_contract()


@pytest.fixture
def substitutive_technique_contract(
    substitutive_harness_contract: HarnessContract,
) -> TechniqueContract:
    return make_technique_contract(
        parent_harness_contract_hash=substitutive_harness_contract.envelope.content_hash,
    )


@pytest.fixture
def substitutive_manifest(
    substitutive_harness_contract: HarnessContract,
) -> HarnessAPIManifest:
    return make_substitutive_manifest(
        parent_harness_contract_hash=substitutive_harness_contract.envelope.content_hash,
    )
