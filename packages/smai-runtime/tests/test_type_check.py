"""Manifest-driven runtime type check tests (§6.1 / §6.2)."""

from __future__ import annotations

from typing import Any

import pytest
from smai_runtime import (
    HarnessAPIManifest,
    HarnessExtensionPoint,
    TechniqueOutputContractError,
    check_technique_output,
    freeze_manifest,
)


def _manifest(*eps: HarnessExtensionPoint) -> HarnessAPIManifest:
    return freeze_manifest(
        HarnessAPIManifest(
            extension_points=list(eps),
            integration_pattern_summary="t",
            harness_version_hash="hv",
            parent_harness_contract_hash="pc",
            manifest_schema_version=1,
            runtime_template_version="1.0.0",
        )
    )


def test_unknown_key_raises() -> None:
    m = _manifest(
        HarnessExtensionPoint(
            key="train_transforms",
            type_signature="list[Callable]",
            purpose="x",
            optional=True,
            integration_pattern="append",
        )
    )
    with pytest.raises(TechniqueOutputContractError) as exc:
        check_technique_output({"unknown_key": []}, m)
    assert exc.value.code == "unknown_key"
    assert exc.value.extension_point_key == "unknown_key"


def test_missing_required_key_raises() -> None:
    m = _manifest(
        HarnessExtensionPoint(
            key="model_wrapper",
            type_signature="Callable[[nn.Module], nn.Module]",
            purpose="x",
            optional=False,
            integration_pattern="replace",
        )
    )
    with pytest.raises(TechniqueOutputContractError) as exc:
        check_technique_output({}, m)
    assert exc.value.code == "missing_required_key"
    assert exc.value.extension_point_key == "model_wrapper"


def test_type_signature_mismatch_callable() -> None:
    m = _manifest(
        HarnessExtensionPoint(
            key="loss_fn",
            type_signature="Callable[[Tensor, Tensor], Tensor]",
            purpose="x",
            optional=False,
            integration_pattern="replace",
        )
    )
    with pytest.raises(TechniqueOutputContractError) as exc:
        check_technique_output({"loss_fn": "not_a_callable"}, m)
    assert exc.value.code == "type_signature_mismatch"
    assert exc.value.extension_point_key == "loss_fn"


def test_type_signature_mismatch_list() -> None:
    m = _manifest(
        HarnessExtensionPoint(
            key="train_transforms",
            type_signature="list[Callable]",
            purpose="x",
            optional=True,
            integration_pattern="append",
        )
    )
    with pytest.raises(TechniqueOutputContractError) as exc:
        check_technique_output({"train_transforms": {"not": "a list"}}, m)
    assert exc.value.code == "type_signature_mismatch"


def test_list_of_callables_pass() -> None:
    m = _manifest(
        HarnessExtensionPoint(
            key="train_transforms",
            type_signature="list[Callable]",
            purpose="x",
            optional=True,
            integration_pattern="append",
        )
    )
    check_technique_output(
        {"train_transforms": [lambda x: x, lambda x: x + 1]},
        m,
    )


def test_dict_signature_check() -> None:
    m = _manifest(
        HarnessExtensionPoint(
            key="training_overrides",
            type_signature="dict[str, Any]",
            purpose="x",
            optional=True,
            integration_pattern="override_dict",
        )
    )
    check_technique_output({"training_overrides": {"epochs": 10}}, m)
    with pytest.raises(TechniqueOutputContractError):
        check_technique_output({"training_overrides": [1, 2, 3]}, m)


def test_unknown_class_name_degrades_to_true() -> None:
    """§6.1 — unknown class names accept anything."""
    m = _manifest(
        HarnessExtensionPoint(
            key="model_wrapper",
            type_signature="ZGormoth",  # unknown class
            purpose="x",
            optional=False,
            integration_pattern="replace",
        )
    )
    check_technique_output({"model_wrapper": object()}, m)


def test_replay_byte_equal_outcome() -> None:
    """§6.4 — same triple → same outcome."""
    m = _manifest(
        HarnessExtensionPoint(
            key="train_transforms",
            type_signature="list[Callable]",
            purpose="x",
            optional=True,
            integration_pattern="append",
        )
    )
    payload: dict[str, Any] = {"train_transforms": "wrong"}
    e1: TechniqueOutputContractError | None = None
    e2: TechniqueOutputContractError | None = None
    try:
        check_technique_output(payload, m)
    except TechniqueOutputContractError as e:
        e1 = e
    try:
        check_technique_output(payload, m)
    except TechniqueOutputContractError as e:
        e2 = e
    assert e1 is not None and e2 is not None
    assert e1.to_dict() == e2.to_dict()
