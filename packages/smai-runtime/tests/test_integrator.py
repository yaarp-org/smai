"""Integrator (technique → harness components) tests (§3.5 / §8.4)."""

from __future__ import annotations

from smai_runtime import (
    HarnessAPIManifest,
    HarnessComponents,
    HarnessExtensionPoint,
    freeze_manifest,
    integrate_technique_output,
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


def test_append_pattern_appends_to_list() -> None:
    base = HarnessComponents(train_transforms=[lambda x: x])

    def t1(x: int) -> int:
        return x + 1

    def t2(x: int) -> int:
        return x * 2

    m = _manifest(
        HarnessExtensionPoint(
            key="train_transforms",
            type_signature="list[Callable]",
            purpose="x",
            optional=True,
            integration_pattern="append",
        )
    )
    out = integrate_technique_output(base, {"train_transforms": [t1, t2]}, m)
    assert len(out.train_transforms) == 3


def test_replace_pattern_replaces_default_loss() -> None:
    base = HarnessComponents(default_loss=lambda y, p: 0.0)

    def my_loss(y: float, p: float) -> float:
        return 1.0

    m = _manifest(
        HarnessExtensionPoint(
            key="loss_fn",
            type_signature="Callable[[Tensor, Tensor], Tensor]",
            purpose="x",
            optional=False,
            integration_pattern="replace",
        )
    )
    out = integrate_technique_output(base, {"loss_fn": my_loss}, m)
    assert out.default_loss is my_loss


def test_wrap_pattern_composes_factory() -> None:
    from typing import Any

    def base_factory() -> dict[str, Any]:
        return {"weights": [1, 2, 3]}

    base = HarnessComponents(model_factory=base_factory)

    def freeze_layers(model: dict[str, Any]) -> dict[str, Any]:
        return {**model, "frozen": True}

    m = _manifest(
        HarnessExtensionPoint(
            key="model_wrapper",
            type_signature="Callable[[nn.Module], nn.Module]",
            purpose="x",
            optional=False,
            integration_pattern="wrap",
        )
    )
    out = integrate_technique_output(base, {"model_wrapper": freeze_layers}, m)
    assert out.model_factory is not None
    built = out.model_factory()
    assert built == {"weights": [1, 2, 3], "frozen": True}


def test_override_dict_pattern_dict_merges() -> None:
    base = HarnessComponents(training_config={"epochs": 100, "lr": 0.01})
    m = _manifest(
        HarnessExtensionPoint(
            key="training_overrides",
            type_signature="dict[str, Any]",
            purpose="x",
            optional=True,
            integration_pattern="override_dict",
        )
    )
    out = integrate_technique_output(base, {"training_overrides": {"lr": 0.001, "warmup": 5}}, m)
    assert out.training_config == {"epochs": 100, "lr": 0.001, "warmup": 5}


def test_absent_optional_key_leaves_default() -> None:
    base = HarnessComponents(callbacks=[lambda: None])
    m = _manifest(
        HarnessExtensionPoint(
            key="callbacks",
            type_signature="list[Callable]",
            purpose="x",
            optional=True,
            integration_pattern="append",
        )
    )
    out = integrate_technique_output(base, {}, m)
    assert len(out.callbacks) == 1  # untouched
