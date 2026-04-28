"""Tests for the ``Registries`` container."""

from __future__ import annotations

from smai_core import (
    AtomicMetricEntry,
    MetricRegistry,
    ParametricFamily,
    Registries,
    TechniqueRef,
)


def _registry() -> MetricRegistry:
    return MetricRegistry(
        atomic={
            "accuracy": AtomicMetricEntry(
                canonical_name="accuracy",
                aliases=[],
                task_types=["classification"],
                direction="higher_is_better",
                input_shape="scalar",
                frequency=100,
                category="quality",
            )
        },
        parametric={
            "top_k_accuracy": ParametricFamily(
                family_name="top_k_accuracy",
                parameter="k",
                parameter_values_seen=[1, 5],
                parameter_type="int",
                task_types=["classification"],
                direction="higher_is_better",
                frequency=50,
                category="quality",
            )
        },
    )


def _technique() -> TechniqueRef:
    return TechniqueRef(
        id="tech_resnet50",
        name="ResNet-50",
        description="Residual network with 50 layers",
        category="architecture",
        compatible_factor_types=["substitutive"],
        standard=True,
        affects_extension_points=["model"],
    )


def test_registries_validates() -> None:
    reg = Registries(
        technique_registry={"tech_resnet50": _technique()},
        metric_registry=_registry(),
        factor_type_plugins={},
    )
    assert "tech_resnet50" in reg.technique_registry
    assert reg.metric_registry.get_atomic("accuracy") is not None


def test_registries_admits_protocol_conforming_plugins() -> None:
    """Any object that satisfies ``FactorTypePlugin`` is admitted (Task 1.3)."""
    from smai_core import AdditivePlugin

    plugin = AdditivePlugin()
    reg = Registries(
        technique_registry={},
        metric_registry=_registry(),
        factor_type_plugins={"additive": plugin},
    )
    assert isinstance(reg.factor_type_plugins["additive"], AdditivePlugin)
