"""Tests for ``load_default_registries()`` (Task 1.4 — default-Registries factory)."""

from __future__ import annotations

from smai_core import (
    AdditivePlugin,
    MetricRegistry,
    Registries,
    SubstitutivePlugin,
    TechniqueRef,
    load_default_registries,
)


def test_returns_registries_instance() -> None:
    registries = load_default_registries()
    assert isinstance(registries, Registries)


def test_metric_registry_is_loaded_v1_set() -> None:
    registries = load_default_registries()
    assert isinstance(registries.metric_registry, MetricRegistry)
    assert len(registries.metric_registry.atomic) == 58
    assert len(registries.metric_registry.parametric) == 14


def test_factor_type_plugins_include_v1_builtins() -> None:
    registries = load_default_registries()
    assert "additive" in registries.factor_type_plugins
    assert "substitutive" in registries.factor_type_plugins
    assert isinstance(registries.factor_type_plugins["additive"], AdditivePlugin)
    assert isinstance(registries.factor_type_plugins["substitutive"], SubstitutivePlugin)


def test_aggregation_rule_registry_has_v1_closed_set() -> None:
    registries = load_default_registries()
    assert set(registries.aggregation_rule_registry) == {"mean", "median"}


def test_comparison_rule_registry_has_v1_closed_set() -> None:
    registries = load_default_registries()
    assert set(registries.comparison_rule_registry) == {
        "compare_to_baseline",
        "compare_to_target",
    }


def test_technique_registry_defaults_to_empty() -> None:
    """Per 02-dsl-and-contracts.md §4.1 — technique registry is *input* to the
    methodology layer; the v1 starter set is populated by callers."""
    registries = load_default_registries()
    assert registries.technique_registry == {}


def test_technique_registry_accepts_caller_supplied_input() -> None:
    tech = TechniqueRef(
        id="tech_resnet50_cifar10",
        name="ResNet-50",
        description="ResNet-50 with CIFAR-10 head.",
        category="architecture",
        compatible_factor_types=["substitutive"],
        standard=True,
        affects_extension_points=["model_wrapper"],
    )
    registries = load_default_registries(technique_registry={tech.id: tech})
    assert registries.technique_registry == {tech.id: tech}


def test_default_registries_share_underlying_metric_registry_instance() -> None:
    """The metric registry loader is cached, so two factory calls share state."""
    a = load_default_registries()
    b = load_default_registries()
    assert a.metric_registry is b.metric_registry


def test_default_registries_provide_independent_dict_copies_of_rule_registries() -> None:
    """Mutation of one factory's rule registry must not leak to another's."""
    a = load_default_registries()
    b = load_default_registries()
    assert a.aggregation_rule_registry is not b.aggregation_rule_registry
    assert a.comparison_rule_registry is not b.comparison_rule_registry


def test_registries_default_field_factories_populate_rule_registries() -> None:
    """Direct ``Registries(...)`` construction without rule registries still
    yields the v1 closed set via the model's ``default_factory``."""
    from smai_core import load_metric_registry

    registries = Registries(
        technique_registry={},
        metric_registry=load_metric_registry(),
        factor_type_plugins={},
    )
    assert set(registries.aggregation_rule_registry) == {"mean", "median"}
    assert set(registries.comparison_rule_registry) == {
        "compare_to_baseline",
        "compare_to_target",
    }
