"""Tests for the ``FactorTypePlugin`` Protocol and the entry-point loader.

Per ``designs/smai/02-dsl-and-contracts.md`` §3.1 / §3.5 and Task 1.3.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint, EntryPoints, entry_points
from typing import TYPE_CHECKING

import pytest
from smai_core import (
    AdditivePlugin,
    FactorTypePlugin,
    FactorTypePluginError,
    SubstitutivePlugin,
    load_builtin_factor_type_plugins,
)

if TYPE_CHECKING:
    from smai_core.entities.experiment import ExperimentDefinition
    from smai_core.entities.registries import Registries
    from smai_core.entities.validation_report import ValidationError


def test_additive_satisfies_protocol() -> None:
    assert isinstance(AdditivePlugin(), FactorTypePlugin)


def test_substitutive_satisfies_protocol() -> None:
    assert isinstance(SubstitutivePlugin(), FactorTypePlugin)


def test_additive_has_required_attributes() -> None:
    plugin = AdditivePlugin()
    assert plugin.name == "additive"
    assert isinstance(plugin.description, str)
    assert plugin.description  # non-empty


def test_substitutive_has_required_attributes() -> None:
    plugin = SubstitutivePlugin()
    assert plugin.name == "substitutive"
    assert isinstance(plugin.description, str)
    assert plugin.description


def test_entry_points_registers_both_builtins() -> None:
    """Built-in plugins are discoverable via ``smai.factor_types`` entry points."""
    discovered = entry_points(group="smai.factor_types")
    names = {ep.name for ep in discovered}
    assert {"additive", "substitutive"}.issubset(names)


def test_loader_returns_dict_keyed_by_name() -> None:
    plugins = load_builtin_factor_type_plugins()
    assert "additive" in plugins
    assert "substitutive" in plugins
    assert plugins["additive"].name == "additive"
    assert plugins["substitutive"].name == "substitutive"


def test_loader_returns_protocol_conforming_instances() -> None:
    plugins = load_builtin_factor_type_plugins()
    for plugin in plugins.values():
        assert isinstance(plugin, FactorTypePlugin)


# ---------------------------------------------------------------------------
# Stubs used to exercise discovery edge cases without planting real packages.
# Module-level (not nested) so pyright strict mode can resolve their types.
# ---------------------------------------------------------------------------


class _ThirdPartyPlugin:
    name: str = "third_party_test"
    description: str = "Test-only third-party factor type."

    def validate(
        self,
        experiment: ExperimentDefinition,
        registries: Registries,
    ) -> list[ValidationError]:
        del experiment, registries
        return []


class _CollidingAdditivePlugin:
    """Same ``name`` as the built-in additive plugin — must trigger a collision."""

    name: str = "additive"
    description: str = "Test colliding additive."

    def validate(
        self,
        experiment: ExperimentDefinition,
        registries: Registries,
    ) -> list[ValidationError]:
        del experiment, registries
        return []


class _NotAPlugin:
    """Missing ``name`` and ``validate`` — fails Protocol conformance."""


def test_loader_third_party_plugin_via_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """A third-party plugin registered via a fixture entry-point is loaded."""
    fake_ep = EntryPoint(
        name="third_party_test",
        value="tests.fake:third_party_plugin",
        group="smai.factor_types",
    )
    real_load = EntryPoint.load
    third_party_instance = _ThirdPartyPlugin()

    def patched_load(self: EntryPoint) -> object:
        if self is fake_ep:
            return third_party_instance
        return real_load(self)

    monkeypatch.setattr(EntryPoint, "load", patched_load)

    real_eps = entry_points(group="smai.factor_types")
    fake_eps = EntryPoints((*real_eps, fake_ep))

    def fake_entry_points(*, group: str) -> EntryPoints:
        if group == "smai.factor_types":
            return fake_eps
        return entry_points(group=group)

    monkeypatch.setattr("smai_core.factor_types._loader.entry_points", fake_entry_points)

    plugins = load_builtin_factor_type_plugins()
    assert "third_party_test" in plugins
    assert plugins["third_party_test"].name == "third_party_test"
    # Built-ins still discoverable alongside the third-party plugin.
    assert "additive" in plugins
    assert "substitutive" in plugins


def test_loader_rejects_non_conforming_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """An entry-point that yields a non-Protocol object errors clearly."""
    fake_ep = EntryPoint(
        name="bad",
        value="tests.fake:bad",
        group="smai.factor_types",
    )
    bad_instance = _NotAPlugin()

    def patched_load(self: EntryPoint) -> object:
        del self
        return bad_instance

    monkeypatch.setattr(EntryPoint, "load", patched_load)

    def fake_entry_points(*, group: str) -> EntryPoints:
        if group == "smai.factor_types":
            return EntryPoints((fake_ep,))
        return entry_points(group=group)

    monkeypatch.setattr("smai_core.factor_types._loader.entry_points", fake_entry_points)

    with pytest.raises(FactorTypePluginError, match="did not yield a FactorTypePlugin"):
        load_builtin_factor_type_plugins()


def test_loader_rejects_name_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two plugins claiming the same ``name`` are a startup error."""
    fake_ep = EntryPoint(
        name="collision",
        value="tests.fake:collision",
        group="smai.factor_types",
    )
    real_load = EntryPoint.load
    colliding_instance = _CollidingAdditivePlugin()

    def patched_load(self: EntryPoint) -> object:
        if self is fake_ep:
            return colliding_instance
        return real_load(self)

    monkeypatch.setattr(EntryPoint, "load", patched_load)

    real_eps = entry_points(group="smai.factor_types")
    fake_eps = EntryPoints((*real_eps, fake_ep))

    def fake_entry_points(*, group: str) -> EntryPoints:
        if group == "smai.factor_types":
            return fake_eps
        return entry_points(group=group)

    monkeypatch.setattr("smai_core.factor_types._loader.entry_points", fake_entry_points)

    with pytest.raises(FactorTypePluginError, match="Duplicate factor-type plugin name"):
        load_builtin_factor_type_plugins()
