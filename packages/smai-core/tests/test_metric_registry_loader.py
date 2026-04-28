"""Tests for the bundled v1 metric-registry JSON loader (Task 1.4)."""

from __future__ import annotations

from importlib.resources import files
from typing import get_args

import pytest
from smai_core import (
    AtomicMetricEntry,
    MetricRefNotRegistered,
    MetricRegistry,
    ParametricFamily,
    load_metric_registry,
)
from smai_core.data import (
    METRIC_REGISTRY_RESOURCE_NAME,
    METRIC_REGISTRY_SCHEMA_VERSION,
    MetricRegistryDataError,
)

# Counts pinned to dev/smai_metric_registry_v1.md.
EXPECTED_ATOMIC = 58
EXPECTED_PARAMETRIC = 14

# A representative sample of canonical names + family names that the loader
# must produce. If the dev source ever rotates entries out of these slots, the
# test names need updating, not just the counts.
_REQUIRED_ATOMIC = (
    "accuracy",
    "perplexity",
    "psnr",
    "f1",
    "params",
    "flops",
    "latency",
    "ate",
)
_REQUIRED_PARAMETRIC = (
    "top_k_accuracy",
    "average_precision_at_iou",
    "min_traj_error_at_k",
    "compute_cost",
    "mrr_at_k",
    "f1_variant",
)


def test_resource_file_is_present_in_package() -> None:
    """The JSON ships next to the loader module, not as an external resource."""
    bundled = files("smai_core.data") / METRIC_REGISTRY_RESOURCE_NAME
    assert bundled.is_file()


def test_load_metric_registry_returns_validated_instance() -> None:
    registry = load_metric_registry()
    assert isinstance(registry, MetricRegistry)


def test_atomic_count_matches_dev_source() -> None:
    registry = load_metric_registry()
    assert len(registry.atomic) == EXPECTED_ATOMIC


def test_parametric_count_matches_dev_source() -> None:
    registry = load_metric_registry()
    assert len(registry.parametric) == EXPECTED_PARAMETRIC


@pytest.mark.parametrize("name", _REQUIRED_ATOMIC)
def test_required_atomic_canonical_name_present(name: str) -> None:
    registry = load_metric_registry()
    entry = registry.get_atomic(name)
    assert entry is not None
    assert isinstance(entry, AtomicMetricEntry)
    assert entry.canonical_name == name


@pytest.mark.parametrize("family", _REQUIRED_PARAMETRIC)
def test_required_parametric_family_name_present(family: str) -> None:
    registry = load_metric_registry()
    entry = registry.get_family(family)
    assert entry is not None
    assert isinstance(entry, ParametricFamily)
    assert entry.family_name == family


def test_atomic_entries_have_valid_direction_literal() -> None:
    registry = load_metric_registry()
    direction_field = AtomicMetricEntry.model_fields["direction"]
    valid_directions = set(get_args(direction_field.annotation))
    assert valid_directions  # sanity — Literal yielded the enum
    for entry in registry.atomic.values():
        assert entry.direction in valid_directions


def test_parametric_uniform_direction_is_valid_literal() -> None:
    registry = load_metric_registry()
    valid_uniform = {"higher_is_better", "lower_is_better", "ambiguous"}
    valid_per_value = {"higher_is_better", "lower_is_better"}
    for family in registry.parametric.values():
        if isinstance(family.direction, str):
            assert family.direction in valid_uniform
        else:
            assert isinstance(family.direction, dict)
            for value, direction in family.direction.items():
                assert isinstance(value, str)
                assert direction in valid_per_value


def test_min_traj_error_at_k_uses_per_value_direction() -> None:
    """The canonical multi-direction example from `dev/smai_metric_registry_v1.md`."""
    registry = load_metric_registry()
    family = registry.get_family("min_traj_error_at_k")
    assert family is not None
    assert isinstance(family.direction, dict)
    assert family.direction["DAC"] == "higher_is_better"
    assert family.direction["minADE"] == "lower_is_better"


def test_min_traj_error_at_k_uses_multi_parameter_shape() -> None:
    registry = load_metric_registry()
    family = registry.get_family("min_traj_error_at_k")
    assert family is not None
    assert isinstance(family.parameter, list)
    assert isinstance(family.parameter_values_seen, dict)
    assert "error_kind" in family.parameter_values_seen
    assert "k" in family.parameter_values_seen


def test_every_parametric_family_has_non_empty_parameter_values_seen() -> None:
    """Acceptance: ``parameter_values_seen`` must be non-empty everywhere."""
    registry = load_metric_registry()
    for family in registry.parametric.values():
        seen = family.parameter_values_seen
        if isinstance(seen, list):
            assert len(seen) >= 1, family.family_name
        else:
            assert len(seen) >= 1, family.family_name
            for param, vals in seen.items():
                assert len(vals) >= 1, f"{family.family_name}.{param}"


def test_every_atomic_has_category_quality_or_cost() -> None:
    registry = load_metric_registry()
    valid = {"quality", "cost"}
    for entry in registry.atomic.values():
        assert entry.category in valid


def test_known_cost_metrics_are_tagged_cost() -> None:
    """Cost-tagged entries auto-include into ``optional_telemetry`` per
    02-dsl-and-contracts.md §6.6."""
    registry = load_metric_registry()
    for name in ("params", "flops", "latency"):
        entry = registry.get_atomic(name)
        assert entry is not None
        assert entry.category == "cost", name
    cost_family = registry.get_family("compute_cost")
    assert cost_family is not None
    assert cost_family.category == "cost"


def test_quality_metric_sample_tagged_quality() -> None:
    registry = load_metric_registry()
    entry = registry.get_atomic("accuracy")
    assert entry is not None
    assert entry.category == "quality"


def test_registry_invariant_no_double_underscore_or_equals() -> None:
    """Registry-policy invariant from 01-data-model.md §3.8.1."""
    registry = load_metric_registry()
    forbidden = ("=", "__")
    for name in registry.atomic:
        for token in forbidden:
            assert token not in name, name
    for name in registry.parametric:
        for token in forbidden:
            assert token not in name, name
    for family in registry.parametric.values():
        seen = family.parameter_values_seen
        flat: list[str | int | float] = []
        if isinstance(seen, list):
            flat.extend(seen)
        else:
            for vals in seen.values():
                flat.extend(vals)
        for v in flat:
            if isinstance(v, str):
                for token in forbidden:
                    assert token not in v, (family.family_name, v)


def test_load_metric_registry_is_cached() -> None:
    """``functools.cache`` guarantees the same instance on repeat calls."""
    a = load_metric_registry()
    b = load_metric_registry()
    assert a is b


def test_schema_version_constant_matches_bundled_file() -> None:
    """The constant the loader checks against == the value embedded in the JSON."""
    import json

    raw = (files("smai_core.data") / METRIC_REGISTRY_RESOURCE_NAME).read_text(encoding="utf-8")
    assert json.loads(raw)["schema_version"] == METRIC_REGISTRY_SCHEMA_VERSION


def test_loader_rejects_invalid_data_is_a_data_error_subclass() -> None:
    """``MetricRegistryDataError`` is the canonical error for malformed bundles."""
    assert issubclass(MetricRegistryDataError, ValueError)


def test_lookup_returns_atomic_entry_for_registered_atomic_ref() -> None:
    from smai_core import AtomicMetricRef

    registry = load_metric_registry()
    ref = AtomicMetricRef(ref="accuracy")
    entry = registry.lookup(ref)
    assert isinstance(entry, AtomicMetricEntry)
    assert entry.canonical_name == "accuracy"


def test_lookup_returns_family_for_registered_parametric_ref() -> None:
    from smai_core import ParametricMetricRef

    registry = load_metric_registry()
    ref = ParametricMetricRef(family="top_k_accuracy", parameters={"k": 1})
    entry = registry.lookup(ref)
    assert isinstance(entry, ParametricFamily)
    assert entry.family_name == "top_k_accuracy"


def test_lookup_raises_metric_ref_not_registered_for_unknown_atomic() -> None:
    from smai_core import AtomicMetricRef

    registry = load_metric_registry()
    with pytest.raises(MetricRefNotRegistered) as excinfo:
        registry.lookup(AtomicMetricRef(ref="not_a_real_metric"))
    assert "not_a_real_metric" in str(excinfo.value)


def test_lookup_raises_metric_ref_not_registered_for_unknown_family() -> None:
    from smai_core import ParametricMetricRef

    registry = load_metric_registry()
    with pytest.raises(MetricRefNotRegistered) as excinfo:
        registry.lookup(ParametricMetricRef(family="not_a_real_family", parameters={"k": 1}))
    assert "not_a_real_family" in str(excinfo.value)


def test_metric_ref_not_registered_is_key_error_subclass() -> None:
    """Tier B integrators can catch either ``KeyError`` or the specific subclass."""
    assert issubclass(MetricRefNotRegistered, KeyError)


def test_canonical_get_helpers_return_none_per_canonical_doc_contract() -> None:
    """Per 01-data-model.md §3.8.1 the canonical contract is ``None``-on-miss
    so verification rules can emit a structured ``ValidationError``."""
    registry = load_metric_registry()
    assert registry.get_atomic("not_a_real_metric") is None
    assert registry.get_family("not_a_real_family") is None
