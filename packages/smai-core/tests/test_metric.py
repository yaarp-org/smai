"""Tests for ``MetricRef``, ``MetricRegistry``, and friends."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from smai_core import (
    AtomicMetricEntry,
    AtomicMetricRef,
    MetricRefAdapter,
    MetricRegistry,
    ParametricFamily,
    ParametricMetricRef,
)


def test_atomic_metric_validates() -> None:
    ref = AtomicMetricRef(ref="accuracy")
    assert ref.kind == "atomic"
    assert ref.ref == "accuracy"


def test_parametric_metric_validates() -> None:
    ref = ParametricMetricRef(family="top_k_accuracy", parameters={"k": 5})
    assert ref.kind == "parametric"
    assert ref.parameters == {"k": 5}


def test_atomic_round_trip() -> None:
    ref = AtomicMetricRef(ref="psnr")
    payload = ref.model_dump(mode="json")
    assert AtomicMetricRef.model_validate(payload) == ref


def test_parametric_round_trip() -> None:
    ref = ParametricMetricRef(family="average_precision_at_iou", parameters={"iou_or_area": "0.75"})
    payload = ref.model_dump(mode="json")
    assert ParametricMetricRef.model_validate(payload) == ref


def test_discriminator_routes_to_atomic() -> None:
    parsed = MetricRefAdapter.validate_python({"kind": "atomic", "ref": "f1"})
    assert isinstance(parsed, AtomicMetricRef)


def test_discriminator_routes_to_parametric() -> None:
    parsed = MetricRefAdapter.validate_python(
        {"kind": "parametric", "family": "top_k_accuracy", "parameters": {"k": 5}}
    )
    assert isinstance(parsed, ParametricMetricRef)


def test_discriminator_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        MetricRefAdapter.validate_python({"kind": "novel", "ref": "x"})


def test_metric_ref_adapter_json_round_trip_atomic() -> None:
    ref = AtomicMetricRef(ref="accuracy")
    payload = ref.model_dump(mode="json")
    parsed = MetricRefAdapter.validate_python(payload)
    assert parsed == ref


def test_metric_ref_adapter_json_round_trip_parametric() -> None:
    ref = ParametricMetricRef(family="top_k_accuracy", parameters={"k": 1})
    payload = ref.model_dump(mode="json")
    parsed = MetricRefAdapter.validate_python(payload)
    assert parsed == ref


def test_atomic_metric_entry_validates() -> None:
    entry = AtomicMetricEntry(
        canonical_name="accuracy",
        aliases=["acc", "top1"],
        task_types=["image-classification"],
        direction="higher_is_better",
        input_shape="scalar",
        frequency=1234,
        category="quality",
    )
    assert entry.canonical_name == "accuracy"


def test_parametric_family_validates_single_parameter() -> None:
    fam = ParametricFamily(
        family_name="top_k_accuracy",
        parameter="k",
        parameter_values_seen=[1, 5, 10],
        parameter_type="int",
        task_types=["image-classification"],
        direction="higher_is_better",
        frequency=42,
        category="quality",
    )
    assert fam.parameter == "k"


def test_parametric_family_validates_multi_parameter() -> None:
    fam = ParametricFamily(
        family_name="min_traj_error_at_k",
        parameter=["error_kind", "k"],
        parameter_values_seen={"error_kind": ["minADE", "DAC"], "k": [1, 5]},
        parameter_type="enum + int",
        task_types=["motion-prediction"],
        direction={"minADE": "lower_is_better", "DAC": "higher_is_better"},
        frequency=10,
        category="quality",
    )
    assert isinstance(fam.parameter, list)
    assert isinstance(fam.direction, dict)


def test_metric_registry_lookup() -> None:
    registry = MetricRegistry(
        atomic={
            "accuracy": AtomicMetricEntry(
                canonical_name="accuracy",
                aliases=[],
                task_types=["image-classification"],
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
                task_types=["image-classification"],
                direction="higher_is_better",
                frequency=50,
                category="quality",
            )
        },
    )
    assert registry.get_atomic("accuracy") is not None
    assert registry.get_atomic("missing") is None
    assert registry.get_family("top_k_accuracy") is not None
    assert registry.get_family("missing") is None


def test_metric_registry_round_trip() -> None:
    registry = MetricRegistry(
        atomic={
            "psnr": AtomicMetricEntry(
                canonical_name="psnr",
                aliases=[],
                task_types=["image-restoration"],
                direction="higher_is_better",
                input_shape="scalar",
                frequency=10,
                category="quality",
                notes="dB",
            )
        },
        parametric={},
    )
    payload = registry.model_dump(mode="json")
    assert MetricRegistry.model_validate(payload) == registry
