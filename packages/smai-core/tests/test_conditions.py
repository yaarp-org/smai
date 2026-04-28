"""Tests for ``ControlledConditions`` — the plugin-extensibility surface."""

from __future__ import annotations

from smai_core import ControlledConditions


def _basic() -> ControlledConditions:
    return ControlledConditions(
        dataset={"name": "cifar10", "split": "train", "version": "v1"},
        optimization={"optimizer": "sgd", "lr": 0.1},
        seeds=[1, 2, 3],
    )


def test_basic_validates() -> None:
    cc = _basic()
    assert cc.seeds == [1, 2, 3]


def test_with_extras_validates() -> None:
    cc = ControlledConditions(
        dataset={"name": "cifar10", "split": "train", "version": "v1"},
        optimization={"optimizer": "sgd", "lr": 0.1},
        seeds=[1, 2],
        architecture={"name": "resnet50"},  # type: ignore[call-arg]
    )
    assert cc.has_field("architecture")
    assert cc.get_field("architecture") == {"name": "resnet50"}


def test_has_field_finds_declared() -> None:
    cc = _basic()
    assert cc.has_field("dataset")
    assert cc.has_field("optimization")
    assert cc.has_field("seeds")


def test_has_field_finds_extras() -> None:
    cc = ControlledConditions.model_validate(
        {
            "dataset": {"name": "x", "split": "y", "version": "z"},
            "optimization": {"optimizer": "sgd"},
            "seeds": [1],
            "pruning_method": "rigl",
        }
    )
    assert cc.has_field("pruning_method")
    assert cc.get_field("pruning_method") == "rigl"


def test_has_field_finds_additional_fixed() -> None:
    cc = ControlledConditions(
        dataset={"name": "x", "split": "y", "version": "z"},
        optimization={"optimizer": "sgd"},
        seeds=[1],
        additional_fixed={"activation_function": "relu"},
    )
    assert cc.has_field("activation_function")
    assert cc.get_field("activation_function") == "relu"


def test_has_field_returns_false_for_missing() -> None:
    cc = _basic()
    assert not cc.has_field("nonexistent_key")
    assert cc.get_field("nonexistent_key") is None


def test_round_trip_preserves_extras() -> None:
    cc = ControlledConditions.model_validate(
        {
            "dataset": {"name": "cifar10", "split": "train", "version": "v1"},
            "optimization": {"optimizer": "sgd"},
            "seeds": [1, 2, 3],
            "architecture": {"name": "resnet50"},
            "additional_fixed": {"normalization": "batch"},
        }
    )
    payload = cc.model_dump(mode="json")
    parsed = ControlledConditions.model_validate(payload)
    assert parsed.has_field("architecture")
    assert parsed.has_field("normalization")
    assert parsed.get_field("architecture") == {"name": "resnet50"}


def test_additional_fixed_not_returned_as_declared_field() -> None:
    cc = ControlledConditions(
        dataset={"name": "x", "split": "y", "version": "z"},
        optimization={"optimizer": "sgd"},
        seeds=[1],
        additional_fixed={"k": "v"},
    )
    # additional_fixed is the bucket itself, not a "field" by name
    assert cc.has_field("k")
    assert cc.get_field("k") == "v"
