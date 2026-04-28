"""Tests for ``Factor``, ``Level``, ``Entry``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from smai_core import Entry, Factor, Level, NumericValue


def test_factor_validates() -> None:
    f = Factor(name="augmentation", type="additive", description="random crops on/off")
    assert f.type == "additive"


def test_factor_round_trip() -> None:
    f = Factor(name="architecture", type="substitutive", description="model family")
    payload = f.model_dump(mode="json")
    assert Factor.model_validate(payload) == f


def test_factor_rejects_invalid_type() -> None:
    with pytest.raises(ValidationError):
        Factor.model_validate({"name": "x", "type": "pipeline", "description": "y"})


def test_level_substitutive_validates() -> None:
    level = Level(factor="architecture", name="resnet50", technique_id="tech_resnet50")
    assert level.technique_id == "tech_resnet50"


def test_level_additive_baseline_validates() -> None:
    level = Level(factor="augmentation", name="absent", technique_id=None)
    assert level.technique_id is None


def test_level_with_value_validates() -> None:
    level = Level(
        factor="sparsity",
        name="0.7",
        technique_id="tech_rigl",
        value=NumericValue(value=0.7, kind="continuous", min=0.0, max=1.0),
    )
    assert level.value is not None
    assert level.value.value == 0.7


def test_level_with_params_validates() -> None:
    level = Level(
        factor="augmentation",
        name="cutout_16",
        technique_id="tech_cutout",
        technique_params={"patch_size": 16, "fill": "mean", "active": True, "alpha": 0.5},
    )
    assert level.technique_params == {
        "patch_size": 16,
        "fill": "mean",
        "active": True,
        "alpha": 0.5,
    }


def test_level_round_trip_minimal() -> None:
    level = Level(factor="aug", name="absent")
    payload = level.model_dump(mode="json")
    assert Level.model_validate(payload) == level


def test_level_round_trip_full() -> None:
    level = Level(
        factor="sparsity",
        name="0.5",
        description="medium sparsity",
        technique_id="tech_rigl",
        technique_params={"sparsity": 0.5},
        value=NumericValue(value=0.5, kind="continuous"),
    )
    payload = level.model_dump(mode="json")
    assert Level.model_validate(payload) == level


def test_entry_validates() -> None:
    entry = Entry(
        id="entry_1",
        is_baseline=True,
        level=Level(factor="aug", name="absent"),
    )
    assert entry.is_baseline is True


def test_entry_round_trip() -> None:
    entry = Entry(
        id="entry_2",
        is_baseline=False,
        level=Level(factor="aug", name="cutout", technique_id="tech_cutout"),
    )
    payload = entry.model_dump(mode="json")
    assert Entry.model_validate(payload) == entry
