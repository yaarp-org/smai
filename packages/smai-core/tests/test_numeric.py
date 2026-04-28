"""Tests for ``NumericValue``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from smai_core import NumericValue


def test_continuous_value_validates() -> None:
    nv = NumericValue(value=0.7, kind="continuous", unit="ratio", min=0.0, max=1.0)
    assert nv.value == 0.7
    assert nv.kind == "continuous"


def test_ordinal_value_validates() -> None:
    nv = NumericValue(value=16, kind="ordinal", unit="pixels")
    assert nv.value == 16
    assert nv.kind == "ordinal"
    assert nv.min is None and nv.max is None


def test_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        NumericValue.model_validate({"value": 1, "kind": "categorical"})


def test_round_trip_continuous() -> None:
    nv = NumericValue(value=0.5, kind="continuous", min=0.0, max=1.0)
    payload = nv.model_dump(mode="json")
    assert NumericValue.model_validate(payload) == nv


def test_round_trip_ordinal_no_unit() -> None:
    nv = NumericValue(value=4, kind="ordinal")
    payload = nv.model_dump(mode="json")
    assert NumericValue.model_validate(payload) == nv
