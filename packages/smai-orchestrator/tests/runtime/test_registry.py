"""Tests for the process-local :class:`PipelineSpec` registry.

Per `05-orchestrator.md` §5.1 / §7.1: register-once-per-process,
lookup-by-name. The :func:`reset_registry` helper exists for test
isolation; the autouse fixture in ``conftest.py`` calls it before /
after every test.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import pytest
from smai_core.plugins import EntityKind
from smai_orchestrator.engine import (
    ConcurrencyPool,
    EdgeDef,
    StateDef,
)
from smai_orchestrator.runtime import (
    DuplicateSpecError,
    PipelineSpec,
    SpecNotRegisteredError,
    get_pipeline_spec,
    list_registered_specs,
    register_pipeline_spec,
)

_ENGINE_TESTS_DIR = Path(__file__).resolve().parent.parent / "engine"
sys.path.insert(0, str(_ENGINE_TESTS_DIR))

from _helpers import make_gate  # type: ignore[import-not-found] # noqa: E402


def _spec(name: str) -> PipelineSpec:
    return PipelineSpec(
        name=name,
        entity_kind=cast(EntityKind, "cg"),
        initial_state="a",
        states=[
            StateDef(name="a"),
            StateDef(name="b", is_terminal=True),
        ],
        edges=[
            EdgeDef(
                name="advance",
                from_state="a",
                target_state="b",
                gate_rule=make_gate(advance=True),
            ),
        ],
        pools=[ConcurrencyPool(name="p", limit=1)],
        scheduling_queries={},
    )


def test_register_then_get() -> None:
    spec = _spec("alpha")
    register_pipeline_spec(spec)
    assert get_pipeline_spec("alpha") is spec


def test_get_unregistered_raises_with_available() -> None:
    register_pipeline_spec(_spec("alpha"))
    register_pipeline_spec(_spec("beta"))
    with pytest.raises(SpecNotRegisteredError) as excinfo:
        get_pipeline_spec("ghost")
    assert excinfo.value.name == "ghost"
    assert sorted(excinfo.value.available) == ["alpha", "beta"]


def test_duplicate_registration_rejected() -> None:
    register_pipeline_spec(_spec("alpha"))
    with pytest.raises(DuplicateSpecError, match="already registered"):
        register_pipeline_spec(_spec("alpha"))


def test_list_registered_specs_returns_sorted() -> None:
    assert list_registered_specs() == []
    register_pipeline_spec(_spec("zeta"))
    register_pipeline_spec(_spec("alpha"))
    register_pipeline_spec(_spec("mu"))
    assert list_registered_specs() == ["alpha", "mu", "zeta"]


def test_reset_registry_isolates_tests() -> None:
    """Demonstrates the autouse fixture in conftest works.

    By the time this test starts the registry is empty even though the
    previous tests registered specs.
    """
    assert list_registered_specs() == []
    register_pipeline_spec(_spec("only_in_this_test"))
    assert "only_in_this_test" in list_registered_specs()
