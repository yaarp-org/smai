"""Tests for :class:`PipelineSpec` — structural validators + engine
substrate projection.

Per ``05-orchestrator.md`` §5.1 / §5.2 and Task 2.C3's deliverable.
The spec validators reject malformed configurations at construction
time so the engine never needs to defensively check during a poll
cycle.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError
from smai_core.plugins import EntityKind
from smai_orchestrator.engine import (
    ConcurrencyPool,
    DispatchAction,
    EdgeDef,
    SchedulingQueryRef,
    StateDef,
)
from smai_orchestrator.runtime import PipelineSpec

# Re-mount the engine helpers — same builders the engine tests use.
_ENGINE_TESTS_DIR = Path(__file__).resolve().parent.parent / "engine"
sys.path.insert(0, str(_ENGINE_TESTS_DIR))

from _helpers import (  # type: ignore[import-not-found] # noqa: E402
    make_dispatch,
    make_gate,
    make_job_handle,
)


def _two_state_minimal_spec(name: str = "test_spec") -> PipelineSpec:
    """A minimal valid spec — used as a starting point for permutation
    tests that mutate one field to provoke a validator."""
    return PipelineSpec(
        name=name,
        entity_kind=cast(EntityKind, "cg"),
        initial_state="draft",
        states=[
            StateDef(name="draft"),
            StateDef(
                name="implementing",
                on_entry_dispatch=DispatchAction(
                    name="harness_build",
                    handler=make_dispatch(handle=make_job_handle("h1")),
                    pool="agents",
                    handle_field="harness_job_handle",
                ),
            ),
            StateDef(name="implemented", is_terminal=True),
        ],
        edges=[
            EdgeDef(
                name="advance",
                from_state="draft",
                target_state="implementing",
                gate_rule=make_gate(advance=True),
            ),
            EdgeDef(
                name="job-succeeded",
                from_state="implementing",
                target_state="implemented",
                gate_rule=make_gate(advance=True),
                fires_on="job_succeeded",
            ),
        ],
        pools=[ConcurrencyPool(name="agents", limit=4)],
        scheduling_queries={},
    )


# === Happy path =============================================================


def test_minimal_spec_constructs() -> None:
    spec = _two_state_minimal_spec()
    assert spec.name == "test_spec"
    assert spec.initial_state == "draft"
    assert len(spec.states) == 3
    assert len(spec.edges) == 2
    assert spec.pools[0].name == "agents"


def test_state_def_lookup() -> None:
    spec = _two_state_minimal_spec()
    assert spec.state_def("draft").name == "draft"
    with pytest.raises(KeyError):
        spec.state_def("nonexistent")


def test_edges_from_filters_by_phase_trigger() -> None:
    spec = _two_state_minimal_spec()
    dispatch_edges = spec.edges_from("draft", "dispatch_time")
    succeeded_edges = spec.edges_from("implementing", "job_succeeded")
    failed_edges = spec.edges_from("implementing", "job_failed")
    assert len(dispatch_edges) == 1
    assert len(succeeded_edges) == 1
    assert len(failed_edges) == 0


# === Validator: initial_state must resolve ==================================


def test_initial_state_must_resolve_in_states() -> None:
    with pytest.raises(ValidationError, match="initial_state"):
        PipelineSpec(
            name="bad",
            entity_kind=cast(EntityKind, "cg"),
            initial_state="nonexistent",
            states=[StateDef(name="draft")],
            edges=[],
            pools=[],
            scheduling_queries={},
        )


# === Validator: state names must be unique ==================================


def test_duplicate_state_names_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate names"):
        PipelineSpec(
            name="bad",
            entity_kind=cast(EntityKind, "cg"),
            initial_state="a",
            states=[
                StateDef(name="a"),
                StateDef(name="a"),
                StateDef(name="b", is_terminal=True),
            ],
            edges=[],
            pools=[],
            scheduling_queries={},
        )


# === Validator: pool names must be unique ===================================


def test_duplicate_pool_names_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate names"):
        PipelineSpec(
            name="bad",
            entity_kind=cast(EntityKind, "cg"),
            initial_state="a",
            states=[StateDef(name="a"), StateDef(name="b", is_terminal=True)],
            edges=[],
            pools=[
                ConcurrencyPool(name="agents", limit=4),
                ConcurrencyPool(name="agents", limit=2),
            ],
            scheduling_queries={},
        )


# === Validator: edge state references must resolve ==========================


def test_edge_from_state_must_resolve() -> None:
    with pytest.raises(ValidationError, match="unknown from_state"):
        PipelineSpec(
            name="bad",
            entity_kind=cast(EntityKind, "cg"),
            initial_state="a",
            states=[StateDef(name="a"), StateDef(name="b", is_terminal=True)],
            edges=[
                EdgeDef(
                    name="advance",
                    from_state="ghost",
                    target_state="b",
                    gate_rule=make_gate(advance=True),
                ),
            ],
            pools=[],
            scheduling_queries={},
        )


def test_edge_target_state_must_resolve() -> None:
    with pytest.raises(ValidationError, match="unknown target_state"):
        PipelineSpec(
            name="bad",
            entity_kind=cast(EntityKind, "cg"),
            initial_state="a",
            states=[StateDef(name="a"), StateDef(name="b", is_terminal=True)],
            edges=[
                EdgeDef(
                    name="advance",
                    from_state="a",
                    target_state="ghost",
                    gate_rule=make_gate(advance=True),
                ),
            ],
            pools=[],
            scheduling_queries={},
        )


def test_terminal_states_have_no_outgoing_edges() -> None:
    with pytest.raises(ValidationError, match="terminal state"):
        PipelineSpec(
            name="bad",
            entity_kind=cast(EntityKind, "cg"),
            initial_state="a",
            states=[
                StateDef(name="a"),
                StateDef(name="b", is_terminal=True),
            ],
            edges=[
                EdgeDef(
                    name="forbidden",
                    from_state="b",  # terminal
                    target_state="a",
                    gate_rule=make_gate(advance=True),
                ),
            ],
            pools=[],
            scheduling_queries={},
        )


# === Validator: dispatch pool refs must resolve =============================


def test_dispatch_pool_must_resolve() -> None:
    with pytest.raises(ValidationError, match="unknown pool"):
        PipelineSpec(
            name="bad",
            entity_kind=cast(EntityKind, "cg"),
            initial_state="a",
            states=[
                StateDef(name="a"),
                StateDef(
                    name="b",
                    on_entry_dispatch=DispatchAction(
                        name="d",
                        handler=make_dispatch(handle=make_job_handle("h")),
                        pool="ghost-pool",
                    ),
                ),
                StateDef(name="c", is_terminal=True),
            ],
            edges=[
                EdgeDef(
                    name="advance",
                    from_state="a",
                    target_state="b",
                    gate_rule=make_gate(advance=True),
                ),
            ],
            pools=[ConcurrencyPool(name="real-pool", limit=1)],
            scheduling_queries={},
        )


# === Validator: scheduling-query state refs must resolve ====================


def test_scheduling_query_state_must_resolve() -> None:
    spec_kwargs = {
        "name": "bad",
        "entity_kind": cast(EntityKind, "cg"),
        "initial_state": "a",
        "states": [StateDef(name="a"), StateDef(name="b", is_terminal=True)],
        "edges": [],
        "pools": [],
    }
    bogus_query = SchedulingQueryRef(name="ghost", fn=lambda store: _empty_records())
    with pytest.raises(ValidationError, match="no matching StateDef"):
        PipelineSpec(
            **spec_kwargs,  # type: ignore[arg-type]
            scheduling_queries={"ghost-state": bogus_query},
        )


def test_scheduling_query_terminal_state_rejected() -> None:
    bogus_query = SchedulingQueryRef(name="bogus", fn=lambda store: _empty_records())
    with pytest.raises(ValidationError, match="terminal"):
        PipelineSpec(
            name="bad",
            entity_kind=cast(EntityKind, "cg"),
            initial_state="a",
            states=[
                StateDef(name="a"),
                StateDef(name="terminal", is_terminal=True),
            ],
            edges=[],
            pools=[],
            scheduling_queries={"terminal": bogus_query},
        )


async def _empty_records():  # type: ignore[no-untyped-def]
    return []


# === engine_spec() projection: phase partitioning ==========================


def test_engine_spec_partitions_by_outgoing_edge_trigger() -> None:
    """States with an outgoing ``job_succeeded`` / ``job_failed`` edge
    end up in ``phase1_queries``; the rest in ``phase2_queries``."""
    q_in_flight = SchedulingQueryRef(name="in_flight", fn=lambda store: _empty_records())
    q_ready = SchedulingQueryRef(name="ready", fn=lambda store: _empty_records())
    spec = PipelineSpec(
        name="partition_test",
        entity_kind=cast(EntityKind, "cg"),
        initial_state="draft",
        states=[
            StateDef(name="draft"),
            StateDef(
                name="implementing",
                on_entry_dispatch=DispatchAction(
                    name="d",
                    handler=make_dispatch(handle=make_job_handle("h")),
                    pool="agents",
                    handle_field="harness_job_handle",
                ),
            ),
            StateDef(name="implemented", is_terminal=True),
        ],
        edges=[
            EdgeDef(
                name="advance",
                from_state="draft",
                target_state="implementing",
                gate_rule=make_gate(advance=True),
            ),
            EdgeDef(
                name="job-succeeded",
                from_state="implementing",
                target_state="implemented",
                gate_rule=make_gate(advance=True),
                fires_on="job_succeeded",
            ),
        ],
        pools=[ConcurrencyPool(name="agents", limit=2)],
        scheduling_queries={
            "draft": q_ready,
            "implementing": q_in_flight,
        },
    )
    engine = spec.engine_spec()
    assert "draft" in engine.phase2_queries
    assert "implementing" in engine.phase1_queries
    assert "draft" not in engine.phase1_queries
    assert "implementing" not in engine.phase2_queries


def test_engine_spec_preserves_states_and_edges() -> None:
    spec = _two_state_minimal_spec()
    engine = spec.engine_spec()
    assert [s.name for s in engine.states] == [s.name for s in spec.states]
    assert [e.name for e in engine.edges] == [e.name for e in spec.edges]
    assert engine.entity_kind == spec.entity_kind
    assert engine.initial_state == spec.initial_state
    assert engine.pools == spec.pools


# === Inspection dump =========================================================


def test_model_dump_for_inspection_stringifies_callables() -> None:
    spec = _two_state_minimal_spec()
    dump = spec.model_dump_for_inspection()
    # gate_rule shows as a qualified name string (callable refs can't
    # JSON-encode through Pydantic's default model_dump).
    advance_edge = next(e for e in dump["edges"] if e["name"] == "advance")
    assert isinstance(advance_edge["gate_rule"], str)
    # State with on_entry_dispatch carries handler qualname.
    impl_state = next(s for s in dump["states"] if s["name"] == "implementing")
    assert impl_state["on_entry_dispatch"] is not None
    assert isinstance(impl_state["on_entry_dispatch"]["handler"], str)


# === extra="forbid" sanity ==================================================


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineSpec(
            name="x",
            entity_kind=cast(EntityKind, "cg"),
            initial_state="a",
            states=[StateDef(name="a")],
            edges=[],
            pools=[],
            scheduling_queries={},
            unknown_field="oops",  # type: ignore[call-arg]
        )
