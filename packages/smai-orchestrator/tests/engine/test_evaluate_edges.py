"""Unit tests for :func:`evaluate_outgoing_edges`.

Per `05` §1.2: declaration-order evaluation, first-passing-edge wins,
phase-keyed evaluation (edges filter on ``fires_on``).
"""

from __future__ import annotations

import pytest
from _helpers import (
    FakeArtifactStore,
    make_gate,
)
from smai_orchestrator.engine import (
    EdgeDef,
    EngineConfig,
    EngineSpec,
    GateContext,
    StateDef,
    evaluate_outgoing_edges,
)


def _build_spec(*edges: EdgeDef) -> EngineSpec:
    """Construct a small spec exercising the supplied edges."""
    return EngineSpec(
        entity_kind="cg",
        initial_state="draft",
        states=[
            StateDef(name="draft"),
            StateDef(name="implementing"),
            StateDef(name="implementation_failed", is_terminal=True),
            StateDef(name="implemented", is_terminal=True),
        ],
        edges=list(edges),
    )


@pytest.fixture
def gate_context(sqlite_store, fake_artifact_store: FakeArtifactStore) -> GateContext:
    return GateContext(
        entity_kind="cg",
        entity_id="cg_1",
        entity_state="draft",
        entity_version=0,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        config=EngineConfig(),
    )


async def test_first_passing_edge_wins(gate_context: GateContext) -> None:
    """Multiple edges from one state — declaration order picks the
    first whose gate returns ``advance=True`` (`05` §1.2)."""
    calls: list[str] = []
    e1 = EdgeDef(
        name="rule-A",
        from_state="draft",
        target_state="implementing",
        gate_rule=make_gate(advance=False, on_call=lambda: calls.append("A")),
    )
    e2 = EdgeDef(
        name="rule-B",
        from_state="draft",
        target_state="implementation_failed",
        gate_rule=make_gate(advance=True, on_call=lambda: calls.append("B")),
    )
    e3 = EdgeDef(
        name="rule-C",
        from_state="draft",
        target_state="implemented",
        gate_rule=make_gate(advance=True, on_call=lambda: calls.append("C")),
    )
    spec = _build_spec(e1, e2, e3)
    fired = await evaluate_outgoing_edges(
        spec=spec,
        entity_state="draft",
        phase="dispatch_time",
        gate_context=gate_context,
    )
    # The third edge would also pass but is not evaluated — `B` wins.
    assert fired is e2
    assert calls == ["A", "B"]


async def test_no_match_returns_none(gate_context: GateContext) -> None:
    e1 = EdgeDef(
        name="rule-A",
        from_state="draft",
        target_state="implementing",
        gate_rule=make_gate(advance=False),
    )
    e2 = EdgeDef(
        name="rule-B",
        from_state="draft",
        target_state="implementation_failed",
        gate_rule=make_gate(advance=False),
    )
    spec = _build_spec(e1, e2)
    fired = await evaluate_outgoing_edges(
        spec=spec,
        entity_state="draft",
        phase="dispatch_time",
        gate_context=gate_context,
    )
    assert fired is None


async def test_phase_filter_respects_fires_on(gate_context: GateContext) -> None:
    """Phase-3 ignores phase-1 edges and vice versa (`05` §1.2)."""
    p3_edge = EdgeDef(
        name="p3",
        from_state="draft",
        target_state="implementing",
        gate_rule=make_gate(advance=True),
        fires_on="dispatch_time",
    )
    p1_succ = EdgeDef(
        name="p1-ok",
        from_state="draft",
        target_state="implemented",
        gate_rule=make_gate(advance=True),
        fires_on="job_succeeded",
    )
    p1_fail = EdgeDef(
        name="p1-fail",
        from_state="draft",
        target_state="implementation_failed",
        gate_rule=make_gate(advance=True),
        fires_on="job_failed",
    )
    spec = _build_spec(p3_edge, p1_succ, p1_fail)

    # Phase-3 sees only the p3_edge.
    fired = await evaluate_outgoing_edges(
        spec=spec,
        entity_state="draft",
        phase="dispatch_time",
        gate_context=gate_context,
    )
    assert fired is p3_edge

    # Phase-1 success-path picks the success edge.
    fired = await evaluate_outgoing_edges(
        spec=spec,
        entity_state="draft",
        phase="job_succeeded",
        gate_context=gate_context,
    )
    assert fired is p1_succ

    # Phase-1 failure-path picks the failure edge.
    fired = await evaluate_outgoing_edges(
        spec=spec,
        entity_state="draft",
        phase="job_failed",
        gate_context=gate_context,
    )
    assert fired is p1_fail


async def test_unknown_state_returns_none(gate_context: GateContext) -> None:
    """`evaluate_outgoing_edges` is robust to a state with no outgoing
    edges (e.g., a terminal); returns ``None``."""
    spec = _build_spec()  # no edges
    fired = await evaluate_outgoing_edges(
        spec=spec,
        entity_state="draft",
        phase="dispatch_time",
        gate_context=gate_context,
    )
    assert fired is None
