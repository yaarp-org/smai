"""Structural validation tests for the SMAI Phase-2 :class:`PipelineSpec`s.

Per Task 2.C4 acceptance: ``cg_execution_spec`` validates against
:class:`PipelineSpec`'s 7 structural validators (state references,
edge from/to, pool refs, scheduling-query refs all consistent). Edge
declaration order is correct. All required pools / queries exist.

These tests construct the specs via the factory functions and assert
on the resulting :class:`PipelineSpec` shape — they do NOT drive the
state machine. Engine-level behavior is covered by the per-state tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _specs_fakes import StubLlmProvider  # type: ignore[import-not-found]
from smai_orchestrator.specs.cg_entries import build_cg_entries_spec
from smai_orchestrator.specs.cg_execution import (
    CG_ENTRIES_SPEC_NAME,
    CG_EXECUTION_SPEC_NAME,
    POOL_AGENTS,
    POOL_INLINE,
    POOL_RUNS,
    build_cg_execution_spec,
)


@pytest.fixture
def cg_spec(tmp_path: Path):  # type: ignore[no-untyped-def]
    review_llm = StubLlmProvider([])
    contextual_llm = StubLlmProvider([])
    return build_cg_execution_spec(
        workspace_root=tmp_path / "workspaces",
        llm_for_code_reviewer=review_llm,  # type: ignore[arg-type]
        llm_for_contextual_evaluator=contextual_llm,  # type: ignore[arg-type]
    )


@pytest.fixture
def entry_spec(tmp_path: Path):  # type: ignore[no-untyped-def]
    return build_cg_entries_spec(workspace_root=tmp_path / "workspaces")


def test_cg_spec_name_and_entity_kind(cg_spec) -> None:  # type: ignore[no-untyped-def]
    assert cg_spec.name == CG_EXECUTION_SPEC_NAME
    assert cg_spec.entity_kind == "cg"
    assert cg_spec.initial_state == "draft"


def test_entry_spec_name_and_entity_kind(entry_spec) -> None:  # type: ignore[no-untyped-def]
    assert entry_spec.name == CG_ENTRIES_SPEC_NAME
    assert entry_spec.entity_kind == "entry"
    assert entry_spec.initial_state == "pending"


def test_cg_spec_states_match_dec_034_failure_terminal_set(cg_spec) -> None:  # type: ignore[no-untyped-def]
    """Per `03-state-machine.md` §3.1 / DEC-034 #1, the CG spec carries
    nine states: six progress + three failure terminals."""
    state_names = {s.name for s in cg_spec.states}
    assert state_names == {
        "draft",
        "implementing",
        "implemented",
        "running",
        "evaluating",
        "complete",
        "implementation_failed",
        "running_failed",
        "evaluation_failed",
    }


def test_cg_spec_terminal_states_have_no_outgoing_edges(cg_spec) -> None:  # type: ignore[no-untyped-def]
    """The 7th structural validator: terminal states reject outgoing edges."""
    terminal_names = {s.name for s in cg_spec.states if s.is_terminal}
    assert terminal_names == {
        "complete",
        "implementation_failed",
        "running_failed",
        "evaluation_failed",
    }
    outgoing_from_terminal = [e for e in cg_spec.edges if e.from_state in terminal_names]
    assert outgoing_from_terminal == []


def test_cg_spec_pools_declared(cg_spec) -> None:  # type: ignore[no-untyped-def]
    """Three pools — runs / agents / inline — declared per `03` §2.6 / DEC-034 #4."""
    pool_names = {p.name for p in cg_spec.pools}
    assert pool_names == {POOL_RUNS, POOL_AGENTS, POOL_INLINE}


def test_cg_spec_pool_priorities_match_dec_034_4(cg_spec) -> None:  # type: ignore[no-untyped-def]
    """Per DEC-034 #4: runs > inline > agents (Phase 2 simplification of
    the canonical four-pool layout — proposal_pipeline / paper_ingestion
    pools land in Phase 3)."""
    priority = {p.name: p.priority for p in cg_spec.pools}
    assert priority[POOL_RUNS] > priority[POOL_AGENTS]


def test_cg_spec_dispatch_pool_refs_resolve(cg_spec) -> None:  # type: ignore[no-untyped-def]
    """Every state's on_entry_dispatch references a declared pool."""
    pool_names = {p.name for p in cg_spec.pools}
    for state in cg_spec.states:
        if state.on_entry_dispatch is None:
            continue
        assert state.on_entry_dispatch.pool in pool_names, (
            f"state {state.name!r} on_entry_dispatch.pool "
            f"{state.on_entry_dispatch.pool!r} not in {sorted(pool_names)}"
        )


def test_cg_spec_scheduling_queries_declared(cg_spec) -> None:  # type: ignore[no-untyped-def]
    """Every non-terminal CG state has a scheduling query."""
    expected_keys = {"draft", "implementing", "implemented", "running", "evaluating"}
    assert set(cg_spec.scheduling_queries.keys()) == expected_keys


def test_cg_spec_edge_declaration_order_success_first(cg_spec) -> None:  # type: ignore[no-untyped-def]
    """Per `03` §2.4 success edges are declared before retry / failure
    edges within the same (from_state, fires_on) group."""
    # Sub-PR E cutover: ``implementing`` edges fire on phase-1
    # ``job_succeeded`` / ``job_failed`` (the harness builder runs in the
    # sandboxed ``smai-agent-runtime`` container, polled by phase-1).
    # Success → implemented BEFORE no-survivors → impl_failed within the
    # ``job_succeeded`` group; ``job_failed`` carries the sandbox-crash
    # terminal. No dispatch_time edges leave ``implementing`` now.
    assert cg_spec.edges_from("implementing", fires_on="dispatch_time") == []

    impl_job_succeeded = cg_spec.edges_from("implementing", fires_on="job_succeeded")
    assert len(impl_job_succeeded) == 2
    assert impl_job_succeeded[0].target_state == "implemented"
    assert impl_job_succeeded[1].target_state == "implementation_failed"

    impl_job_failed = cg_spec.edges_from("implementing", fires_on="job_failed")
    assert len(impl_job_failed) == 1
    assert impl_job_failed[0].target_state == "implementation_failed"

    # implemented dispatch_time: running BEFORE implementing-retry BEFORE impl_failed
    implemented_dispatch = cg_spec.edges_from("implemented", fires_on="dispatch_time")
    assert len(implemented_dispatch) == 3
    assert implemented_dispatch[0].target_state == "running"
    assert implemented_dispatch[1].target_state == "implementing"
    assert implemented_dispatch[2].target_state == "implementation_failed"


def test_cg_spec_three_failure_terminal_routes_per_dec_034_1(cg_spec) -> None:  # type: ignore[no-untyped-def]
    """Each phase-specific failure terminal has at least one inbound edge."""
    terminals_with_inbound = {
        e.target_state
        for e in cg_spec.edges
        if e.target_state in {"implementation_failed", "running_failed", "evaluation_failed"}
    }
    assert terminals_with_inbound == {
        "implementation_failed",
        "running_failed",
        "evaluation_failed",
    }


def test_cg_spec_engine_spec_projection_partitions_phases(cg_spec) -> None:  # type: ignore[no-untyped-def]
    """:meth:`PipelineSpec.engine_spec` partitions per-state queries by
    presence of phase-1 trigger edges (``job_succeeded`` / ``job_failed``).

    Sub-PR E cutover: ``implementing`` carries phase-1 ``job_*`` edges
    now (the sandboxed harness-builder dispatch is a polled Compute
    job), so its scheduling query partitions into ``phase1_queries``.
    Every other dispatch state still advances via ``dispatch_time``
    edges and stays in ``phase2_queries``.
    """
    eng = cg_spec.engine_spec()
    assert "implementing" in eng.phase1_queries
    assert "implementing" not in eng.phase2_queries
    assert "draft" in eng.phase2_queries
    assert "implemented" in eng.phase2_queries
    assert "running" in eng.phase2_queries
    assert "evaluating" in eng.phase2_queries


def test_entry_spec_states_set(entry_spec) -> None:  # type: ignore[no-untyped-def]
    """Per `03` §3.1, the entry spec covers ``pending → implementing →
    implemented | implementation_failed``."""
    state_names = {s.name for s in entry_spec.states}
    assert state_names == {"pending", "implementing", "implemented", "implementation_failed"}


def test_entry_spec_edges_success_first(entry_spec) -> None:  # type: ignore[no-untyped-def]
    """Per `03` §2.4: validation-pass success edge BEFORE validation-fail
    terminal edge within the same (from_state, fires_on) group.

    Round 14: the entry ``implementing`` edges fire on ``dispatch_time``
    (the technique implementer runs in-process); the ``job failed``
    edge is gone (routed through the RetryPolicy)."""
    impl_dispatch = entry_spec.edges_from("implementing", fires_on="dispatch_time")
    assert impl_dispatch[0].target_state == "implemented"
    assert impl_dispatch[1].target_state == "implementation_failed"
    assert entry_spec.edges_from("implementing", fires_on="job_succeeded") == []
    assert entry_spec.edges_from("implementing", fires_on="job_failed") == []


def test_entry_spec_pool_assignment(entry_spec) -> None:  # type: ignore[no-untyped-def]
    """Entry implementer dispatches go to the ``agents`` pool."""
    state = entry_spec.state_def("implementing")
    assert state.on_entry_dispatch is not None
    assert state.on_entry_dispatch.pool == POOL_AGENTS
    assert state.on_entry_dispatch.handle_field == "implementation_job_handle"


def test_register_smai_specs_round_trips(tmp_path: Path) -> None:
    """The :func:`register_smai_specs` helper registers both specs and
    returns them; subsequent ``get_pipeline_spec`` calls succeed."""
    from smai_orchestrator.runtime import get_pipeline_spec, list_registered_specs
    from smai_orchestrator.specs import register_smai_specs

    review_llm = StubLlmProvider([])
    contextual_llm = StubLlmProvider([])
    cg_spec, entry_spec = register_smai_specs(
        workspace_root=tmp_path / "workspaces",
        llm_for_code_reviewer=review_llm,  # type: ignore[arg-type]
        llm_for_contextual_evaluator=contextual_llm,  # type: ignore[arg-type]
    )
    assert cg_spec.name == CG_EXECUTION_SPEC_NAME
    assert entry_spec.name == CG_ENTRIES_SPEC_NAME
    assert set(list_registered_specs()) == {
        CG_EXECUTION_SPEC_NAME,
        CG_ENTRIES_SPEC_NAME,
    }
    assert get_pipeline_spec(CG_EXECUTION_SPEC_NAME).entity_kind == "cg"
    assert get_pipeline_spec(CG_ENTRIES_SPEC_NAME).entity_kind == "entry"
