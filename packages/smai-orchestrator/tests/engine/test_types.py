"""Unit tests for engine primitives — shape and default behavior.

Per Task 2.C1's acceptance: GateContext / EdgeDef / DispatchContext /
DispatchAction shape; EngineConfig defaults; time_provider substitution
works.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest
from _helpers import (
    FakeClock,
    FakeMonotonic,
    make_dispatch,
    make_gate,
)
from smai_core.plugins import JobHandle
from smai_orchestrator.engine import (
    DispatchAction,
    EdgeDef,
    EngineConfig,
    EngineSpec,
    GateContext,
    GateOutcome,
    RetryPolicy,
    StateDef,
    default_time_provider,
    default_wall_clock,
)


def test_engine_config_defaults() -> None:
    cfg = EngineConfig()
    assert cfg.orphan_grace_seconds == 600
    assert cfg.lease_seconds == 120
    assert cfg.retry_policies == {}
    # Defaults are the production callables.
    assert cfg.time_provider is default_time_provider
    assert cfg.wall_clock is default_wall_clock


def test_engine_config_supervisor_defaults() -> None:
    """Task 3.G4: supervisor is default-on; cadence default is every turn."""
    cfg = EngineConfig()
    assert cfg.supervisor_enabled is True
    assert cfg.supervisor_check_every_n_turns == 1


def test_engine_config_supervisor_can_be_disabled() -> None:
    """Deployment override flips the master switch."""
    cfg = EngineConfig(supervisor_enabled=False)
    assert cfg.supervisor_enabled is False
    # ``supervisor_check_every_n_turns`` is still tunable but ignored
    # by handlers when ``supervisor_enabled=False``.
    assert cfg.supervisor_check_every_n_turns == 1


def test_engine_config_supervisor_cadence_override() -> None:
    """Deployment can change the cadence without disabling."""
    cfg = EngineConfig(supervisor_check_every_n_turns=5)
    assert cfg.supervisor_enabled is True
    assert cfg.supervisor_check_every_n_turns == 5


def test_time_provider_substitution() -> None:
    fake = FakeMonotonic(start=10.0)
    cfg = EngineConfig(time_provider=fake)
    assert cfg.time_provider() == 10.0
    fake.advance(5)
    assert cfg.time_provider() == 15.0


def test_wall_clock_substitution() -> None:
    fake = FakeClock(start=datetime(2026, 1, 1, tzinfo=UTC))
    cfg = EngineConfig(wall_clock=fake)
    assert cfg.wall_clock() == datetime(2026, 1, 1, tzinfo=UTC)
    fake.advance(60)
    assert cfg.wall_clock() == datetime(2026, 1, 1, 0, 1, tzinfo=UTC)


def test_default_time_provider_is_monotonic() -> None:
    """Production default uses :func:`time.monotonic` per the brief."""
    assert default_time_provider is time.monotonic


def test_retry_policy_defaults() -> None:
    p = RetryPolicy(max_attempts=3)
    assert p.backoff_seconds == 30
    assert p.backoff_multiplier == 2.0


def test_edge_def_defaults_to_dispatch_time_phase() -> None:
    """Edges default to phase-3 (`dispatch_time`) per `05` §1.2."""
    edge = EdgeDef(
        name="advance",
        from_state="draft",
        target_state="implementing",
        gate_rule=make_gate(advance=True),
    )
    assert edge.fires_on == "dispatch_time"


def test_edge_def_phase_one_triggers() -> None:
    """The two phase-1 triggers must round-trip through Pydantic."""
    s = EdgeDef(
        name="ok",
        from_state="implementing",
        target_state="implemented",
        gate_rule=make_gate(advance=True),
        fires_on="job_succeeded",
    )
    f = EdgeDef(
        name="retry",
        from_state="implementing",
        target_state="draft",
        gate_rule=make_gate(advance=True),
        fires_on="job_failed",
    )
    assert s.fires_on == "job_succeeded"
    assert f.fires_on == "job_failed"


def test_state_def_terminal_default() -> None:
    """States default to non-terminal."""
    s = StateDef(name="draft")
    assert s.is_terminal is False
    assert s.on_entry_dispatch is None


def test_state_def_with_dispatch_action() -> None:
    handle = JobHandle(plugin="fake", handle="h1")
    action = DispatchAction(
        name="build",
        handler=make_dispatch(handle=handle),
        pool="agents",
        handle_field="harness_job_handle",
    )
    s = StateDef(name="implementing", on_entry_dispatch=action)
    assert s.on_entry_dispatch is action
    assert action.handle_field == "harness_job_handle"


def test_dispatch_action_inline_has_no_handle_field() -> None:
    """Inline dispatches (single LLM call, no external compute) leave
    `handle_field` as ``None`` (`05` §1.4)."""
    inline = DispatchAction(
        name="code_review",
        handler=make_dispatch(),
        pool="agents",
    )
    assert inline.handle_field is None


def test_engine_spec_state_def_lookup() -> None:
    spec = EngineSpec(
        entity_kind="cg",
        initial_state="draft",
        states=[StateDef(name="draft"), StateDef(name="complete", is_terminal=True)],
        edges=[],
    )
    assert spec.state_def("draft").name == "draft"
    assert spec.state_def("complete").is_terminal is True
    with pytest.raises(KeyError):
        spec.state_def("nonexistent")


def test_engine_spec_edges_from_filters_by_phase() -> None:
    """`edges_from` returns only edges leaving the named state with
    matching `fires_on` (`05` §1.2 phase-keyed evaluation)."""
    e_dispatch = EdgeDef(
        name="advance",
        from_state="draft",
        target_state="implementing",
        gate_rule=make_gate(advance=True),
        fires_on="dispatch_time",
    )
    e_succeeded = EdgeDef(
        name="ok",
        from_state="implementing",
        target_state="implemented",
        gate_rule=make_gate(advance=True),
        fires_on="job_succeeded",
    )
    e_failed = EdgeDef(
        name="retry",
        from_state="implementing",
        target_state="draft",
        gate_rule=make_gate(advance=True),
        fires_on="job_failed",
    )
    spec = EngineSpec(
        entity_kind="cg",
        initial_state="draft",
        states=[
            StateDef(name="draft"),
            StateDef(name="implementing"),
            StateDef(name="implemented", is_terminal=True),
        ],
        edges=[e_dispatch, e_succeeded, e_failed],
    )

    assert spec.edges_from("draft", fires_on="dispatch_time") == [e_dispatch]
    assert spec.edges_from("draft", fires_on="job_succeeded") == []
    assert spec.edges_from("implementing", fires_on="job_succeeded") == [e_succeeded]
    assert spec.edges_from("implementing", fires_on="job_failed") == [e_failed]


async def test_gate_context_carries_job_outcome_only_phase_one(
    sqlite_store, fake_artifact_store
) -> None:
    """`GateContext.job_outcome` defaults to ``None`` (phase-3 case);
    callers populate it for phase-1 evaluations per `05` §9 #9."""
    cfg = EngineConfig()
    ctx = GateContext(
        entity_kind="cg",
        entity_id="cg_1",
        entity_state="draft",
        entity_version=0,
        metadata_store=sqlite_store,
        artifact_store=fake_artifact_store,
        config=cfg,
    )
    assert ctx.job_outcome is None


async def test_gate_outcome_round_trips() -> None:
    o1 = GateOutcome(advance=True)
    assert o1.reason is None
    o2 = GateOutcome(advance=False, reason="contract not satisfied")
    assert o2.reason == "contract not satisfied"
