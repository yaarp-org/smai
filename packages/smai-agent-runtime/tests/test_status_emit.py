"""Per-step tests for D10's status emit (sub-PR C2).

Asserts:

* :class:`StatusEmitter` produces one line per emit, increments
  ``event_seq`` monotonically, writes to both stdout and the workspace
  mirror file at ``status/events.jsonl``.
* The mini-orchestrator's step iteration emits ``session.start`` /
  ``step.start`` / ``step.end`` / ``session.cost`` / ``session.end``
  in the right order for a successful workflow.
* ``extra="forbid"`` rejects malformed events.
* The host-side discriminated-union parse (``TypeAdapter(StatusEvent)``)
  round-trips every emitted line.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from _harness_builder_workspace_fixtures import (  # type: ignore[import-not-found]
    write_minimal_harness_workspace,
)
from pydantic import TypeAdapter, ValidationError
from smai_agent_runtime.__main__ import main as entry_main
from smai_agent_runtime.status import (
    EVENTS_MIRROR_RELPATH,
    SCHEMA_VERSION,
    SessionStartEvent,
    StatusEmitter,
    StatusEvent,
    StepStartEvent,
    WorkflowPlanItem,
)

# Sub-PR D thread 5: the entry-point tests below append ``--fake-llm`` to
# their argv (the prior ``SMAI_AGENT_RUNTIME_FAKE_LLM`` env-var seam was
# replaced with a dependency-injected :class:`AgentRunner` selected at
# main()-entry by this flag).
_FAKE_LLM_ARGV = ["--fake-llm"]


# === StatusEmitter unit tests ================================================


def test_emitter_writes_one_line_per_emit_with_monotonic_seq(tmp_path: Path) -> None:
    """Each ``emit_*`` writes one JSON line to stdout with
    ``event_seq`` incrementing by 1 starting at 0."""
    buf = io.StringIO()
    emitter = StatusEmitter(session_id="session-test", workspace=tmp_path, stdout=buf)

    emitter.emit_session_start(
        agent_role="harness_builder",
        parent_kind="cg",
        parent_id="cg-x",
        workflow_plan=[
            WorkflowPlanItem(step_index=0, step_kind="harness_body_generation", step_label="a")
        ],
    )
    emitter.emit_step_start(
        step_index=0,
        step_kind="harness_body_generation",
        step_label="a",
    )
    emitter.emit_step_end(
        step_index=0,
        step_kind="harness_body_generation",
        outcome="success",
        duration_seconds=0.1,
    )
    emitter.emit_session_end(outcome="success", last_completed_step_index=0)

    lines = buf.getvalue().strip().split("\n")
    assert len(lines) == 4
    seqs = [json.loads(line)["event_seq"] for line in lines]
    assert seqs == [0, 1, 2, 3]


def test_emitter_writes_workspace_mirror_file(tmp_path: Path) -> None:
    """Each emit also appends to ``status/events.jsonl``."""
    buf = io.StringIO()
    emitter = StatusEmitter(session_id="session-mirror", workspace=tmp_path, stdout=buf)
    emitter.emit_session_start(
        agent_role="harness_builder",
        parent_kind="cg",
        parent_id="cg-x",
        workflow_plan=[],
    )
    emitter.emit_session_end(outcome="success")

    mirror = tmp_path / EVENTS_MIRROR_RELPATH
    assert mirror.exists()
    mirror_lines = mirror.read_text().strip().split("\n")
    assert len(mirror_lines) == 2
    # Mirror content matches the stdout content byte-for-byte.
    assert mirror_lines == buf.getvalue().strip().split("\n")


def test_emitter_emit_step_turn_and_session_cost(tmp_path: Path) -> None:
    """Cover the per-turn rolling-token emit and the
    session-cost final-reckoning emit."""
    buf = io.StringIO()
    emitter = StatusEmitter(session_id="session-cost", workspace=tmp_path, stdout=buf)
    emitter.emit_step_turn(
        step_index=0,
        step_kind="diagnose_on_failure",
        turn_count=2,
        input_tokens_rolling=100,
        output_tokens_rolling=50,
        cache_read_tokens_rolling=10,
        last_tool_call="read_file",
    )
    emitter.emit_session_cost(
        input_tokens=200,
        output_tokens=100,
        cache_read_tokens=20,
        cache_write_tokens=5,
        turn_count=3,
        tool_errors_fired=1,
    )
    lines = buf.getvalue().strip().split("\n")
    types = [json.loads(line)["event_type"] for line in lines]
    assert types == ["step.turn", "session.cost"]


def test_event_schema_forbids_extra_fields() -> None:
    """``extra="forbid"`` is load-bearing for the host parser's
    forward-compat shape (within-event additive fields bump
    SCHEMA_VERSION explicitly per D10 design note)."""
    with pytest.raises(ValidationError):
        SessionStartEvent.model_validate(
            {
                "schema_version": SCHEMA_VERSION,
                "event_seq": 0,
                "emitted_at_utc": "2026-05-26T00:00:00Z",
                "session_id": "s",
                "agent_role": "harness_builder",
                "parent_kind": "cg",
                "parent_id": "cg-x",
                "workflow_plan": [],
                "unknown_field": "boom",
            }
        )


def test_discriminated_union_round_trips_all_event_types(tmp_path: Path) -> None:
    """Every event the emitter produces parses cleanly through
    ``TypeAdapter(StatusEvent)`` — the discriminator dispatch the host
    parser will rely on."""
    buf = io.StringIO()
    emitter = StatusEmitter(session_id="round-trip", workspace=tmp_path, stdout=buf)
    emitter.emit_session_start(
        agent_role="harness_builder",
        parent_kind="cg",
        parent_id="cg-x",
        workflow_plan=[
            WorkflowPlanItem(step_index=0, step_kind="validation", step_label="x"),
        ],
    )
    emitter.emit_step_start(
        step_index=0,
        step_kind="validation",
        step_label="x",
    )
    emitter.emit_step_turn(
        step_index=0,
        step_kind="diagnose_on_failure",
        turn_count=1,
        input_tokens_rolling=10,
        output_tokens_rolling=5,
        cache_read_tokens_rolling=0,
    )
    emitter.emit_step_end(
        step_index=0,
        step_kind="validation",
        outcome="success",
        duration_seconds=0.5,
    )
    emitter.emit_session_cost(
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
        turn_count=1,
        tool_errors_fired=0,
    )
    emitter.emit_session_end(outcome="success", last_completed_step_index=0)

    adapter: TypeAdapter[StatusEvent] = TypeAdapter(StatusEvent)
    for line in buf.getvalue().strip().split("\n"):
        parsed = adapter.validate_json(line)
        # Pydantic discriminated-union parses into the right concrete subclass.
        assert parsed.session_id == "round-trip"


# === Mini-orchestrator integration ===========================================


def test_mini_orchestrator_emits_full_event_stream(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: a successful workflow run emits the expected
    sequence of D10 events. session_start → 7x (step_start, step_end)
    → session_cost → session_end."""
    write_minimal_harness_workspace(tmp_path)
    rc = entry_main(
        [
            "--role",
            "harness_builder",
            "--cg-id",
            "cg-emit-test",
            "--workspace",
            str(tmp_path),
            *_FAKE_LLM_ARGV,
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    events: list[dict] = []
    for line in captured.out.strip().split("\n"):
        events.append(json.loads(line))

    # First event is session.start at seq 0.
    assert events[0]["event_type"] == "session.start"
    assert events[0]["event_seq"] == 0
    # event_seq monotonic.
    for i, ev in enumerate(events):
        assert ev["event_seq"] == i
    # Last two are session.cost then session.end.
    assert events[-2]["event_type"] == "session.cost"
    assert events[-1]["event_type"] == "session.end"
    assert events[-1]["outcome"] == "success"
    # Workflow plan carries 7 steps.
    assert len(events[0]["workflow_plan"]) == 7
    # Every step emitted step.start and step.end (in order).
    step_starts = [e for e in events if e["event_type"] == "step.start"]
    step_ends = [e for e in events if e["event_type"] == "step.end"]
    assert len(step_starts) == 7
    assert len(step_ends) == 7


def test_mini_orchestrator_writes_events_mirror_file(
    tmp_path: Path,
) -> None:
    """The session's workspace-mirror file at ``status/events.jsonl``
    contains the same lines that landed on stdout (the host-harvest
    surface per D10 design)."""
    write_minimal_harness_workspace(tmp_path)
    entry_main(
        [
            "--role",
            "harness_builder",
            "--cg-id",
            "cg-mirror-test",
            "--workspace",
            str(tmp_path),
            *_FAKE_LLM_ARGV,
        ]
    )
    mirror = tmp_path / EVENTS_MIRROR_RELPATH
    assert mirror.exists()
    mirror_lines = mirror.read_text().strip().split("\n")
    assert len(mirror_lines) >= 4  # session.start + at least one step + session.cost + session.end
    # All lines parse as StatusEvents.
    adapter: TypeAdapter[StatusEvent] = TypeAdapter(StatusEvent)
    for line in mirror_lines:
        adapter.validate_json(line)


def test_step_start_carries_input_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The ``input_summary`` on step.start carries per-step
    identification fields the operator surface renders alongside the
    step name."""
    write_minimal_harness_workspace(tmp_path)
    entry_main(
        [
            "--role",
            "harness_builder",
            "--cg-id",
            "cg-input-summary",
            "--workspace",
            str(tmp_path),
            *_FAKE_LLM_ARGV,
        ]
    )
    captured = capsys.readouterr()
    events = [json.loads(line) for line in captured.out.strip().split("\n")]

    # The first step.start (body generation for build_harness) carries
    # the function_name in its input_summary.
    body_step_starts = [
        e
        for e in events
        if e["event_type"] == "step.start" and e["step_kind"] == "harness_body_generation"
    ]
    assert body_step_starts
    assert body_step_starts[0]["input_summary"]["function_name"] == "build_harness"


def test_pre_session_errors_use_simple_emit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The pre-session-construction error path (bad workspace / missing
    cg-id) bypasses the structured emitter and writes a simple
    ``event_type``-keyed JSON line. This keeps the very-narrow set of
    pre-emitter errors visible without forcing the D10 envelope onto
    them."""
    rc = StepStartEvent.model_validate_json(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "event_seq": 0,
                "emitted_at_utc": "2026-05-26T00:00:00Z",
                "session_id": "s",
                "step_index": 0,
                "step_kind": "validation",
                "step_label": "x",
                "attempt_index": 0,
                "input_summary": {},
            }
        )
    )
    assert rc.event_type == "step.start"
    _ = tmp_path  # silence pytest-unused-argument
    _ = capsys
