"""Mini-orchestrator stdout-JSON status emit (D10).

Per architectural_decisions.md §7 ("sandbox produces, host attests-and-
persists"), the mini-orchestrator writes one JSON object per line to
stdout and to a workspace mirror file. The host worker reads via
``Compute.logs(handle)`` polling (mid-flight, best-effort) and via the
harvested mirror file (post-exit, byte-perfect retrospective).

Six event types per the D10 design note's catalog:

* ``session.start`` — first emit (event_seq=0). Carries the workflow_plan
  the generator produced so the host can pre-render the operator surface.
* ``step.start`` — fires once per step attempt (retries emit twice with
  different ``attempt_index``).
* ``step.turn`` — fires once per agent turn within an agent-reasoning
  step; rolling token counts on each event for live-cost surfacing.
* ``step.end`` — fires once per step attempt termination.
* ``session.cost`` — fires once per session, just before ``session.end``
  on clean exit. The authoritative token / cost reckoning.
* ``session.end`` — last lifecycle emit.

Forward-compat rules (host parser side; documented here as the schema's
trust contract):

1. Unknown ``event_type`` => structured log + line absorbed-and-dropped.
2. Unknown fields on a known event type => Pydantic ``extra="forbid"``
   validation error (sub-PR D's parser surfaces structurally; within-
   event additive changes bump :data:`SCHEMA_VERSION` explicitly).
3. Duplicate ``event_seq`` => silent drop (substrate buffering re-delivers
   tails per compute_dispatch_decisions §2).
4. Non-JSON lines => silent skip (container infrastructure logs may
   commingle on some substrates).

The :class:`StatusEmitter` class threads the writer / mirror file / event
sequence counter; the mini-orchestrator uses one instance per session.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# Bumped when an event-type's shape changes in a non-additive way.
# Additive changes (new event_type values; new optional fields) do not
# bump the version; the host parser ignores unknown event types and the
# host's own forward-compat policy is documented in the D10 design note.
SCHEMA_VERSION: int = 1

# Workspace mirror file path (relative to the session's workspace root).
# Companion to the stdout channel: byte-perfect even when substrate log
# buffers truncate. Host harvest pulls this back via the round-1
# bidirectional workspace-distribution surface.
EVENTS_MIRROR_RELPATH: str = "status/events.jsonl"


# === Literals ================================================================

WorkflowStepKind = Literal[
    "harness_body_generation",
    "technique_body_generation",
    "baseline_generation",
    "validation",
    "diagnose_on_failure",
    "manifest_emit",
    "apply_review_feedback",
]
"""D9 ``WorkflowStep.step_type`` discriminator values. Mirrors the
literals declared on each :class:`WorkflowStepBase` subclass in
``smai_agent_runtime.workflow.step_types``; adding a new step type is a
two-line change (the step's Literal + this Literal's tuple)."""

SandboxedAgentRole = Literal["harness_builder", "technique_implementer"]
"""The two roles whose mini-orchestrator emits status. Inline roles run
host-side and never emit through this channel."""

SessionOutcome = Literal["success", "failure", "aborted_by_supervisor", "timeout"]
StepOutcome = Literal["success", "failure", "skipped"]


# === Envelope ================================================================


class _BaseEvent(BaseModel):
    """Shared envelope fields. Present on every event_type."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = SCHEMA_VERSION
    event_seq: int = Field(..., ge=0)
    emitted_at_utc: str
    session_id: str


class WorkflowPlanItem(BaseModel):
    """One entry in the workflow plan list emitted on ``session.start``.

    Pre-rendered by the mini-orchestrator from
    ``generate_workflow(contract, role)`` before any step runs; lets the
    host's operator surface display "step 3 of 7" prior to the first
    ``step.start`` event.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_index: int = Field(..., ge=0)
    step_kind: WorkflowStepKind
    step_label: str


class SessionStartEvent(_BaseEvent):
    """First emit. Fires once per session, event_seq=0."""

    event_type: Literal["session.start"] = "session.start"
    agent_role: SandboxedAgentRole
    parent_kind: Literal["cg", "entry"]
    parent_id: str
    workflow_plan: list[WorkflowPlanItem]


class SessionEndEvent(_BaseEvent):
    """Last lifecycle emit. Fires once per session.

    On clean exit, :class:`SessionCostEvent` fires immediately before.
    On a crash the mini-orchestrator best-effort emits whatever event
    was prepared, and the host treats a missing ``session.end`` as a
    synthesized failure outcome.
    """

    event_type: Literal["session.end"] = "session.end"
    outcome: SessionOutcome
    last_completed_step_index: int | None = None
    failure_reason: str | None = None


class SessionCostEvent(_BaseEvent):
    """Final token / cost reckoning. Fires once per session.

    Lives in its own event (not folded into ``session.end``) so a future
    crash path can emit cost without forcing the success-vs-failure
    surface into the same line.
    """

    event_type: Literal["session.cost"] = "session.cost"
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    cache_read_tokens: int = Field(..., ge=0)
    cache_write_tokens: int = Field(..., ge=0)
    turn_count: int = Field(..., ge=0)
    tool_errors_fired: int = Field(..., ge=0)


class StepStartEvent(_BaseEvent):
    """Fires once per step attempt (so retries emit ``step.start`` twice
    with incremented ``attempt_index``)."""

    event_type: Literal["step.start"] = "step.start"
    step_index: int = Field(..., ge=0)
    step_kind: WorkflowStepKind
    step_label: str
    attempt_index: int = Field(default=0, ge=0)
    input_summary: dict[str, str] = Field(default_factory=dict)


class StepEndEvent(_BaseEvent):
    """Fires once per step attempt termination."""

    event_type: Literal["step.end"] = "step.end"
    step_index: int = Field(..., ge=0)
    step_kind: WorkflowStepKind
    outcome: StepOutcome
    failure_reason: str | None = None
    duration_seconds: float = Field(..., ge=0.0)
    captured_stderr: str | None = Field(
        default=None,
        description=(
            "Tail of the subprocess stderr the step captured on failure. "
            "Only ValidationStep populates this today (round-21 finding: "
            "the validation subprocess's stderr was captured to "
            "_StepOutcome.captured_stderr but never surfaced past the "
            "mini-orchestrator, leaving the host parser blind to the "
            "runner diagnostic). Mirrors the round-20 host-side "
            "container-stderr pattern at one level deeper."
        ),
    )


class StepTurnEvent(_BaseEvent):
    """Per-turn emit for agent-reasoning steps only.

    Scripted steps (validation, manifest_emit) do not emit ``step.turn``
    events because there is no per-turn surface inside a scripted step.
    """

    event_type: Literal["step.turn"] = "step.turn"
    step_index: int = Field(..., ge=0)
    step_kind: WorkflowStepKind
    turn_count: int = Field(..., ge=0)
    input_tokens_rolling: int = Field(..., ge=0)
    output_tokens_rolling: int = Field(..., ge=0)
    cache_read_tokens_rolling: int = Field(..., ge=0)
    last_tool_call: str | None = None


StatusEvent = Annotated[
    SessionStartEvent
    | SessionEndEvent
    | SessionCostEvent
    | StepStartEvent
    | StepEndEvent
    | StepTurnEvent,
    Field(discriminator="event_type"),
]
"""Discriminated union over every event type. The host parser binds
``TypeAdapter(StatusEvent)`` against this for O(1) parse dispatch."""


# === Emitter =================================================================


def _now_utc_iso() -> str:
    """ISO 8601 UTC timestamp.

    Mirrors the existing ``AgentSession.wall_clock_utc`` format the
    round-5 status surface already consumes (smai_cli/runtime.py).
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class StatusEmitter:
    """Single-session emit channel: stdout + workspace mirror file.

    Construct one per session; the mini-orchestrator holds the instance
    and calls ``emit_*`` for each lifecycle / step transition. Each emit
    increments :attr:`event_seq` and writes the line to both stdout and
    the mirror file (eager flush so polling sees progress).
    """

    def __init__(
        self,
        *,
        session_id: str,
        workspace: Path,
        stdout: IO[str] | None = None,
    ) -> None:
        self.session_id = session_id
        self.workspace = workspace
        self._stdout: IO[str] = stdout if stdout is not None else sys.stdout
        self.event_seq: int = 0
        self._mirror_path = workspace / EVENTS_MIRROR_RELPATH
        # Pre-create the mirror directory so the first emit's open() in
        # append mode does not race against missing parents.
        self._mirror_path.parent.mkdir(parents=True, exist_ok=True)

    # === Public emit methods =================================================

    def emit_session_start(
        self,
        *,
        agent_role: SandboxedAgentRole,
        parent_kind: Literal["cg", "entry"],
        parent_id: str,
        workflow_plan: list[WorkflowPlanItem],
    ) -> SessionStartEvent:
        event = SessionStartEvent(
            event_seq=self._next_seq(),
            emitted_at_utc=_now_utc_iso(),
            session_id=self.session_id,
            agent_role=agent_role,
            parent_kind=parent_kind,
            parent_id=parent_id,
            workflow_plan=workflow_plan,
        )
        self._write(event)
        return event

    def emit_step_start(
        self,
        *,
        step_index: int,
        step_kind: WorkflowStepKind,
        step_label: str,
        attempt_index: int = 0,
        input_summary: dict[str, str] | None = None,
    ) -> StepStartEvent:
        event = StepStartEvent(
            event_seq=self._next_seq(),
            emitted_at_utc=_now_utc_iso(),
            session_id=self.session_id,
            step_index=step_index,
            step_kind=step_kind,
            step_label=step_label,
            attempt_index=attempt_index,
            input_summary=input_summary or {},
        )
        self._write(event)
        return event

    def emit_step_end(
        self,
        *,
        step_index: int,
        step_kind: WorkflowStepKind,
        outcome: StepOutcome,
        duration_seconds: float,
        failure_reason: str | None = None,
        captured_stderr: str | None = None,
    ) -> StepEndEvent:
        event = StepEndEvent(
            event_seq=self._next_seq(),
            emitted_at_utc=_now_utc_iso(),
            session_id=self.session_id,
            step_index=step_index,
            step_kind=step_kind,
            outcome=outcome,
            failure_reason=failure_reason,
            duration_seconds=duration_seconds,
            captured_stderr=captured_stderr,
        )
        self._write(event)
        return event

    def emit_step_turn(
        self,
        *,
        step_index: int,
        step_kind: WorkflowStepKind,
        turn_count: int,
        input_tokens_rolling: int,
        output_tokens_rolling: int,
        cache_read_tokens_rolling: int,
        last_tool_call: str | None = None,
    ) -> StepTurnEvent:
        event = StepTurnEvent(
            event_seq=self._next_seq(),
            emitted_at_utc=_now_utc_iso(),
            session_id=self.session_id,
            step_index=step_index,
            step_kind=step_kind,
            turn_count=turn_count,
            input_tokens_rolling=input_tokens_rolling,
            output_tokens_rolling=output_tokens_rolling,
            cache_read_tokens_rolling=cache_read_tokens_rolling,
            last_tool_call=last_tool_call,
        )
        self._write(event)
        return event

    def emit_session_cost(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        turn_count: int,
        tool_errors_fired: int,
    ) -> SessionCostEvent:
        event = SessionCostEvent(
            event_seq=self._next_seq(),
            emitted_at_utc=_now_utc_iso(),
            session_id=self.session_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            turn_count=turn_count,
            tool_errors_fired=tool_errors_fired,
        )
        self._write(event)
        return event

    def emit_session_end(
        self,
        *,
        outcome: SessionOutcome,
        last_completed_step_index: int | None = None,
        failure_reason: str | None = None,
    ) -> SessionEndEvent:
        event = SessionEndEvent(
            event_seq=self._next_seq(),
            emitted_at_utc=_now_utc_iso(),
            session_id=self.session_id,
            outcome=outcome,
            last_completed_step_index=last_completed_step_index,
            failure_reason=failure_reason,
        )
        self._write(event)
        return event

    # === Internals ===========================================================

    def _next_seq(self) -> int:
        seq = self.event_seq
        self.event_seq += 1
        return seq

    def _write(self, event: BaseModel) -> None:
        """Serialize ``event`` to JSON and write to stdout + mirror file.

        Compact JSON (no whitespace beyond the field separators) keeps
        the per-line overhead small. Each emit flushes stdout so a host
        worker tailing logs sees progress immediately; the mirror file
        is written in append mode and flushed-and-closed per emit so a
        crash mid-session preserves the events written so far.
        """
        line = event.model_dump_json()
        # stdout: flush eagerly for live polling.
        self._stdout.write(line + "\n")
        self._stdout.flush()
        # Mirror file: append per-line.
        with self._mirror_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()


def _step_label_for_kind(step_kind: str, *, function_name: str | None = None) -> str:
    """Render a human-readable label for a workflow step.

    The mini-orchestrator passes the step's ``function_name`` (where
    applicable) so the operator-surface sample reads "fill build_harness"
    rather than the abstract "harness_body_generation."
    """
    if step_kind == "harness_body_generation":
        if function_name:
            return f"fill {function_name}"
        return "fill harness body"
    if step_kind == "technique_body_generation":
        return "fill technique body"
    if step_kind == "baseline_generation":
        return "fill baseline"
    if step_kind == "validation":
        return "validation smoke"
    if step_kind == "diagnose_on_failure":
        return "diagnose validation failure"
    if step_kind == "manifest_emit":
        return "emit harness manifest"
    if step_kind == "apply_review_feedback":
        return "apply review feedback"
    return step_kind


__all__ = [
    "EVENTS_MIRROR_RELPATH",
    "SCHEMA_VERSION",
    "SandboxedAgentRole",
    "SessionCostEvent",
    "SessionEndEvent",
    "SessionOutcome",
    "SessionStartEvent",
    "StatusEmitter",
    "StatusEvent",
    "StepEndEvent",
    "StepOutcome",
    "StepStartEvent",
    "StepTurnEvent",
    "WorkflowPlanItem",
    "WorkflowStepKind",
    "_step_label_for_kind",
]
