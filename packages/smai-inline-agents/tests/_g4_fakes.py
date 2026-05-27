"""Cross-test fixtures for Task 3.G4 (supervisor agent).

* :func:`canned_supervisor_decision` — :class:`ModelResponse` builder
  for the supervisor's ``submit_decision`` tool-use payload.
* :func:`stuck_planner_response` — a canned tool-use response that the
  stuck-agent fixture replays turn after turn (same tool, same input).
"""

from __future__ import annotations

from typing import Any, Literal

from _agent_helpers import model_response  # type: ignore[import-not-found]
from smai_core.plugins import ModelResponse


def canned_supervisor_decision(
    *,
    action: Literal["continue", "intervene", "abort"],
    reason: str,
    nudge: str | None = None,
    tool_use_id: str = "sup-tu-1",
) -> ModelResponse:
    """Build a :class:`ModelResponse` carrying a supervisor decision.

    The supervisor's prompt-config base ships
    ``structured_output_tool.name = "submit_decision"`` and the schema
    is :class:`smai_inline_agents.SupervisorDecision`.
    """
    payload: dict[str, Any] = {"action": action, "reason": reason}
    if nudge is not None:
        payload["nudge"] = nudge
    return model_response(
        tool_uses=[(tool_use_id, "submit_decision", payload)],
        stop_reason="tool_use",
    )


def stuck_agent_loop_response(
    *,
    tool_name: str = "read_file",
    tool_input: dict[str, Any] | None = None,
    tool_use_id: str = "stuck-tu",
) -> ModelResponse:
    """A loop-side response that the stuck-agent fixture replays.

    The same tool call repeated turn after turn produces an identical
    ``recent_tool_call_names`` ring + identical-aside-from-turn-count
    status snapshots — the supervisor's signal for "agent isn't
    progressing." Tests build a `StubLlmProvider` whose queue contains
    several copies of this response.
    """
    return model_response(
        tool_uses=[
            (
                tool_use_id,
                tool_name,
                tool_input if tool_input is not None else {"path": "harness/train.py"},
            )
        ],
        stop_reason="tool_use",
    )


__all__ = ["canned_supervisor_decision", "stuck_agent_loop_response"]
