"""Supervisor single-call agent — ``04-agents.md`` §2.6 / §15 OQ2.

Per §2.6: single-call structured-output agent invoked between turns of
the multi-turn agents (planner, harness builder, technique implementer)
to consume a status snapshot + recent activity summary and decide
whether the running agent should continue, receive a corrective nudge,
or be aborted.

Same shape as the other single-call agents (code reviewer / contextual
evaluator) — :class:`SupervisorInput` Pydantic model + ``run_*(...)``
async function returning the locked :class:`SupervisorDecision` schema.
The wrapper itself does NOT touch the engine, the ``MetadataStore``, or
the ``ArtifactStore``; it consumes the inputs the between-turn hook
assembled and produces the structured decision. The hook
(:func:`smai_inline_agents.between_turn.maybe_run_supervisor_check`) translates
the decision into a side effect — write a nudge file
(``action='intervene'``) or set a session-abort flag
(``action='abort'``).

The supervisor does **not** see the full :file:`conversation-trace.json`
per §2.6: it consumes a structured summary (recent status snapshots,
recent tool-call names, agent-role identity, turn budget) — not the
conversation history.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from smai_core.plugins import LlmProvider

from smai_inline_agents.model_selection import TaskRole
from smai_inline_agents.prompts import PromptConfig, load_prompt_config
from smai_inline_agents.schemas.supervisor import SupervisorDecision
from smai_inline_agents.structured_call import structured_call

_SUPERVISOR_ROLE: TaskRole = "supervisor"


class SupervisorInput(BaseModel):
    """Assembled inputs to :func:`run_supervisor_check` (hook's view).

    Per §2.6 "Inputs": agent role identity, current turn / budget,
    recent status snapshots (NOT the full conversation trace), recent
    tool-call summary, and the optional signal-kind tag identifying
    *what* triggered the check (``'periodic'``, ``'no_progress'``, etc.).

    Per §15 OQ2 the *signal kinds* are an open vocabulary — ``signal_kind``
    is a free-text string the hook populates from the configured
    cadence policy. Phase-3 ships ``'periodic'`` (the only
    cadence-driven kind v2 currently emits); future stuck-detection
    signals (``'status_unchanged_25m'`` / ``'same_error_3x'`` / etc.)
    fit the same string surface.
    """

    model_config = ConfigDict(extra="forbid")

    agent_role: TaskRole
    """The supervised agent's role — e.g., ``'harness_builder'`` /
    ``'technique_implementer'`` / ``'planner'``. The supervisor's prompt
    uses this to frame its decision criteria."""

    entity_id: str
    """Identifier for the entity the supervised agent is working on
    (CG id for the harness builder, entry id for the technique
    implementer, proposal id for the planner). Surfaced verbatim in
    the user message for traceability."""

    current_turn: int
    """Current turn count of the supervised loop. Per
    :class:`smai_inline_agents.AgentSession.turn_count`."""

    turn_budget: int
    """Maximum turns the loop is configured for. Per
    :attr:`smai_inline_agents.AgentLoopConfig.max_turns`."""

    recent_status_snapshots: list[dict[str, object]] = Field(
        default_factory=list[dict[str, object]]
    )
    """Recent status payloads the loop's :func:`write_status` produced,
    most-recent-last. Bounded to the last N (the hook trims). Per §2.6
    the supervisor consumes the structured summary, not the full
    conversation history."""

    recent_tool_calls: list[str] = Field(default_factory=list[str])
    """Names of recently invoked tools (most-recent-last) — gives the
    supervisor a sense of "is the agent stuck reading the same file?"
    without the supervisor needing to see the full conversation trace."""

    signal_kind: str = "periodic"
    """Free-text tag for what triggered this check. Phase-3 ships
    ``'periodic'`` (cadence-driven); §15 OQ2 leaves the open vocabulary
    for stuck-detection signals to a future DEC."""


async def run_supervisor_check(
    *,
    llm: LlmProvider,
    input: SupervisorInput,
    prompt_config: PromptConfig | None = None,
) -> SupervisorDecision:
    """Run the supervisor agent for one between-turn check.

    Single-call structured output via :func:`structured_call`; no
    multi-turn loop. The function is purely a wrapper:

    1. Resolve ``prompt_config`` — caller-supplied or, when ``None``,
       :func:`smai_inline_agents.prompts.load_prompt_config` for the
       ``supervisor`` role (per ``04-agents.md`` §10).
    2. Read the structured-output tool's ``name`` + ``description``
       from :attr:`PromptConfig.structured_output_tool` (§10.2).
    3. Render the user message from ``input``.
    4. Call :func:`structured_call` with
       ``output_schema=SupervisorDecision``.
    5. Return the parsed decision.

    The function inherits :func:`structured_call`'s discipline:
    retry-once-on-text-or-schema-invalid response per DEC-018, and
    :class:`smai_inline_agents.StructuredCallFailed` on second-attempt
    failure (no silent fallback). The :class:`SupervisorDecision`
    validator enforces the ``intervene`` ↔ ``nudge`` invariant.
    """
    cfg = prompt_config if prompt_config is not None else load_prompt_config(role=_SUPERVISOR_ROLE)
    sot = cfg.structured_output_tool
    if sot is None:
        raise ValueError(
            "supervisor prompt config is missing structured_output_tool; "
            "single-call agents require it per 04-agents.md §10.2"
        )
    user_message = _build_user_message(input)

    return await structured_call(
        llm=llm,
        system=cfg.system_prompt,
        user_message=user_message,
        output_schema=SupervisorDecision,
        tool_name=sot.name,
        tool_description=sot.description,
    )


def _build_user_message(inp: SupervisorInput) -> str:
    """Render ``inp`` into the LLM's user-message text.

    Mirrors v1's :file:`packages/agents/src/supervisor.ts` structure —
    role identity + turn budget + recent status snapshots + recent
    tool-call summary — with v2-shaped inputs.
    """
    parts: list[str] = []
    parts.append(f"# Supervisor Check — agent_role: {inp.agent_role}")
    parts.append(f"## Entity: {inp.entity_id}")
    parts.append("")
    parts.append("## Progress")
    parts.append(f"- current_turn: {inp.current_turn}")
    parts.append(f"- turn_budget: {inp.turn_budget}")
    parts.append(f"- turns_remaining: {max(inp.turn_budget - inp.current_turn, 0)}")
    parts.append(f"- signal_kind: {inp.signal_kind}")
    parts.append("")
    parts.append("## Recent Status Snapshots (oldest first)")
    if not inp.recent_status_snapshots:
        parts.append("(no status snapshots recorded yet)")
    else:
        for idx, snap in enumerate(inp.recent_status_snapshots):
            parts.append(f"### Snapshot {idx + 1}")
            parts.append("```json")
            parts.append(_format_snapshot(snap))
            parts.append("```")
    parts.append("")
    parts.append("## Recent Tool Calls (oldest first)")
    if not inp.recent_tool_calls:
        parts.append("(no tool calls recorded yet)")
    else:
        for name in inp.recent_tool_calls:
            parts.append(f"- {name}")
    parts.append("")
    parts.append(
        "Decide one of: continue (the agent is making progress); "
        "intervene (the agent is stuck or off-track and a corrective "
        "nudge would unstick it); abort (the agent cannot recover and "
        "should be terminated). Always set reason. Set nudge iff "
        "action='intervene'."
    )
    return "\n".join(parts)


def _format_snapshot(snap: dict[str, object]) -> str:
    """Render a status-snapshot dict as deterministic JSON-ish text.

    Avoid ``json.dumps`` because the snapshot may contain non-JSON-safe
    values; we render keys in sorted order with ``repr`` of each value
    so the LLM sees stable text it can compare across snapshots.
    """
    lines: list[str] = []
    for key in sorted(snap):
        lines.append(f"  {key}: {snap[key]!r}")
    return "{\n" + "\n".join(lines) + "\n}"


__all__ = [
    "SupervisorInput",
    "run_supervisor_check",
]
