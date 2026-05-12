"""Multi-turn agent loop engine.

Per ``04-agents.md`` §3 (turn structure, between-turn deterministic
logic, tool execution semantics, debug logging). Custom Bedrock-Converse-style
loop, ~200 lines, built on the :class:`smai_core.plugins.LlmProvider`
plugin per DEC-005 / DEC-024 / DEC-025 (no LangGraph runtime, no AgentKit,
no Strands SDK, no Claude Agent SDK).

The loop's job is narrow:

1. Drive a multi-turn conversation by calling the
   :class:`smai_core.plugins.LlmProvider`-routed model for the agent's
   role.
2. Execute tool calls sequentially (per §3.3 — "most tools are I/O-bound
   and ordering matters for file ops").
3. Run the **between-turn deterministic logic** (context truncation,
   status writes, lint-on-write, validation polling, supervisor nudges)
   that v1 settled on (``agent_design.md`` §4) and that DEC-005
   explicitly preserves as the reason for the custom loop in the first
   place.

The loop owns sequencing only; domain logic lives in
:class:`smai_agents.tools.Tool` handlers and in
:mod:`smai_agents.between_turn`.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from smai_core.plugins import (
    ArtifactStore,
    CacheConfig,
    Compute,
    LlmProvider,
    ModelResponse,
    NormalizedMessage,
    TextContent,
    TokenUsage,
    ToolResultContent,
    ToolUseContent,
)

from smai_agents.cache import DEFAULT_CACHE_CONFIG
from smai_agents.model_selection import TaskRole
from smai_agents.tools import (
    FINISH_TOOL_NAME,
    ToolContext,
    ToolRegistry,
)
from smai_agents.truncation import TruncationPolicy

if TYPE_CHECKING:
    pass

_log = logging.getLogger(__name__)


class AgentLoopConfig(BaseModel):
    """Engine-side knobs (§3.1, §3.2).

    Per §3.2 / ``05-orchestrator.md`` §6 — these settings live on engine
    config so deployments override per dispatch.
    """

    model_config = ConfigDict(extra="forbid")

    max_turns: int = 50
    """Maximum LLM turns per session before the loop exits with
    :class:`AgentOutcome` ``kind='exhausted_turns'``. v1's value scaled
    with role; v2 ships a single default the orchestrator overrides per
    role. 50 matches v1's typical harness-builder budget."""

    max_output_tokens: int = 4096
    """Per-call ``max_tokens`` passed through to
    :meth:`smai_core.plugins.LlmProvider.call`. Capability-clamped by the
    plugin (``LlmCapabilities.max_output_tokens``)."""

    status_write_every_turns: int = 1
    """Per §3.2 — "v1 wrote on every turn; v2 inherits this default and
    exposes it as engine config." Set to ``0`` to disable status writes
    entirely."""

    supervisor_nudge_check_every_turns: int = 1
    """How often the loop polls for a supervisor nudge file. Set to
    ``0`` to disable."""

    supervisor_check_every_turns: int = 0
    """How often the loop runs the supervisor between-turn check (Task
    3.G4 / ``04-agents.md`` §2.6). Default ``0`` (disabled at the loop
    level — the supervisor needs an :class:`LlmProvider` for the
    ``supervisor`` role in :attr:`AgentSession.llm_providers`, so an
    ad-hoc test session that doesn't wire one shouldn't accidentally
    activate the check). The orchestrator's dispatch handlers translate
    :class:`smai_orchestrator.engine.config.EngineConfig.supervisor_enabled`
    + :class:`...EngineConfig.supervisor_check_every_n_turns` into this
    field at session-construction time."""

    supervisor_recent_snapshot_count: int = 5
    """How many recent status snapshots the supervisor's input bundle
    carries. Bounded so a long-running session doesn't unboundedly
    grow :attr:`AgentSession.recent_status_snapshots`."""

    supervisor_recent_tool_call_count: int = 10
    """How many recent tool-call names the supervisor's input bundle
    carries. Same bound rationale as :attr:`supervisor_recent_snapshot_count`."""

    truncation_check_every_turns: int = 1
    """How often the loop checks the truncation threshold. Set to ``0``
    to disable."""

    persist_conversation_trace: bool = True
    """Whether :func:`run_loop` serializes :attr:`AgentSession.messages`
    to :attr:`AgentSession.conversation_trace_artifact_path` on loop
    exit (per ``04-agents.md`` §3.1 / DEC-023 — the trace the
    technique-implementer's retry context (:func:`retry_context.load_retry_context`)
    reads). No-op when the path or :attr:`AgentSession.artifact_store`
    is ``None``. Set ``False`` to disable (unit tests that don't care)."""


class AgentOutcome(BaseModel):
    """Discriminated-union return type for :func:`run_loop`.

    Four kinds per §3.1:

    * ``finished`` — agent called the ``finish`` tool (success path).
    * ``finished_without_tool_use`` — model returned ``stop_reason=end_turn``
      without invoking any tool. Per §3.1 this is a stopping condition,
      not necessarily failure (a planning agent that has nothing more to
      say after summarizing can land here).
    * ``truncated_output`` — model hit ``stop_reason=max_tokens``. Not
      itself terminal in v1, but the loop returns and lets the caller
      decide whether to resume.
    * ``exhausted_turns`` — turn budget reached without ``finish``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str
    turn_count: int
    usage_total: TokenUsage
    finish_summary: str | None = None
    finish_success: bool | None = None
    supervisor_reason: str | None = None
    """Populated only when ``kind == 'aborted_by_supervisor'`` (Task
    3.G4) — carries the supervisor's :attr:`SupervisorDecision.reason`
    so the dispatch handler can route the parent entity to a
    ``*_failed`` state with a human-readable diagnostic. ``None`` for
    every other kind."""


class AgentSession(BaseModel):
    """In-memory state for one dispatched agent invocation.

    Per §3.1: "State lives in :class:`AgentSession` (the conversation,
    the LlmProvider instance, the prompt config, accumulated usage, the
    workspace path); transitions happen through tool execution and
    through the between-turn logic; the loop itself owns sequencing but
    no domain logic."

    Per §3.1 paragraph 2: "AgentSession is pipeline-layer state,
    distinct from pipeline-tracking entities — it lives in memory for
    the duration of the dispatched job and is not persisted to
    MetadataStore." What gets persisted is the ``conversation-trace.json``
    (an :class:`ArtifactStore` write at session end and on retry
    boundaries) and the per-call usage aggregation (rolled up into v1's
    ``AgentSessionRecord`` shape per §4.6 of 07).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    system_prompt: str
    messages: list[NormalizedMessage] = Field(default_factory=list[NormalizedMessage])
    tools: ToolRegistry
    llm_providers: dict[TaskRole, LlmProvider]
    """Per §4: "One plugin instance per role per dispatched job, not a
    shared global." The orchestrator's dispatch handler (Task 2.C4)
    constructs one :class:`smai_core.plugins.LlmProvider` per per-task
    model and hands the dict in here."""

    current_role: TaskRole
    """Which role's :class:`LlmProvider` is in use for this session.
    The loop reads ``llm_providers[current_role]`` once per turn."""

    cache_config: CacheConfig = DEFAULT_CACHE_CONFIG
    truncation_policy: TruncationPolicy = Field(default_factory=TruncationPolicy)

    workspace_path: Path
    """Local disk path the agent's tools (file ops, ``execute``) operate
    against. Per ``05-orchestrator.md`` §1.4 the dispatch handler
    establishes the workspace before invoking the loop."""

    artifact_store: ArtifactStore | None = None
    """Optional ArtifactStore for status writes, manifest emission,
    nudge-file polling. ``None`` is permitted for unit tests; the
    loop's between-turn logic skips status writes and nudge polling
    when absent."""

    compute: Compute | None = None
    """Optional Compute plugin for ``run_experiment``-style tools.
    The loop itself doesn't invoke Compute; tool handlers do."""

    status_artifact_path: str | None = None
    """Path under which :func:`smai_agents.between_turn.write_status`
    writes the per-turn status JSON. Per ``04-agents.md`` §3.2 the
    layout is owned by the orchestrator's dispatch handler."""

    nudge_artifact_path: str | None = None
    """Path under which :func:`smai_agents.between_turn.maybe_check_supervisor_nudge`
    looks for a supervisor nudge. Same provenance as
    :attr:`status_artifact_path`."""

    conversation_trace_artifact_path: str | None = None
    """ArtifactStore key under which :func:`run_loop` persists the
    serialized conversation (:attr:`messages`) on loop exit (per
    ``04-agents.md`` §3.1's "the conversation trace is persisted at
    session end"). Per role: planner →
    ``proposals/{proposal_id}/conversation-trace.json``; harness
    builder → ``comparison-groups/{cg_id}/harness/conversation-trace.json``;
    technique implementer →
    ``comparison-groups/{cg_id}/entries/{entry_id}/conversation-trace.json``
    (the key :func:`smai_agents.retry_context.load_retry_context`
    reads on a retry). ``None`` disables persistence; the dispatch
    handler sets it."""

    last_tool_call: str | None = None
    """Name of the most-recently-invoked tool (set in
    :func:`_execute_tool_uses` for every ``tool_use`` block). Surfaced
    in the per-turn status payload so observability consumers can see
    "what is the agent doing right now"."""

    last_tool_error: str | None = None
    """Content string of the most-recent ``is_error=True`` tool result
    (any of the four error branches in :func:`_execute_tool_uses`).
    ``None`` until the first tool error. Lets the supervisor + status
    readers see the last failure without the conversation trace."""

    tool_errors_fired: int = 0
    """Count of ``is_error=True`` tool results across the session.
    Diagnostic counter alongside :attr:`truncations_fired` etc."""

    attempt_index: int | None = None
    """Which (re-)attempt this dispatched session is, for roles that
    carry an attempt counter (technique implementer:
    ``implementation_attempt``). ``None`` for roles with no attempt
    notion (harness builder, planner). Set by the dispatch handler /
    session constructor; surfaced in the status payload."""

    progress_sink: Callable[[AgentSession], Awaitable[None]] | None = Field(
        default=None, exclude=True
    )
    """Persistence-agnostic per-turn callback. Fired by
    :func:`smai_agents.between_turn.write_status` on the same cadence as
    status writes (every turn by default). The loop must NOT import any
    :class:`MetadataStore` type; the orchestrator's dispatch handler
    supplies a closure that UPDATEs the ``agent_sessions`` row. ``None``
    in unit tests / Tier-B usage that doesn't track sessions."""

    turn_count: int = 0
    usage_total: TokenUsage = Field(
        default_factory=lambda: TokenUsage(input_tokens=0, output_tokens=0)
    )
    nudges_consumed: int = 0
    truncations_fired: int = 0
    supervisor_checks_fired: int = 0
    """Diagnostic counters per §3.4's debug logging."""

    recent_status_snapshots: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    """Bounded ring of the most-recent status payloads the loop's
    between-turn :func:`smai_agents.between_turn.write_status`
    produced. Trimmed to
    :attr:`AgentLoopConfig.supervisor_recent_snapshot_count` on append.
    Consumed by the supervisor's between-turn hook (Task 3.G4) so the
    supervisor sees the structured signal without reading
    :class:`ArtifactStore`."""

    recent_tool_call_names: list[str] = Field(default_factory=list[str])
    """Bounded ring of recently invoked tool names (most-recent-last).
    Trimmed to
    :attr:`AgentLoopConfig.supervisor_recent_tool_call_count` on append.
    Lets the supervisor (Task 3.G4) detect "agent stuck reading the
    same file repeatedly" without seeing the full conversation trace."""

    supervisor_aborted: bool = False
    """Set to ``True`` by
    :func:`smai_agents.between_turn.maybe_run_supervisor_check` when
    the supervisor returned ``action='abort'``. The loop checks this
    flag after every between-turn cycle and exits with
    :class:`AgentOutcome` ``kind='aborted_by_supervisor'``."""

    supervisor_abort_reason: str | None = None
    """The :attr:`SupervisorDecision.reason` from the abort decision,
    propagated onto the terminal :class:`AgentOutcome`. ``None`` when
    :attr:`supervisor_aborted` is ``False``."""

    config: AgentLoopConfig = Field(default_factory=AgentLoopConfig)

    time_provider: Callable[[], float] = Field(default=time.monotonic, exclude=True)
    """Per Task 1.9 carry-forward: tests inject a fake clock through
    this seam so 2.C1's eventual fake-clock injection works through the
    loop too. Status writes and any timed between-turn logic should call
    ``session.time_provider()`` rather than using
    :func:`time.monotonic` / :func:`datetime.datetime.now` directly."""

    @property
    def llm(self) -> LlmProvider:
        """The active :class:`LlmProvider` for the current role."""
        return self.llm_providers[self.current_role]

    def append_assistant(self, message: NormalizedMessage) -> None:
        """Append the latest assistant turn to the conversation."""
        self.messages.append(message)

    def append_user_with_tool_results(self, results: list[ToolResultContent]) -> None:
        """Append a user-role message bundling tool_result blocks.

        Per the Bedrock Converse / normalized message shape, tool
        results return as ``role=user, content=[ToolResultContent...]``
        (matching Anthropic's Messages-API convention).
        """
        # Mypy/Pyright: list-invariance — explicit cast at construction.
        self.messages.append(NormalizedMessage(role="user", content=list(results)))

    def aggregate_usage(self, usage: TokenUsage) -> None:
        """Per §3.4: roll per-call usage into the session total."""
        self.usage_total = TokenUsage(
            input_tokens=self.usage_total.input_tokens + usage.input_tokens,
            output_tokens=self.usage_total.output_tokens + usage.output_tokens,
            cache_read_tokens=self.usage_total.cache_read_tokens + usage.cache_read_tokens,
            cache_write_tokens=self.usage_total.cache_write_tokens + usage.cache_write_tokens,
        )


# Rebuild :class:`ToolContext` now that :class:`AgentSession` exists in
# the runtime namespace. The forward-reference (``"AgentSession"`` on
# ``tools.ToolContext.session``) requires Pydantic to see the real
# class to build its validator.
ToolContext.model_rebuild()


# === Loop driver (§3.1) =====================================================


async def run_loop(session: AgentSession) -> AgentOutcome:
    """Drive a multi-turn agent conversation.

    Returns when the agent calls ``finish`` or hits a stopping condition
    (``stop_reason='end_turn'`` without tool use, ``stop_reason='max_tokens'``,
    or ``max_turns`` reached). Per §3.1; pseudocode is followed
    structurally.

    The between-turn deterministic logic lives in
    :mod:`smai_agents.between_turn`; the import is local to break the
    module-import cycle (``between_turn`` types its argument as
    :class:`AgentSession`).

    On exit (any kind, including an exception) the conversation trace is
    persisted to :attr:`AgentSession.conversation_trace_artifact_path`
    when configured (per ``04-agents.md`` §3.1 / DEC-023).
    """
    try:
        return await _run_turn_loop(session)
    finally:
        await _persist_conversation_trace(session)


async def _persist_conversation_trace(session: AgentSession) -> None:
    """Serialize :attr:`AgentSession.messages` to ArtifactStore.

    No-op when persistence is disabled
    (:attr:`AgentLoopConfig.persist_conversation_trace`) or when the
    path / store seam is absent. Best-effort: a failed trace write is
    logged-and-swallowed — observability persistence must never mask
    the loop's actual outcome (or turn a clean exit into a crash)."""
    if not session.config.persist_conversation_trace:
        return
    key = session.conversation_trace_artifact_path
    store = session.artifact_store
    if key is None or store is None:
        return
    try:
        trace = [json.loads(m.model_dump_json()) for m in session.messages]
        await store.put(key, json.dumps(trace, indent=2).encode("utf-8"))
    except Exception:  # noqa: BLE001 — trace persistence is best-effort
        _log.exception("failed to persist conversation trace to %s", key)


async def _run_turn_loop(session: AgentSession) -> AgentOutcome:
    """The turn loop proper — see :func:`run_loop`."""
    # Imported here, not at module top, to avoid the
    # loop ↔ between_turn module-import cycle: the between-turn
    # module references AgentSession at runtime.
    from smai_agents import between_turn  # noqa: PLC0415

    tool_definitions = session.tools.to_provider_definitions()
    finish_tool_present = FINISH_TOOL_NAME in session.tools

    while True:
        # ---- Pre-turn: between-turn deterministic logic (§3.2) ----
        if (
            session.config.truncation_check_every_turns > 0
            and session.turn_count % session.config.truncation_check_every_turns == 0
        ):
            await between_turn.maybe_truncate_context(session)
        if (
            session.config.supervisor_nudge_check_every_turns > 0
            and session.turn_count % session.config.supervisor_nudge_check_every_turns == 0
        ):
            await between_turn.maybe_check_supervisor_nudge(session)
        if (
            session.config.status_write_every_turns > 0
            and session.turn_count % session.config.status_write_every_turns == 0
        ):
            await between_turn.write_status(session)
        if (
            session.config.supervisor_check_every_turns > 0
            and session.turn_count > 0
            and session.turn_count % session.config.supervisor_check_every_turns == 0
        ):
            await between_turn.maybe_run_supervisor_check(session)
            if session.supervisor_aborted:
                return AgentOutcome(
                    kind="aborted_by_supervisor",
                    turn_count=session.turn_count,
                    usage_total=session.usage_total,
                    supervisor_reason=session.supervisor_abort_reason,
                )

        # ---- Turn: model call through the LlmProvider plugin ----
        response: ModelResponse = await session.llm.call(
            system=session.system_prompt,
            messages=session.messages,
            tools=tool_definitions or None,
            max_tokens=session.config.max_output_tokens,
            cache_config=session.cache_config,
        )
        session.turn_count += 1
        session.aggregate_usage(response.usage)

        # ---- Post-turn: parse stop reason, execute tools (§3.1) ----
        if response.stop_reason == "end_turn":
            session.append_assistant(response.message)
            return AgentOutcome(
                kind="finished_without_tool_use",
                turn_count=session.turn_count,
                usage_total=session.usage_total,
            )
        if response.stop_reason == "max_tokens":
            session.append_assistant(response.message)
            return AgentOutcome(
                kind="truncated_output",
                turn_count=session.turn_count,
                usage_total=session.usage_total,
            )

        # ``stop_reason == "tool_use"``
        session.append_assistant(response.message)
        finish_payload, tool_results = await _execute_tool_uses(response.message, session)
        session.append_user_with_tool_results(tool_results)

        if finish_tool_present and finish_payload is not None:
            return AgentOutcome(
                kind="finished",
                turn_count=session.turn_count,
                usage_total=session.usage_total,
                finish_summary=finish_payload.get("summary"),
                finish_success=finish_payload.get("success"),
            )

        if session.turn_count >= session.config.max_turns:
            return AgentOutcome(
                kind="exhausted_turns",
                turn_count=session.turn_count,
                usage_total=session.usage_total,
            )


async def _execute_tool_uses(
    assistant_message: NormalizedMessage,
    session: AgentSession,
) -> tuple[dict[str, Any] | None, list[ToolResultContent]]:
    """Execute every ``tool_use`` block in the assistant message.

    Sequential per §3.3 — "most tools are I/O-bound and ordering matters
    for file ops." Returns the parsed ``finish`` payload (if any was
    invoked) plus the full list of :class:`ToolResultContent` to append
    to the conversation. The loop uses the finish payload to populate
    :class:`AgentOutcome` with the agent's stated success/summary.
    """
    context = ToolContext(
        workspace_path=session.workspace_path,
        artifact_store=session.artifact_store,
        compute=session.compute,
        session=session,
    )

    finish_payload: dict[str, Any] | None = None
    results: list[ToolResultContent] = []

    for block in assistant_message.content:
        if not isinstance(block, ToolUseContent):
            continue

        # Record the tool-call name for the supervisor's between-turn
        # check (Task 3.G4). Bounded ring; trim to the configured cap.
        session.recent_tool_call_names.append(block.name)
        cap = session.config.supervisor_recent_tool_call_count
        if cap > 0 and len(session.recent_tool_call_names) > cap:
            del session.recent_tool_call_names[: len(session.recent_tool_call_names) - cap]
        # Most-recent single-valued mirror for the status payload.
        session.last_tool_call = block.name

        tool = session.tools.get(block.name)
        if tool is None:
            _record_tool_error(session, f"unknown tool: {block.name!r}")
            results.append(
                ToolResultContent(
                    tool_use_id=block.id,
                    content=f"unknown tool: {block.name!r}",
                    is_error=True,
                )
            )
            continue

        try:
            parsed = tool.input_schema.model_validate(block.input)
        except ValidationError as exc:
            # Per §3.3: "tool errors are tool results, not loop errors."
            # An input that fails schema validation surfaces back to
            # the agent as an is_error tool_result; the agent can react
            # on the next turn.
            content = f"input validation failed for tool {block.name!r}: {exc}"
            _record_tool_error(session, content)
            results.append(
                ToolResultContent(
                    tool_use_id=block.id,
                    content=content,
                    is_error=True,
                )
            )
            continue

        try:
            result = await tool.handler(parsed, context)
        except Exception as exc:  # noqa: BLE001 — surface every error as tool_result
            content = f"tool {block.name!r} raised {type(exc).__name__}: {exc}"
            _record_tool_error(session, content)
            results.append(
                ToolResultContent(
                    tool_use_id=block.id,
                    content=content,
                    is_error=True,
                )
            )
            continue

        # Backfill ``tool_use_id`` — handlers don't see the assistant's
        # block id, the loop does.
        result = result.model_copy(update={"tool_use_id": block.id})

        for hook in tool.post_result_hooks:
            result = await hook(tool, parsed, result, context)

        if result.is_error:
            _record_tool_error(session, result.content)

        results.append(result)

        if tool.name == FINISH_TOOL_NAME and finish_payload is None:
            # Capture the finish payload for the loop's outcome. If the
            # agent invoked finish multiple times in one turn (unusual
            # but legal under the protocol), the first call wins —
            # matches v1's behavior (``packages/agents/src/loop/engine.ts``).
            finish_payload = {
                "success": _safe_get_bool(parsed, "success"),
                "summary": _safe_get_str(parsed, "summary"),
            }

    return finish_payload, results


def _record_tool_error(session: AgentSession, content: str) -> None:
    """Record the latest ``is_error`` tool-result content on the session.

    Mirrors the four error branches in :func:`_execute_tool_uses`
    (unknown tool, input-validation failure, handler raised, handler
    returned ``is_error``) into the single-valued status fields.
    """
    session.last_tool_error = content
    session.tool_errors_fired += 1


def _safe_get_bool(model: BaseModel, field: str) -> bool | None:
    value = getattr(model, field, None)
    return value if isinstance(value, bool) else None


def _safe_get_str(model: BaseModel, field: str) -> str | None:
    value = getattr(model, field, None)
    return value if isinstance(value, str) else None


# `TextContent` is re-exported for convenient session-construction
# call sites without forcing an extra `smai_core.plugins` import.
__all__ = [
    "AgentLoopConfig",
    "AgentOutcome",
    "AgentSession",
    "TextContent",
    "run_loop",
]
