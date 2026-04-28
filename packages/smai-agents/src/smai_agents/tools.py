"""Tool registry primitives for the multi-turn agent loop.

Per ``04-agents.md`` §3.3 and §12. This module ships **structural**
primitives only — :class:`Tool`, :class:`ToolContext`,
:class:`ToolRegistry`, :class:`ToolResultHook`, the loop-exit
:func:`make_finish_tool` factory, and the :func:`truncate_tool_output`
utility. Specific domain tools (``read_file`` / ``write_file`` /
``edit_file`` / ``execute`` / ``search`` / ``list_files`` /
``run_experiment`` / ``emit_harness_manifest``) ship in Task 2.B2 /
2.B3.

Per §3.3:

* Tool execution is **sequential** within an assistant tool_use block.
* Tool errors become :class:`ToolResultContent` with ``is_error=True``,
  not loop crashes.
* :func:`finish` is the only loop-exit tool.
* Aggressive output truncation for ``execute``-shaped tools — last 200
  lines, capped at 5000 tokens — is a utility, not embedded in the
  loop.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field
from smai_core.plugins import (
    ArtifactStore,
    Compute,
    ToolDefinition,
    ToolResultContent,
)

if TYPE_CHECKING:
    from smai_agents.loop import AgentSession

# v1 settled the cap at 200 lines / 5000 tokens (``agent_design.md`` §4
# / 04-agents.md §3.3 / §12.2). Inherited verbatim.
EXECUTE_MAX_LINES = 200
EXECUTE_MAX_TOKENS = 5000

# Char-count heuristic — same factor used in :mod:`smai_agents.truncation`.
_CHARS_PER_TOKEN = 4

# The handler signature is type-erased on its input — the loop validates
# the JSON payload against ``input_schema`` before dispatching, so the
# handler always receives an instance of its declared schema. Type-erasure
# at the registry boundary lets one :class:`ToolRegistry` hold tools with
# heterogeneous input types.
ToolHandler = Callable[[BaseModel, "ToolContext"], Awaitable[ToolResultContent]]


class ToolContext(BaseModel):
    """Per-call context the loop hands to each tool handler.

    Tools that touch the workspace, ArtifactStore, or Compute use this
    rather than reaching for module globals. The :class:`AgentSession`
    is included as a back-reference (forward-declared via
    ``TYPE_CHECKING``) so handlers can append to the conversation if
    they want — e.g., ``finish`` doesn't need it but a future tool that
    wants to push a structured note into the conversation does.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    workspace_path: Path
    artifact_store: ArtifactStore | None = None
    compute: Compute | None = None
    session: AgentSession


@runtime_checkable
class ToolResultHook(Protocol):
    """Hook signature run after a tool handler returns.

    Per §3.2's lint-on-write description: "the linter output is
    appended to the tool result." A hook receives the tool that ran,
    the parsed input, the result, and the context; it returns the
    (possibly-mutated) result.

    Typing-only — :class:`Tool` stores its hook list as plain
    callables (``_ToolResultHookCallable``); this Protocol exists so
    consumers can declare hook implementations against a typed
    surface without subclassing a class.
    """

    async def __call__(
        self,
        tool: Tool,
        parsed_input: BaseModel,
        result: ToolResultContent,
        context: ToolContext,
    ) -> ToolResultContent: ...


# Pydantic v2 cannot validate a :class:`Protocol` field via ``isinstance``,
# so the field annotation uses a plain :class:`Callable` alias. The
# :class:`ToolResultHook` Protocol above documents the same shape for
# consumers writing hooks.
_ToolResultHookCallable = Callable[
    ["Tool", BaseModel, ToolResultContent, ToolContext],
    Awaitable[ToolResultContent],
]


class Tool(BaseModel):
    """A registered tool the agent can invoke.

    The shape is Pydantic so a registry of tools can be inspected,
    serialized for prompt-trace logging, and validated at registration
    time. The ``handler`` is an ``async`` callable receiving the
    parsed Pydantic input and a :class:`ToolContext`. Tool errors are
    returned as a :class:`ToolResultContent` with ``is_error=True``
    rather than raised — per §3.3, "tool errors are tool results, not
    loop errors."

    The ``post_result_hooks`` list lets B2 register the
    lint-on-write hook (see
    :func:`smai_agents.between_turn.lint_after_python_write`) without
    B1 knowing about the specific hook.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    description: str
    input_schema: type[BaseModel]
    handler: ToolHandler
    post_result_hooks: list[_ToolResultHookCallable] = Field(
        default_factory=list[_ToolResultHookCallable]
    )

    def to_provider_definition(self) -> ToolDefinition:
        """Render the registered tool as a provider-shaped tool definition.

        The :class:`smai_core.plugins.LlmProvider` Protocol's ``call``
        accepts ``list[ToolDefinition]``; the agent loop calls this once
        per session to materialize the wire-shaped tool list once
        (tools are static within a session).
        """
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema.model_json_schema(),
        )


class ToolRegistry:
    """Mutable map of tool name → :class:`Tool`.

    Tools can be added at any time but only *before* the loop starts —
    the wire-shaped tool list is materialized once per session, so
    adding tools mid-session would silently fail to register them with
    the provider.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def to_provider_definitions(self) -> list[ToolDefinition]:
        return [t.to_provider_definition() for t in self._tools.values()]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


# === finish — the only loop-exit tool (§12.4) ================================


class FinishInput(BaseModel):
    """Schema for the :func:`finish` tool (§12.4).

    The agent calls ``finish(success=..., summary=...)`` once it has
    completed its task or determined it cannot. The loop's post-turn
    handler detects the call and returns the corresponding
    :class:`smai_agents.loop.AgentOutcome`.
    """

    model_config = ConfigDict(extra="forbid")

    success: bool
    summary: str


FINISH_TOOL_NAME = "finish"


async def _finish_handler(
    parsed_input: BaseModel,
    context: ToolContext,
) -> ToolResultContent:
    """Handler for :func:`finish`.

    Returns a structural acknowledgement; the loop never feeds the
    return value back to the model because :func:`finish` triggers
    loop exit before the next turn runs. The handler exists so the
    registry surface is uniform (every tool has a handler, including
    the sentinel) and so an out-of-band caller invoking the tool via
    the registry directly gets a well-formed result.
    """
    del context  # nothing to do
    if not isinstance(parsed_input, FinishInput):
        raise TypeError(
            f"finish handler expected FinishInput, got {type(parsed_input).__name__}"
        )
    return ToolResultContent(
        tool_use_id="",  # populated by the loop dispatcher
        content=f"finished (success={parsed_input.success}): {parsed_input.summary}",
        is_error=not parsed_input.success,
    )


def make_finish_tool() -> Tool:
    """Build the loop-exit ``finish`` tool (§12.4)."""
    return Tool(
        name=FINISH_TOOL_NAME,
        description=(
            "Exit the agent loop. Call this once the task is complete or "
            "you have determined you cannot complete it. After this call "
            "the loop terminates with the corresponding outcome (success "
            "or failure) and the orchestrator's gate rule reads the "
            "summary."
        ),
        input_schema=FinishInput,
        handler=_finish_handler,
    )


# === Output truncation utility (§3.3 / §12.2) ================================


def truncate_tool_output(
    text: str,
    *,
    max_lines: int = EXECUTE_MAX_LINES,
    max_tokens: int = EXECUTE_MAX_TOKENS,
) -> str:
    """Aggressive last-N-lines + token-cap truncation.

    Per §3.3 / §12.2: "execute's output is truncated to the last 200
    lines and capped at 5000 tokens." Token estimation is the same
    char-count heuristic used by :mod:`smai_agents.truncation` (~4
    chars per token) — exactness isn't worth a tokenizer dep here.

    Output starts with a single-line marker indicating truncation
    happened, so the agent sees the elision rather than guessing
    whether the output was complete. v1 settled this convention in
    ``agent_design.md`` §4.
    """
    if max_lines <= 0:
        raise ValueError("max_lines must be > 0")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")

    char_budget = max_tokens * _CHARS_PER_TOKEN
    lines = text.splitlines()
    truncated_by_lines = False
    truncated_by_tokens = False

    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        truncated_by_lines = True
    body = "\n".join(lines)

    if len(body) > char_budget:
        body = body[-char_budget:]
        truncated_by_tokens = True

    if not truncated_by_lines and not truncated_by_tokens:
        return text

    reasons: list[str] = []
    if truncated_by_lines:
        reasons.append(f"last {max_lines} lines")
    if truncated_by_tokens:
        reasons.append(f"last ~{max_tokens} tokens")
    marker = f"[output truncated — kept {' / '.join(reasons)}]"
    return f"{marker}\n{body}"


__all__ = [
    "EXECUTE_MAX_LINES",
    "EXECUTE_MAX_TOKENS",
    "FINISH_TOOL_NAME",
    "FinishInput",
    "Tool",
    "ToolContext",
    "ToolHandler",
    "ToolRegistry",
    "ToolResultHook",
    "make_finish_tool",
    "truncate_tool_output",
]
