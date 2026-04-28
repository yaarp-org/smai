""":func:`smai_agents.run_loop` turn cycle, finish, max-turns, errors."""

from __future__ import annotations

from pathlib import Path

import pytest
from _agent_fakes import StubLlmProvider  # type: ignore[import-not-found]
from _agent_helpers import model_response  # type: ignore[import-not-found]
from pydantic import BaseModel
from smai_agents import (
    AgentLoopConfig,
    AgentSession,
    Tool,
    ToolContext,
    ToolRegistry,
    make_finish_tool,
    run_loop,
)
from smai_core.plugins import LlmProvider, ToolResultContent


class _EchoInput(BaseModel):
    text: str


async def _echo_handler(
    parsed_input: BaseModel, context: ToolContext
) -> ToolResultContent:
    del context
    if not isinstance(parsed_input, _EchoInput):
        raise TypeError("expected EchoInput")
    return ToolResultContent(
        tool_use_id="",
        content=f"echo:{parsed_input.text}",
        is_error=False,
    )


class _RaisingInput(BaseModel):
    why: str


async def _raising_handler(
    parsed_input: BaseModel, context: ToolContext
) -> ToolResultContent:
    del context
    if not isinstance(parsed_input, _RaisingInput):
        raise TypeError("expected RaisingInput")
    raise RuntimeError(f"intentional crash: {parsed_input.why}")


def _build_session(
    *,
    llm: LlmProvider,
    workspace: Path,
    tools: ToolRegistry,
    config: AgentLoopConfig | None = None,
) -> AgentSession:
    return AgentSession(
        system_prompt="you are a test agent",
        tools=tools,
        llm_providers={"planner": llm},
        current_role="planner",
        workspace_path=workspace,
        config=config or AgentLoopConfig(status_write_every_turns=0),
    )


# --- finish path ------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_terminates_on_finish(tmp_path: Path) -> None:
    """When the agent invokes ``finish``, the loop returns
    ``AgentOutcome(kind='finished')`` with the parsed payload."""
    canned = model_response(
        tool_uses=[("tu-1", "finish", {"success": True, "summary": "done"})],
        stop_reason="tool_use",
    )
    llm = StubLlmProvider([canned])

    registry = ToolRegistry()
    registry.register(make_finish_tool())

    session = _build_session(llm=llm, workspace=tmp_path, tools=registry)
    outcome = await run_loop(session)

    assert outcome.kind == "finished"
    assert outcome.finish_success is True
    assert outcome.finish_summary == "done"
    assert outcome.turn_count == 1


# --- end_turn (no tool use) -------------------------------------------------


@pytest.mark.asyncio
async def test_loop_returns_finished_without_tool_use(tmp_path: Path) -> None:
    """``stop_reason='end_turn'`` without tool_use returns the
    ``finished_without_tool_use`` outcome (§3.1)."""
    canned = model_response(text="i have nothing more", stop_reason="end_turn")
    llm = StubLlmProvider([canned])

    registry = ToolRegistry()
    registry.register(make_finish_tool())

    session = _build_session(llm=llm, workspace=tmp_path, tools=registry)
    outcome = await run_loop(session)

    assert outcome.kind == "finished_without_tool_use"


# --- max_tokens -------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_returns_truncated_output_on_max_tokens(tmp_path: Path) -> None:
    canned = model_response(
        text="output cut off mid-",
        stop_reason="max_tokens",
    )
    llm = StubLlmProvider([canned])

    registry = ToolRegistry()
    registry.register(make_finish_tool())

    session = _build_session(llm=llm, workspace=tmp_path, tools=registry)
    outcome = await run_loop(session)
    assert outcome.kind == "truncated_output"


# --- exhausted_turns --------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_exits_with_exhausted_turns_on_budget(tmp_path: Path) -> None:
    """A loop that calls a non-finish tool indefinitely should exit
    after ``max_turns`` rather than running forever."""
    echo_call = lambda i: model_response(  # noqa: E731
        tool_uses=[(f"tu-{i}", "echo", {"text": f"loop {i}"})],
        stop_reason="tool_use",
    )
    canned = [echo_call(i) for i in range(5)]
    llm = StubLlmProvider(canned)

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="echo",
            description="Echo back the text payload.",
            input_schema=_EchoInput,
            handler=_echo_handler,
        )
    )
    # No finish tool — agent has no way to gracefully exit.

    config = AgentLoopConfig(max_turns=3, status_write_every_turns=0)
    session = _build_session(
        llm=llm, workspace=tmp_path, tools=registry, config=config
    )
    outcome = await run_loop(session)

    assert outcome.kind == "exhausted_turns"
    assert outcome.turn_count == 3
    # The conversation captures all three rounds (assistant + tool result).
    assert len(session.messages) == 6  # 3 assistant turns + 3 tool_result blocks


# --- tool errors are tool results, not loop crashes (§3.3) ------------------


@pytest.mark.asyncio
async def test_tool_handler_exception_becomes_is_error_tool_result(
    tmp_path: Path,
) -> None:
    """§3.3: 'Tool errors are tool results, not loop errors.'"""
    canned = [
        model_response(
            tool_uses=[("tu-1", "raise", {"why": "test"})],
            stop_reason="tool_use",
        ),
        model_response(
            tool_uses=[("tu-2", "finish", {"success": False, "summary": "done"})],
            stop_reason="tool_use",
        ),
    ]
    llm = StubLlmProvider(canned)

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="raise",
            description="Always raises.",
            input_schema=_RaisingInput,
            handler=_raising_handler,
        )
    )
    registry.register(make_finish_tool())

    session = _build_session(llm=llm, workspace=tmp_path, tools=registry)
    outcome = await run_loop(session)

    assert outcome.kind == "finished"
    # Walk the conversation and find the error tool_result the loop produced.
    crash_results = [
        block
        for msg in session.messages
        if msg.role == "user"
        for block in msg.content
        if isinstance(block, ToolResultContent)
        and block.is_error
        and "intentional crash" in block.content
    ]
    assert len(crash_results) == 1


@pytest.mark.asyncio
async def test_unknown_tool_name_becomes_error_result(tmp_path: Path) -> None:
    """The model can hallucinate a tool name; the loop surfaces it as
    an error tool_result rather than crashing."""
    canned = [
        model_response(
            tool_uses=[("tu-1", "no_such_tool", {})],
            stop_reason="tool_use",
        ),
        model_response(
            tool_uses=[("tu-2", "finish", {"success": True, "summary": "ok"})],
            stop_reason="tool_use",
        ),
    ]
    llm = StubLlmProvider(canned)

    registry = ToolRegistry()
    registry.register(make_finish_tool())

    session = _build_session(llm=llm, workspace=tmp_path, tools=registry)
    outcome = await run_loop(session)
    assert outcome.kind == "finished"


@pytest.mark.asyncio
async def test_invalid_tool_input_becomes_validation_error_result(
    tmp_path: Path,
) -> None:
    """A tool_use whose input fails the schema is a tool_result error."""
    canned = [
        model_response(
            # Missing ``text`` field.
            tool_uses=[("tu-1", "echo", {})],
            stop_reason="tool_use",
        ),
        model_response(
            tool_uses=[("tu-2", "finish", {"success": True, "summary": "ok"})],
            stop_reason="tool_use",
        ),
    ]
    llm = StubLlmProvider(canned)

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="echo",
            description="Echo a text payload.",
            input_schema=_EchoInput,
            handler=_echo_handler,
        )
    )
    registry.register(make_finish_tool())

    session = _build_session(llm=llm, workspace=tmp_path, tools=registry)
    outcome = await run_loop(session)
    assert outcome.kind == "finished"

    error_results = [
        block
        for msg in session.messages
        if msg.role == "user"
        for block in msg.content
        if isinstance(block, ToolResultContent) and block.is_error
    ]
    assert len(error_results) == 1
    assert "input validation failed" in error_results[0].content


# --- usage aggregation (§3.4) -----------------------------------------------


@pytest.mark.asyncio
async def test_loop_aggregates_per_call_usage(tmp_path: Path) -> None:
    """§3.4: per-call usage aggregates into ``session.usage_total``."""
    canned = [
        model_response(
            tool_uses=[("tu-1", "echo", {"text": "a"})],
            stop_reason="tool_use",
            input_tokens=10,
            output_tokens=3,
        ),
        model_response(
            tool_uses=[("tu-2", "finish", {"success": True, "summary": "x"})],
            stop_reason="tool_use",
            input_tokens=20,
            output_tokens=5,
        ),
    ]
    llm = StubLlmProvider(canned)

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="echo",
            description="Echo.",
            input_schema=_EchoInput,
            handler=_echo_handler,
        )
    )
    registry.register(make_finish_tool())

    session = _build_session(llm=llm, workspace=tmp_path, tools=registry)
    outcome = await run_loop(session)

    assert outcome.kind == "finished"
    assert session.usage_total.input_tokens == 30
    assert session.usage_total.output_tokens == 8


# --- multiple tool_use in one turn ------------------------------------------


@pytest.mark.asyncio
async def test_loop_executes_multiple_tool_uses_sequentially(
    tmp_path: Path,
) -> None:
    """§3.3: 'Tool execution is sequential.' Two echo calls + finish in
    one assistant turn produce three tool_result blocks in one user turn."""
    canned = [
        model_response(
            tool_uses=[
                ("tu-1", "echo", {"text": "first"}),
                ("tu-2", "echo", {"text": "second"}),
                ("tu-3", "finish", {"success": True, "summary": "fin"}),
            ],
            stop_reason="tool_use",
        ),
    ]
    llm = StubLlmProvider(canned)

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="echo",
            description="Echo.",
            input_schema=_EchoInput,
            handler=_echo_handler,
        )
    )
    registry.register(make_finish_tool())

    session = _build_session(llm=llm, workspace=tmp_path, tools=registry)
    outcome = await run_loop(session)

    assert outcome.kind == "finished"
    # Last user message bundles three tool_result blocks.
    user_msgs = [m for m in session.messages if m.role == "user"]
    assert len(user_msgs) == 1
    assert len(user_msgs[0].content) == 3
