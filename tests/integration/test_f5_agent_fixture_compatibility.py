"""Phase-3 Task 3.F5 acceptance: existing agent fixtures from Phase 2
run successfully against the new :class:`AnthropicProvider` and
:class:`OpenAIProvider` plugins (mocked at the SDK seam).

Per the §3.4 Task 3.F5 brief: "the existing agent fixtures from
Phase 2 run successfully against each provider."

The plugin packages cannot depend on ``smai-agents`` (the dependency-
allowlist forbids it — plugins live below the pipeline layer). So the
compatibility check lives at the workspace integration-test layer
where the agents loop and the new plugins both can be imported.

The test drives :func:`smai_agents.run_loop` against each provider's
fake-client seam, exercising the full Protocol surface the loop
relies on:

1. ``call(system, messages, tools, max_tokens, ...)`` is invoked.
2. The response's ``stop_reason`` and ``content`` blocks survive the
   round-trip and feed the loop's tool-execution branch.
3. ``ToolUseContent`` blocks the plugin emits parse correctly into
   the loop's tool-call dispatch.
4. ``ToolResultContent`` blocks the loop appends to ``messages``
   round-trip back through the plugin's request-encoder without
   raising.
5. Token usage aggregates into the session's running total.

These five guarantees are the agent loop's contract on
:class:`LlmProvider`; passing them against AnthropicProvider's
FakeAnthropicClient and OpenAIProvider's FakeOpenAIClient is the
acceptance bar.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Any  # noqa: F401  # used in canned response builders

import pytest
from smai_inline_agents import (
    AgentLoopConfig,
    AgentSession,
    ToolRegistry,
    make_finish_tool,
    run_loop,
)

# Plugin tests/ directories add themselves to sys.path via their own
# conftests (so the per-plugin ``_f5_*_fakes`` module is importable
# inside the plugin's own test tree). At the integration-test layer
# we have to do that ourselves — the plugin tests/ dirs are not on
# the path by default. Per the brief's filename-hygiene rule, the
# fakes are named ``_f5_anthropic_fakes`` / ``_f5_openai_fakes`` so
# they don't collide.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "plugins" / "smai-llm-anthropic" / "tests"))
sys.path.insert(0, str(_REPO_ROOT / "plugins" / "smai-llm-openai" / "tests"))

from _f5_anthropic_fakes import FakeAnthropicClient  # noqa: E402  # type: ignore[import-not-found]
from _f5_openai_fakes import FakeOpenAIClient  # noqa: E402  # type: ignore[import-not-found]
from smai_llm_anthropic import AnthropicProvider  # noqa: E402
from smai_llm_openai import OpenAIProvider  # noqa: E402


def _build_session(*, llm: Any, workspace: Path) -> AgentSession:
    registry = ToolRegistry()
    registry.register(make_finish_tool())
    return AgentSession(
        system_prompt="you are a fixture agent",
        tools=registry,
        llm_providers={"planner": llm},
        current_role="planner",
        workspace_path=workspace,
        config=AgentLoopConfig(status_write_every_turns=0),
    )


def _anthropic_finish_response() -> dict[str, Any]:
    """Canned Anthropic Message that calls the loop's ``finish`` tool."""
    return {
        "id": "msg_finish",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "tu_finish",
                "name": "finish",
                "input": {"success": True, "summary": "done via anthropic"},
            }
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 4,
            "output_tokens": 2,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


def _openai_finish_response() -> dict[str, Any]:
    """Canned OpenAI ChatCompletion that calls the loop's ``finish`` tool."""
    return {
        "id": "chatcmpl_finish",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_finish",
                            "type": "function",
                            "function": {
                                "name": "finish",
                                "arguments": ('{"success": true, "summary": "done via openai"}'),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "total_tokens": 6,
        },
    }


@pytest.mark.asyncio
async def test_run_loop_against_anthropic_provider_finishes_cleanly(
    tmp_path: Path,
) -> None:
    """The Phase-2 ``finish`` tool fixture round-trips through
    :class:`AnthropicProvider` cleanly: tool_use block from the canned
    response parses, the loop dispatches to ``finish``, the outcome
    is ``finished`` with the parsed payload."""
    fake = FakeAnthropicClient()
    fake.messages._conformance_queue = deque([_anthropic_finish_response()])

    provider = AnthropicProvider(
        model_id="claude-opus-4-7",
        anthropic_client=fake,
        transient_backoff_seconds=0.0,
    )
    session = _build_session(llm=provider, workspace=tmp_path)

    outcome = await run_loop(session)

    assert outcome.kind == "finished"
    assert outcome.finish_success is True
    assert outcome.finish_summary == "done via anthropic"
    # Token usage round-tripped into the session aggregate.
    assert outcome.usage_total.input_tokens == 4
    assert outcome.usage_total.output_tokens == 2
    # The loop made exactly one call to the plugin.
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_run_loop_against_openai_provider_finishes_cleanly(
    tmp_path: Path,
) -> None:
    """Same shape as the AnthropicProvider acceptance — but OpenAI's
    ``tool_calls`` / JSON-encoded-arguments shape exercises a
    materially different translation path on the way back into the
    loop's normalized :class:`ToolUseContent`."""
    fake = FakeOpenAIClient()
    fake.chat.completions._conformance_queue = deque([_openai_finish_response()])

    provider = OpenAIProvider(
        model_id="gpt-4o",
        openai_client=fake,
        transient_backoff_seconds=0.0,
    )
    session = _build_session(llm=provider, workspace=tmp_path)

    outcome = await run_loop(session)

    assert outcome.kind == "finished"
    assert outcome.finish_success is True
    assert outcome.finish_summary == "done via openai"
    assert outcome.usage_total.input_tokens == 4
    assert outcome.usage_total.output_tokens == 2
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_anthropic_provider_round_trips_tool_result_messages(
    tmp_path: Path,
) -> None:
    """Two-turn fixture: the model first calls a non-finish tool; the
    loop appends the tool_result to the conversation; the second turn
    of the model invokes ``finish``. This exercises the plugin's
    request-encoding of a tool_result-bearing user message — the
    primary mutation the loop performs between turns."""
    fake = FakeAnthropicClient()
    # Turn 1: model invokes echo tool; turn 2: model invokes finish.
    fake.messages._conformance_queue = deque(
        [
            {
                "id": "msg_echo",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_echo",
                        "name": "echo",
                        "input": {"text": "hi"},
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
            _anthropic_finish_response(),
        ]
    )

    provider = AnthropicProvider(
        model_id="claude-opus-4-7",
        anthropic_client=fake,
        transient_backoff_seconds=0.0,
    )

    # Register a no-op echo tool alongside finish so the first turn
    # has somewhere to dispatch.
    from pydantic import BaseModel
    from smai_core.plugins import ToolResultContent
    from smai_inline_agents import Tool, ToolContext

    class _EchoInput(BaseModel):
        text: str

    async def _echo_handler(parsed_input: BaseModel, context: ToolContext) -> ToolResultContent:
        del context
        assert isinstance(parsed_input, _EchoInput)
        return ToolResultContent(
            tool_use_id="",
            content=f"echoed:{parsed_input.text}",
            is_error=False,
        )

    registry = ToolRegistry()
    registry.register(make_finish_tool())
    registry.register(
        Tool(
            name="echo",
            description="Echo input back",
            input_schema=_EchoInput,
            handler=_echo_handler,
        )
    )

    session = AgentSession(
        system_prompt="you are a fixture agent",
        tools=registry,
        llm_providers={"planner": provider},
        current_role="planner",
        workspace_path=tmp_path,
        config=AgentLoopConfig(status_write_every_turns=0),
    )

    outcome = await run_loop(session)

    assert outcome.kind == "finished"
    assert outcome.turn_count == 2
    # Two model calls; the second one's request body must include the
    # tool_result the loop appended after turn 1 — verifies the plugin
    # successfully translated a tool_result-bearing user message.
    assert len(fake.calls) == 2
    second_request = fake.calls[1]
    second_messages = second_request["messages"]
    # The last user message (just before the model's second turn)
    # carries the tool_result the loop synthesized from echo's output.
    user_with_result = next(m for m in reversed(second_messages) if m["role"] == "user")
    tool_result_blocks = [b for b in user_with_result["content"] if b.get("type") == "tool_result"]
    assert len(tool_result_blocks) == 1
    assert tool_result_blocks[0]["tool_use_id"] == "tu_echo"
    assert "echoed:hi" in tool_result_blocks[0]["content"]
