"""End-to-end agent loop with :class:`BedrockProvider` + :class:`FakeBedrockClient`.

Per the Task 2.B1 brief: "an integration test demonstrating a full
multi-turn loop end-to-end with truncation + retry-context." This
exercises the loop against the real
:class:`smai_llm_bedrock.BedrockProvider` plugin (Task 2.A1) wired to
the bedrock plugin's :class:`FakeBedrockClient` test stand-in — so the
test is fully deterministic, never reaches AWS, and hits the same
boto3-shaped translation path production runs through.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from _agent_fakes import StubArtifactStore  # type: ignore[import-not-found]
from smai_agents import (
    PREV_CONVERSATION_TRACE_FILENAME,
    AgentLoopConfig,
    AgentSession,
    Tool,
    ToolContext,
    ToolRegistry,
    TruncationPolicy,
    load_retry_context,
    make_finish_tool,
    run_loop,
)
from smai_core.plugins import (
    LlmCapabilities,
    NormalizedMessage,
    TextContent,
    ToolResultContent,
)

# Reach for the bedrock plugin's FakeBedrockClient. The path is fixed
# by the workspace layout; importing it lazily here (rather than via
# the smai-agents conftest) keeps the bedrock tests dir off
# ``sys.path`` for the smai-agents test session by default — only
# this one integration test needs the fake client.
_BEDROCK_TESTS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "plugins"
    / "smai-llm-bedrock"
    / "tests"
)
if str(_BEDROCK_TESTS_DIR) not in sys.path:
    sys.path.append(str(_BEDROCK_TESTS_DIR))

# ruff: noqa: E402 — must come after the sys.path tweak above.
from _fakes import FakeBedrockClient  # type: ignore[import-not-found]
from smai_llm_bedrock import BedrockProvider


def _bedrock_response(text: str) -> dict[str, object]:
    """Bedrock Converse response wrapping a single text block."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 7,
            "outputTokens": 3,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokens": 0,
        },
    }


def _bedrock_tool_use(
    *,
    tool_use_id: str,
    name: str,
    payload: dict[str, object],
    stop_reason: str = "tool_use",
) -> dict[str, object]:
    """Bedrock Converse response wrapping a tool_use block."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": tool_use_id,
                            "name": name,
                            "input": dict(payload),
                        }
                    }
                ],
            }
        },
        "stopReason": stop_reason,
        "usage": {
            "inputTokens": 9,
            "outputTokens": 4,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokens": 0,
        },
    }


# --- Multi-turn loop with finish ---------------------------------------------


@pytest.mark.asyncio
async def test_multi_turn_loop_terminates_via_finish(tmp_path: Path) -> None:
    """A two-turn conversation: assistant calls ``finish`` on the
    first turn; the loop returns ``AgentOutcome(kind='finished')`` and
    the conversation trace is well-formed."""
    fake = FakeBedrockClient()
    fake._conformance_queue.append(  # noqa: SLF001 — stand-in seam
        _bedrock_tool_use(
            tool_use_id="tu-fin",
            name="finish",
            payload={"success": True, "summary": "all done"},
        )
    )
    provider = BedrockProvider(
        region="us-east-1",
        model_id="us.anthropic.claude-opus-4-7-v1",
        bedrock_client=fake,
        transient_backoff_seconds=0.0,
    )

    registry = ToolRegistry()
    registry.register(make_finish_tool())

    session = AgentSession(
        system_prompt="multi-turn test",
        messages=[NormalizedMessage(role="user", content=[TextContent(text="hi")])],
        tools=registry,
        llm_providers={"planner": provider},
        current_role="planner",
        workspace_path=tmp_path,
        config=AgentLoopConfig(status_write_every_turns=0),
    )

    outcome = await run_loop(session)
    assert outcome.kind == "finished"
    assert outcome.finish_success is True
    assert outcome.finish_summary == "all done"
    # The loop materialized one Bedrock Converse request — request shape
    # confirmed by the fake's ``calls`` capture.
    assert len(fake.calls) == 1


# --- Truncation in a real session -------------------------------------------


@pytest.mark.asyncio
async def test_truncation_fires_before_provider_call(tmp_path: Path) -> None:
    """A session whose conversation already exceeds the truncation
    threshold has the middle dropped before the next provider call."""
    fake = FakeBedrockClient()
    fake._conformance_queue.append(  # noqa: SLF001
        _bedrock_tool_use(
            tool_use_id="tu-fin",
            name="finish",
            payload={"success": True, "summary": "ok"},
        )
    )
    # Pin the capability surface to a small window so the heuristic
    # threshold is reachable with synthetic content.
    capabilities = LlmCapabilities(
        supports_caching=True,
        context_window=4_000,
        max_output_tokens=1_024,
        supports_tool_use=True,
        model_id="us.anthropic.claude-opus-4-7-v1",
    )
    provider = BedrockProvider(
        region="us-east-1",
        model_id="us.anthropic.claude-opus-4-7-v1",
        bedrock_client=fake,
        capabilities=capabilities,
        transient_backoff_seconds=0.0,
    )

    # Build a conversation with 16 user/assistant pairs, each ~200
    # chars (~50 tokens). Total ~16 * 200 = 3.2K chars / ~800 tokens —
    # need more to cross 4K * 0.90 * 0.85 = 3060 token threshold.
    # Use 30 messages of 1000 chars each (~7500 tokens total).
    bulk_messages: list[NormalizedMessage] = []
    for i in range(30):
        role = "user" if i % 2 == 0 else "assistant"
        text = f"message {i:03d}: " + ("y" * 1000)
        bulk_messages.append(
            NormalizedMessage(
                role=role,
                content=[TextContent(text=text)],
            )
        )

    registry = ToolRegistry()
    registry.register(make_finish_tool())
    session = AgentSession(
        system_prompt="truncation test",
        messages=list(bulk_messages),
        tools=registry,
        llm_providers={"planner": provider},
        current_role="planner",
        workspace_path=tmp_path,
        truncation_policy=TruncationPolicy(),
        config=AgentLoopConfig(status_write_every_turns=0),
    )

    outcome = await run_loop(session)
    assert outcome.kind == "finished"
    assert session.truncations_fired >= 1
    # Truncation kept head + tail: shorter than the original 30 messages
    # (plus the assistant's tool_use turn + user tool_result that the
    # loop appended after the call).
    assert len(session.messages) < len(bulk_messages) + 2


# --- Retry-context passes through on second invocation ----------------------


@pytest.mark.asyncio
async def test_retry_context_writes_workspace_file(tmp_path: Path) -> None:
    """DEC-023: on retry, the previous attempt's conversation trace is
    written to ``<workspace>/prev-conversation-trace.json``. Demonstrates
    the file shows up in the workspace before a (subsequent) loop runs."""
    store = StubArtifactStore()
    artifact_path = "comparison-groups/cg-fixture/entries/e1/conversation-trace.json"
    canned = b'{"turn": 1, "role": "technique_implementer", "content": "..."}'
    await store.put(artifact_path, canned)

    workspace = tmp_path / "ws"
    written = await load_retry_context(
        workspace_path=workspace,
        artifact_store=store,
        artifact_path=artifact_path,
    )
    assert written is not None
    assert written == workspace / PREV_CONVERSATION_TRACE_FILENAME
    assert written.read_bytes() == canned

    # Now drive a one-turn loop where a tool reads the prev trace —
    # this demonstrates the file is in place before the loop runs and
    # is reachable through the workspace_path the loop hands to tool
    # handlers via :class:`ToolContext`.
    fake = FakeBedrockClient()
    fake._conformance_queue.append(  # noqa: SLF001
        _bedrock_tool_use(
            tool_use_id="tu-read",
            name="read_prev",
            payload={"path": PREV_CONVERSATION_TRACE_FILENAME},
        )
    )
    fake._conformance_queue.append(  # noqa: SLF001
        _bedrock_tool_use(
            tool_use_id="tu-fin",
            name="finish",
            payload={"success": True, "summary": "consumed retry context"},
        )
    )

    captured_content: list[str] = []

    from pydantic import BaseModel  # noqa: PLC0415

    class _ReadInput(BaseModel):
        path: str

    async def _read_handler(
        parsed_input: BaseModel, context: ToolContext
    ) -> ToolResultContent:
        if not isinstance(parsed_input, _ReadInput):
            raise TypeError("expected _ReadInput")
        target = context.workspace_path / parsed_input.path
        body = target.read_text()
        captured_content.append(body)
        return ToolResultContent(
            tool_use_id="",
            content=body,
            is_error=False,
        )

    provider = BedrockProvider(
        region="us-east-1",
        model_id="us.anthropic.claude-opus-4-7-v1",
        bedrock_client=fake,
        transient_backoff_seconds=0.0,
    )

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="read_prev",
            description="Read the previous conversation trace.",
            input_schema=_ReadInput,
            handler=_read_handler,
        )
    )
    registry.register(make_finish_tool())

    session = AgentSession(
        system_prompt="retry agent",
        messages=[
            NormalizedMessage(
                role="user",
                content=[TextContent(text="please review prev trace and finish")],
            )
        ],
        tools=registry,
        llm_providers={"technique_implementer": provider},
        current_role="technique_implementer",
        workspace_path=workspace,
        config=AgentLoopConfig(status_write_every_turns=0),
    )

    outcome = await run_loop(session)
    assert outcome.kind == "finished"
    assert outcome.finish_summary == "consumed retry context"
    assert len(captured_content) == 1
    assert captured_content[0] == canned.decode("utf-8")
