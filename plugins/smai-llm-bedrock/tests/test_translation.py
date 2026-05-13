"""Round-trip translation tests beyond what conformance covers.

The :class:`LlmProviderConformance` base verifies the contract surface;
these tests pin down the Bedrock-Converse-specific request shape (cache
points, tool config, message ordering, role mapping) that the agent
loop will rely on but that the conformance suite is provider-agnostic
about.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from _bedrock_fakes import FakeBedrockClient  # type: ignore[import-not-found]
from smai_core.plugins import (
    CacheConfig,
    LlmCapabilities,
    NormalizedMessage,
    TextContent,
    ToolDefinition,
    ToolResultContent,
    ToolUseContent,
)
from smai_llm_bedrock import BedrockProvider
from smai_llm_bedrock._translation import (
    apply_cache_points,
    to_converse_messages,
    to_converse_tool_config,
)

# --- pure translation -------------------------------------------------------


def test_text_message_translates_to_text_block() -> None:
    out = to_converse_messages(
        [NormalizedMessage(role="user", content=[TextContent(text="hello")])]
    )
    assert out == [{"role": "user", "content": [{"text": "hello"}]}]


def test_tool_use_message_translates_to_tooluse_block() -> None:
    out = to_converse_messages(
        [
            NormalizedMessage(
                role="assistant",
                content=[
                    ToolUseContent(id="tu1", name="echo", input={"text": "hi"}),
                ],
            )
        ]
    )
    assert out == [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "tu1", "name": "echo", "input": {"text": "hi"}}}],
        }
    ]


def test_tool_result_translates_with_status_and_text_wrap() -> None:
    out = to_converse_messages(
        [
            NormalizedMessage(
                role="user",
                content=[
                    ToolResultContent(tool_use_id="tu1", content="ok", is_error=False),
                    ToolResultContent(tool_use_id="tu2", content="bad", is_error=True),
                ],
            )
        ]
    )
    assert out == [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "tu1",
                        "content": [{"text": "ok"}],
                        "status": "success",
                    }
                },
                {
                    "toolResult": {
                        "toolUseId": "tu2",
                        "content": [{"text": "bad"}],
                        "status": "error",
                    }
                },
            ],
        }
    ]


def test_tool_config_translation() -> None:
    out = to_converse_tool_config(
        [
            ToolDefinition(
                name="t",
                description="d",
                input_schema={"type": "object", "properties": {}},
            )
        ]
    )
    assert out == {
        "tools": [
            {
                "toolSpec": {
                    "name": "t",
                    "description": "d",
                    "inputSchema": {"json": {"type": "object", "properties": {}}},
                }
            }
        ]
    }


# --- cache-point placement --------------------------------------------------


def _capabilities(*, supports_caching: bool) -> LlmCapabilities:
    return LlmCapabilities(
        supports_caching=supports_caching,
        context_window=200_000,
        max_output_tokens=4096,
        supports_tool_use=True,
        model_id="test",
    )


_CP = {"cachePoint": {"type": "default"}}


def _count_cache_points(
    system_blocks: list[dict[str, object]],
    converse_messages: list[dict[str, object]],
    tool_config: dict[str, object] | None,
) -> int:
    total = sum(1 for b in system_blocks if b == _CP)
    if tool_config is not None:
        tools = tool_config.get("tools")
        if isinstance(tools, list):
            total += sum(1 for b in tools if b == _CP)
    for msg in converse_messages:
        content = msg.get("content")
        if isinstance(content, list):
            total += sum(1 for b in content if b == _CP)
    return total


def test_apply_cache_points_static_prefix_marks_tools_only_not_system() -> None:
    system_blocks: list[dict[str, object]] = [{"text": "sys"}]
    converse_messages: list[dict[str, object]] = [{"role": "user", "content": [{"text": "hi"}]}]
    tool_config: dict[str, object] = {"tools": [{"toolSpec": {"name": "t"}}]}
    apply_cache_points(
        system_blocks=system_blocks,
        converse_messages=converse_messages,
        tool_config=tool_config,
        cache_config=CacheConfig(cache_static_prefix=True),
        capabilities=_capabilities(supports_caching=True),
    )
    tools = tool_config["tools"]
    assert isinstance(tools, list)
    assert tools[-1] == _CP
    # Bedrock evaluates tools -> system, so the system block is NOT marked.
    assert system_blocks == [{"text": "sys"}]


def test_apply_cache_points_static_prefix_falls_back_to_system_when_no_tools() -> None:
    system_blocks: list[dict[str, object]] = [{"text": "sys"}]
    converse_messages: list[dict[str, object]] = [{"role": "user", "content": [{"text": "hi"}]}]
    apply_cache_points(
        system_blocks=system_blocks,
        converse_messages=converse_messages,
        tool_config=None,
        cache_config=CacheConfig(cache_static_prefix=True),
        capabilities=_capabilities(supports_caching=True),
    )
    assert system_blocks[-1] == _CP


def test_apply_cache_points_initial_message() -> None:
    converse_messages: list[dict[str, object]] = [
        {"role": "user", "content": [{"text": "hi"}]},
        {"role": "assistant", "content": [{"text": "ok"}]},
    ]
    apply_cache_points(
        system_blocks=[{"text": "sys"}],
        converse_messages=converse_messages,
        tool_config=None,
        cache_config=CacheConfig(cache_initial_message=True),
        capabilities=_capabilities(supports_caching=True),
    )
    first_content = converse_messages[0]["content"]
    assert isinstance(first_content, list)
    assert first_content[-1] == {"cachePoint": {"type": "default"}}


def test_apply_cache_points_rolling_skips_initial_and_caps_at_count() -> None:
    converse_messages: list[dict[str, object]] = [
        {"role": "user", "content": [{"text": "u0"}]},
        {"role": "assistant", "content": [{"text": "a1"}]},
        {"role": "user", "content": [{"text": "u2"}]},
        {"role": "assistant", "content": [{"text": "a3"}]},
        {"role": "user", "content": [{"text": "u4"}]},
    ]
    apply_cache_points(
        system_blocks=[{"text": "sys"}],
        converse_messages=converse_messages,
        tool_config=None,
        cache_config=CacheConfig(rolling_cache_count=2),
        capabilities=_capabilities(supports_caching=True),
    )
    # Initial message (index 0) untouched.
    initial = converse_messages[0]["content"]
    assert isinstance(initial, list)
    assert all(b != {"cachePoint": {"type": "default"}} for b in initial)
    # Last two user messages (indexes 2 and 4) carry cachePoint.
    for idx in (2, 4):
        content = converse_messages[idx]["content"]
        assert isinstance(content, list)
        assert content[-1] == {"cachePoint": {"type": "default"}}


def test_apply_cache_points_noop_when_caps_off() -> None:
    converse_messages: list[dict[str, object]] = [{"role": "user", "content": [{"text": "hi"}]}]
    system_blocks: list[dict[str, object]] = [{"text": "sys"}]
    apply_cache_points(
        system_blocks=system_blocks,
        converse_messages=converse_messages,
        tool_config={"tools": [{"toolSpec": {"name": "t"}}]},
        cache_config=CacheConfig(
            cache_static_prefix=True,
            cache_initial_message=True,
            rolling_cache_count=4,
        ),
        capabilities=_capabilities(supports_caching=False),
    )
    assert system_blocks == [{"text": "sys"}]
    first_content = converse_messages[0]["content"]
    assert isinstance(first_content, list)
    assert first_content == [{"text": "hi"}]


def test_apply_cache_points_default_multiturn_emits_exactly_four() -> None:
    # The agent-loop default: 1 static (tools) + 1 initial-message + 2 rolling.
    converse_messages: list[dict[str, object]] = [
        {"role": "user", "content": [{"text": "u0"}]},
        {"role": "assistant", "content": [{"text": "a1"}]},
        {"role": "user", "content": [{"text": "u2"}]},
        {"role": "assistant", "content": [{"text": "a3"}]},
        {"role": "user", "content": [{"text": "u4"}]},
    ]
    system_blocks: list[dict[str, object]] = [{"text": "sys"}]
    tool_config: dict[str, object] = {"tools": [{"toolSpec": {"name": "t"}}]}
    apply_cache_points(
        system_blocks=system_blocks,
        converse_messages=converse_messages,
        tool_config=tool_config,
        cache_config=CacheConfig(
            cache_static_prefix=True,
            cache_initial_message=True,
            rolling_cache_count=2,
        ),
        capabilities=_capabilities(supports_caching=True),
    )
    tools = tool_config["tools"]
    assert isinstance(tools, list)
    assert tools[-1] == _CP  # static marker on tools
    assert all(b != _CP for b in system_blocks)  # NOT on system
    assert converse_messages[0]["content"][-1] == _CP  # initial message  # type: ignore[index]
    assert converse_messages[2]["content"][-1] == _CP  # rolling  # type: ignore[index]
    assert converse_messages[4]["content"][-1] == _CP  # rolling  # type: ignore[index]
    assert _count_cache_points(system_blocks, converse_messages, tool_config) == 4


def test_apply_cache_points_clamps_rolling_to_budget(
    caplog: pytest.LogCaptureFixture,
) -> None:
    converse_messages: list[dict[str, object]] = [
        {"role": "user", "content": [{"text": f"u{i}"}]}
        if i % 2 == 0
        else {"role": "assistant", "content": [{"text": f"a{i}"}]}
        for i in range(11)
    ]
    system_blocks: list[dict[str, object]] = [{"text": "sys"}]
    tool_config: dict[str, object] = {"tools": [{"toolSpec": {"name": "t"}}]}
    with caplog.at_level(logging.WARNING):
        apply_cache_points(
            system_blocks=system_blocks,
            converse_messages=converse_messages,
            tool_config=tool_config,
            cache_config=CacheConfig(
                cache_static_prefix=True,
                cache_initial_message=True,
                rolling_cache_count=4,
            ),
            capabilities=_capabilities(supports_caching=True),
        )
    # 1 static + 1 initial leaves room for 2 rolling -> 4 total, not 6.
    assert _count_cache_points(system_blocks, converse_messages, tool_config) == 4
    # The clamp dropped the oldest rolling markers: only the two newest
    # user messages (indexes 10 and 8) carry a rolling cachePoint.
    assert converse_messages[10]["content"][-1] == _CP  # type: ignore[index]
    assert converse_messages[8]["content"][-1] == _CP  # type: ignore[index]
    assert converse_messages[6]["content"][-1] != _CP  # type: ignore[index]
    assert any("clamped rolling_cache_count" in r.message for r in caplog.records)


def test_apply_cache_points_single_call_config_emits_two() -> None:
    # rolling=0 -> 1 static (tools) + 1 initial-message = 2.
    converse_messages: list[dict[str, object]] = [{"role": "user", "content": [{"text": "hi"}]}]
    system_blocks: list[dict[str, object]] = [{"text": "sys"}]
    tool_config: dict[str, object] = {"tools": [{"toolSpec": {"name": "t"}}]}
    apply_cache_points(
        system_blocks=system_blocks,
        converse_messages=converse_messages,
        tool_config=tool_config,
        cache_config=CacheConfig(
            cache_static_prefix=True,
            cache_initial_message=True,
            rolling_cache_count=0,
        ),
        capabilities=_capabilities(supports_caching=True),
    )
    assert _count_cache_points(system_blocks, converse_messages, tool_config) == 2


# --- end-to-end via FakeBedrockClient --------------------------------------


def test_end_to_end_request_shape_carries_cache_points_and_inference_config() -> None:
    fake = FakeBedrockClient()
    provider = BedrockProvider(
        region="us-east-1",
        model_id="us.anthropic.claude-opus-4-6-v1",
        bedrock_client=fake,
        transient_backoff_seconds=0.0,
    )
    asyncio.run(
        provider.call(
            system="You are a research assistant.",
            messages=[NormalizedMessage(role="user", content=[TextContent(text="ping")])],
            tools=[ToolDefinition(name="t", description="d", input_schema={"type": "object"})],
            max_tokens=512,
            temperature=0.2,
            cache_config=CacheConfig(
                cache_static_prefix=True,
                cache_initial_message=True,
                rolling_cache_count=0,
            ),
        )
    )
    assert len(fake.calls) == 1
    sent = fake.calls[0]
    # Inference config flows through.
    assert sent["inferenceConfig"] == {"maxTokens": 512, "temperature": 0.2}
    # Tools were sent and wear the single static cachePoint marker; the
    # system block is NOT marked (Bedrock caches it via the tools checkpoint).
    assert sent["toolConfig"]["tools"][-1] == _CP
    assert all(b != _CP for b in sent["system"])
    # Initial user message has cachePoint appended.
    assert sent["messages"][0]["content"][-1] == _CP
    # Model id is what we constructed with.
    assert sent["modelId"] == "us.anthropic.claude-opus-4-6-v1"
    # 1 static (tools) + 1 initial-message, rolling=0.
    assert _count_cache_points(sent["system"], sent["messages"], sent["toolConfig"]) == 2


def test_call_does_not_mutate_input_messages() -> None:
    fake = FakeBedrockClient()
    provider = BedrockProvider(
        region="us-east-1",
        model_id="us.anthropic.claude-opus-4-6-v1",
        bedrock_client=fake,
        transient_backoff_seconds=0.0,
    )
    msgs = [NormalizedMessage(role="user", content=[TextContent(text="hi")])]
    snapshot = [m.model_copy(deep=True) for m in msgs]
    asyncio.run(provider.call(system="s", messages=msgs))
    assert msgs == snapshot
