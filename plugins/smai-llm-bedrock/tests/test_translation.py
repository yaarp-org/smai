"""Round-trip translation tests beyond what conformance covers.

The :class:`LlmProviderConformance` base verifies the contract surface;
these tests pin down the Bedrock-Converse-specific request shape (cache
points, tool config, message ordering, role mapping) that the agent
loop will rely on but that the conformance suite is provider-agnostic
about.
"""

from __future__ import annotations

import asyncio

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


def test_apply_cache_points_static_prefix_marks_tools_and_system() -> None:
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
    assert system_blocks[-1] == {"cachePoint": {"type": "default"}}
    tools = tool_config["tools"]
    assert isinstance(tools, list)
    assert tools[-1] == {"cachePoint": {"type": "default"}}


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


# --- end-to-end via FakeBedrockClient --------------------------------------


def test_end_to_end_request_shape_carries_cache_points_and_inference_config() -> None:
    fake = FakeBedrockClient()
    provider = BedrockProvider(
        region="us-east-1",
        model_id="us.anthropic.claude-opus-4-7-v1",
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
    # System block has the cachePoint marker (cache_static_prefix=True).
    assert sent["system"][-1] == {"cachePoint": {"type": "default"}}
    # Tools were sent and wear the cachePoint marker too.
    assert sent["toolConfig"]["tools"][-1] == {"cachePoint": {"type": "default"}}
    # Initial user message has cachePoint appended.
    assert sent["messages"][0]["content"][-1] == {"cachePoint": {"type": "default"}}
    # Model id is what we constructed with.
    assert sent["modelId"] == "us.anthropic.claude-opus-4-7-v1"


def test_call_does_not_mutate_input_messages() -> None:
    fake = FakeBedrockClient()
    provider = BedrockProvider(
        region="us-east-1",
        model_id="us.anthropic.claude-opus-4-7-v1",
        bedrock_client=fake,
        transient_backoff_seconds=0.0,
    )
    msgs = [NormalizedMessage(role="user", content=[TextContent(text="hi")])]
    snapshot = [m.model_copy(deep=True) for m in msgs]
    asyncio.run(provider.call(system="s", messages=msgs))
    assert msgs == snapshot
