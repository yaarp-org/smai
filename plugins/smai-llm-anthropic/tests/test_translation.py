"""Round-trip translation tests beyond what conformance covers.

The :class:`LlmProviderConformance` base verifies the contract surface;
these tests pin down the Anthropic-native-API-specific request shape
(cache_control markers, tool definitions, message ordering, role
mapping, system-prompt promotion) that the agent loop will rely on
but that the conformance suite is provider-agnostic about.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from _f5_anthropic_fakes import FakeAnthropicClient  # type: ignore[import-not-found]
from smai_core.plugins import (
    CacheConfig,
    LlmCapabilities,
    NormalizedMessage,
    TextContent,
    ToolDefinition,
    ToolResultContent,
    ToolUseContent,
)
from smai_llm_anthropic import AnthropicProvider
from smai_llm_anthropic._translation import (
    apply_cache_control,
    to_anthropic_messages,
    to_anthropic_tools,
)

# --- pure translation -------------------------------------------------------


def test_text_message_translates_to_text_block() -> None:
    out = to_anthropic_messages(
        [NormalizedMessage(role="user", content=[TextContent(text="hello")])]
    )
    assert out == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]


def test_tool_use_message_translates_to_tool_use_block() -> None:
    out = to_anthropic_messages(
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
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu1",
                    "name": "echo",
                    "input": {"text": "hi"},
                }
            ],
        }
    ]


def test_tool_result_translates_with_is_error_flag() -> None:
    out = to_anthropic_messages(
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
                    "type": "tool_result",
                    "tool_use_id": "tu1",
                    "content": "ok",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "tu2",
                    "content": "bad",
                    "is_error": True,
                },
            ],
        }
    ]


def test_tool_definitions_translate_to_flat_anthropic_shape() -> None:
    out = to_anthropic_tools(
        [
            ToolDefinition(
                name="t",
                description="d",
                input_schema={"type": "object", "properties": {}},
            )
        ]
    )
    assert out == [
        {
            "name": "t",
            "description": "d",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


# --- cache_control placement ------------------------------------------------


def _capabilities(*, supports_caching: bool) -> LlmCapabilities:
    return LlmCapabilities(
        supports_caching=supports_caching,
        context_window=200_000,
        max_output_tokens=4096,
        supports_tool_use=True,
        model_id="test",
    )


_EPHEMERAL = {"type": "ephemeral"}


def _count_markers(
    system_blocks: list[dict[str, object]] | None,
    anthropic_messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None,
) -> int:
    total = 0
    for seq in (system_blocks or [], tools or []):
        total += sum(1 for b in seq if b.get("cache_control") == _EPHEMERAL)
    for msg in anthropic_messages:
        content = msg.get("content")
        if isinstance(content, list):
            total += sum(
                1 for b in content if isinstance(b, dict) and b.get("cache_control") == _EPHEMERAL
            )
    return total


def test_cache_control_static_prefix_marks_tools_only_not_system() -> None:
    system_blocks: list[dict[str, object]] = [{"type": "text", "text": "sys"}]
    anthropic_messages: list[dict[str, object]] = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ]
    tools: list[dict[str, object]] = [{"name": "t"}]
    apply_cache_control(
        system_blocks=system_blocks,
        anthropic_messages=anthropic_messages,
        tools=tools,
        cache_config=CacheConfig(cache_static_prefix=True),
        capabilities=_capabilities(supports_caching=True),
    )
    assert tools[-1].get("cache_control") == _EPHEMERAL
    # tools -> system prefix order: one marker on tools already caches system.
    assert "cache_control" not in system_blocks[-1]


def test_cache_control_static_prefix_falls_back_to_system_when_no_tools() -> None:
    system_blocks: list[dict[str, object]] = [{"type": "text", "text": "sys"}]
    anthropic_messages: list[dict[str, object]] = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ]
    apply_cache_control(
        system_blocks=system_blocks,
        anthropic_messages=anthropic_messages,
        tools=None,
        cache_config=CacheConfig(cache_static_prefix=True),
        capabilities=_capabilities(supports_caching=True),
    )
    assert system_blocks[-1].get("cache_control") == _EPHEMERAL


def test_cache_control_default_multiturn_emits_exactly_four() -> None:
    anthropic_messages: list[dict[str, object]] = [
        {"role": "user", "content": [{"type": "text", "text": "u0"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
        {"role": "user", "content": [{"type": "text", "text": "u2"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a3"}]},
        {"role": "user", "content": [{"type": "text", "text": "u4"}]},
    ]
    system_blocks: list[dict[str, object]] = [{"type": "text", "text": "sys"}]
    tools: list[dict[str, object]] = [{"name": "t"}]
    apply_cache_control(
        system_blocks=system_blocks,
        anthropic_messages=anthropic_messages,
        tools=tools,
        cache_config=CacheConfig(
            cache_static_prefix=True,
            cache_initial_message=True,
            rolling_cache_count=2,
        ),
        capabilities=_capabilities(supports_caching=True),
    )
    assert tools[-1].get("cache_control") == _EPHEMERAL
    assert "cache_control" not in system_blocks[-1]
    assert _count_markers(system_blocks, anthropic_messages, tools) == 4


def test_cache_control_clamps_rolling_to_budget(
    caplog: pytest.LogCaptureFixture,
) -> None:
    anthropic_messages: list[dict[str, object]] = [
        {"role": "user", "content": [{"type": "text", "text": f"u{i}"}]}
        if i % 2 == 0
        else {"role": "assistant", "content": [{"type": "text", "text": f"a{i}"}]}
        for i in range(11)
    ]
    system_blocks: list[dict[str, object]] = [{"type": "text", "text": "sys"}]
    tools: list[dict[str, object]] = [{"name": "t"}]
    with caplog.at_level(logging.WARNING):
        apply_cache_control(
            system_blocks=system_blocks,
            anthropic_messages=anthropic_messages,
            tools=tools,
            cache_config=CacheConfig(
                cache_static_prefix=True,
                cache_initial_message=True,
                rolling_cache_count=4,
            ),
            capabilities=_capabilities(supports_caching=True),
        )
    assert _count_markers(system_blocks, anthropic_messages, tools) == 4
    last10 = anthropic_messages[10]["content"]
    last8 = anthropic_messages[8]["content"]
    last6 = anthropic_messages[6]["content"]
    assert isinstance(last10, list) and last10[-1].get("cache_control") == _EPHEMERAL
    assert isinstance(last8, list) and last8[-1].get("cache_control") == _EPHEMERAL
    assert isinstance(last6, list) and "cache_control" not in last6[-1]
    assert any("clamped rolling_cache_count" in r.message for r in caplog.records)


def test_cache_control_initial_message_marks_first_user_block() -> None:
    anthropic_messages: list[dict[str, object]] = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
    ]
    apply_cache_control(
        system_blocks=None,
        anthropic_messages=anthropic_messages,
        tools=None,
        cache_config=CacheConfig(cache_initial_message=True),
        capabilities=_capabilities(supports_caching=True),
    )
    first_content = anthropic_messages[0]["content"]
    assert isinstance(first_content, list)
    assert first_content[-1].get("cache_control") == {"type": "ephemeral"}


def test_cache_control_rolling_skips_initial_and_caps_at_count() -> None:
    anthropic_messages: list[dict[str, object]] = [
        {"role": "user", "content": [{"type": "text", "text": "u0"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
        {"role": "user", "content": [{"type": "text", "text": "u2"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a3"}]},
        {"role": "user", "content": [{"type": "text", "text": "u4"}]},
    ]
    apply_cache_control(
        system_blocks=None,
        anthropic_messages=anthropic_messages,
        tools=None,
        cache_config=CacheConfig(rolling_cache_count=2),
        capabilities=_capabilities(supports_caching=True),
    )
    initial = anthropic_messages[0]["content"]
    assert isinstance(initial, list)
    assert all("cache_control" not in b for b in initial)
    for idx in (2, 4):
        content = anthropic_messages[idx]["content"]
        assert isinstance(content, list)
        assert content[-1].get("cache_control") == {"type": "ephemeral"}


def test_cache_control_noop_when_caps_off() -> None:
    anthropic_messages: list[dict[str, object]] = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ]
    system_blocks: list[dict[str, object]] = [{"type": "text", "text": "sys"}]
    apply_cache_control(
        system_blocks=system_blocks,
        anthropic_messages=anthropic_messages,
        tools=[{"name": "t"}],
        cache_config=CacheConfig(
            cache_static_prefix=True,
            cache_initial_message=True,
            rolling_cache_count=4,
        ),
        capabilities=_capabilities(supports_caching=False),
    )
    assert "cache_control" not in system_blocks[0]
    first = anthropic_messages[0]["content"]
    assert isinstance(first, list)
    assert all("cache_control" not in b for b in first)


# --- end-to-end via FakeAnthropicClient -------------------------------------


def test_end_to_end_request_shape_carries_cache_markers_and_inference_config() -> None:
    fake = FakeAnthropicClient()
    provider = AnthropicProvider(
        model_id="claude-opus-4-7",
        anthropic_client=fake,
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
    # Inference args flow through.
    assert sent["max_tokens"] == 512
    assert sent["temperature"] == 0.2
    assert sent["model"] == "claude-opus-4-7"
    # System block was promoted to typed-list shape but is NOT marked when
    # tools are present (the tools checkpoint already caches it).
    system = sent["system"]
    assert isinstance(system, list)
    assert "cache_control" not in system[-1]
    # Tools list carries the single static cache_control on the last entry.
    assert sent["tools"][-1].get("cache_control") == {"type": "ephemeral"}
    # Initial user message has cache_control on its last block.
    assert sent["messages"][0]["content"][-1].get("cache_control") == {"type": "ephemeral"}
    # 1 static (tools) + 1 initial-message, rolling=0.
    assert _count_markers(system, sent["messages"], sent["tools"]) == 2


def test_call_does_not_mutate_input_messages() -> None:
    fake = FakeAnthropicClient()
    provider = AnthropicProvider(
        model_id="claude-opus-4-7",
        anthropic_client=fake,
        transient_backoff_seconds=0.0,
    )
    msgs = [NormalizedMessage(role="user", content=[TextContent(text="hi")])]
    snapshot = [m.model_copy(deep=True) for m in msgs]
    asyncio.run(provider.call(system="s", messages=msgs))
    assert msgs == snapshot


def test_default_system_prompt_sent_as_plain_string() -> None:
    """When caching is not requested, the system prompt is passed as a
    plain string (the simpler Anthropic API shape)."""
    fake = FakeAnthropicClient()
    provider = AnthropicProvider(
        model_id="claude-opus-4-7",
        anthropic_client=fake,
        transient_backoff_seconds=0.0,
    )
    asyncio.run(
        provider.call(
            system="You are a fixture.",
            messages=[NormalizedMessage(role="user", content=[TextContent(text="hi")])],
        )
    )
    sent = fake.calls[0]
    assert sent["system"] == "You are a fixture."


def test_no_tools_argument_omits_tools_key() -> None:
    fake = FakeAnthropicClient()
    provider = AnthropicProvider(
        model_id="claude-opus-4-7",
        anthropic_client=fake,
        transient_backoff_seconds=0.0,
    )
    asyncio.run(
        provider.call(
            system="s",
            messages=[NormalizedMessage(role="user", content=[TextContent(text="hi")])],
        )
    )
    sent = fake.calls[0]
    assert "tools" not in sent
