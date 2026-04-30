"""Round-trip translation tests beyond what conformance covers.

The :class:`LlmProviderConformance` base verifies the contract surface;
these tests pin down the OpenAI-chat-completions-specific request
shape (system as a role-message, ``tool_calls`` as a flat list, JSON-
encoded function arguments, ``tool_call_id`` on tool-result messages,
``finish_reason`` collapse) — all of which differ from the Anthropic
shape and would silently break the agent loop's tool-execution branch
if regressed.
"""

from __future__ import annotations

import asyncio
import json

from _f5_openai_fakes import FakeOpenAIClient  # type: ignore[import-not-found]
from smai_core.plugins import (
    NormalizedMessage,
    TextContent,
    ToolDefinition,
    ToolResultContent,
    ToolUseContent,
)
from smai_llm_openai import OpenAIProvider
from smai_llm_openai._translation import (
    from_openai_response,
    to_openai_messages,
    to_openai_tools,
)

# --- pure translation: NormalizedMessage → OpenAI messages ------------------


def test_system_prompt_becomes_leading_system_role_message() -> None:
    out = to_openai_messages(
        system="You are a fixture.",
        messages=[NormalizedMessage(role="user", content=[TextContent(text="hi")])],
    )
    assert out[0] == {"role": "system", "content": "You are a fixture."}


def test_empty_system_prompt_omits_system_message() -> None:
    out = to_openai_messages(
        system="",
        messages=[NormalizedMessage(role="user", content=[TextContent(text="hi")])],
    )
    # Only the user message; no leading system entry.
    assert out == [{"role": "user", "content": "hi"}]


def test_assistant_text_becomes_content_string() -> None:
    out = to_openai_messages(
        system="",
        messages=[NormalizedMessage(role="assistant", content=[TextContent(text="reply")])],
    )
    assert out == [{"role": "assistant", "content": "reply"}]


def test_assistant_tool_use_becomes_flat_tool_calls_list() -> None:
    """OpenAI distinguishes assistant text from tool calls — text is on
    ``content``, tool_use lifts to a top-level ``tool_calls`` list with
    JSON-encoded ``arguments`` strings."""
    out = to_openai_messages(
        system="",
        messages=[
            NormalizedMessage(
                role="assistant",
                content=[
                    TextContent(text="thinking..."),
                    ToolUseContent(id="call_1", name="echo", input={"text": "hi"}),
                ],
            )
        ],
    )
    assert out == [
        {
            "role": "assistant",
            "content": "thinking...",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "arguments": json.dumps({"text": "hi"}),
                    },
                }
            ],
        }
    ]


def test_assistant_tool_use_only_sets_content_to_none() -> None:
    """OpenAI accepts ``content=None`` when only tool calls are emitted."""
    out = to_openai_messages(
        system="",
        messages=[
            NormalizedMessage(
                role="assistant",
                content=[ToolUseContent(id="call_1", name="echo", input={"text": "hi"})],
            )
        ],
    )
    assert out[0]["content"] is None
    assert len(out[0]["tool_calls"]) == 1


def test_tool_results_become_separate_role_tool_messages() -> None:
    """A user message carrying tool_result blocks splits into one
    ``role=tool`` message per result, keyed by ``tool_call_id``."""
    out = to_openai_messages(
        system="",
        messages=[
            NormalizedMessage(
                role="user",
                content=[
                    ToolResultContent(tool_use_id="call_1", content="ok", is_error=False),
                    ToolResultContent(tool_use_id="call_2", content="bad", is_error=True),
                ],
            )
        ],
    )
    assert out == [
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {"role": "tool", "tool_call_id": "call_2", "content": "bad"},
    ]


def test_user_text_plus_tool_results_emit_user_then_tool_messages() -> None:
    """Mixed text + tool_result on a user message splits into a leading
    user message + one tool message per result."""
    out = to_openai_messages(
        system="",
        messages=[
            NormalizedMessage(
                role="user",
                content=[
                    TextContent(text="follow-up"),
                    ToolResultContent(tool_use_id="call_1", content="ok"),
                ],
            )
        ],
    )
    assert out == [
        {"role": "user", "content": "follow-up"},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]


def test_tool_definitions_wrap_in_function_object() -> None:
    """OpenAI's tool definitions wrap in ``{"type": "function",
    "function": {...}}``; the JSON-schema field is ``parameters``
    (not Anthropic's ``input_schema``)."""
    out = to_openai_tools(
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
            "type": "function",
            "function": {
                "name": "t",
                "description": "d",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


# --- pure translation: OpenAI response → ModelResponse ----------------------


def test_response_text_translates_to_text_content() -> None:
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "hello back"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }
    out = from_openai_response(response)
    assert out.message.role == "assistant"
    assert len(out.message.content) == 1
    block = out.message.content[0]
    assert isinstance(block, TextContent)
    assert block.text == "hello back"
    assert out.stop_reason == "end_turn"
    assert out.usage.input_tokens == 4
    assert out.usage.output_tokens == 2


def test_response_tool_calls_translate_with_json_decoded_args() -> None:
    """The wire format encodes ``arguments`` as a JSON string; the
    plugin must decode it back into a dict."""
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_99",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"text": "hi"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8},
    }
    out = from_openai_response(response)
    assert out.stop_reason == "tool_use"
    assert len(out.message.content) == 1
    block = out.message.content[0]
    assert isinstance(block, ToolUseContent)
    assert block.id == "call_99"
    assert block.name == "echo"
    assert block.input == {"text": "hi"}


def test_malformed_tool_call_arguments_fall_back_to_empty_dict() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_x",
                            "type": "function",
                            "function": {"name": "echo", "arguments": "not json"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    out = from_openai_response(response)
    block = out.message.content[0]
    assert isinstance(block, ToolUseContent)
    assert block.input == {}


def test_finish_reason_length_maps_to_max_tokens() -> None:
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "truncated"},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    out = from_openai_response(response)
    assert out.stop_reason == "max_tokens"


def test_cached_tokens_surface_under_cache_read_field() -> None:
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 80},
        },
    }
    out = from_openai_response(response)
    assert out.usage.cache_read_tokens == 80
    # OpenAI does not surface a "cache write" counter — automatic
    # caching is opaque, so we don't fabricate one.
    assert out.usage.cache_write_tokens == 0


# --- end-to-end via FakeOpenAIClient ----------------------------------------


def test_end_to_end_request_shape_carries_messages_and_tools() -> None:
    fake = FakeOpenAIClient()
    provider = OpenAIProvider(
        model_id="gpt-4o",
        openai_client=fake,
        transient_backoff_seconds=0.0,
    )
    asyncio.run(
        provider.call(
            system="You are a research assistant.",
            messages=[NormalizedMessage(role="user", content=[TextContent(text="ping")])],
            tools=[ToolDefinition(name="t", description="d", input_schema={"type": "object"})],
            max_tokens=512,
            temperature=0.2,
        )
    )
    assert len(fake.calls) == 1
    sent = fake.calls[0]
    assert sent["model"] == "gpt-4o"
    assert sent["max_tokens"] == 512
    assert sent["temperature"] == 0.2
    # System prompt rolls into the leading messages entry.
    assert sent["messages"][0] == {
        "role": "system",
        "content": "You are a research assistant.",
    }
    assert sent["messages"][1] == {"role": "user", "content": "ping"}
    # Tool definition is wrapped.
    assert sent["tools"][0]["type"] == "function"
    assert sent["tools"][0]["function"]["name"] == "t"


def test_call_does_not_mutate_input_messages() -> None:
    fake = FakeOpenAIClient()
    provider = OpenAIProvider(
        model_id="gpt-4o",
        openai_client=fake,
        transient_backoff_seconds=0.0,
    )
    msgs = [NormalizedMessage(role="user", content=[TextContent(text="hi")])]
    snapshot = [m.model_copy(deep=True) for m in msgs]
    asyncio.run(provider.call(system="s", messages=msgs))
    assert msgs == snapshot


def test_cache_config_silently_ignored() -> None:
    """Per §4.3 contract: when ``supports_caching`` is False, the
    plugin silently ignores ``cache_config``. No exception, no crash,
    no markers in the request."""
    from smai_core.plugins import CacheConfig

    fake = FakeOpenAIClient()
    provider = OpenAIProvider(
        model_id="gpt-4o",
        openai_client=fake,
        transient_backoff_seconds=0.0,
    )
    asyncio.run(
        provider.call(
            system="s",
            messages=[NormalizedMessage(role="user", content=[TextContent(text="hi")])],
            cache_config=CacheConfig(
                cache_static_prefix=True,
                cache_initial_message=True,
                rolling_cache_count=4,
            ),
        )
    )
    sent = fake.calls[0]
    # No cache markers should appear anywhere in the request.
    assert "cache_control" not in str(sent)
    assert "cachePoint" not in str(sent)


def test_no_tools_argument_omits_tools_key() -> None:
    fake = FakeOpenAIClient()
    provider = OpenAIProvider(
        model_id="gpt-4o",
        openai_client=fake,
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
