""":func:`smai_agents.structured_call` per ``04-agents.md`` §6 / DEC-018."""

from __future__ import annotations

import pytest
from _agent_fakes import StubLlmProvider  # type: ignore[import-not-found]
from _agent_helpers import model_response  # type: ignore[import-not-found]
from pydantic import BaseModel
from smai_agents import StructuredCallFailed, structured_call


class _CodeReviewResult(BaseModel):
    overall_pass: bool
    summary: str


@pytest.mark.asyncio
async def test_structured_call_returns_parsed_pydantic_on_first_attempt() -> None:
    """Happy path: model returns ``stop_reason='tool_use'`` with valid input."""
    canned = model_response(
        tool_uses=[
            (
                "tu-1",
                "submit_review",
                {"overall_pass": True, "summary": "looks good"},
            )
        ],
        stop_reason="tool_use",
    )
    llm = StubLlmProvider([canned])

    result = await structured_call(
        llm=llm,
        system="reviewer system prompt",
        user_message="please review",
        output_schema=_CodeReviewResult,
        tool_name="submit_review",
        tool_description="Submit your structured code review verdict.",
    )

    assert isinstance(result, _CodeReviewResult)
    assert result.overall_pass is True
    assert result.summary == "looks good"
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_structured_call_retries_once_on_text_response() -> None:
    """§6 step 2: text response triggers retry-once with explicit instruction."""
    text_only = model_response(text="looks good", stop_reason="end_turn")
    valid = model_response(
        tool_uses=[
            (
                "tu-2",
                "submit_review",
                {"overall_pass": False, "summary": "fix me"},
            )
        ],
        stop_reason="tool_use",
    )
    llm = StubLlmProvider([text_only, valid])

    result = await structured_call(
        llm=llm,
        system="reviewer",
        user_message="please review",
        output_schema=_CodeReviewResult,
        tool_name="submit_review",
        tool_description="Submit verdict",
    )
    assert result.overall_pass is False
    assert len(llm.calls) == 2

    # Second call should carry the retry instruction in its messages.
    second_messages = llm.calls[1]["messages"]
    assert isinstance(second_messages, list)
    last_user = second_messages[-1]
    assert last_user["role"] == "user"
    last_user_content = last_user["content"]
    assert isinstance(last_user_content, list)
    assert "submit_review" in last_user_content[0]["text"]


@pytest.mark.asyncio
async def test_structured_call_raises_after_two_failed_attempts() -> None:
    """§6 step 3: second-attempt failure raises :class:`StructuredCallFailed`."""
    text_only_a = model_response(text="not a tool call", stop_reason="end_turn")
    text_only_b = model_response(text="still not", stop_reason="end_turn")
    llm = StubLlmProvider([text_only_a, text_only_b])

    with pytest.raises(StructuredCallFailed) as exc_info:
        await structured_call(
            llm=llm,
            system="reviewer",
            user_message="please review",
            output_schema=_CodeReviewResult,
            tool_name="submit_review",
            tool_description="Submit verdict",
        )
    err = exc_info.value
    assert err.tool_name == "submit_review"
    assert err.attempt_count == 2
    assert err.last_assistant_text == "still not"


@pytest.mark.asyncio
async def test_structured_call_retries_on_schema_invalid_payload() -> None:
    """§6 step 2 also covers payloads that fail schema validation —
    the tool was used but the input didn't match. DEC-018: no silent
    fallback; retry, then fail loudly."""
    bad_payload = model_response(
        tool_uses=[
            (
                "tu-3",
                "submit_review",
                # Missing ``summary`` (required field).
                {"overall_pass": True},
            )
        ],
        stop_reason="tool_use",
    )
    bad_payload_2 = model_response(
        tool_uses=[
            (
                "tu-4",
                "submit_review",
                # Missing ``overall_pass`` this time.
                {"summary": "still wrong"},
            )
        ],
        stop_reason="tool_use",
    )
    llm = StubLlmProvider([bad_payload, bad_payload_2])

    with pytest.raises(StructuredCallFailed) as exc_info:
        await structured_call(
            llm=llm,
            system="reviewer",
            user_message="please review",
            output_schema=_CodeReviewResult,
            tool_name="submit_review",
            tool_description="Submit verdict",
        )
    assert exc_info.value.last_validation_error is not None


@pytest.mark.asyncio
async def test_structured_call_uses_single_call_cache_defaults() -> None:
    """§5 final paragraph: rolling=0 for single-call agents."""
    canned = model_response(
        tool_uses=[
            (
                "tu-5",
                "submit_review",
                {"overall_pass": True, "summary": "ok"},
            )
        ],
        stop_reason="tool_use",
    )
    llm = StubLlmProvider([canned])

    await structured_call(
        llm=llm,
        system="reviewer",
        user_message="please review",
        output_schema=_CodeReviewResult,
        tool_name="submit_review",
        tool_description="Submit verdict",
    )
    cache_config = llm.calls[0]["cache_config"]
    assert isinstance(cache_config, dict)
    assert cache_config["rolling_cache_count"] == 0
    assert cache_config["cache_static_prefix"] is True
    assert cache_config["cache_initial_message"] is True
